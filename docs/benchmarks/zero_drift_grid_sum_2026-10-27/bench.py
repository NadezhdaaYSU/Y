"""The cost of an exact grid-wide sum (`drift_add_to`), against torch.sum and a plain-f32 two-pass sum.

Correctness first: every `zd_sum` launch must store the exact total computed here (each term rounded
to Q32.32, ties away from zero, summed in Python integers, read back through f64), and three launch
shapes must agree bit for bit. Then each variant is timed with CUDA events, interleaved, REPS rounds;
each interval times INNER launches back to back (the per-launch mean is recorded); the minimum ranks them and the median is kept beside it.

Usage: bench.py <grid_sum.ptx> <out dir>
"""
import json, os, subprocess, sys, time
import numpy as np
import torch

PTX, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python"))
from y_lang.torch_interop import TorchKernel

dev = torch.device("cuda")
torch.zeros(1, device=dev)
REPS = 60
INNER = 10  # launches per timed interval, back to back, so the clocks stay up
SIZES = [1 << 20, 1 << 24, 1 << 26]
SMS = torch.cuda.get_device_properties(0).multi_processor_count
BLOCK = 256
GRID = SMS * 8


def load(name):
    k = TorchKernel.__new__(TorchKernel)
    k.source, k.kernel_name, k.target_sm, k.ptx = None, name, "sm_89", open(PTX).read()
    k._module = k._function = None
    k._ensure_loaded()
    return k


def util():
    return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm,clocks.mem", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


def who():
    """Per-process SM use over 5 samples (`nvidia-smi pmon`): which process loads the GPU."""
    out = subprocess.run(["nvidia-smi", "pmon", "-c", "5", "-s", "u"], capture_output=True, text=True).stdout
    rows = {}
    for line in out.splitlines():
        w = line.split()
        if len(w) >= 10 and not line.startswith("#") and w[3] != "-":
            rows.setdefault(f"{w[1]} {w[-1]}", []).append(int(w[3]))
    return {k: max(v) for k, v in sorted(rows.items())}


def exact(x):
    y = x.double() * 2.0 ** 32
    q = (torch.sign(y) * torch.floor(y.abs() + 0.5)).to(torch.int64)
    t = int(q.cpu().numpy().astype(object).sum())
    return t, np.float32(np.float64(t) * 2.0 ** -32)


zd, part = load("zd_sum"), load("f32_partials")
acc = torch.zeros(3, dtype=torch.int64, device=dev)
res = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "block": BLOCK, "grid": GRID,
       "reps": REPS, "inner": INNER, "util_before": util(), "sm_by_process_before": who(), "sizes": {}}
# Bring the clocks up first: the first size used to be timed while they were still ramping
# (run_2026-10-28 first attempt: torch.sum took 28.9 us on 4 MB where it takes 7.4).
warm = torch.empty(1 << 26, device=dev)
t_warm = time.time()
while time.time() - t_warm < 1.5:
    warm.sum()
torch.cuda.synchronize()
del warm
res["clocks_after_warmup"] = util()
g = torch.Generator(device="cpu").manual_seed(7)
for n in SIZES:
    x = (torch.randn(n, generator=g) * 0.3).float().to(dev)
    t, want = exact(x)
    out = torch.empty(1, dtype=torch.float32, device=dev)
    p = torch.empty(2, dtype=torch.int32, device=dev)

    def run_zd(grid=GRID, block=BLOCK):
        threads = grid * block
        p.copy_(torch.tensor([(n + threads - 1) // threads, n], dtype=torch.int32))
        zd.launch((grid, 1, 1), (block, 1, 1), [x, acc, out, p])

    partials = torch.empty(GRID * BLOCK, dtype=torch.float32, device=dev)
    p2 = torch.tensor([(n + GRID * BLOCK - 1) // (GRID * BLOCK), n], dtype=torch.int32, device=dev)

    def run_two_pass():
        part.launch((GRID, 1, 1), (BLOCK, 1, 1), [x, acc, partials, p2])
        return partials.sum()

    # correctness: three shapes, bit for bit, and the exact total
    seen = set()
    for grid, block in [(GRID, BLOCK), (37, 1000), (5 * SMS, 64)]:
        out.fill_(7.0)
        run_zd(grid, block)
        torch.cuda.synchronize()
        seen.add(out.view(torch.int32).item())
    assert seen == {int(np.array(want).view(np.int32))}, (seen, want)
    assert acc.abs().sum().item() == 0
    p.copy_(torch.tensor([(n + GRID * BLOCK - 1) // (GRID * BLOCK), n], dtype=torch.int32))
    variants = {"torch.sum": lambda: x.sum(), "f32 two-pass": run_two_pass, "zd_sum": lambda: zd.launch((GRID, 1, 1), (BLOCK, 1, 1), [x, acc, out, p])}
    times = {k: [] for k in variants}
    for _ in range(5):
        for f in variants.values():
            f()
    torch.cuda.synchronize()
    for r in range(REPS):
        for k, f in variants.items():
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(INNER):
                f()
            e1.record()
            torch.cuda.synchronize()
            times[k].append(e0.elapsed_time(e1) * 1000.0 / INNER)
    # every timed launch left the right total and a zero ACC behind
    assert out.view(torch.int32).item() == int(np.array(want).view(np.int32)) and acc.abs().sum().item() == 0
    ts = x.sum().item()
    tp = run_two_pass().item()
    row = {"exact_total_raw": t, "zd_out": float(want), "torch_sum": ts, "two_pass": tp,
           "torch_sum_err": ts - float(t) * 2.0 ** -32, "two_pass_err": tp - float(t) * 2.0 ** -32,
           "us": {k: {"min": min(v), "median": float(np.median(v)), "all": v} for k, v in times.items()}}
    gb = n * 4 / 1e9
    for k, v in times.items():
        print(f"n={n:>9} {k:>13}: min {min(v):8.1f} us  median {np.median(v):8.1f} us  ({gb / (min(v) * 1e-6):6.1f} GB/s)", flush=True)
    print(f"            exact {float(want)!r}  torch.sum {ts!r}  two-pass {tp!r}", flush=True)
    res["sizes"][str(n)] = row
res["util_after"] = util()
res["sm_by_process_after"] = who()
print("util before", res["util_before"], "after", res["util_after"])
print("SM % by process (max of 5 samples), before", res["sm_by_process_before"], "after", res["sm_by_process_after"])
os.makedirs(OUT, exist_ok=True)
json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)
