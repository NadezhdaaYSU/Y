"""Where a `@ZeroDrift` loop stops, before and after it moved to the loop's end (2026-10-28).

Compiles `rowsum.ysu` and `grid_sum.ysu` with the compiler before the change and the one after,
and for each kernel records its SASS (loads and traps). Then, before any timing:
- row sums: every `rowsum_zd` sum, over 10 orders, equals the documented semantics, for both
  compilers;
- grid sums: `zd_sum` and `zd_split` store the exact total at three launch shapes, both compilers;
- stops: both compilers' `rowsum_zd` stop on a term of 1e30, a NaN, a total out of range, and
  partial sums out of range with a total of 0, and run on good input; the grid sums stop on a
  NaN. Each case in a child process, as a trap leaves the device unusable for its process.
Then the timings: row sums as `docs/benchmarks/zero_drift_conversion_2026-10-27/` (minimum of 25
interleaved rounds of 20 launches, after a 2 s clock ramp) and grid sums as
`docs/benchmarks/zero_drift_grid_sum_2026-10-27/` (60 rounds of 10, after 1.5 s), with the
GPU's state, any game and the load per process before and after.

    python3 bench.py <old Y> <new Y> <.ysu_hw_profile> <out dir> [games script]
"""
import json, os, re, shutil, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python"))


def kernel(ptx, name):
    from y_lang.torch_interop import TorchKernel
    k = TorchKernel.__new__(TorchKernel)
    k.source, k.kernel_name, k.target_sm, k.ptx = None, name, "sm_89", ptx
    k._module = k._function = None
    k._ensure_loaded()
    return k


def stop_case(ptx_path, name, case):
    """One launch in this process (a child): prints STOPPED or the row's value."""
    import torch
    dev = torch.device("cuda")
    if name.startswith("rowsum"):
        n = 978
        z = np.zeros((4096, n), np.float32)
        if case == "term": z[100, 500] = 1e30
        if case == "nan": z[100, 500] = np.nan
        if case == "total": z[100, :] = 20.0
        if case == "inside": z[100, :489] = 25.0; z[100, 489:] = -25.0
        if case == "good": z[100, :] = 0.5
        zt = torch.from_numpy(np.ascontiguousarray(z.T)).to(dev)
        out = torch.full((4096,), 7.0, device=dev)
        args = [zt, torch.arange(n, dtype=torch.int32, device=dev), out, 4096, n]
        grid, block = (16, 1, 1), (256, 1, 1)
    else:
        n = 100_003
        x = torch.full((n,), 0.25, device=dev)
        if case == "nan": x[77_777] = float("nan")
        acc, out = torch.zeros(3, dtype=torch.int64, device=dev), torch.full((1,), 7.0, device=dev)
        threads = 37 * 256
        args = [x, acc, out, torch.tensor([(n + threads - 1) // threads, n, n // threads], dtype=torch.int32, device=dev)]
        grid, block = (37, 1, 1), (256, 1, 1)
    try:
        kernel(open(ptx_path).read(), name).launch(grid, block, args)
        torch.cuda.synchronize()
        print("RAN", out[100 if name.startswith("rowsum") else 0].item())
    except Exception as e:
        print("STOPPED", type(e).__name__)


if len(sys.argv) > 1 and sys.argv[1] == "--stop":
    stop_case(*sys.argv[2:5])
    sys.exit(0)

OLD, NEW, PROFILE, OUT = sys.argv[1:5]
GAMES = sys.argv[5] if len(sys.argv) > 5 else None
import torch

dev = torch.device("cuda")
torch.zeros(1, device=dev)
os.makedirs(OUT, exist_ok=True)
res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}


def sass_stats(cubin, name):
    sass = subprocess.run(["cuobjdump", "-sass", "-fun", name, cubin], capture_output=True, text=True).stdout
    ops = [m.group(2) for m in re.finditer(r"^\s+/\*[0-9a-f]+\*/\s+(@!?P\d+\s+)?([A-Z0-9_.]+)", sass, re.M)]
    return {"instructions": len([o for o in ops if o != "NOP"]), "LDG": sum(o.startswith("LDG") for o in ops),
            "BPT.TRAP": sum(o.startswith("BPT.TRAP") for o in ops)}


def build(tag, ybin):
    d = os.path.join(OUT, f"build_{tag}")
    os.makedirs(d, exist_ok=True)
    shutil.copy(PROFILE, os.path.join(d, ".ysu_hw_profile"))
    ptx = {}
    for src in ("rowsum", "grid_sum"):
        shutil.copy(os.path.join(HERE, f"{src}.ysu"), d)
        subprocess.run([ybin, f"{src}.ysu", "--emit-ptx", "-o", f"{src}.ptx"], cwd=d, check=True, capture_output=True)
        p = os.path.join(d, f"{src}.ptx")
        shutil.copy(p, os.path.join(OUT, f"{src}_{tag}.ptx"))
        subprocess.run(["ptxas", "-arch=sm_89", p, "-o", p + ".cubin"], check=True)
        names = ["rowsum_f32", "rowsum_zd"] if src == "rowsum" else ["zd_sum", "zd_split", "f32_partials"]
        for name in names:
            res.setdefault("sass", {})[f"{name} {tag}"] = sass_stats(p + ".cubin", name)
        ptx[src] = open(p).read()
    shutil.rmtree(d)
    return ptx


P = {"old": build("old", OLD), "new": build("new", NEW)}

# ---- stops, in children
res["stops"] = {}
for tag in P:
    for name, src, cases in (("rowsum_zd", "rowsum", ["term", "nan", "total", "inside", "good"]),
                             ("zd_sum", "grid_sum", ["nan", "good"]), ("zd_split", "grid_sum", ["nan", "good"])):
        for case in cases:
            r = subprocess.run([sys.executable, os.path.abspath(__file__), "--stop", os.path.join(OUT, f"{src}_{tag}.ptx"), name, case],
                               capture_output=True, text=True)
            line = [l for l in r.stdout.splitlines() if l.startswith(("RAN", "STOPPED"))]
            res["stops"][f"{name} {tag} {case}"] = line[0] if line else "NO ANSWER: " + r.stderr[-300:]

# ---- row sums: correctness
R, N = 46226, 978
Z = np.clip(np.random.default_rng(2027).standard_normal((R, N)) * 3.0, -10, 10).astype(np.float32)
zq = Z.astype(np.float64) * 2.0 ** 32
spec = ((np.sign(zq) * np.floor(np.abs(zq) + 0.5)).astype(np.int64).sum(1).astype(np.float64) / 2.0 ** 32).astype(np.float32)
rng = np.random.default_rng(2026)
orders = [np.arange(N), np.arange(N)[::-1]] + [rng.permutation(N) for _ in range(8)]
ZT = torch.from_numpy(np.ascontiguousarray(Z.T)).to(dev)
ZTs = torch.from_numpy(np.ascontiguousarray(Z[:4096].T)).to(dev)
B = 256
RK = {"f32": kernel(P["new"]["rowsum"], "rowsum_f32"), "zd old": kernel(P["old"]["rowsum"], "rowsum_zd"),
      "zd new": kernel(P["new"]["rowsum"], "rowsum_zd")}


def row_launch(k, zt, rows, perm):
    out = torch.empty(rows, dtype=torch.float32, device=dev)
    p = torch.from_numpy(np.ascontiguousarray(perm, dtype=np.int32)).to(dev)
    args = [zt, p, out, rows, N]
    return out, (lambda: k.launch(((rows + B - 1) // B, 1, 1), (B, 1, 1), args))


for tag in ("zd old", "zd new"):
    ok = 0
    for perm in orders:
        out, f = row_launch(RK[tag], ZT, R, perm)
        f()
        torch.cuda.synchronize()
        ok += int((out.cpu().numpy().view(np.int32) == spec.view(np.int32)).sum())
    res[f"row sums equal to the semantics, {tag}"] = f"{ok}/{R * len(orders)}"

# ---- grid sums: correctness
SMS = torch.cuda.get_device_properties(0).multi_processor_count
GRID = SMS * 8
GK = {f"{n} {t}": kernel(P[t]["grid_sum"], n) for n in ("zd_sum", "zd_split") for t in ("old", "new")}
part = kernel(P["new"]["grid_sum"], "f32_partials")


def exact(x):
    y = x.double() * 2.0 ** 32
    q = (torch.sign(y) * torch.floor(y.abs() + 0.5)).to(torch.int64)
    t = int(q.cpu().numpy().astype(object).sum())
    return t, np.float32(np.float64(t) * 2.0 ** -32)


def params(n, threads):
    return torch.tensor([(n + threads - 1) // threads, n, n // threads], dtype=torch.int32, device=dev)


g = torch.Generator(device="cpu").manual_seed(7)
SIZES = [1 << 20, 1 << 24, 1 << 26]
X = {n: (torch.randn(n, generator=g) * 0.3).float().to(dev) for n in SIZES}
acc = torch.zeros(3, dtype=torch.int64, device=dev)
gout = torch.empty(1, dtype=torch.float32, device=dev)
res["grid sums exact"] = {}
for n, x in X.items():
    _, want = exact(x)
    for name, k in GK.items():
        seen = set()
        for grid, block in [(GRID, B), (37, 1000), (5 * SMS, 64)]:
            gout.fill_(7.0)
            k.launch((grid, 1, 1), (block, 1, 1), [x, acc, gout, params(n, grid * block)])
            torch.cuda.synchronize()
            seen.add(gout.view(torch.int32).item())
        res["grid sums exact"][f"{n} {name}"] = seen == {int(np.array(want).view(np.int32))} and acc.abs().sum().item() == 0


def state():
    smi = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm,clocks.mem", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    out = subprocess.run(["nvidia-smi", "pmon", "-c", "5", "-s", "u"], capture_output=True, text=True).stdout
    rows = {}
    for line in out.splitlines():
        w = line.split()
        if len(w) >= 10 and not line.startswith("#") and w[3] != "-":
            rows.setdefault(f"{w[1]} {w[-1]}", []).append(int(w[3]))
    games = subprocess.run(["bash", GAMES], capture_output=True, text=True).stdout.strip() if GAMES else "not checked"
    return {"nvidia_smi": smi, "sm_by_process": {k: max(v) for k, v in sorted(rows.items())}, "games": games}


def timed(f, iters):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        f()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0


res["state_before"] = state()
# ---- row sums: timing
cands = {}
for regime, zt, rows in (("46,226 rows (bandwidth-bound)", ZT, R), ("4,096 rows (latency-bound)", ZTs, 4096)):
    for tag in RK:
        cands[(regime, tag)] = row_launch(RK[tag], zt, rows, orders[0])[1]
t_end = time.time() + 2.0
while time.time() < t_end:
    for f in cands.values():
        f()
torch.cuda.synchronize()
allt = {k: [] for k in cands}
for _ in range(25):
    for k, f in cands.items():
        allt[k].append(timed(f, 20))
res["row_us"] = {}
for (regime, tag), ts in allt.items():
    base = min(allt[(regime, "f32")])
    res["row_us"].setdefault(regime, {})[tag] = {"min": min(ts), "median": float(np.median(ts)), "x_f32": min(ts) / base, "all": ts}
# ---- grid sums: timing
warm = torch.empty(1 << 26, device=dev)
t_end = time.time() + 1.5
while time.time() < t_end:
    warm.sum()
torch.cuda.synchronize()
del warm
res["grid_us"] = {}
for n, x in X.items():
    pp = params(n, GRID * B)
    partials = torch.empty(GRID * B, dtype=torch.float32, device=dev)
    variants = {"torch.sum": lambda: x.sum(),
                "f32 two-pass": lambda: (part.launch((GRID, 1, 1), (B, 1, 1), [x, acc, partials, pp]), partials.sum())}
    for name, k in GK.items():
        variants[name] = (lambda k=k: k.launch((GRID, 1, 1), (B, 1, 1), [x, acc, gout, pp]))
    times = {k: [] for k in variants}
    for _ in range(5):
        for f in variants.values():
            f()
    torch.cuda.synchronize()
    for _ in range(60):
        for k, f in variants.items():
            times[k].append(timed(f, 10))
    _, want = exact(x)
    assert gout.view(torch.int32).item() == int(np.array(want).view(np.int32)) and acc.abs().sum().item() == 0
    base = min(times["torch.sum"])
    res["grid_us"][str(n)] = {k: {"min": min(v), "median": float(np.median(v)), "x_torch_sum": min(v) / base, "all": v}
                              for k, v in times.items()}
res["state_after"] = state()

for k, v in res.items():
    if k not in ("row_us", "grid_us"):
        print(f"{k}: {v}")
for regime, d in res["row_us"].items():
    for tag, v in d.items():
        print(f"{regime:32s} {tag:7s} min {v['min']:8.1f} us  median {v['median']:8.1f} us  x{v['x_f32']:.2f} of f32")
for n, d in res["grid_us"].items():
    for tag, v in d.items():
        print(f"n={int(n):>9} {tag:13s} min {v['min']:8.1f} us  median {v['median']:8.1f} us  "
              f"{int(n) * 4 / 1e9 / (v['min'] * 1e-6):6.1f} GB/s  x{v['x_torch_sum']:.2f} of torch.sum")
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
