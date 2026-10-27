"""@ZeroDrift's f32 term conversion, before and after it stopped going through f64.

Compiles `rowsum.ysu` with two compilers (the one before the change and the one after), checks
that their `rowsum_zd` give the same bits as each other and as the documented semantics, over
10 summation orders, then times plain f32, the old and the new @ZeroDrift kernel.

    python3 bench.py <old Y binary> <new Y binary> <.ysu_hw_profile> <out dir>

Data: z-score-like f32 values, N(0, 3) clipped to +-10, 978 columns, seeded; 46,226 rows
(bandwidth-bound) and 4,096 rows (16 blocks of 256: latency-bound). Timing: minimum over 25
interleaved rounds of 20 launches after a 2 s clock ramp; the GPU's state and any running game
are recorded before and after.
"""
import json, os, re, shutil, subprocess, sys, time
import numpy as np
import torch

OLD, NEW, PROFILE, OUT = sys.argv[1:5]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python"))
from y_lang.torch_interop import TorchKernel

dev = torch.device("cuda")
torch.zeros(1, device=dev)
os.makedirs(OUT, exist_ok=True)
res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}


def build(tag, ybin):
    d = os.path.join(OUT, f"build_{tag}")
    os.makedirs(d, exist_ok=True)
    shutil.copy(os.path.join(HERE, "rowsum.ysu"), d)
    shutil.copy(PROFILE, os.path.join(d, ".ysu_hw_profile"))
    subprocess.run([ybin, "rowsum.ysu", "--emit-ptx", "-o", "rowsum.ptx"], cwd=d, check=True, capture_output=True)
    ptx = os.path.join(d, "rowsum.ptx")
    shutil.copy(ptx, os.path.join(OUT, f"rowsum_{tag}.ptx"))
    subprocess.run(["ptxas", "-arch=sm_89", ptx, "-o", ptx + ".cubin"], check=True)
    sass = subprocess.run(["cuobjdump", "-sass", "-fun", "rowsum_zd", ptx + ".cubin"], capture_output=True, text=True).stdout
    names = [m.group(2) for m in re.finditer(r"^\s+/\*[0-9a-f]+\*/\s+(@!?P\d+\s+)?([A-Z0-9_.]+)", sass, re.M)]
    fp64 = [n for n in names if n.startswith(("DADD", "DMUL", "DFMA", "DSETP")) or ".F64" in n]
    res[f"sass_rowsum_zd_{tag}"] = {"instructions": len(names), "fp64": len(fp64)}
    shutil.rmtree(d)
    return ptx + ".cubin", open(os.path.join(OUT, f"rowsum_{tag}.ptx")).read()


def kernel(ptx, name):
    k = TorchKernel.__new__(TorchKernel)
    k.source, k.kernel_name, k.target_sm, k.ptx = None, name, "sm_89", ptx
    k._module = k._function = None
    k._ensure_loaded()
    return k


_, ptx_old = build("old", OLD)
_, ptx_new = build("new", NEW)
K = {"f32": kernel(ptx_new, "rowsum_f32"), "zd old": kernel(ptx_old, "rowsum_zd"), "zd new": kernel(ptx_new, "rowsum_zd")}

R, N = 46226, 978
Z = np.clip(np.random.default_rng(2027).standard_normal((R, N)) * 3.0, -10, 10).astype(np.float32)
zq = Z.astype(np.float64) * 2.0 ** 32
spec = ((np.sign(zq) * np.floor(np.abs(zq) + 0.5)).astype(np.int64).sum(1).astype(np.float64) / 2.0 ** 32).astype(np.float32)
rng = np.random.default_rng(2026)
orders = [np.arange(N), np.arange(N)[::-1]] + [rng.permutation(N) for _ in range(8)]
ZT = torch.from_numpy(np.ascontiguousarray(Z.T)).to(dev)
B = 256


def launch(k, zt, rows, perm):
    out = torch.empty(rows, dtype=torch.float32, device=dev)
    p = torch.from_numpy(np.ascontiguousarray(perm, dtype=np.int32)).to(dev)
    args = [zt, p, out, rows, N]
    return out, (lambda: k.launch(((rows + B - 1) // B, 1, 1), (B, 1, 1), args))


bits = {}
for tag in K:
    rows_out = []
    for perm in orders:
        out, f = launch(K[tag], ZT, R, perm)
        f()
        torch.cuda.synchronize()
        rows_out.append(out.cpu().numpy().view(np.int32).copy())
    bits[tag] = np.stack(rows_out)
res["rows"], res["orders"] = R, len(orders)
res["zd_new_equals_zd_old"] = int((bits["zd new"] == bits["zd old"]).sum())
res["zd_new_equals_semantics"] = int((bits["zd new"] == spec.view(np.int32)).sum())
res["rows_depending_on_order"] = {t: int((b != b[0]).any(0).sum()) for t, b in bits.items()}
res["sums_compared"] = int(bits["zd new"].size)


def state():
    games = subprocess.run(["bash", "-c", os.environ.get("GAMES_CMD", "true")], capture_output=True, text=True).stdout.strip()
    smi = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return {"games": games, "nvidia_smi": smi}


ZTs = torch.from_numpy(np.ascontiguousarray(Z[:4096].T)).to(dev)
cands = {}
for regime, zt, rows in (("46,226 rows (bandwidth-bound)", ZT, R), ("4,096 rows (latency-bound)", ZTs, 4096)):
    for tag in K:
        cands[(regime, tag)] = launch(K[tag], zt, rows, orders[0])[1]


def timed(f, iters=20):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        f()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0


res["state_before"] = state()
t_end = time.time() + 2.0
while time.time() < t_end:
    for f in cands.values():
        f()
torch.cuda.synchronize()
allt = {k: [] for k in cands}
for _ in range(25):
    for k, f in cands.items():
        allt[k].append(timed(f))
res["state_after"] = state()
res["timing_us"] = {}
for (regime, tag), ts in allt.items():
    base = min(allt[(regime, "f32")])
    res["timing_us"].setdefault(regime, {})[tag] = {"min": min(ts), "median": float(np.median(ts)), "min_vs_f32": min(ts) / base,
                                                     "all": ts}
for k, v in res.items():
    if k != "timing_us":
        print(f"{k}: {v}")
for regime, d in res["timing_us"].items():
    for tag, v in d.items():
        print(f"{regime:32s} {tag:7s} min {v['min']:8.1f} us  median {v['median']:8.1f} us  x{v['min_vs_f32']:.2f} of f32")
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
