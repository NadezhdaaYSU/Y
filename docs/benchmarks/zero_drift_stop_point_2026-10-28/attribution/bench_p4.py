"""Plan #4 probe: does deferring rowsum_zd's two in-loop traps to one after the loop recover
the latency-bound cost? Kernels: f32, zd (compiled), zd_flag, zd_nocheck (hand-edited).
Usage: bench_p4.py <rowsum.ptx> <var.ptx> <out json> [--trap CASE]"""
import json, os, subprocess, sys, time
import numpy as np
import torch
sys.path.insert(0, os.environ["Y_PYTHON"])
from y_lang.torch_interop import TorchKernel
dev = torch.device("cuda"); torch.zeros(1, device=dev)
def kernel(path, name):
    k = TorchKernel.__new__(TorchKernel)
    k.source, k.kernel_name, k.target_sm, k.ptx = None, name, "sm_89", open(path).read()
    k._module = k._function = None; k._ensure_loaded(); return k
A, V = sys.argv[1], sys.argv[2]
N, B = 978, 256
if "--trap" in sys.argv:
    case, name = sys.argv[sys.argv.index("--trap") + 1], sys.argv[sys.argv.index("--trap") + 2]
    k = kernel(A if name == "rowsum_zd" else V, name)
    Z = np.zeros((4096, N), np.float32)
    if case == "term": Z[100, 500] = 1e30
    if case == "nan": Z[100, 500] = np.nan
    if case == "total": Z[100, :] = 20.0
    if case == "inside": Z[100, :489] = 25.0; Z[100, 489:] = -25.0   # partial sums leave the bounds, total 0
    ZT = torch.from_numpy(np.ascontiguousarray(Z.T)).to(dev)
    out = torch.full((4096,), 7.0, device=dev); p = torch.arange(N, dtype=torch.int32, device=dev)
    k.launch((16, 1, 1), (B, 1, 1), [ZT, p, out, 4096, N])
    torch.cuda.synchronize()
    print("RESULT no stop; row 100 =", out[100].item()); sys.exit(0)
K = {"f32": kernel(A, "rowsum_f32"), "zd": kernel(A, "rowsum_zd"),
     "zd_flag": kernel(V, "rowsum_zd_flag"), "zd_nocheck": kernel(V, "rowsum_zd_nocheck")}
R = 46226
Z = np.clip(np.random.default_rng(2027).standard_normal((R, N)) * 3.0, -10, 10).astype(np.float32)
zq = Z.astype(np.float64) * 2.0 ** 32
spec = ((np.sign(zq) * np.floor(np.abs(zq) + 0.5)).astype(np.int64).sum(1).astype(np.float64) / 2.0 ** 32).astype(np.float32)
orders = [np.arange(N), np.arange(N)[::-1]] + [np.random.default_rng(2026 + i).permutation(N) for i in range(3)]
res = {}
ZT = torch.from_numpy(np.ascontiguousarray(Z.T)).to(dev)
ZTs = torch.from_numpy(np.ascontiguousarray(Z[:4096].T)).to(dev)
def launch(k, zt, rows, perm):
    out = torch.empty(rows, dtype=torch.float32, device=dev)
    p = torch.from_numpy(np.ascontiguousarray(perm, dtype=np.int32)).to(dev)
    args = [zt, p, out, rows, N]
    return out, (lambda: k.launch(((rows + B - 1) // B, 1, 1), (B, 1, 1), args))
for tag in ("zd", "zd_flag", "zd_nocheck"):
    ok = 0
    for perm in orders:
        out, f = launch(K[tag], ZT, R, perm); f(); torch.cuda.synchronize()
        ok += int((out.cpu().numpy().view(np.int32) == spec.view(np.int32)).sum())
    res[f"{tag}_equal_to_semantics"] = f"{ok}/{R * len(orders)}"
def state():
    g = subprocess.run(["bash", os.environ["GAMES"]], capture_output=True, text=True).stdout.strip()
    s = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return {"games": g, "smi": s}
cands = {}
for regime, zt, rows in (("46226 rows", ZT, R), ("4096 rows", ZTs, 4096)):
    for tag in K: cands[(regime, tag)] = launch(K[tag], zt, rows, orders[0])[1]
def timed(f, iters=20):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): f()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0
res["state_before"] = state()
t_end = time.time() + 2.0
while time.time() < t_end:
    for f in cands.values(): f()
torch.cuda.synchronize()
allt = {k: [] for k in cands}
for _ in range(25):
    for k, f in cands.items(): allt[k].append(timed(f))
res["state_after"] = state()
res["timing_us"] = {f"{r} {t}": {"min": min(v), "median": float(np.median(v)), "x_f32": min(v) / min(allt[(r, 'f32')]), "all": v} for (r, t), v in allt.items()}
for k, v in res.items():
    if k != "timing_us": print(k, v)
for k, v in res["timing_us"].items(): print(f"{k:24s} min {v['min']:7.1f} us  median {v['median']:7.1f}  x{v['x_f32']:.2f} of f32")
json.dump(res, open(sys.argv[3], "w"), indent=1)
