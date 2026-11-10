"""The default (Y_EXACT_NORM=torch) norm, compiled: residual add + norm + the
int8 row quantiser shape of the decode layer, old sum (float64 tree) vs new
(sum_of_squares), 32 x 3584, CUDA graphs of 50 calls, 9 shuffled rounds, min.
Run with this checkout's python/ on PYTHONPATH."""
import random, torch
from y_lang.exact import model as em
H = 3584
d = torch.tensor(float(H), dtype=torch.float64, device="cuda")
eps = 1e-6
def norm_old(x, r, w):
    h = x + r
    xd = h.to(torch.float64)
    var = xd.pow(2).sum(-1, keepdim=True) / d
    y = (xd / torch.sqrt(var + eps)).to(torch.float32) * w
    a = y.abs().amax(-1, keepdim=True)
    return h, torch.round(y / a * 127).to(torch.int8)
def norm_new(x, r, w):
    h = x + r
    xd = h.to(torch.float64)
    var = em.sum_of_squares(h) / d
    y = (xd / torch.sqrt(var + eps)).to(torch.float32) * w
    a = y.abs().amax(-1, keepdim=True)
    return h, torch.round(y / a * 127).to(torch.int8)
torch.manual_seed(0)
x = torch.randn(32, H, device="cuda"); r = torch.randn(32, H, device="cuda"); w = torch.rand(H, device="cuda") + 0.5
fs = {"old": torch.compile(norm_old, dynamic=False), "new": torch.compile(norm_new, dynamic=False)}
for f in fs.values():
    for _ in range(3): f(x, r, w)
torch.cuda.synchronize()
graphs = {}
for k, f in fs.items():
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3): f(x, r, w)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(50): f(x, r, w)
    graphs[k] = g
import time
spin = torch.randn(4096, 4096, device="cuda")
res = {k: [] for k in graphs}
for _ in range(15):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 0.3:
        spin @ spin
    torch.cuda.synchronize()
    order = list(graphs); random.shuffle(order)
    for k in order:
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); graphs[k].replay(); b.record(); torch.cuda.synchronize()
        res[k].append(a.elapsed_time(b) * 1000 / 50)
for k, v in res.items():
    print(f"{k}: min {min(v):.2f} us  median {sorted(v)[7]:.2f} us")
o, n = fs["old"](x, r, w)[1], fs["new"](x, r, w)[1]
print("int8 outputs differ in", int((o != n).sum()), "of", o.numel())
from torch._inductor.utils import run_and_get_code
import re
for k, f in (("old", norm_old), ("new", norm_new)):
    torch._dynamo.reset()
    _, code = run_and_get_code(torch.compile(f, dynamic=False), x, r, w)
    print(k, "kernels:", re.findall(r"^(triton_\w+) = async_compile", "\n".join(code), re.M))
