"""norm_kernel before (argv[1]) and after (argv[2]) the integer sum of squares, at
32 x 3584, CUDA graphs of 100 launches, 15 shuffled rounds after a clock warm-up; min and median.
Usage: norm_bench.py OLD/norm_kernel.py NEW/norm_kernel.py"""
import importlib.util, random, torch
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m
import sys
old = load("nk_old", sys.argv[1])
new = load("nk_new", sys.argv[2])
torch.manual_seed(0)
x = torch.randn(32, 3584, device="cuda") * 0.5; x[:, ::97] *= 60
w = torch.rand(3584, device="cuda") + 0.5
for m in (old, new): m.rms_norm(x, w, 1e-6)
torch.cuda.synchronize()
graphs = {}
for name, m in (("old", old), ("new", new)):
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3): m.rms_norm(x, w, 1e-6)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for _ in range(100): m.rms_norm(x, w, 1e-6)
    graphs[name] = g
import time
spin = torch.randn(4096, 4096, device="cuda")
res = {k: [] for k in graphs}
for r in range(15):
    t0 = time.perf_counter()                # hold the clocks up between rounds
    while time.perf_counter() - t0 < 0.3:
        spin @ spin
    torch.cuda.synchronize()
    order = list(graphs); random.shuffle(order)
    for k in order:
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); graphs[k].replay(); b.record(); torch.cuda.synchronize()
        res[k].append(a.elapsed_time(b) * 1000 / 100)
for k, v in res.items():
    print(f"{k}: min {min(v):.2f} us  median {sorted(v)[7]:.2f} us")
print("bits: old vs new outputs differ in", int((old.rms_norm(x, w, 1e-6).view(torch.int32) != new.rms_norm(x, w, 1e-6).view(torch.int32)).sum()), "of", x.numel())
