"""redtime.py file.ptx ...: time each split kernel's main entry alone, its reduce alone, and both, as CUDA graphs of CALLS calls, interleaved, minimum."""
import re, sys, torch
from y_lang.torch_interop import TorchKernel
CALLS, ROUNDS = 10, 15
torch.zeros(1, device="cuda")
def kern(ptx, name):
    k = TorchKernel.__new__(TorchKernel); k.source, k.kernel_name, k.target_sm = None, name, "sm_89"
    k.ptx = ptx; k._module = k._function = None; k._ensure_loaded(); return k
fns = {}
for path in sys.argv[1:]:
    ptx = open(path).read()
    names = re.findall(r"\.visible \.entry (\w+)\(", ptx)
    name = [x for x in names if not x.endswith("_reduce")][0]
    k = int(re.search(r"x(\d+)_splitk", name).group(1)); m = n = 512
    cm, cn, wm, wn = (int(x) for x in re.search(r"CTA (\d+)x(\d+)x\d+ \| (\d+)x(\d+) warps", ptx).groups())
    s = int(re.search(r"in (\d+) splits", ptx).group(1))
    main, red = kern(ptx, name), kern(ptx, name + "_reduce")
    a = torch.randn(m, k, device="cuda").half(); b = torch.randn(k, n, device="cuda").half()
    c = torch.empty(m, n, device="cuda"); p = torch.empty(s, m, n, device="cuda")
    lm = lambda main=main, a=a, b=b, c=c, p=p, s=s, cm=cm, cn=cn, t=wm*wn*32: main.launch((n // cn, m // cm, s), (t, 1, 1), [a, b, c, p])
    lr = lambda red=red, a=a, b=b, c=c, p=p: red.launch((m * n // 4 // 256, 1, 1), (256, 1, 1), [a, b, c, p])
    tag = path.split("/")[-1][:-4]
    fns[tag + " main"] = lm; fns[tag + " reduce"] = lr; fns[tag + " both"] = lambda lm=lm, lr=lr: (lm(), lr())
graphs = {}
for t, f in fns.items():
    f(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(CALLS): f()
    graphs[t] = g
w = torch.randn(4096, 4096, device="cuda")
for _ in range(30): w @ w
samples = {t: [] for t in graphs}
order = list(graphs)
for r in range(ROUNDS + 2):
    for t in order[r % len(order):] + order[:r % len(order)]:
        s0, e0 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s0.record(); graphs[t].replay(); e0.record(); e0.synchronize()
        if r >= 2: samples[t].append(s0.elapsed_time(e0) * 1000 / CALLS)
for t in fns: print(f"  {t:40s} min {min(samples[t]):8.2f} us")
