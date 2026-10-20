"""sbench.py OUT.json M N K label=file.ptx ...: Y's F16 GEMMs (f16 A, B -> f32 C),
unsplit or split K, against cuBLAS at one shape.

A PTX file holding a `<name>_reduce` entry is a split-K kernel: each call is
its main entry (grid (N/cn, M/cm, S)) then the reduce entry, into a workspace
of S x M x N floats. Every column is first checked equal to float64 A @ B on
small integers (exact) on a NaN-filled output, then each is captured as a
CUDA graph of CALLS calls and the graphs replay in interleaved rounds (each
round a rotated order), so no host launch cost is timed; outputs are checked
again after the graphs ran. Minimum and median per column.
"""
import json, re, statistics, subprocess, sys
import torch
from y_lang.torch_interop import TorchKernel

OPTS = dict(a.split("=", 1) for a in sys.argv if a.startswith(("calls=", "data=")))
CALLS = int(OPTS.get("calls", 20))
DATA = OPTS.get("data", "int")  # timing inputs: the small integers, or f16 normal (after the exact check)
ROUNDS, WARM_ROUNDS = 21, 3

def kern(ptx, name):
    k = TorchKernel.__new__(TorchKernel)
    k.source, k.kernel_name, k.target_sm = None, name, "sm_89"
    k.ptx = ptx
    k._module = k._function = None
    k._ensure_loaded()
    return k

def util():
    return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm,power.draw", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()

def warm():
    w = torch.randn(4096, 4096, device="cuda")
    for _ in range(30):
        w @ w
    torch.cuda.synchronize()

def main():
    out_path, m, n, kk = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    specs = [a.split("=", 1) for a in sys.argv[5:] if "=" in a and not a.startswith(("calls=", "data="))]
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    gen = torch.Generator(device="cuda").manual_seed(kk)
    a = torch.randint(-4, 5, (m, kk), device="cuda", generator=gen).half()
    b = torch.randint(-4, 5, (kk, n), device="cuda", generator=gen).half()
    ref = (a.double() @ b.double()).float()
    cols, outs, info = {}, {}, {}
    for label, path in specs:
        ptx = open(path).read()
        names = re.findall(r"\.visible \.entry (\w+)\(", ptx)
        name = [x for x in names if not x.endswith("_reduce")][0]
        cm, cn, wm, wn = (int(x) for x in re.search(r"CTA (\d+)x(\d+)x\d+ \| (\d+)x(\d+) warps", ptx).groups())
        sm = re.search(r"Dynamic shared memory required: (\d+)", ptx).group(1)
        o = torch.full((m, n), float("nan"), device="cuda")
        outs[label] = o
        k = kern(ptx, name)
        if name + "_reduce" in names:
            sk = re.search(r"over (\d+) CTAs .* Partial\[0\.\.(\d+)\]", ptx)
            if sk:  # stream K: grid (P, 1, 1), workspace of the most pieces a tile has
                s, slots, grid = int(sk.group(1)), int(sk.group(2)), (int(sk.group(1)), 1, 1)
            else:
                s = slots = int(re.search(r"in (\d+) splits", ptx).group(1))
                grid = (-(-n // cn), -(-m // cm), s)
            r = kern(ptx, name + "_reduce")
            p = torch.full((slots, m, n), float("nan"), device="cuda")
            def f(k=k, r=r, p=p, o=o, grid=grid, t=wm * wn * 32):
                k.launch(grid, (t, 1, 1), [a, b, o, p])
                r.launch((-(-(m * n // 4) // 256), 1, 1), (256, 1, 1), [a, b, o, p])
        else:
            s = 1
            def f(k=k, o=o, cm=cm, cn=cn, t=wm * wn * 32):
                k.launch((-(-n // cn), -(-m // cm), 1), (t, 1, 1), [a, b, o])
        cols[label] = f
        info[label] = {"path": path, "tile": f"{cm}x{cn} {wm}x{wn}", "splits": s, "smem": int(sm),
                       "kind": "P" if "STREAM-K GEMM]" in ptx else "S",
                       "f32_mode": "{%f0,%f0,%f0,%f0}" in ptx or bool(re.search(r"\{(%f\d+),\1,\1,\1\};", ptx))}
    outs["cuBLAS"] = torch.full((m, n), float("nan"), device="cuda")
    cols["cuBLAS"] = lambda o=outs["cuBLAS"]: torch.mm(a, b, out_dtype=torch.float32, out=o)
    for f in cols.values():
        f()
    torch.cuda.synchronize()
    exact = {c: bool(torch.equal(o, ref)) for c, o in outs.items()}
    result = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "calls": CALLS, "shape": [m, n, kk],
              "util_before": util(), "columns": info, "exact": exact}
    if not all(exact.values()):
        print(f"not exact: {exact}", flush=True)
        json.dump(result, open(out_path, "w"), indent=1)
        sys.exit(1)
    graphs = {}
    for c, f in cols.items():
        for _ in range(3):
            f()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(CALLS):
                f()
        graphs[c] = g
    for o in outs.values():
        o.fill_(float("nan"))
    for g in graphs.values():
        g.replay()
    torch.cuda.synchronize()
    result["exact_after_graph"] = {c: bool(torch.equal(o, ref)) for c, o in outs.items()}
    result["timing_data"] = DATA
    if DATA == "normal":  # same buffers, so the captured graphs read them
        g2 = torch.Generator(device="cuda").manual_seed(kk + 1)
        a.copy_(torch.randn(m, kk, device="cuda", generator=g2).half())
        b.copy_(torch.randn(kk, n, device="cuda", generator=g2).half())
        for g in graphs.values():
            g.replay()
        torch.cuda.synchronize()
        r2 = a.double() @ b.double()
        result["normal_rel_fro"] = {c: float((o.double() - r2).norm() / r2.norm()) for c, o in outs.items()}
    samples, clocks = {c: [] for c in cols}, []
    order = list(cols)
    warm()
    for r in range(WARM_ROUNDS + ROUNDS):
        for c in order[r % len(order):] + order[:r % len(order)]:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); graphs[c].replay(); e.record(); e.synchronize()
            if r >= WARM_ROUNDS:
                samples[c].append(s.elapsed_time(e) * 1000.0 / CALLS)
        if r >= WARM_ROUNDS and r % 5 == 0:
            clocks.append(util())
    result.update(samples=samples, during=clocks, util_after=util())
    mins = {c: min(v) for c, v in samples.items()}
    print(f"{m}x{n}x{kk} data={DATA} exact {all(exact.values())} graph {all(result['exact_after_graph'].values())} | during {clocks[0]} .. {clocks[-1]}", flush=True)
    for c in sorted(mins, key=mins.get):
        i = info.get(c, {})
        print(f"  {c:28s} min {mins[c]:8.2f} med {statistics.median(samples[c]):8.2f} us  cuBLAS/this {mins['cuBLAS'] / mins[c]:.3f}  {i.get('tile', '')} {i.get('kind', '')}={i.get('splits', '')} {'f32' if i.get('f32_mode') else ''}", flush=True)
    json.dump(result, open(out_path, "w"), indent=1)

main()
