"""ktiles.py <label=ptx-dir,...> <out.json> [K,K,...]: Y's F16 GEMM (f16 A, B -> f32 C)
at M = N = 512 against cuBLAS, over K, to split the time into a part that
grows with K (the K loop) and a part that does not (prologue, epilogue,
launch).

Each ptx-dir holds g512x512x<K>.ptx (kernel gemm_f16_512x512x<K>). Every column
is checked equal to float64 A @ B first (small integers: exact) on a
NaN-filled output, then each column is captured as a CUDA graph of CALLS
launches and the graphs are replayed in interleaved rounds, so neither
side's host launch cost is timed. Minimum and median per column; the cuBLAS
kernel's name at each K from the profiler.
"""
import json
import re
import statistics
import subprocess
import sys

import torch

from y_lang.torch_interop import TorchKernel

CALLS = 50
ROUNDS = 21
WARM_ROUNDS = 3


def kernel_from_ptx(path, name):
    k = TorchKernel.__new__(TorchKernel)
    k.source, k.kernel_name, k.target_sm = None, name, "sm_89"
    k.ptx = open(path).read()
    k._module = k._function = None
    k._ensure_loaded()
    return k


def util():
    return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()


def warm():
    w = torch.randn(4096, 4096, device="cuda")
    for _ in range(30):
        w @ w
    torch.cuda.synchronize()


def cublas_kernel(fn):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    return sorted({e.name for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA})


def main():
    dirs = [tuple(x.split("=", 1)) for x in sys.argv[1].split(",")]
    out_path = sys.argv[2]
    ks = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "256,512,1024,2048,4096,8192").split(",")]
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    m = n = 512
    result = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "calls": CALLS,
              "util_before": util(), "k": {}}
    for kk in ks:
        name = f"512x512x{kk}"
        gen = torch.Generator(device="cuda").manual_seed(kk)
        a = torch.randint(-4, 5, (m, kk), device="cuda", generator=gen).half()
        b = torch.randint(-4, 5, (kk, n), device="cuda", generator=gen).half()
        ref = (a.double() @ b.double()).float()
        cols, outs, tiles = {}, {}, {}
        for label, d in dirs:
            k = kernel_from_ptx(f"{d}/g{name}.ptx", f"gemm_f16_{name}")
            cm, cn, wm, wn = (int(x) for x in re.search(r"CTA (\d+)x(\d+)x\d+ \| (\d+)x(\d+) warps", k.ptx).groups())
            tiles[label] = f"{cm}x{cn} {wm}x{wn}"
            outs[label] = torch.full((m, n), float("nan"), device="cuda")
            cols[label] = (lambda k=k, cm=cm, cn=cn, wm=wm, wn=wn, o=outs[label]:
                           k.launch((-(-n // cn), -(-m // cm), 1), (wm * wn * 32, 1, 1), [a, b, o]))
        outs["cuBLAS"] = torch.full((m, n), float("nan"), device="cuda")
        cols["cuBLAS"] = lambda o=outs["cuBLAS"]: torch.mm(a, b, out_dtype=torch.float32, out=o)
        for f in cols.values():
            f()
        torch.cuda.synchronize()
        exact = {c: bool(torch.equal(o, ref)) for c, o in outs.items()}
        if not all(exact.values()):
            print(f"K={kk}: not exact {exact}", flush=True)
            result["k"][kk] = {"exact": exact}
            continue
        names = cublas_kernel(cols["cuBLAS"])
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
        # The graphs write the same outputs; check them again after replay.
        for o in outs.values():
            o.fill_(float("nan"))
        for g in graphs.values():
            g.replay()
        torch.cuda.synchronize()
        exact_graph = {c: bool(torch.equal(o, ref)) for c, o in outs.items()}
        samples = {c: [] for c in cols}
        clocks = []
        order = list(cols)
        warm()
        for r in range(WARM_ROUNDS + ROUNDS):
            for c in order[r % len(order):] + order[:r % len(order)]:
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                graphs[c].replay()
                e.record()
                e.synchronize()
                if r >= WARM_ROUNDS:
                    samples[c].append(s.elapsed_time(e) * 1000.0 / CALLS)
                    clocks.append(torch.cuda.clock_rate(0))
        mins = {c: min(v) for c, v in samples.items()}
        result["k"][kk] = {"tiles": tiles, "exact": exact, "exact_after_graph": exact_graph,
                           "cublas_kernels": names, "samples": samples, "sm_clock_mhz": clocks}
        print(f"K={kk:5d} exact {all(exact.values())} graph {all(exact_graph.values())} SM {min(clocks)}-{max(clocks)} MHz | cuBLAS {names}", flush=True)
        for c in sorted(mins, key=mins.get):
            print(f"      {c:18s} min {mins[c]:8.2f} median {statistics.median(samples[c]):8.2f} us  cuBLAS/this {mins['cuBLAS'] / mins[c]:.3f}", flush=True)
    result["util_after"] = util()
    json.dump(result, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
