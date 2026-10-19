"""ksweep.py <ptx-dir> <out.json> [K,K,...]: Y's F16 GEMM (f16 A, B -> f32 C)
at M = N = 512 against cuBLAS, over K, to split the time into a part that
grows with K (the K loop) and a part that does not (prologue, epilogue,
launch).

<ptx-dir> holds g512x512x<K>.ptx (kernel gemm_f16_512x512x<K>). Every column
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
    d, out_path = sys.argv[1], sys.argv[2]
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
        k = kernel_from_ptx(f"{d}/g{name}.ptx", f"gemm_f16_{name}")
        cm, cn, wm, wn = (int(x) for x in re.search(r"CTA (\d+)x(\d+)x\d+ \| (\d+)x(\d+) warps", k.ptx).groups())
        y_out = torch.full((m, n), float("nan"), device="cuda")
        c_out = torch.full((m, n), float("nan"), device="cuda")
        cols = {
            "Y": lambda: k.launch((-(-n // cn), -(-m // cm), 1), (wm * wn * 32, 1, 1), [a, b, y_out]),
            "cuBLAS": lambda: torch.mm(a, b, out_dtype=torch.float32, out=c_out),
        }
        for f in cols.values():
            f()
        torch.cuda.synchronize()
        exact = {"Y": bool(torch.equal(y_out, ref)), "cuBLAS": bool(torch.equal(c_out, ref))}
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
        y_out.fill_(float("nan")); c_out.fill_(float("nan"))
        for g in graphs.values():
            g.replay()
        torch.cuda.synchronize()
        exact_graph = {"Y": bool(torch.equal(y_out, ref)), "cuBLAS": bool(torch.equal(c_out, ref))}
        samples = {c: [] for c in cols}
        clocks = []
        order = list(cols)
        warm()
        for r in range(WARM_ROUNDS + ROUNDS):
            for c in order[r % 2:] + order[:r % 2]:
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                graphs[c].replay()
                e.record()
                e.synchronize()
                if r >= WARM_ROUNDS:
                    samples[c].append(s.elapsed_time(e) * 1000.0 / CALLS)
                    clocks.append(torch.cuda.clock_rate(0))
        mins = {c: min(v) for c, v in samples.items()}
        result["k"][kk] = {"tile": f"{cm}x{cn} {wm}x{wn}", "exact": exact, "exact_after_graph": exact_graph,
                           "cublas_kernels": names, "samples": samples, "sm_clock_mhz": clocks}
        print(f"K={kk:5d} exact {exact} graph {exact_graph} SM {min(clocks)}-{max(clocks)} MHz | "
              f"Y min {mins['Y']:.2f} med {statistics.median(samples['Y']):.2f} | "
              f"cuBLAS min {mins['cuBLAS']:.2f} med {statistics.median(samples['cuBLAS']):.2f} | "
              f"cuBLAS/Y {mins['cuBLAS'] / mins['Y']:.3f} | {names}", flush=True)
    result["util_after"] = util()
    json.dump(result, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
