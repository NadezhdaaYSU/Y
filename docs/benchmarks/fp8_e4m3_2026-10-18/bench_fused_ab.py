"""Y's e4m3-input @tile GEMM (f32 A [M][K], B [K][N] -> f32 C, quantised in the kernel) from PTX
variants, against torch._scaled_mm (cuBLASLt) on the same bytes.

    python3 bench_e4m3.py <label=ptx-dir,...> <out.json> [sizes]

Each directory holds g<N>.ptx (kernel gemm_e4m3_<N>). Inputs are small
integers times 1/8, so quantising them is exact and every correct column
equals the float64 product bit for bit; that is checked first. Timing:
interleaved rounds, clocks brought up per size, SM clock read per sample.
"""
import json
import re
import statistics
import subprocess
import sys

import torch

from y_lang.torch_interop import TorchKernel
sys.path.insert(0, sys.argv.pop(1))
from run_head_to_head_comparison import triton_fp8_gemm


def kernel_from_ptx(path, name):
    k = TorchKernel.__new__(TorchKernel)
    k.source, k.kernel_name, k.target_sm = None, name, "sm_89"
    k.ptx = open(path).read()
    k._module = k._function = None
    k._ensure_loaded()
    return k


def event_us(fn, calls):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(calls):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) * 1000.0 / calls


def warm():
    w = torch.randn(4096, 4096, device="cuda")
    for _ in range(30):
        w @ w
    torch.cuda.synchronize()


def util():
    return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()


def required_launch(ptx):
    m = re.search(r"Launch grid required: \((\d+), (\d+), (\d+)\) CTAs, block \((\d+),(\d+),(\d+)\)", ptx)
    v = [int(x) for x in m.groups()]
    return tuple(v[:3]), tuple(v[3:])


def main():
    dirs = [tuple(x.split("=", 1)) for x in sys.argv[1].split(",")]
    out_path = sys.argv[2]
    sizes = (sys.argv[3] if len(sys.argv) > 3 else "1024,2048,4096,8192").split(",")
    rounds = 17
    result = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "util_before": util(), "sizes": {}}
    sa = sb = 0.125
    for size in sizes:
        m, n, kk = (int(x) for x in size.split("x")) if "x" in size else (int(size),) * 3
        gen = torch.Generator(device="cuda").manual_seed(m * 3 + n)
        a = torch.randint(-8, 9, (m, kk), device="cuda", generator=gen).float() * sa
        b = torch.randint(-8, 9, (kk, n), device="cuda", generator=gen).float() * sb
        ref = a.double() @ b.double()
        cols, outs, tiles = {}, {}, {}
        for label, d in dirs:
            k = kernel_from_ptx(f"{d}/g{size}.ptx", f"gemm_fp8_{size}")
            tiles[label] = re.search(r"CTA \S+ \(\w+\) \| \S+ warps", k.ptx).group(0)
            grid, block = required_launch(k.ptx)
            outs[label] = torch.full((m, n), float("nan"), device="cuda")
            cols[label] = (lambda k=k, grid=grid, block=block, o=outs[label]:
                           k.launch(grid, block, [a, b, sa, sb, o]))
        outs["Triton"] = torch.full((m, n), float("nan"), device="cuda")
        cols["Triton"] = lambda: triton_fp8_gemm[lambda meta: (m // meta["BM"], n // meta["BN"])](
            a, b, outs["Triton"], m, n, kk, 1.0 / sa, 1.0 / sb, sa * sb)
        for f in cols.values():
            f()
        torch.cuda.synchronize()
        exact = {name: bool(torch.equal(o.double(), ref)) for name, o in outs.items()}
        calls = max(2, min(200, int(5e-3 / max(2 * m * n * kk / 150e12, 2e-6))))
        names = [c for c in cols if exact[c]]
        samples = {c: [] for c in names}
        clocks = {c: [] for c in names}
        warm()
        for r in range(rounds):
            for c in names[r % len(names):] + names[:r % len(names)]:
                t = event_us(cols[c], calls)
                clk = torch.cuda.clock_rate(0)
                if r >= 2:
                    samples[c].append(t)
                    clocks[c].append(clk)
        mins = {c: min(s) for c, s in samples.items()}
        result["sizes"][size] = {"tiles": tiles, "exact": exact, "calls": calls,
                                 "samples": samples, "sm_clock_mhz": clocks}
        allclk = [x for v in clocks.values() for x in v]
        print(f"{size:>14s} {tiles} | exact {exact} | SM {min(allclk)}-{max(allclk)} MHz", flush=True)
        for c in sorted(mins, key=mins.get):
            tf = 2 * m * n * kk / mins[c] / 1e6
            print(f"      {c:12s} min {mins[c]:9.1f} median {statistics.median(samples[c]):9.1f} us"
                  f"  {tf:6.1f} TFLOPS"
                  + (f"  {mins[c] / mins['Triton']:.3f}x Triton" if "Triton" in mins else ""), flush=True)
    result["util_after"] = util()
    json.dump(result, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
