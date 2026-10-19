"""Y's @tile F16 GEMM (f16 A, B -> f32 C) from two compilers, against Triton
and cuBLAS, on square sizes.

    python3 bench_ab.py <label=ptx-dir,label=ptx-dir,...> <out.json> [sizes]

Each PTX directory holds g<N>.ptx (kernel gemm_f16_<N>), emitted by that
compiler's CLI from g<N>.ysu against the same .ysu_hw_profile. Every column
is checked equal to float64 A @ B first (small integers: exact), then timed
in interleaved rounds, the clocks brought up before each size and the SM
clock read after every sample. Minimum and median per column.
"""
import json
import re
import statistics
import subprocess
import sys

import torch
import triton
import triton.language as tl

from y_lang.torch_interop import TorchKernel


@triton.autotune(
    configs=[
        triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=4, num_stages=4),
        triton.Config({"BM": 128, "BN": 128, "BK": 64}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 256, "BK": 32}, num_warps=8, num_stages=3),
        triton.Config({"BM": 256, "BN": 128, "BK": 32}, num_warps=8, num_stages=3),
        triton.Config({"BM": 64, "BN": 128, "BK": 32}, num_warps=4, num_stages=4),
        triton.Config({"BM": 64, "BN": 64, "BK": 64}, num_warps=4, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def triton_gemm(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptr + rm[:, None] * K + (k0 + rk)[None, :])
        b = tl.load(b_ptr + (k0 + rk)[:, None] * N + rn[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)


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
    out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return out


def main():
    dirs = [tuple(x.split("=", 1)) for x in sys.argv[1].split(",")]
    out_path = sys.argv[2]
    sizes = (sys.argv[3] if len(sys.argv) > 3 else "512,1024,2048,4096,8192").split(",")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    result = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "triton": triton.__version__,
              "util_before": util(), "sizes": {}}
    for size in sizes:
        m, n, kk = (int(x) for x in size.split("x")) if "x" in size else (int(size),) * 3
        gen = torch.Generator(device="cuda").manual_seed(m * 3 + n)
        a = torch.randint(-4, 5, (m, kk), device="cuda", generator=gen).half()
        b = torch.randint(-4, 5, (kk, n), device="cuda", generator=gen).half()
        ref = (a.double() @ b.double()).float()
        cols, outs, tiles = {}, {}, {}
        for label, d in dirs:
            path = f"{d}/g{size}.ptx"
            k = kernel_from_ptx(path, f"gemm_f16_{size}")
            cm, cn, wm, wn = (int(x) for x in re.search(r"CTA (\d+)x(\d+)x\d+ \| (\d+)x(\d+) warps", k.ptx).groups())
            tiles[label] = f"{cm}x{cn} {wm}x{wn}"
            outs[label] = torch.full((m, n), float("nan"), device="cuda")
            cols[label] = (lambda k=k, cm=cm, cn=cn, wm=wm, wn=wn, o=outs[label]:
                           k.launch((-(-n // cn), -(-m // cm), 1), (wm * wn * 32, 1, 1), [a, b, o]))
        if m == n == kk:
            outs["Triton"] = torch.full((m, n), float("nan"), device="cuda")
            cols["Triton"] = lambda: triton_gemm[lambda meta: (triton.cdiv(m, meta["BM"]), triton.cdiv(n, meta["BN"]))](
                a, b, outs["Triton"], m, n, kk)
        cols["cuBLAS"] = lambda: torch.mm(a, b, out_dtype=torch.float32)
        for name, f in cols.items():
            r = f()
            if name == "cuBLAS":
                outs["cuBLAS"] = r
        torch.cuda.synchronize()
        exact = {name: bool(torch.equal(o, ref)) for name, o in outs.items()}
        calls = max(2, min(200, int(5e-3 / max(2 * m * n * kk / 80e12, 2e-6))))
        names = [c for c in cols if exact[c]]
        samples = {c: [] for c in names}
        clocks = {c: [] for c in names}
        warm()
        for r in range(17):
            for c in names[r % len(names):] + names[:r % len(names)]:
                t = event_us(cols[c], calls)
                clk = torch.cuda.clock_rate(0)
                if r >= 2:
                    samples[c].append(t)
                    clocks[c].append(clk)
        mins = {c: min(s) for c, s in samples.items()}
        result["sizes"][size] = {"tiles": tiles, "triton_config": str(getattr(triton_gemm, "best_config", None)) if "Triton" in cols else None, "exact": exact,
                              "calls": calls, "samples": samples, "sm_clock_mhz": clocks}
        allclk = [x for v in clocks.values() for x in v]
        print(f"{size:>14s} Y {tiles} | exact {exact} | Triton {result['sizes'][size]['triton_config']} | SM {min(allclk)}-{max(allclk)} MHz",
              flush=True)
        for c in sorted(mins, key=mins.get):
            print(f"      {c:9s} min {mins[c]:9.1f} median {statistics.median(samples[c]):9.1f} us"
                  + (f"  {mins[c] / mins['Triton']:.3f}x Triton" if "Triton" in mins else ""), flush=True)
    result["util_after"] = util()
    json.dump(result, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
