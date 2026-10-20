"""accuracy.py <bench_ab.py dir> <ptx dir> <out.json> <shape> ...: the
accuracy of Y's F16 GEMM (f16 A, B -> f32 C), cuBLAS's (torch.mm with a
float32 output) and Triton's, against float64.

Each shape is MxNxK (or N for N^3); <ptx dir>/g<shape>.ptx holds kernel
gemm_f16_<shape>. For each of two input distributions - N(0, 1) and
|N(0, 1)| (same sign, so no cancellation hides a short accumulator) - as
f16, every output C[i,j] is compared with the float64 product R[i,j] (f16
products are exact in float64, so R is the true sum to ~1e-16). The error
is scaled by S[i,j] = sum_k |A[i,k]| |B[k,j]| (float64): e = (C - R) / S,
the backward-error yardstick; for same-sign inputs S = R and e is the
relative error. Reported: mean e (its sign is the bias: a truncating
accumulator sits below), mean |e|, max |e|, the fraction of outputs equal
to the correctly rounded float32 of R, the mean distance from it in float32
ulps, and the fraction bitwise equal to cuBLAS's output.
"""
import json
import re
import sys

import torch

sys.path.insert(0, sys.argv[1])
from bench_ab import kernel_from_ptx, triton_gemm  # noqa: E402

import triton  # noqa: E402


def ulps(x, y):
    """|x - y| in float32 ulps (both float32, finite), as float64."""
    xi = x.view(torch.int32).to(torch.int64)
    yi = y.view(torch.int32).to(torch.int64)
    # map to a monotonic integer line
    xi = torch.where(xi < 0, -(xi & 0x7FFFFFFF), xi)
    yi = torch.where(yi < 0, -(yi & 0x7FFFFFFF), yi)
    return (xi - yi).abs().double()


def stats(c, r, s, rn, cb):
    e = (c.double() - r) / s
    return {
        "mean_e": e.mean().item(),
        "mean_abs_e": e.abs().mean().item(),
        "max_abs_e": e.abs().max().item(),
        "frac_correctly_rounded": (c == rn).double().mean().item(),
        "mean_ulps_from_rounded": ulps(c, rn).mean().item(),
        "max_ulps_from_rounded": ulps(c, rn).max().item(),
        "frac_bitwise_cublas": (c.view(torch.int32) == cb.view(torch.int32)).double().mean().item(),
    }


def main():
    ptx_dir, out_path = sys.argv[2], sys.argv[3]
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.zeros(1, device="cuda")
    result = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "triton": triton.__version__, "shapes": {}}
    for shape in sys.argv[4:]:
        m, n, k = (int(x) for x in shape.split("x")) if "x" in shape else (int(shape),) * 3
        kern = kernel_from_ptx(f"{ptx_dir}/g{shape}.ptx", f"gemm_f16_{shape}")
        cm, cn, wm, wn = (int(x) for x in re.search(r"CTA (\d+)x(\d+)x\d+ \| (\d+)x(\d+) warps", kern.ptx).groups())
        tile = re.search(r"CTA (\d+x\d+x\d+ \| \d+x\d+ warps)", kern.ptx).group(1)
        entry = {"tile": tile}
        for dist in ("normal", "same_sign"):
            g = torch.Generator(device="cuda").manual_seed(m * 7 + n * 3 + k)
            a = torch.randn(m, k, device="cuda", generator=g).half()
            b = torch.randn(k, n, device="cuda", generator=g).half()
            if dist == "same_sign":
                a, b = a.abs(), b.abs()
            r = a.double() @ b.double()
            s = a.double().abs() @ b.double().abs()
            rn = r.float()  # correctly rounded float32 (round to nearest even)
            y = torch.full((m, n), float("nan"), device="cuda")
            kern.launch((-(-n // cn), -(-m // cm), 1), (wm * wn * 32, 1, 1), [a, b, y])
            torch.cuda.synchronize()
            cb = torch.mm(a, b, out_dtype=torch.float32)
            cols = {"Y": y, "cuBLAS": cb}
            if m % 256 == 0 and n % 256 == 0 and k % 64 == 0:  # the Triton kernel has no masks
                t = torch.full((m, n), float("nan"), device="cuda")
                triton_gemm[lambda meta: (triton.cdiv(m, meta["BM"]), triton.cdiv(n, meta["BN"]))](a, b, t, m, n, k)
                torch.cuda.synchronize()
                cols["Triton"] = t
            # cuBLAS's kernel, from the profiler
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                torch.mm(a, b, out_dtype=torch.float32)
                torch.cuda.synchronize()
            names = sorted({e.name[:90] for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA})
            assert all(torch.isfinite(c).all() for c in cols.values()), shape
            entry[dist] = {name: stats(c, r, s, rn, cb) for name, c in cols.items()}
            entry[dist]["cublas_kernels"] = names
            del r, s, rn, a, b
            torch.cuda.empty_cache()
            line = " | ".join(f"{name} mean e {v['mean_e']:+.2e} mean|e| {v['mean_abs_e']:.2e} max|e| {v['max_abs_e']:.2e} "
                              f"rounded {v['frac_correctly_rounded']:.3f} ulps {v['mean_ulps_from_rounded']:.2f} =cuBLAS {v['frac_bitwise_cublas']:.3f}"
                              for name, v in entry[dist].items() if name != "cublas_kernels")
            print(f"{shape:>14s} {dist:9s} {tile} | {line} | {names}", flush=True)
        result["shapes"][shape] = entry
    json.dump(result, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
