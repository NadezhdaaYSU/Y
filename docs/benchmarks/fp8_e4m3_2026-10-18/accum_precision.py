"""How precisely the e4m3 tensor-core GEMMs accumulate: Y's e4m3 GEMM and
torch._scaled_mm on the same e4m3 bytes, each against the float64 product of
those bytes (every product of two e4m3 values is exact in float32, so any
difference is in the summation).

    python3 accum_precision.py <ptx-dir>    (g<N>.ptx, kernel gemm_e4m3_<N>)
"""
import re
import sys

import torch
from y_lang.torch_interop import TorchKernel


def kernel_from_ptx(path, name):
    k = TorchKernel.__new__(TorchKernel)
    k.source, k.kernel_name, k.target_sm = None, name, "sm_89"
    k.ptx = open(path).read()
    k._module = k._function = None
    k._ensure_loaded()
    return k


torch.zeros(1, device="cuda")
d = sys.argv[1]
one = torch.tensor(1.0, device="cuda")
for n in (1024, 4096):
    k = kernel_from_ptx(f"{d}/g{n}.ptx", f"gemm_e4m3_{n}")
    g = re.search(r"Launch grid required: \((\d+), (\d+), (\d+)\) CTAs, block \((\d+),(\d+),(\d+)\)", k.ptx)
    v = [int(x) for x in g.groups()]
    gen = torch.Generator(device="cuda").manual_seed(n)
    cases = {
        # Random normal values, as activations and weights come.
        "randn": (torch.randn(n, n, device="cuda", generator=gen) * 8, torch.randn(n, n, device="cuda", generator=gen) * 8),
        # All positive: partial sums grow, so a short accumulator drops bits.
        "positive": (torch.rand(n, n, device="cuda", generator=gen) * 16, torch.rand(n, n, device="cuda", generator=gen) * 16),
    }
    for name, (a, b) in cases.items():
        a8, b8 = a.to(torch.float8_e4m3fn), b.to(torch.float8_e4m3fn)
        ref = a8.double() @ b8.double().t()
        cy = torch.empty(n, n, device="cuda")
        k.launch(tuple(v[:3]), tuple(v[3:]), [a8, b8, 1.0, 1.0, cy])
        ct = torch._scaled_mm(a8, b8.t(), scale_a=one, scale_b=one, out_dtype=torch.float32)
        # The float32 rounding of the exact sum: the best a float32 result can be.
        best = ref.float().double()
        torch.cuda.synchronize()
        for who, c in (("Y", cy), ("_scaled_mm", ct), ("exact sum rounded to f32", best)):
            err = (c.double() - ref).abs() / ref.abs().clamp_min(1e-30)
            print(f"{n} {name:9s} {who:26s} rel err: max {err.max().item():.2e} median {err.median().item():.2e} "
                  f"| differs from the rounded exact sum in {int((c.double() != best).sum())} of {n * n}", flush=True)
