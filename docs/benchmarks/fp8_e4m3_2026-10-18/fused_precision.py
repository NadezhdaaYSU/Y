"""Accumulation error of the fused FP8 GEMM (float32 in, quantised in the
kernel) and the head-to-head's Triton kernel, on inputs already exact in e4m3
(so quantising them is exact), against the float64 product."""
import re
import sys
import torch
import y_lang
sys.path.insert(0, sys.argv[1])
from run_head_to_head_comparison import triton_fp8_gemm

for n in (1024, 4096):
    src = f"@tile({n}, {n}, {n})\nkernel gemm_fp8_{n}(A: GlobalMemory<F32>, B: GlobalMemory<F32>, scale_a: F32, scale_b: F32, C: GlobalMemory<F32>) {{\n}}\n"
    k = y_lang.TorchKernel(src, kernel_name=f"gemm_fp8_{n}", target_sm="sm_89")
    g = re.search(r"Launch grid required: \((\d+), (\d+), (\d+)\) CTAs, block \((\d+),(\d+),(\d+)\)", k.ptx)
    v = [int(x) for x in g.groups()]
    gen = torch.Generator(device="cuda").manual_seed(n)
    a = (torch.rand(n, n, device="cuda", generator=gen) * 16).to(torch.float8_e4m3fn).float()
    b = (torch.rand(n, n, device="cuda", generator=gen) * 16).to(torch.float8_e4m3fn).float()
    ref = a.double() @ b.double()
    cy, ct = torch.empty(n, n, device="cuda"), torch.empty(n, n, device="cuda")
    k.launch(tuple(v[:3]), tuple(v[3:]), [a, b, 1.0, 1.0, cy])
    triton_fp8_gemm[lambda meta: (n // meta["BM"], n // meta["BN"])](a, b, ct, n, n, n, 1.0, 1.0, 1.0)
    torch.cuda.synchronize()
    for who, c in (("Y fused", cy), ("Triton", ct)):
        err = (c.double() - ref).abs() / ref.abs()
        print(f"{n} positive {who:8s} rel err max {err.max().item():.2e} median {err.median().item():.2e}", flush=True)
