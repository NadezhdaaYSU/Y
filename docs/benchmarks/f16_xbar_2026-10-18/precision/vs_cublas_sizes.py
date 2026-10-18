#!/usr/bin/env python3
"""vs_cublas_sizes.py <bench_ab dir> <tests dir>: the committed tests/gemm_f16_<N>.ptx
against torch.mm(out_dtype=float32), bit for bit, on random normal and
same-sign (|normal|) f16 inputs, 256 to 8192."""
import sys, re, torch
sys.path.insert(0, sys.argv[1])
from bench_ab import kernel_from_ptx
torch.zeros(1, device="cuda")
for n in (256, 512, 1024, 2048, 4096, 8192):
    k = kernel_from_ptx(f"{sys.argv[2]}/gemm_f16_{n}.ptx", f"gemm_f16_{n}")
    cm, cn, wm, wn = (int(x) for x in re.search(r"CTA (\d+)x(\d+)x\d+ \| (\d+)x(\d+) warps", k.ptx).groups())
    g = torch.Generator(device="cuda").manual_seed(n)
    for kind in ("normal", "same-sign"):
        a = torch.randn(n, n, device="cuda", generator=g).half(); b = torch.randn(n, n, device="cuda", generator=g).half()
        if kind == "same-sign":
            a, b = a.abs(), b.abs()
        c = torch.full((n, n), float("nan"), device="cuda")
        k.launch((-(-n // cn), -(-n // cm), 1), (wm * wn * 32, 1, 1), [a, b, c]); torch.cuda.synchronize()
        cb = torch.mm(a, b, out_dtype=torch.float32)
        d = int((c.view(torch.int32) != cb.view(torch.int32)).sum())
        print(f"{n} {kind:9s} tile {cm}x{cn}: {d} of {n * n} elements differ in their bits from cuBLAS's", flush=True)
