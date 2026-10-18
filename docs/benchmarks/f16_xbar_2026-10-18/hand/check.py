#!/usr/bin/env python3
"""check.py <bench_ab dir> <base ptx> <variant ptx> ...: each variant against the
committed kernel, bit for bit on random normal f16 inputs (same MMA order), and
exact against float64 on small integers; NaN-filled outputs."""
import sys, re
import torch
sys.path.insert(0, sys.argv[1])
from bench_ab import kernel_from_ptx
torch.zeros(1, device="cuda")
n = 512
ks = [kernel_from_ptx(p, "gemm_f16_512") for p in sys.argv[2:]]
def run(k, a, b):
    o = torch.full((n, n), float("nan"), device="cuda")
    k.launch((n // 64, n // 64, 1), (256, 1, 1), [a, b, o]); torch.cuda.synchronize()
    return o
g = torch.Generator(device="cuda").manual_seed(11)
for trial in range(3):
    a = torch.randn(n, n, device="cuda", generator=g).half(); b = torch.randn(n, n, device="cuda", generator=g).half()
    base = run(ks[0], a, b)
    assert not base.isnan().any()
    for p, k in zip(sys.argv[3:], ks[1:]):
        o = run(k, a, b)
        print(p.split("/")[-2], "random trial", trial, "bitwise equal to committed:", torch.equal(o.view(torch.int32), base.view(torch.int32)))
    a = torch.randint(-4, 5, (n, n), device="cuda", generator=g).half(); b = torch.randint(-4, 5, (n, n), device="cuda", generator=g).half()
    ref = (a.double() @ b.double()).float()
    for p, k in zip(sys.argv[2:], ks):
        print(p.split("/")[-2], "integer trial", trial, "exact:", torch.equal(run(k, a, b), ref))
