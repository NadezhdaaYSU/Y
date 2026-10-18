#!/usr/bin/env python3
"""prof512.py: Y's gemm_f16_512 (committed PTX) and cuBLAS's torch.mm (f32 out) at 512^3, a few calls each, for ncu."""
import sys, re
import torch
sys.path.insert(0, sys.argv[1])          # dir holding bench_ab.py
from bench_ab import kernel_from_ptx
ptx = sys.argv[2]
n = 512
gen = torch.Generator(device="cuda").manual_seed(n * 4)
a = torch.randint(-4, 5, (n, n), device="cuda", generator=gen).half()
b = torch.randint(-4, 5, (n, n), device="cuda", generator=gen).half()
k = kernel_from_ptx(ptx, "gemm_f16_512")
cm, cn, wm, wn = (int(x) for x in re.search(r"CTA (\d+)x(\d+)x\d+ \| (\d+)x(\d+) warps", k.ptx).groups())
o = torch.full((n, n), float("nan"), device="cuda")
ref = (a.double() @ b.double()).float()
for _ in range(3):
    k.launch((-(-n // cn), -(-n // cm), 1), (wm * wn * 32, 1, 1), [a, b, o])
    c = torch.mm(a, b, out_dtype=torch.float32)
torch.cuda.synchronize()
print("Y exact", torch.equal(o, ref), "cuBLAS exact", torch.equal(c, ref))
