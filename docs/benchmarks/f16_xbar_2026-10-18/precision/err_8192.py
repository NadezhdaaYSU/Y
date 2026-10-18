#!/usr/bin/env python3
"""err_8192.py <bench_ab dir> <tests dir>: at 8192^3, the committed kernel and
cuBLAS against float64 on same-sign f16 inputs (rows 0-255 of the reference),
and the same comparison at 4096, where the two agree bit for bit."""
import sys, re, torch
sys.path.insert(0, sys.argv[1])
from bench_ab import kernel_from_ptx
torch.zeros(1, device="cuda")
for n in (4096, 8192):
    k = kernel_from_ptx(f"{sys.argv[2]}/gemm_f16_{n}.ptx", f"gemm_f16_{n}")
    cm, cn, wm, wn = (int(x) for x in re.search(r"CTA (\d+)x(\d+)x\d+ \| (\d+)x(\d+) warps", k.ptx).groups())
    g = torch.Generator(device="cuda").manual_seed(n + 1)
    a = torch.randn(n, n, device="cuda", generator=g).abs().half(); b = torch.randn(n, n, device="cuda", generator=g).abs().half()
    c = torch.full((n, n), float("nan"), device="cuda")
    k.launch((-(-n // cn), -(-n // cm), 1), (wm * wn * 32, 1, 1), [a, b, c]); torch.cuda.synchronize()
    cb = torch.mm(a, b, out_dtype=torch.float32)
    ref = a[:256].double() @ b.double()
    rel = lambda x: ((x[:256].double() - ref).abs() / ref)
    signed = lambda x: ((x[:256].double() - ref) / ref)
    ry, rc = rel(c), rel(cb)
    for name, x in (("Y", c), ("cuBLAS", cb)):
        s = signed(x)
        print(f"{n}: {name} signed mean {s.mean().item():+.3e}, below float64 in {(s < 0).double().mean().item() * 100:.2f}% of elements", flush=True)
    seq = torch.zeros(256, n, device="cuda", dtype=torch.float32)
    for kk in range(0, n, 16):
        seq += (a[:256, kk:kk + 16].double() @ b[kk:kk + 16, :].double()).float()
    print(f"{n}: rows 0-255 vs float64: Y max {ry.max().item():.3e} mean {ry.mean().item():.3e}; "
          f"cuBLAS max {rc.max().item():.3e} mean {rc.mean().item():.3e}; f32 sum of exact k16 partials max {rel(seq).max().item():.3e}; "
          f"bits differ Y vs cuBLAS {int((c[:256].view(torch.int32) != cb[:256].view(torch.int32)).sum())} of {256 * n}", flush=True)
