#!/usr/bin/env python3
"""ab.py <examples dir> <bench_ab dir> <committed ptx> <variant ptx>...: rmsnorm_residual_4096
variants against the committed kernel (Out and NewResidual bit for bit, random
inputs) and Triton (checked against float64), timed like probe_nr.py."""
import sys
sys.argv, extra = sys.argv[:3] + [sys.argv[3]], sys.argv[3:]
import torch, triton
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)) + "/..")
import probe_nr_lib as P
from bench_ab import kernel_from_ptx
D = 4096
ks = {("committed" if i == 0 else f"v{i}:" + p.split("/")[-1]): kernel_from_ptx(p, "rmsnorm_residual_4096") for i, p in enumerate(extra)}
gen = torch.Generator(device="cuda").manual_seed(17)
w = (torch.rand(D, device="cuda", generator=gen) + 0.5).half()
for rows in (128, 512, 1024, 3072):
    x = torch.randn(rows, D, device="cuda", generator=gen).half(); res = torch.randn(rows, D, device="cuda", generator=gen).half()
    h64 = x.double() + res.double()
    out64 = h64 * torch.rsqrt((h64 * h64).mean(dim=1, keepdim=True) + 1e-5) * w.double()
    o = {c: (torch.full_like(x, float("nan")), torch.full_like(x, float("nan"))) for c in list(ks) + ["Triton"]}
    cols = {c: (lambda k=k, c=c: k.launch((rows, 1, 1), (256, 1, 1), [x, res, w, o[c][0], o[c][1]])) for c, k in ks.items()}
    cols["Triton"] = lambda: P.t_add_rmsnorm[(rows,)](x, res, w, o["Triton"][0], o["Triton"][1], D=D, num_warps=8)
    def checks(c):
        out, nres = o[c]
        if c == "Triton":
            assert torch.equal(nres, h64.half()) and (out.double() - out64).abs().max().item() <= 2 * 2 ** -10 * out64.abs().max().item(), c
        else:
            assert torch.equal(out.view(torch.int16), o["committed"][0].view(torch.int16)), (c, "Out bits")
            assert torch.equal(nres.view(torch.int16), o["committed"][1].view(torch.int16)), (c, "NewResidual bits")
    P.bench(f"rmsnorm_residual_4096 rows {rows}", cols, checks, rows * D * 8 + D * 2)
