#!/usr/bin/env python3
"""probe_warps.py <examples dir> <bench_ab dir> <warps dir> <d,d,...> <MB,MB,...>: rmsnorm_residual_<d> compiled with 1, 2, 4, 8 and 16 warps a row (w<n>/r<d>_xbar.ptx, the scratch override) against Triton, each checked against float64 (NewResidual exact). Adapted from probe_dims.py:
rmsnorm_residual_<d> as the parent compiler emits it (r<d>_integ.ptx) and as
this change does (r<d>_xbar.ptx), against Triton. The two Y kernels must agree
bit for bit on Out and NewResidual (random normal inputs) where both exist;
Triton is checked against float64. Rows per footprint (every byte read or
written once: X, Residual, Out, NewResidual). Interleaved rounds, minimum."""
import os, re, statistics, sys
import torch, triton
import triton.language as tl
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
import run_head_to_head_comparison as H
from bench_ab import kernel_from_ptx
P = sys.argv[3]
dims = [int(x) for x in sys.argv[4].split(",")]
mbs = [int(x) for x in sys.argv[5].split(",")]
torch.zeros(1, device="cuda")
gen = torch.Generator(device="cuda").manual_seed(31)


@triton.jit
def t_add_rmsnorm(x_ptr, r_ptr, w_ptr, o_ptr, nr_ptr, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < D
    h = tl.load(x_ptr + row * D + offs, mask=m, other=0.0).to(tl.float32) + tl.load(r_ptr + row * D + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(nr_ptr + row * D + offs, h.to(tl.float16), mask=m)
    rstd = 1.0 / tl.sqrt(tl.sum(h * h, axis=0) / D + 1e-5)
    w = tl.load(w_ptr + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(o_ptr + row * D + offs, (h * rstd * w).to(tl.float16), mask=m)


def load(path, name):
    if not os.path.exists(path):
        return None, None
    k = kernel_from_ptx(path, name)
    threads = int(re.search(r"warps \((\d+) threads\)", k.ptx).group(1))
    return k, threads


for d in dims:
    name = f"rmsnorm_residual_{d}"
    kern = {f"w{n}": load(f"{P}/w{n}/r{d}_xbar.ptx", name) for n in (1, 2, 4, 8, 16, 32) if os.path.exists(f"{P}/w{n}/r{d}_xbar.ptx")}
    w = (torch.rand(d, device="cuda", generator=gen) + 0.5).half()
    block = triton.next_power_of_2(d)
    for mb in mbs:
        rows = max(1, round(mb * 1e6 / (8 * d)))
        x = torch.randn(rows, d, device="cuda", generator=gen).half(); res = torch.randn(rows, d, device="cuda", generator=gen).half()
        h64 = x.double() + res.double()
        out64 = h64 * torch.rsqrt((h64 * h64).mean(dim=1, keepdim=True) + 1e-5) * w.double()
        o, cols = {}, {}
        for label, (k, thr) in kern.items():
            o[label] = (torch.full_like(x, float("nan")), torch.full_like(x, float("nan")))
            cols[label] = (lambda k=k, thr=thr, label=label: k.launch((rows, 1, 1), (thr, 1, 1), [x, res, w, o[label][0], o[label][1]]))
        o["Triton"] = (torch.full_like(x, float("nan")), torch.full_like(x, float("nan")))
        cols["Triton"] = lambda: t_add_rmsnorm[(rows,)](x, res, w, o["Triton"][0], o["Triton"][1], D=d, BLOCK=block, num_warps=8)
        for f in cols.values():
            f()
        torch.cuda.synchronize()
        tol = 2 * 2 ** -10 * out64.abs().max().item()
        for c in cols:
            assert torch.equal(o[c][1], h64.half()), (d, rows, c, "NewResidual")
            assert (o[c][0].double() - out64).abs().max().item() <= tol, (d, rows, c, "Out")
        H.warm_clocks()
        one = min(H.event_time_us(lambda f=f: [f() for _ in range(10)], 10) for f in cols.values() for _ in range(3))
        calls = max(3, int(3000 / max(one, 1)))
        samples = {c: [] for c in cols}
        clocks = []
        names = list(cols)
        for r in range(17):
            for c in names[r % len(names):] + names[:r % len(names)]:
                t = H.event_time_us(lambda f=cols[c]: [f() for _ in range(calls)], calls)
                if r >= 2:
                    samples[c].append(t)
            clocks.append(torch.cuda.clock_rate(0))
        nbytes = rows * d * 8 + d * 2
        print(f"== d={d} rows {rows}  {nbytes / 1e6:.0f} MB  checked  calls {calls}  SM {min(clocks)}-{max(clocks)} MHz", flush=True)
        best = min(min(v) for v in samples.values())
        for c in sorted(samples, key=lambda c: min(samples[c])):
            m = min(samples[c])
            print(f"   {c:9s} min {m:9.2f} median {statistics.median(samples[c]):9.2f} us  {nbytes / m / 1e3:7.1f} GB/s  {m / best:.3f}x the fastest", flush=True)
