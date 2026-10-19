#!/usr/bin/env python3
"""probe_final.py <repo tests dir> <bench_ab dir> <ptx dir> <d,...> <MB,...>:
rmsnorm_residual_<d> as the parent compiler emits it (r<d>_before.ptx, where
ptxas accepts it) and as this change does (r<d>_after.ptx), against Triton and
FlashInfer's fused_add_rmsnorm (in place; its buffers are restored before each
sample, outside the timed region). Every column is checked first against
float64 (NewResidual = f16(X + Residual) exactly, Out within 2^-10 of the
largest output). Rows per footprint: X, Residual, Out and NewResidual each
counted once. CUDA events over back-to-back calls, 17 interleaved rounds, the
first two dropped; minimum and median of 15."""
import os, re, statistics, sys
import torch, triton
import triton.language as tl
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[1]), "python", "examples"))
from bench_ab import kernel_from_ptx
import run_head_to_head_comparison as H
from benchmark_y_paged_decode_attention import import_flashinfer
flashinfer = import_flashinfer()
P = sys.argv[3]
dims = [int(x) for x in sys.argv[4].split(",")]
mbs = [int(x) for x in sys.argv[5].split(",")]
torch.zeros(1, device="cuda")
gen = torch.Generator(device="cuda").manual_seed(41)


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
        return None
    try:
        k = kernel_from_ptx(path, name)
    except Exception as e:
        print(f"   {os.path.basename(path)} does not load: {e}"[:200], flush=True)
        return None
    return k, int(re.search(r"warps \((\d+) threads\)", k.ptx).group(1))


def time_cols(cols):
    """cols: name -> (restore, run). Interleaved rounds; restore runs before each sample, untimed."""
    for restore, run in cols.values():
        restore(); run()
    torch.cuda.synchronize()
    H.warm_clocks()
    one = min(H.event_time_us(lambda f=run: [f() for _ in range(10)], 10) for _, run in cols.values() for _ in range(3))
    calls = max(3, int(3000 / max(one, 1)))
    samples = {c: [] for c in cols}
    clocks = []
    names = list(cols)
    for r in range(17):
        for c in names[r % len(names):] + names[:r % len(names)]:
            restore, run = cols[c]
            restore(); torch.cuda.synchronize()
            t = H.event_time_us(lambda f=run: [f() for _ in range(calls)], calls)
            if r >= 2:
                samples[c].append(t)
        clocks.append(torch.cuda.clock_rate(0))
    return samples, calls, clocks


noop = lambda: None
for d in dims:
    name = f"rmsnorm_residual_{d}"
    ys = {"Y before": load(f"{P}/r{d}_before.ptx", name), "Y after": load(f"{P}/r{d}_after.ptx", name)}
    ys = {k: v for k, v in ys.items() if v is not None}
    w = (torch.rand(d, device="cuda", generator=gen) + 0.5).half()
    block = triton.next_power_of_2(d)
    for mb in mbs:
        rows = max(1, round(mb * 1e6 / (8 * d)))
        x = torch.randn(rows, d, device="cuda", generator=gen).half(); res = torch.randn(rows, d, device="cuda", generator=gen).half()
        h64 = x.double() + res.double()
        out64 = h64 * torch.rsqrt((h64 * h64).mean(dim=1, keepdim=True) + 1e-5) * w.double()
        tol = 2 * 2 ** -10 * out64.abs().max().item()
        o, cols = {}, {}
        for label, (k, thr) in ys.items():
            o[label] = (torch.full_like(x, float("nan")), torch.full_like(x, float("nan")))
            cols[label] = (noop, lambda k=k, thr=thr, label=label: k.launch((rows, 1, 1), (thr, 1, 1), [x, res, w, o[label][0], o[label][1]]))
        o["Triton"] = (torch.full_like(x, float("nan")), torch.full_like(x, float("nan")))
        cols["Triton"] = (noop, lambda: t_add_rmsnorm[(rows,)](x, res, w, o["Triton"][0], o["Triton"][1], D=d, BLOCK=block, num_warps=8))
        fi_in, fi_res = x.clone(), res.clone()
        o["FlashInfer"] = (fi_in, fi_res)
        cols["FlashInfer"] = (lambda: (fi_in.copy_(x), fi_res.copy_(res)), lambda: flashinfer.fused_add_rmsnorm(fi_in, fi_res, w, eps=1e-5))
        for restore, run in cols.values():
            restore(); run()
        torch.cuda.synchronize()
        for c in cols:
            assert torch.equal(o[c][1], h64.half()), (d, rows, c, "NewResidual")
            err = (o[c][0].double() - out64).abs().max().item()
            assert err <= tol, (d, rows, c, "Out", err, tol)
        same = ("Y before" in o) and torch.equal(o["Y before"][0].view(torch.int16), o["Y after"][0].view(torch.int16))
        samples, calls, clocks = time_cols(cols)
        nbytes = rows * d * 8 + d * 2
        thr = {k: v[1] for k, v in ys.items()}
        print(f"== d={d} rows {rows}  {nbytes / 1e6:.0f} MB  all checked; threads {thr}; Out bits before == after: {same}  calls {calls}  SM {min(clocks)}-{max(clocks)} MHz", flush=True)
        best = min(min(v) for v in samples.values())
        for c in sorted(samples, key=lambda c: min(samples[c])):
            m = min(samples[c])
            print(f"   {c:10s} min {m:9.2f} median {statistics.median(samples[c]):9.2f} us  {nbytes / m / 1e3:7.1f} GB/s  {m / best:.3f}x the fastest", flush=True)
