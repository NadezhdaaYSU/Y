#!/usr/bin/env python3
"""probe_nr.py <examples dir> <bench_ab dir> <tests dir>: Y's name-specialized
F16 kernels, the committed tests/rmsnorm_residual_4096.ptx and
tests/rope_128.ptx, against Triton kernels doing the same arithmetic and
PyTorch eager, at footprints inside, near and well above the 48 MB L2.
Every column is checked against float64 first; interleaved rounds, minimum.

rmsnorm_residual: h = X + Residual (f32), NewResidual = f16(h),
  Out = f16(h * rsqrt(mean(h^2) + 1e-5) * W). rope: interleaved pairs,
  theta_i = pos * 10000^(-2i/d), out[2i] = x0 cos - x1 sin, out[2i+1] = x0 sin + x1 cos."""
import math, statistics, sys
import torch, triton
import triton.language as tl
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
import run_head_to_head_comparison as H
from bench_ab import kernel_from_ptx
torch.zeros(1, device="cuda")
gen = torch.Generator(device="cuda").manual_seed(13)


@triton.jit
def t_add_rmsnorm(x_ptr, r_ptr, w_ptr, o_ptr, nr_ptr, D: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, D)
    h = tl.load(x_ptr + row * D + offs).to(tl.float32) + tl.load(r_ptr + row * D + offs).to(tl.float32)
    tl.store(nr_ptr + row * D + offs, h.to(tl.float16))
    rstd = 1.0 / tl.sqrt(tl.sum(h * h, axis=0) / D + 1e-5)
    w = tl.load(w_ptr + offs).to(tl.float32)
    tl.store(o_ptr + row * D + offs, (h * rstd * w).to(tl.float16))


@triton.jit
def t_rope(x_ptr, p_ptr, o_ptr, rows, D: tl.constexpr, ROWS: tl.constexpr):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    i = tl.arange(0, D // 2)
    m = r < rows
    pos = tl.load(p_ptr + r, mask=m, other=0).to(tl.float32)
    inv = tl.exp2(-(2.0 * i.to(tl.float32) / D) * 13.287712379549449)  # log2(10000)
    th = pos[:, None] * inv[None, :]
    c, s = tl.cos(th), tl.sin(th)
    base = r[:, None] * D + 2 * i[None, :]
    mm = m[:, None]
    x0 = tl.load(x_ptr + base, mask=mm, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + base + 1, mask=mm, other=0.0).to(tl.float32)
    tl.store(o_ptr + base, (x0 * c - x1 * s).to(tl.float16), mask=mm)
    tl.store(o_ptr + base + 1, (x0 * s + x1 * c).to(tl.float16), mask=mm)


def bench(name, cols, checks, nbytes):
    for f in cols.values():
        f()
    torch.cuda.synchronize()
    for c in cols:
        checks(c)
    H.warm_clocks()
    one = min(H.event_time_us(lambda f=f: [f() for _ in range(10)], 10) for f in cols.values() for _ in range(3))
    calls = max(3, int(3000 / max(one, 1)))
    samples = {c: [] for c in cols}
    clocks = []
    names = list(cols)
    for r in range(17):
        for c in names[r % len(names):] + names[:r % len(names)]:
            f = cols[c]
            t = H.event_time_us(lambda f=f: [f() for _ in range(calls)], calls)
            if r >= 2:
                samples[c].append(t)
        clocks.append(torch.cuda.clock_rate(0))
    print(f"== {name}  {nbytes / 1e6:.0f} MB  all checked  calls {calls}  SM {min(clocks)}-{max(clocks)} MHz", flush=True)
    best = min(min(v) for v in samples.values())
    for c in sorted(samples, key=lambda c: min(samples[c])):
        m = min(samples[c])
        print(f"   {c:14s} min {m:9.2f} median {statistics.median(samples[c]):9.2f} us  {nbytes / m / 1e3:7.1f} GB/s  {m / best:.3f}x the fastest", flush=True)


