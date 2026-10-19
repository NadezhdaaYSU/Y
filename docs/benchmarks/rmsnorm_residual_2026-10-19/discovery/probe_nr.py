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
T = sys.argv[3]
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


D = 4096
kr = kernel_from_ptx(f"{T}/rmsnorm_residual_4096.ptx", "rmsnorm_residual_4096")
w = (torch.rand(D, device="cuda", generator=gen) + 0.5).half()
for rows in (128, 512, 1024, 1536, 3072, 8192):
    x = torch.randn(rows, D, device="cuda", generator=gen).half(); res = torch.randn(rows, D, device="cuda", generator=gen).half()
    h64 = x.double() + res.double()
    out64 = h64 * torch.rsqrt((h64 * h64).mean(dim=1, keepdim=True) + 1e-5) * w.double()
    o = {c: (torch.full_like(x, float("nan")), torch.full_like(x, float("nan"))) for c in ("Y", "Triton")}
    cols = {"Y": lambda: kr.launch((rows, 1, 1), (256, 1, 1), [x, res, w, o["Y"][0], o["Y"][1]]),
            "Triton": lambda: t_add_rmsnorm[(rows,)](x, res, w, o["Triton"][0], o["Triton"][1], D=D, num_warps=8)}
    def eager():
        h = x.float() + res.float()
        o["eager"] = ((h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5) * w.float()).half(), h.half())
    cols["PyTorch"] = eager
    def checks(c):
        out, nres = o["eager" if c == "PyTorch" else c]
        assert torch.equal(nres, h64.half()), (c, "NewResidual")
        err = (out.double() - out64).abs().max().item()
        assert err <= 2 * 2 ** -10 * out64.abs().max().item(), (c, "Out", err)
    bench(f"rmsnorm_residual_4096 rows {rows}", cols, checks, rows * D * 2 * 4 + D * 2)

HD = 128
kp = kernel_from_ptx(f"{T}/rope_128.ptx", "rope_128")
for rows in (4096, 32768, 65536, 98304, 196608, 524288):
    x = torch.randn(rows, HD, device="cuda", generator=gen).half()
    pos = torch.randint(0, 4096, (rows,), device="cuda", generator=gen, dtype=torch.int32)
    inv = 10000.0 ** (-torch.arange(0, HD, 2, device="cuda", dtype=torch.float64) / HD)
    th = pos.double()[:, None] * inv[None, :]
    xv = x.double().view(rows, HD // 2, 2)
    ref = torch.stack([xv[..., 0] * th.cos() - xv[..., 1] * th.sin(), xv[..., 0] * th.sin() + xv[..., 1] * th.cos()], -1).reshape(rows, HD)
    o = {c: torch.full_like(x, float("nan")) for c in ("Y", "Triton")}
    cols = {"Y": lambda: kp.launch((rows, 1, 1), (32, 1, 1), [x, pos, o["Y"]]),
            "Triton": lambda: t_rope[(triton.cdiv(rows, 16),)](x, pos, o["Triton"], rows, D=HD, ROWS=16, num_warps=4)}
    invf = inv.float()
    def eager():
        t = pos.float()[:, None] * invf[None, :]
        xf = x.float().view(rows, HD // 2, 2)
        o["eager"] = torch.stack([xf[..., 0] * t.cos() - xf[..., 1] * t.sin(), xf[..., 0] * t.sin() + xf[..., 1] * t.cos()], -1).reshape(rows, HD).half()
    cols["PyTorch"] = eager
    def checks(c):
        out = o["eager" if c == "PyTorch" else c]
        err = (out.double() - ref).abs().max().item()
        assert torch.isfinite(out).all() and err <= 4e-3 * ref.abs().max().item(), (c, err)
    bench(f"rope_128 rows {rows}", cols, checks, rows * (HD * 2 * 2 + 4))
