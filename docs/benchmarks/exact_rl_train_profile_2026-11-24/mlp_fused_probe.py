"""Probe (2026-11-27): the MLP's forward glue in the gate/up GEMM's epilogue.

The kernel here is the first version of `python/y_lang/exact/mlp_kernel.py`,
kept as the measurement of the speed quoted in README.md.

One Triton kernel computes, per tile, both frozen int8 products (gate, up) on
the shared x8 tile, the adapters' rank-r products from their q8 operands,
the same float32 steps as the eager path (acc * sx * sw + z * scale, the
table SiLU, a * yu), writes g, yu, a, m, and the row maxima of |m|'s bits
(atomic max). Checked bit for bit against lora._lora_forward + ExactSiLU.

python mlp_fused_probe.py [M] [K] [N] [r]
"""
import os
import sys
import time

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

Y = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")   # the repository
sys.path.insert(0, os.path.join(Y, "python"))
from y_lang.exact import lora, model as M  # noqa: E402
from y_lang.exact.train_lm import q8  # noqa: E402


@triton.jit
def _z_tile(qu_ptr, su_ptr, qb_ptr, sb_ptr, rm, rn, Mx, Nx, R, s_um, s_bn, BR: tl.constexpr):
    """float32 [BM][BN]: qmm's one-digit product of q8(u) and q8(B) rows."""
    rr = tl.arange(0, BR)
    qu = tl.load(qu_ptr + rm[:, None] * s_um + rr[None, :], mask=(rm[:, None] < Mx) & (rr[None, :] < R), other=0)
    qb = tl.load(qb_ptr + rn[:, None] * s_bn + rr[None, :], mask=(rn[:, None] < Nx) & (rr[None, :] < R), other=0)
    c = tl.dot(qu.to(tl.int8), tl.trans(qb.to(tl.int8)), out_dtype=tl.int32).to(tl.int64)
    ea = tl.load(su_ptr + rm, mask=rm < Mx, other=0).to(tl.int64)
    eb = tl.load(sb_ptr + rn, mask=rn < Nx, other=0).to(tl.int64)
    e = ea[:, None] + eb[None, :]
    ok = (e >= -126) & (e <= 127)
    two_e32 = ((tl.where(ok, e, 0) + 127) << 23).to(tl.int32).to(tl.float32, bitcast=True)
    out = c.to(tl.float32) * two_e32
    if tl.max((ok == 0).to(tl.int32)) > 0:
        two_e = ((e + 1023) << 52).to(tl.float64, bitcast=True)
        out = tl.where(ok, out, (c.to(tl.float64) * two_e).to(tl.float32))
    return out


@triton.jit
def _sigmoid_tile(x, t_ptr, d_ptr, TN):
    """ExactSiLU.sigmoid: the table's float32 sigmoid of float32 x."""
    xf = tl.where(x != x, 0.0, x)
    xf = tl.minimum(tl.maximum(xf, -16.0), 16.0)
    q = libdevice.nearbyint(xf * 1048576.0).to(tl.int32) + (16 << 20)
    i = tl.minimum(q >> 12, TN - 1)
    s = tl.load(t_ptr + i) + ((tl.load(d_ptr + i) * (q & 4095)) >> 12)
    sig = s.to(tl.float32) * 5.960464477539063e-08
    sig = tl.where(xf <= -16.0, 0.0, sig)
    return tl.where(xf >= 16.0, 1.0, sig)


@triton.jit
def _mlp_kernel(x_ptr, sx_ptr, wg_ptr, wu_ptr, swg_ptr, swu_ptr,
                qug_ptr, sug_ptr, qbg_ptr, sbg_ptr, quu_ptr, suu_ptr, qbu_ptr, sbu_ptr,
                t_ptr, d_ptr, g_ptr, yu_ptr, a_ptr, m_ptr, amax_ptr,
                Mx, Nx, K, R, TN, s_xm, s_wk, s_um, s_bn, s_om, scale_g, scale_u,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, BR: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    accg = tl.zeros((BM, BN), dtype=tl.int32)
    accu = tl.zeros((BM, BN), dtype=tl.int32)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        x = tl.load(x_ptr + rm[:, None] * s_xm + kk[None, :], mask=(rm[:, None] < Mx) & (kk[None, :] < K), other=0)
        wmask = (kk[:, None] < K) & (rn[None, :] < Nx)
        wg = tl.load(wg_ptr + kk[:, None] * s_wk + rn[None, :], mask=wmask, other=0)
        wu = tl.load(wu_ptr + kk[:, None] * s_wk + rn[None, :], mask=wmask, other=0)
        accg += tl.dot(x, wg, out_dtype=tl.int32)
        accu += tl.dot(x, wu, out_dtype=tl.int32)
    sx = tl.load(sx_ptr + rm, mask=rm < Mx, other=1.0)
    swg = tl.load(swg_ptr + rn, mask=rn < Nx, other=0.0)
    swu = tl.load(swu_ptr + rn, mask=rn < Nx, other=0.0)
    zg = _z_tile(qug_ptr, sug_ptr, qbg_ptr, sbg_ptr, rm, rn, Mx, Nx, R, s_um, s_bn, BR)
    g = (accg.to(tl.float32) * sx[:, None]) * swg[None, :] + zg * scale_g
    zu = _z_tile(quu_ptr, suu_ptr, qbu_ptr, sbu_ptr, rm, rn, Mx, Nx, R, s_um, s_bn, BR)
    yu = (accu.to(tl.float32) * sx[:, None]) * swu[None, :] + zu * scale_u
    a = g * _sigmoid_tile(g, t_ptr, d_ptr, TN)
    m = a * yu
    omask = (rm[:, None] < Mx) & (rn[None, :] < Nx)
    off = rm[:, None] * s_om + rn[None, :]
    tl.store(g_ptr + off, g, mask=omask)
    tl.store(yu_ptr + off, yu, mask=omask)
    tl.store(a_ptr + off, a, mask=omask)
    tl.store(m_ptr + off, m, mask=omask)
    bits = tl.where(omask, m.to(tl.int32, bitcast=True) & 0x7FFFFFFF, 0)
    tl.atomic_max(amax_ptr + rm, tl.max(bits, axis=1), mask=rm < Mx)


def fused(x8, sx, wg, wu, swg, swu, zg_ops, zu_ops, t, d, scale_g, scale_u, BM=64, BN=128, BK=64, warps=8, stages=3):
    Mx, K = x8.shape
    Nx = wg.shape[1]
    (qug, sug, qbg, sbg), (quu, suu, qbu, sbu) = zg_ops, zu_ops
    R = qug.shape[1]
    BR = max(32, triton.next_power_of_2(R))      # int8 dots take K >= 32
    out = [torch.empty((Mx, Nx), dtype=torch.float32, device=x8.device) for _ in range(4)]
    amax = torch.zeros(Mx, dtype=torch.int32, device=x8.device)
    grid = (triton.cdiv(Mx, BM), triton.cdiv(Nx, BN))
    _mlp_kernel[grid](x8, sx, wg, wu, swg, swu,
                      qug, sug.reshape(-1).contiguous(), qbg, sbg.reshape(-1).contiguous(),
                      quu, suu.reshape(-1).contiguous(), qbu, sbu.reshape(-1).contiguous(),
                      t, d, *out, amax, Mx, Nx, K, R, t.numel(), x8.stride(0), wg.stride(0),
                      qug.stride(0), qbg.stride(0), out[0].stride(0), float(scale_g), float(scale_u),
                      BM=BM, BN=BN, BK=BK, BR=BR, num_warps=warps, num_stages=stages, enable_fp_fusion=False)
    return (*out, amax)


def bits_eq(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


def main():
    Mx, K, Nx, R = [int(v) for v in (sys.argv[1:] + ["990", "896", "4864", "16"][len(sys.argv) - 1:])][:4]
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    M.GEMM = "torch"
    gate = M.ExactLinear(torch.nn.Linear(K, Nx, bias=False)).to(dev)
    up = M.ExactLinear(torch.nn.Linear(K, Nx, bias=False)).to(dev)
    act = M.ExactSiLU().to(dev)
    x2 = torch.randn(Mx, K, device=dev) * 1.7
    x2[3] = 0.0                                                   # an all-zero row (sx = 1)
    Ag, Au = (torch.randn(R, K, dtype=torch.float64, device=dev) * 0.02 for _ in range(2))
    Bg, Bu = (torch.randn(Nx, R, dtype=torch.float64, device=dev) * 0.05 for _ in range(2))
    scale = 1.0
    w = (12, 8, lora.mm_y_any, torch.float32)
    # The eager path the compiled one is tested against.
    (g_ref, _, ug), _, _ = lora._lora_forward(x2, Ag, Bg, gate, scale, *w)
    (yu_ref, _, uu), _, _ = lora._lora_forward(x2, Au, Bu, up, scale, *w)
    a_ref = act(g_ref)
    m_ref = a_ref * yu_ref
    # The fused kernel's inputs: x quantized as ExactLinear does, the
    # adapters' second products' q8 operands as _exact_mm makes them.
    xq, sx = M._q_per_row(x2)
    x8 = xq.to(torch.int8).contiguous()
    sx = sx.reshape(-1).to(torch.float32).contiguous()
    zg_ops = (*q8(ug, bits=8), *q8(Bg, bits=8))
    zu_ops = (*q8(uu, bits=8), *q8(Bu, bits=8))
    zg_ops = tuple(t.contiguous() for t in zg_ops)
    zu_ops = tuple(t.contiguous() for t in zu_ops)
    args = (x8, sx, gate.w8t.contiguous(), up.w8t.contiguous(), gate.sw.reshape(-1).float().contiguous(),
            up.sw.reshape(-1).float().contiguous(), zg_ops, zu_ops, act.t.contiguous(), act.d.contiguous(), scale, scale)
    g, yu, a, m, amax = fused(*args)
    torch.cuda.synchronize()
    want_amax = (m_ref.view(torch.int32) & 0x7FFFFFFF).amax(-1)
    for name, x, y in (("g", g, g_ref), ("yu", yu, yu_ref), ("a", a, a_ref), ("m", m, m_ref)):
        same = bits_eq(x, y)
        n = (x.view(torch.int32) != y.contiguous().view(torch.int32)).sum().item()
        print(f"{name}: {'the same bits' if same else f'DIFFERENT at {n} of {x.numel()}'}")
    print("row max of |m| bits:", "the same" if torch.equal(amax, want_amax) else "DIFFERENT")

    # Speed: the fused kernel against the pieces it replaces (two frozen
    # products, two rank-r products, the glue), each timed alone.
    def t_ms(fn, n=20):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        best = float("inf")
        for _ in range(n):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            best = min(best, s.elapsed_time(e))
        return best
    print(f"fused kernel: {t_ms(lambda: fused(*args)) * 1000:.1f} us")
    for cfg in ((64, 64, 64, 4, 3), (64, 128, 64, 8, 3), (128, 64, 64, 8, 3), (128, 128, 64, 8, 3), (64, 128, 128, 8, 2)):
        try:
            us = t_ms(lambda: fused(*args, BM=cfg[0], BN=cfg[1], BK=cfg[2], warps=cfg[3], stages=cfg[4])) * 1000
            print(f"   tile {cfg}: {us:.1f} us")
        except Exception as ex:                                    # noqa: BLE001
            print(f"   tile {cfg}: {type(ex).__name__}")

    def unfused():
        (gg, _, u1), _, _ = lora._lora_forward(x2, Ag, Bg, gate, scale, *w)
        (yy, _, u2), _, _ = lora._lora_forward(x2, Au, Bu, up, scale, *w)
        aa = act(gg)
        return aa * yy
    print(f"eager pieces (not the compiled glue): {t_ms(unfused) * 1000:.1f} us")
    import torch._dynamo as dyn
    dyn.config.specialize_int = True
    dyn.config.specialize_float = True
    cu = torch.compile(unfused, dynamic=False)
    m_c = cu()
    print("compiled pieces' m:", "the same bits" if bits_eq(m_c, m_ref) else "DIFFERENT")
    print(f"compiled pieces (the glue fused by Inductor): {t_ms(cu) * 1000:.1f} us")
    print(f"   two frozen products alone: {t_ms(lambda: (gate(x2), up(x2))) * 1000:.1f} us")


if __name__ == "__main__":
    main()
