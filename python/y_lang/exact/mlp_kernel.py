"""The MLP's gate and up projections and their glue in one Triton kernel.

Exact training's MLP forward (`lora._layer_forward`) computes, for the
input x of a layer's MLP,

    g  = gate(x) + (u_g B_g^T) * scale_g      u_g = x A_g^T   (lora._lora_forward)
    yu = up(x)   + (u_u B_u^T) * scale_u
    a  = silu(g)                              (model.ExactSiLU, a table)
    m  = a * yu

with gate and up frozen `ExactLinear`s (int8 weights, `acc * sx * sw`) and
each adapter's second product `qmm(q8(u), q8(B))`. Compiled, that is two
int8 GEMMs writing int32 results, two adapter products writing float32
results and a glue kernel reading all four: the glue runs at 85-95% of the
card's DRAM bandwidth (measured 2026-11-27, RTX 4070 Ti SUPER), so its time
is those bytes. Here one kernel does all of it per output tile: both int8
products on the shared x8 tile (int32, exact), the adapters' rank-r products
from their q8 operands (qmm's one-digit epilogue), and the same float32
operations in the same order, with FMA contraction off. Nothing but g, yu, a
and m is written.

The same bits as the separate path: every integer sum is exact, so any tile
gives the same integers; each float32 operation is the one the separate path
performs, on the same operands, one rounding each (`acc * sx * sw` as every
`ExactLinear` GEMM backend computes it, then `+ z * scale`; the table's
sigmoid in integers, as `ExactSiLU.sigmoid`; `g * sig`; `a * yu`).
python/tests/test_exact_mlp_kernel.py compares them bit for bit.

Needs Triton's int8 dot (CUDA, sm_80 or later; `qmm.available`). Elsewhere,
and with `Y_EXACT_MLP_FUSED=0`, `lora` keeps the separate path.
"""

import os

import torch

from . import model as _model
from . import qmm as _qmm
from .train_lm import q8

FUSED = os.environ.get("Y_EXACT_MLP_FUSED", "1") != "0"
_LIB = globals().get("_LIB")

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:                                        # pragma: no cover
    triton = None


if triton is not None:
    @triton.jit
    def _z_tile(qu_ptr, su_ptr, qb_ptr, sb_ptr, rm, rn, M, N, R, s_um, s_bn, BR: tl.constexpr):
        """float32 [BM][BN]: `qmm(q8(u), q8(B))` for one-digit operands."""
        rr = tl.arange(0, BR)
        qu = tl.load(qu_ptr + rm[:, None] * s_um + rr[None, :],
                     mask=(rm[:, None] < M) & (rr[None, :] < R), other=0)
        qb = tl.load(qb_ptr + rn[:, None] * s_bn + rr[None, :],
                     mask=(rn[:, None] < N) & (rr[None, :] < R), other=0)
        c = tl.dot(qu.to(tl.int8), tl.trans(qb.to(tl.int8)), out_dtype=tl.int32).to(tl.int64)
        ea = tl.load(su_ptr + rm, mask=rm < M, other=0).to(tl.int64)
        eb = tl.load(sb_ptr + rn, mask=rn < N, other=0).to(tl.int64)
        e = ea[:, None] + eb[None, :]
        # qmm's epilogue: float32 where 2^e is a normal float32, else float64.
        ok = (e >= -126) & (e <= 127)
        two_e32 = ((tl.where(ok, e, 0) + 127) << 23).to(tl.int32).to(tl.float32, bitcast=True)
        out = c.to(tl.float32) * two_e32
        if tl.max((ok == 0).to(tl.int32)) > 0:
            two_e = ((e + 1023) << 52).to(tl.float64, bitcast=True)
            out = tl.where(ok, out, (c.to(tl.float64) * two_e).to(tl.float32))
        return out

    @triton.jit
    def _sigmoid_tile(x, t_ptr, d_ptr, TN):
        """`ExactSiLU.sigmoid` of float32 x (SILU_LIM 16, SILU_QX 20,
        SILU_KNOT 8, SILU_QS 24; `_check_constants` holds them)."""
        xf = tl.where(x != x, 0.0, x)                        # nan_to_num(nan=0)
        xf = tl.minimum(tl.maximum(xf, -16.0), 16.0)
        q = libdevice.nearbyint(xf * 1048576.0).to(tl.int32) + (16 << 20)
        i = tl.minimum(q >> 12, TN - 1)
        s = tl.load(t_ptr + i) + ((tl.load(d_ptr + i) * (q & 4095)) >> 12)
        sig = s.to(tl.float32) * 5.960464477539063e-08       # 2^-24
        sig = tl.where(xf <= -16.0, 0.0, sig)
        return tl.where(xf >= 16.0, 1.0, sig)

    @triton.autotune(
        configs=[
            triton.Config({"BM": 64, "BN": 128, "BK": 64}, num_warps=8, num_stages=3),
            triton.Config({"BM": 64, "BN": 64, "BK": 64}, num_warps=4, num_stages=3),
            triton.Config({"BM": 32, "BN": 128, "BK": 64}, num_warps=4, num_stages=3),
            triton.Config({"BM": 32, "BN": 64, "BK": 64}, num_warps=4, num_stages=2),
        ],
        key=["M_BUCKET", "N", "K"],
    )
    @triton.jit
    def _mlp_kernel(x_ptr, sx_ptr, wg_ptr, wu_ptr, swg_ptr, swu_ptr,
                    qug_ptr, sug_ptr, qbg_ptr, sbg_ptr, quu_ptr, suu_ptr, qbu_ptr, sbu_ptr,
                    t_ptr, d_ptr, g_ptr, yu_ptr, a_ptr, m_ptr,
                    M, N, K, R, TN, s_xm, s_xk, s_wk, s_wn, s_ugm, s_bgn, s_uum, s_bun, s_om,
                    scale_g, scale_u, M_BUCKET,
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
            x = tl.load(x_ptr + rm[:, None] * s_xm + kk[None, :] * s_xk,
                        mask=(rm[:, None] < M) & (kk[None, :] < K), other=0)
            wmask = (kk[:, None] < K) & (rn[None, :] < N)
            woff = kk[:, None] * s_wk + rn[None, :] * s_wn
            accg += tl.dot(x, tl.load(wg_ptr + woff, mask=wmask, other=0), out_dtype=tl.int32)
            accu += tl.dot(x, tl.load(wu_ptr + woff, mask=wmask, other=0), out_dtype=tl.int32)
        sx = tl.load(sx_ptr + rm, mask=rm < M, other=1.0)
        swg = tl.load(swg_ptr + rn, mask=rn < N, other=0.0)
        swu = tl.load(swu_ptr + rn, mask=rn < N, other=0.0)
        zg = _z_tile(qug_ptr, sug_ptr, qbg_ptr, sbg_ptr, rm, rn, M, N, R, s_ugm, s_bgn, BR)
        g = (accg.to(tl.float32) * sx[:, None]) * swg[None, :] + zg * scale_g
        zu = _z_tile(quu_ptr, suu_ptr, qbu_ptr, sbu_ptr, rm, rn, M, N, R, s_uum, s_bun, BR)
        yu = (accu.to(tl.float32) * sx[:, None]) * swu[None, :] + zu * scale_u
        a = g * _sigmoid_tile(g, t_ptr, d_ptr, TN)
        m = a * yu
        omask = (rm[:, None] < M) & (rn[None, :] < N)
        off = rm[:, None] * s_om + rn[None, :]
        tl.store(g_ptr + off, g, mask=omask)
        tl.store(yu_ptr + off, yu, mask=omask)
        tl.store(a_ptr + off, a, mask=omask)
        tl.store(m_ptr + off, m, mask=omask)


def _check_constants():
    """The kernel's sigmoid hard-codes ExactSiLU's constants."""
    got = (_model.SILU_LIM, _model.SILU_QX, _model.SILU_KNOT, _model.SILU_QS)
    if got != (16, 20, 8, 24):
        raise RuntimeError(f"mlp_kernel: ExactSiLU's constants changed to {got}; the kernel has (16, 20, 8, 24)")


def _bucket(n):
    return 1 << max(4, (int(n) - 1).bit_length())


def _gate_up(x8, sx, wg, wu, swg, swu, qug, sug, qbg, sbg, quu, suu, qbu, sbu, t, d,
             scale_g: float, scale_u: float):
    M, K = x8.shape
    N = wg.shape[1]
    R = qug.shape[1]
    out = [torch.empty((M, N), dtype=torch.float32, device=x8.device) for _ in range(4)]
    if M == 0 or N == 0:
        return tuple(out)
    BR = max(32, triton.next_power_of_2(R))                 # int8 dots take K >= 32
    grid = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))  # noqa: E731
    _mlp_kernel[grid](x8, sx, wg, wu, swg, swu, qug, sug, qbg, sbg, quu, suu, qbu, sbu, t, d, *out,
                      M, N, K, R, t.numel(), x8.stride(0), x8.stride(1), wg.stride(0), wg.stride(1),
                      qug.stride(0), qbg.stride(0), quu.stride(0), qbu.stride(0), out[0].stride(0),
                      float(scale_g), float(scale_u), _bucket(M), BR=BR, enable_fp_fusion=False)
    return tuple(out)


def _register():
    global _LIB
    if triton is None or not torch.cuda.is_available():
        return None
    if not hasattr(torch.ops.y_lang, "exact_mlp_gate_up"):
        lib = torch.library.Library("y_lang", "FRAGMENT")
        lib.define("exact_mlp_gate_up(Tensor x8, Tensor sx, Tensor wg, Tensor wu, Tensor swg, Tensor swu, "
                   "Tensor qug, Tensor sug, Tensor qbg, Tensor sbg, Tensor quu, Tensor suu, Tensor qbu, "
                   "Tensor sbu, Tensor t, Tensor d, float scale_g, float scale_u) "
                   "-> (Tensor, Tensor, Tensor, Tensor)")
        lib.impl("exact_mlp_gate_up", _gate_up, "CUDA")

        def fake(x8, sx, wg, *rest):
            shape = (x8.shape[0], wg.shape[1])
            return tuple(x8.new_empty(shape, dtype=torch.float32) for _ in range(4))
        torch.library.register_fake("y_lang::exact_mlp_gate_up", fake, lib=lib)
        _LIB = lib
    return torch.ops.y_lang.exact_mlp_gate_up


_OP = _register()


def usable(gate, up, act, x, grad_bits, bwd_bits, mm, act_dt) -> bool:
    """Whether `gate_up_silu` gives the separate path's result for these
    modules: the conditions under which that path's adapter products take
    `qmm` (`lora._exact_mm`) and its frozen products `acc * sx * sw` with no
    bias and one int8 digit of x."""
    from . import lora
    return (FUSED and _OP is not None and x.is_cuda and x.dtype == torch.float32
            and lora.QMM and mm is lora.mm_y_any and act_dt is torch.float32
            and _qmm.available(x.device)
            and all(isinstance(b, _model.ExactLinear) and not b.split_act and b.bias is None
                    for b in (gate, up))
            and gate.in_features == up.in_features and gate.out_features == up.out_features
            # The int32 accumulators hold K terms of 127 * 127 only to
            # K = 133,144 (`proofs/ExactMlpKernel.v`); past qmm's bound, the
            # separate path, which splits or refuses such K.
            and gate.in_features <= _qmm.MAX_K
            and isinstance(act, _model.ExactSiLU))


def gate_up_silu(x, Ag, Bg, Au, Bu, gate, up, act, scale_g, scale_u, grad_bits, bwd_bits, mm, act_dt):
    """`lora._layer_forward`'s g, yu, a = silu(g), m = a * yu, the adapters'
    x as float32 rows and u_g, u_u, as the separate path makes them:
    ((g, yu, a, m, xf, u_g, u_u), check conditions, their messages). The
    caller has checked `usable`."""
    from . import lora
    _check_constants()
    bad = []
    shape = x.shape
    K, N = gate.in_features, gate.out_features
    xf = x.reshape(-1, K).to(torch.float32)
    # The frozen products' x8 and sx, as `ExactLinear.forward` makes them.
    xq, sx = _model._q_per_row(xf)
    x8 = xq.to(torch.int8)
    sx = sx.reshape(-1).to(torch.float32)
    # The adapters' first products and their second products' operands, as
    # `_exact_mm` makes them (forward widths: 8 and 8), with their checks in
    # the separate path's order (gate's two products, then up's).
    w = (grad_bits, bwd_bits, mm, act_dt)
    u_g = lora._exact_mm(xf, Ag, None, *w, bad)
    qug, sug = q8(u_g, bits=8, bad=bad)
    qbg, sbg = q8(Bg, bits=8, bad=bad)
    u_u = lora._exact_mm(xf, Au, None, *w, bad)
    quu, suu = q8(u_u, bits=8, bad=bad)
    qbu, sbu = q8(Bu, bits=8, bad=bad)
    g, yu, a, m = _OP(x8, sx, gate.w8t, up.w8t, gate.sw.reshape(-1), up.sw.reshape(-1),
                      qug, sug.reshape(-1), qbg, sbg.reshape(-1), quu, suu.reshape(-1), qbu, sbu.reshape(-1),
                      act.t, act.d, float(scale_g), float(scale_u))
    out_shape = (*shape[:-1], N)
    g, yu, a, m = (t.reshape(out_shape) for t in (g, yu, a, m))
    return (g, yu, a, m, xf, u_g, u_u), [c for c, _ in bad], [msg for _, msg in bad]
