"""Exact attention's softmax integers without float64 on the common path.

    p, l = weights(s, a, sk, mask, table, ln2, recip6)

`attention.exact_softmax_weights` forms, per score, the logit
`L = RN64(RN64(s * a) * sk)` (s the integer score, a = sq * sc * log2(e) * 2^16
per query row, sk the key's scale), the row maximum m of the unblocked L, and
`tq = round(RN64(m - L))` (half to even), clamped to 0 .. 2^30, then the
integer exp2 and the row sums. Every step is float64, which a GeForce GPU
runs at 1/64 of its float32 rate (a T4 at 1/32); it was 9% of a training step
on an RTX 4070 Ti.

This kernel returns the same integers p and l. It forms L and m in double
float32 (a pair hi + lo; the products error-free by Dekker's split, without
FMA, so Triton's CPU interpreter computes the same thing), whose error
against the float64 values is far below 2^-16 for |L| < 2^24. tq is then
determined unless the difference lies within 2^-16 of a half-integer, a
logit is 2^24 or more (or not a number), or the key is far enough out
(tq past 30 * 2^16) that its weight is 0 whatever the rounding. A row block
with any undetermined element recomputes in float64, the Python's arithmetic
in its order (FMA off). proofs/ExactAttnWeights.v proves the double float's
error bound and the guard's decision, ExactAttnDiff.v the float32
difference's error, ExactAttnPairMax.v the row maximum of the pairs and
ExactAttnRows.v the row indices; python/tests/test_exact_attn_weights_kernel.py
compares it with the Python on ties, near ties, blocked rows and random rows,
under the interpreter and on a GPU. `Y_EXACT_ATTN_P_KERNEL=0` turns it off.
"""

import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:                          # pragma: no cover - CPU-only installs
    triton = None

ENABLED = os.environ.get("Y_EXACT_ATTN_P_KERNEL", "1") != "0"
MAX_T = 4096                                 # a row of keys held at once
# The dtypes whose values float32 holds exactly, so the kernel's float32
# loads read the values the Python widens to float64.
_NARROW = (torch.float32, torch.float16, torch.bfloat16)


if triton is not None:
    @triton.jit
    def _split(a):
        c = a * 4097.0
        hi = c - (c - a)
        return hi, a - hi


    @triton.jit
    def _two_prod(a, b):
        p = a * b
        ah, al = _split(a)
        bh, bl = _split(b)
        e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
        return p, e


    @triton.jit
    def _fast_two_sum(a, b):
        s = a + b
        return s, b - (s - a)


    @triton.jit
    def _two_sum(a, b):
        s = a + b
        bb = s - a
        return s, (a - (s - bb)) + (b - bb)


    @triton.jit
    def _rint32(x):
        ax = tl.abs(x)
        r = tl.where(ax < 8388608.0, (ax + 8388608.0) - 8388608.0, ax)
        return tl.where(x < 0, -r, r)


    @triton.jit
    def _rint64_nonneg(x):
        return tl.where(x < 4503599627370496.0, (x + 4503599627370496.0) - 4503599627370496.0, x)


    @triton.jit
    def _exp2_q16(ti, table_ptr, LN2: tl.constexpr, RECIP6: tl.constexpr):
        ti = tl.maximum(ti, 0)
        n = ti >> 16
        f = ti & 0xFFFF
        idx = tl.minimum(tl.maximum(f >> 10, 0), 63)
        base = tl.load(table_ptr + idx)
        d = f & 0x3FF
        y = (d * LN2) >> 16
        y2 = (y * y) >> 30
        y3 = (y2 * y) >> 30
        y3o6 = (y3 * RECIP6) >> 32
        corr = 1073741824 - y + (y2 >> 1) - y3o6
        g = (base * corr) >> 30
        sh = tl.minimum(2 + n, 62)
        out = (g + (tl.full(sh.shape, 1, tl.int64) << (sh - 1))) >> sh
        return tl.where(n >= 30, 0, out)


    @triton.jit
    def _attn_p_kernel(s_ptr, a_ptr, sk_ptr, mask_ptr, table_ptr, p_ptr, l_ptr, fb_ptr,
                       R, T, Q, NH,
                       HAS_MASK: tl.constexpr, FORCE64: tl.constexpr, BT: tl.constexpr, BR: tl.constexpr,
                       LN2: tl.constexpr, RECIP6: tl.constexpr):
        rows = tl.program_id(0) * BR + tl.arange(0, BR)
        cols = tl.arange(0, BT)
        rmask = rows < R
        cmask = cols < T
        m2 = rmask[:, None] & cmask[None, :]
        offs = rows[:, None].to(tl.int64) * T + cols[None, :]
        s = tl.load(s_ptr + offs, mask=m2, other=0.0)
        a64 = tl.load(a_ptr + rows, mask=rmask, other=1.0)
        skrow = (rows // Q).to(tl.int64)
        sk = tl.load(sk_ptr + skrow[:, None] * T + cols[None, :], mask=m2, other=0.0)
        if HAS_MASK:
            mrow = ((rows // (NH * Q)) * Q + rows % Q).to(tl.int64)
            mv = tl.load(mask_ptr + mrow[:, None] * T + cols[None, :], mask=m2, other=0.0)
            blocked = (mv < -1.0) | (m2 == 0)
        else:
            blocked = m2 == 0

        # L in double float32.
        a_hi = a64.to(tl.float32)
        a_lo = (a64 - a_hi.to(tl.float64)).to(tl.float32)
        p1, e1 = _two_prod(s, a_hi[:, None])
        e1 = e1 + s * a_lo[:, None]
        x_hi, x_lo = _fast_two_sum(p1, e1)
        q1, f1 = _two_prod(x_hi, sk)
        f1 = f1 + x_lo * sk
        l_hi, l_lo = _fast_two_sum(q1, f1)
        lh = tl.where(blocked, float("-inf"), l_hi)
        m_hi = tl.max(lh, axis=1)
        m_lo = tl.max(tl.where(lh == m_hi[:, None], l_lo, float("-inf")), axis=1)
        m_lo = tl.where(m_hi == float("-inf"), 0.0, m_lo)
        m_hi_s = tl.where(m_hi == float("-inf"), 0.0, m_hi)
        d_hi, d_e = _two_sum(m_hi_s[:, None], -l_hi)
        d_lo = d_e + (m_lo[:, None] - l_lo)
        k1 = _rint32(d_hi)
        k2 = _rint32(d_lo)
        r = (d_hi - k1) + (d_lo - k2)
        step = tl.where(r > 0.5, 1, tl.where(r < -0.5, -1, 0))
        k = k1.to(tl.int64) + k2.to(tl.int64) + step.to(tl.int64)
        near = tl.abs(tl.abs(r) - 0.5) <= 0.0000152587890625            # GUARD = 2^-16
        big = (tl.abs(l_hi) >= 16777216.0) | (l_hi != l_hi)          # 2^24
        far = (k1.to(tl.int64) + k2.to(tl.int64)) >= 30 * 65536 + 2
        unsafe = (blocked == 0) & (big | (near & (far == 0)))
        tq = tl.where(blocked, 1073741824, tl.minimum(tl.maximum(k, 0), 1073741824))
        tq = tl.where((blocked == 0) & far & (big == 0), 1073741824, tq)
        n_unsafe = tl.max(tl.max(unsafe.to(tl.int32), axis=1), axis=0)
        if FORCE64 or n_unsafe > 0:
            s64 = s.to(tl.float64)
            sk64 = sk.to(tl.float64)
            logit = (s64 * a64[:, None]) * sk64
            m64 = tl.max(tl.where(blocked, -1.7976931348623157e308, logit), axis=1)
            dd = m64[:, None] - logit
            t64 = _rint64_nonneg(tl.maximum(dd, 0.0))
            t64 = tl.where(blocked, 1073741824.0, tl.minimum(t64, 1073741824.0))
            tq = t64.to(tl.int64)
            tl.store(fb_ptr + rows, 1, mask=rmask)
        p = _exp2_q16(tq, table_ptr, LN2, RECIP6)
        p = tl.where(blocked, 0, p)
        tl.store(p_ptr + offs, p, mask=m2)
        tl.store(l_ptr + rows, tl.sum(p, axis=1), mask=rmask)


def launch(s, a, sk, mask, table, Q, NH, ln2, recip6, force64=False, BR=1):
    """p int64 [R][T], l int64 [R], fallback flags int8 [R], for s float32
    [R][T] (integer scores, rows in (b, h, q) order), a float64 [R], sk
    float32 [b * h][T], mask float32 [b * Q][T] or None."""
    R, T = s.shape
    p = torch.empty((R, T), dtype=torch.int64, device=s.device)
    l = torch.empty((R,), dtype=torch.int64, device=s.device)
    fb = torch.zeros((R,), dtype=torch.int8, device=s.device)
    if R == 0:
        return p, l, fb
    kw = {} if s.device.type == "cpu" else {"enable_fp_fusion": False}
    _attn_p_kernel[(triton.cdiv(R, BR),)](
        s, a, sk, mask if mask is not None else s, table, p, l, fb, R, T, Q, NH,
        HAS_MASK=mask is not None, FORCE64=force64, BT=triton.next_power_of_2(T), BR=BR,
        LN2=ln2, RECIP6=recip6, **kw)
    return p, l, fb


def _impl(s, a, sk, mask, table, Q, NH, ln2, recip6):
    with torch.cuda.device(s.device):
        p, l, _ = launch(s, a, sk, mask, table, Q, NH, ln2, recip6)
    return p, l


def _fake(s, a, sk, mask, table, Q, NH, ln2, recip6):
    return s.new_empty(s.shape, dtype=torch.int64), s.new_empty(s.shape[:1], dtype=torch.int64)


_LIB = globals().get("_LIB")


def _register_op():
    global _LIB
    if triton is None:
        return None
    if not hasattr(torch.ops.y_lang, "exact_attn_p"):
        lib = torch.library.Library("y_lang", "FRAGMENT")
        lib.define("exact_attn_p(Tensor s, Tensor a, Tensor sk, Tensor? mask, Tensor table, "
                   "SymInt Q, SymInt NH, int ln2, int recip6) -> (Tensor, Tensor)")
        lib.impl("exact_attn_p", _impl, "CUDA")
        torch.library.register_fake("y_lang::exact_attn_p", _fake, lib=lib)
        _LIB = lib
    return torch.ops.y_lang.exact_attn_p


_OP = _register_op()


def usable(s, mask, sk=None):
    """The kernel takes this call: CUDA scores [b, h, q, T] with T <= MAX_T,
    a mask of shape [b, 1, q, T] or none, and scores, mask and key scales in
    a dtype float32 holds exactly (a float64 mask or scale keeps the Python
    path, whose float64 compare and product would see bits float32 drops)."""
    if not ENABLED or _OP is None or not s.is_cuda or s.dim() != 4 or s.shape[-1] > MAX_T:
        return False
    if s.dtype not in _NARROW or (sk is not None and sk.dtype not in _NARROW):
        return False
    if mask is None:
        return True
    b, _, q, t = s.shape
    return tuple(mask.shape) == (b, 1, q, t) and mask.dtype in _NARROW


def weights(s, a, sk, mask, table, ln2, recip6):
    """p [b, h, q, T] and l [b, h, q, 1] (int64) for s float32 [b, h, q, T],
    a float64 [b, h, q, 1], sk float32 [b, h, 1, T], mask [b, 1, q, T] or
    None: `attention.exact_softmax_weights`' integers."""
    b, h, q, t = s.shape
    m = None if mask is None else mask.reshape(b * q, t).to(torch.float32).contiguous()
    p, l = _OP(s.reshape(b * h * q, t).to(torch.float32).contiguous(),
               a.reshape(-1).to(torch.float64).contiguous(),
               sk.reshape(b * h, t).to(torch.float32).contiguous(), m, table, q, h, ln2, recip6)
    return p.reshape(b, h, q, t), l.reshape(b, h, q, 1)
