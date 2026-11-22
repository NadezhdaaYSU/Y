"""Exact training's causal softmax and the quantizer of its output, in one
kernel.

    P, q, s, bad = softmax_q8(z, T)     # z [R][T] float32, R = batch * heads * T

For float32 scores `z` (rows of T keys; row r is query r mod T), computes
what `train_lm.Ops.softmax(z, causal)` and then `q8(P, bits=8)` compute, the
same bits: the integer exp2 softmax `P` (float32), its int8 copy `q` and
row exponents `s` (int64, [R][1]) for the product with V, and per-row
refusal flags `bad` (1: a softmax input not finite, 2: a P not finite, 4: a
quantizer scale out of range) that the caller hands to `train_lm.refuse`.
It replaces two of Inductor's kernels over the attention-sized tensors, which
on an H100 took 0.86 ms of a width-512 training step's 7.86.

Every operation is the one the Python performs, in its order, so the bits
cannot differ: float32 `z * c` with c the float32 of log2(e) * 2^16 (which is
what PyTorch multiplies a float32 tensor by), the row max, the rounding
of m - logit, the integer exp2 of `attention.exp2_neg_q16_16`, the integer
row sum, a correctly rounded division (`div_rn`; its operands are normal, so
its subnormal flush cannot act), and q8's round(P * 2^a * 2^b). FMA fusion is
off for the launch, so a product feeding a subtraction is rounded first, as
in PyTorch. The roundings to an integer are of values x >= 0: (x + 2^23) -
2^23 below 2^23, where the sum's spacing is 1 and so the addition rounds to
nearest, ties to even, and x itself above (a float32 there is an integer);
a negative x is clamped to 0, as in the Python, whatever it rounds to.
python/tests/test_exact_softmax_kernel.py checks it against the Python under
Triton's interpreter on the CPU and on a GPU.
"""

import math

import torch
import triton
import triton.language as tl

from .attention import LN2_Q30, RECIP6_Q32, _table

LOG2E_Q16_F32 = float(torch.tensor(math.log2(math.e) * 65536.0, dtype=torch.float32))
TWO23 = 8388608.0


@triton.jit
def _rne_nonneg(x):
    """round-half-even of float32 x >= 0 (any value for x < 0, which the
    callers clamp to 0)."""
    return tl.where(x < 8388608.0, (x + 8388608.0) - 8388608.0, x)


@triton.jit
def _pow2f(k):
    return ((k + 127) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _softmax_q8_kernel(z_ptr, p_ptr, q_ptr, s_ptr, bad_ptr, table_ptr, R,
                       T: tl.constexpr, BR: tl.constexpr, LOG2E: tl.constexpr,
                       LN2: tl.constexpr, RECIP6: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    cols = tl.arange(0, T)
    rmask = rows < R
    offs = rows[:, None].to(tl.int64) * T + cols[None, :]
    m2 = rmask[:, None]
    z = tl.load(z_ptr + offs, mask=m2, other=0.0)
    zbits = z.to(tl.int32, bitcast=True) & 0x7FFFFFFF
    bad_z = tl.max(((zbits >> 23) == 255).to(tl.int32), axis=1)
    blocked = cols[None, :] > (rows % T)[:, None]
    # Ops.softmax: logit, row max over the keys a query may see, t.
    logit = z * LOG2E
    m = tl.max(tl.where(blocked, float("-inf"), logit), axis=1)
    t = _rne_nonneg(m[:, None] - logit)
    t = tl.minimum(tl.maximum(t, 0.0), 1073741824.0)
    t = tl.where(blocked, 1073741824.0, t)
    # attention.exp2_neg_q16_16, in int64.
    ti = tl.maximum(t.to(tl.int64), 0)
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
    p = tl.where(n >= 30, 0, out)
    p = tl.where(blocked, 0, p)
    psum = tl.sum(p, axis=1)
    P = tl.math.div_rn(p.to(tl.float32), psum.to(tl.float32)[:, None])
    tl.store(p_ptr + offs, P, mask=m2)
    # q8(P, bits=8): exponent from the bits, the float32 rounding formula.
    pbits = P.to(tl.int32, bitcast=True) & 0x7FFFFFFF
    amax = tl.max(pbits, axis=1)
    field = amax >> 23
    e = tl.where(amax == 0, 0, field - 126)
    s = e - 7
    bad = bad_z | (((field == 255).to(tl.int32)) << 1) | (((s < -511) | (s > 511)).to(tl.int32) << 2)
    a = tl.minimum(tl.maximum(-s, -126), 127)
    b = tl.minimum(tl.maximum(-s - a, -126), 127)
    yq = (P * _pow2f(a)[:, None]) * _pow2f(b)[:, None]
    yq = tl.where(yq < 0, -_rne_nonneg(-yq), _rne_nonneg(yq))
    yq = tl.minimum(tl.maximum(yq, -127.0), 127.0)
    tl.store(q_ptr + offs, yq.to(tl.int8), mask=m2)
    tl.store(s_ptr + rows, s.to(tl.int64), mask=rmask)
    tl.store(bad_ptr + rows, bad.to(tl.int8), mask=rmask)


def supported(T: int) -> bool:
    """The kernel holds a row of T keys: a power of two from 16 to 1024."""
    return 16 <= T <= 1024 and T & (T - 1) == 0


def launch(z, T, table, BR=None):
    """(P, q, s, bad) for float32 z [R][T] on any device Triton runs on (the
    CPU too, under TRITON_INTERPRET=1)."""
    if z.dtype != torch.float32 or z.dim() != 2 or z.shape[1] != T or not supported(T):
        raise ValueError(f"softmax_q8: z must be [R][T] float32 with T a power of two from 16 to "
                         f"1024, not {tuple(z.shape)} {z.dtype}")
    z = z.contiguous()
    R = z.shape[0]
    if BR is None:
        BR = max(1, 2048 // T)
    P = torch.empty_like(z)
    q = torch.empty(z.shape, dtype=torch.int8, device=z.device)
    s = torch.empty((R, 1), dtype=torch.int64, device=z.device)
    bad = torch.empty((R, 1), dtype=torch.int8, device=z.device)
    kw = {} if z.device.type == "cpu" else {"enable_fp_fusion": False}
    _softmax_q8_kernel[(triton.cdiv(R, BR),)](z, P, q, s, bad, table, R, T=T, BR=BR,
                                              LOG2E=LOG2E_Q16_F32, LN2=LN2_Q30, RECIP6=RECIP6_Q32, **kw)
    return P, q, s, bad


_TABLES = {}


def _impl(z, T):
    table = _TABLES.get(z.device)
    if table is None:
        table = _TABLES[z.device] = _table(z.device)
    with torch.cuda.device(z.device):
        return launch(z, T, table)


def _fake(z, T):
    R = z.shape[0]
    return (torch.empty_like(z), z.new_empty(z.shape, dtype=torch.int8),
            z.new_empty((R, 1), dtype=torch.int64), z.new_empty((R, 1), dtype=torch.int8))


_LIB = globals().get("_LIB")


def _register_op():
    global _LIB
    if not hasattr(torch.ops.y_lang, "exact_softmax_q8"):
        lib = torch.library.Library("y_lang", "FRAGMENT")
        lib.define("exact_softmax_q8(Tensor z, int T) -> (Tensor, Tensor, Tensor, Tensor)")
        lib.impl("exact_softmax_q8", _impl, "CUDA")
        torch.library.register_fake("y_lang::exact_softmax_q8", _fake, lib=lib)
        _LIB = lib
    return torch.ops.y_lang.exact_softmax_q8


_OP = _register_op()


def softmax_q8(z: torch.Tensor, T: int):
    """The op, for CUDA z [..., T] (rows of T keys, query = row index mod T):
    P, q of z's shape; s, bad [..., 1]."""
    lead = z.shape[:-1]
    P, q, s, bad = _OP(z.reshape(-1, T), T)
    return P.reshape(z.shape), q.reshape(z.shape), s.reshape(*lead, 1), bad.reshape(*lead, 1)
