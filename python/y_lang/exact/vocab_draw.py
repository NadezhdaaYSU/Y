"""The exact sampler's draw over the vocabulary in one Triton kernel.

`draw(z, r, table, bad)` is `rl.draw_rows` for float32 logits `z` [rows][V]
with nothing blocked: for each row the exact softmax's integers
(`softmax_ints`: logit = z * LOG2E_Q16 in float32, t = round(max - logit)
clamped to [0, 2^30], p = the integer exp2 of -t, Q0.28, l = their sum) and
the token the integer inverse CDF draws with r (the first i whose running
sum of p passes r mod l), with that token's p and l. A logit that is not
finite is refused (to `bad`), as `softmax_ints` refuses it.

`rl.draw_rows` writes p for the whole vocabulary in int64 and reads it back
for the sums and the draw (256 rows of 151,936 logits: 311 MB a decode step).
Here p is never stored. The row maxima are torch's, so under torch.compile they
fuse into the kernel that makes z; then a program per row checks every
logit is finite and takes the sum of each block of `BLOCK` entries (kept: a few hundred
integers), finds the block whose running sum passes r mod l, and computes
that block's p again to find the token. Integer sums, so the same token, p
and l whatever the order or the blocking; the float32 steps are the same
operations in the same order, with FMA contraction off (`m - z * C`
contracted would round once where torch rounds twice). Tests compare it with
`draw_rows`' torch code bit for bit (python/tests/test_exact_vocab_draw.py).
"""

import os

import torch

from .attention import LN2_Q30, RECIP6_Q32
from .train_lm import LOG2E_Q16

BLOCK = 1024
FUSED = os.environ.get("Y_EXACT_DRAW", "fused") != "torch"
_C32 = float(torch.tensor(LOG2E_Q16, dtype=torch.float32))   # the constant as float32 rounds it
_LIB = globals().get("_LIB")

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:                                        # pragma: no cover
    triton = None


if triton is not None:
    _LN2 = tl.constexpr(LN2_Q30)
    _R6 = tl.constexpr(RECIP6_Q32)

    @triton.jit
    def _p_of(z, valid, m, C, table_ptr):
        """softmax_ints' p for logits z (int64; 0 where not `valid`)."""
        logit = z * C
        t = libdevice.rint(m - logit)
        t = tl.minimum(tl.maximum(t, 0.0), 1073741824.0)
        t = t.to(tl.int64)
        n = t >> 16
        f = t & 0xFFFF
        idx = tl.minimum(tl.maximum(f >> 10, 0), 63)
        base = tl.load(table_ptr + idx)
        d = f & 0x3FF
        y = (d * _LN2) >> 16
        y2 = (y * y) >> 30
        y3 = (y2 * y) >> 30
        y3o6 = (y3 * _R6) >> 32
        corr = (1 << 30) - y + (y2 >> 1) - y3o6
        g = (base * corr) >> 30
        sh = tl.minimum(2 + n, 62)
        one = tl.full(sh.shape, 1, tl.int64)
        out = (g + (one << (sh - 1))) >> sh
        out = tl.where(n >= 30, 0, out)
        return tl.where(valid, out, 0)

    @triton.jit
    def _draw_kernel(z_ptr, m_ptr, r_ptr, table_ptr, cb_ptr, tok_ptr, p_ptr, l_ptr, bad_ptr, V, s_z, C,
                     NB: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK)
        zrow = z_ptr + row * s_z
        nblk = tl.cdiv(V, BLOCK)
        m = tl.load(m_ptr + row)
        total = tl.zeros([], tl.int64)
        badv = tl.zeros([BLOCK], tl.int32)
        for b in range(0, nblk):
            v = b * BLOCK + offs
            valid = v < V
            z = tl.load(zrow + v, mask=valid, other=0.0)
            badv |= (valid & ((z != z) | (tl.abs(z) == float("inf")))).to(tl.int32)
            s = tl.sum(_p_of(z, valid, m, C, table_ptr), axis=0)
            tl.store(cb_ptr + row * NB + b, s)
            total += s
        u = tl.load(r_ptr + row) % total
        bo = tl.arange(0, NB)
        inb = bo < nblk
        cum = tl.cumsum(tl.load(cb_ptr + row * NB + bo, mask=inb, other=0), axis=0)
        k = tl.sum((inb & (cum <= u)).to(tl.int64), axis=0)
        before = tl.sum(tl.where(bo == k - 1, cum, 0), axis=0)
        v = k * BLOCK + offs
        valid = v < V
        z = tl.load(zrow + v, mask=valid, other=0.0)
        p = _p_of(z, valid, m, C, table_ptr)
        ci = tl.cumsum(p, axis=0) + before
        j = tl.sum((ci <= u).to(tl.int64), axis=0)
        tl.store(tok_ptr + row, k * BLOCK + j)
        tl.store(p_ptr + row, tl.sum(tl.where(offs == j, p, 0), axis=0))
        tl.store(l_ptr + row, total)
        tl.store(bad_ptr + row, (tl.max(badv, axis=0) > 0).to(tl.int8))


def _draw(z, m, r, table):
    rows, V = z.shape
    if z.dtype != torch.float32 or z.stride(1) != 1 or m.dtype != torch.float32:
        raise ValueError("vocab_draw: float32 logits with contiguous rows, float32 maxima")
    nb = triton.next_power_of_2(max(1, triton.cdiv(V, BLOCK)))
    dev = z.device
    cb = torch.empty((rows, nb), dtype=torch.int64, device=dev)
    tok = torch.empty(rows, dtype=torch.int64, device=dev)
    p = torch.empty(rows, dtype=torch.int64, device=dev)
    l = torch.empty(rows, dtype=torch.int64, device=dev)
    bad = torch.zeros(rows, dtype=torch.int8, device=dev)
    if rows:
        _draw_kernel[(rows,)](z, m.contiguous(), r.contiguous(), table.contiguous(), cb, tok, p, l, bad, V,
                              z.stride(0), _C32, NB=nb, BLOCK=BLOCK, num_warps=8, enable_fp_fusion=False)
    return tok, p, l, bad.to(torch.bool)


def _register():
    global _LIB
    if triton is None or not torch.cuda.is_available():
        return None
    if not hasattr(torch.ops.y_lang, "exact_vocab_draw"):
        lib = torch.library.Library("y_lang", "FRAGMENT")
        lib.define("exact_vocab_draw(Tensor z, Tensor m, Tensor r, Tensor table) -> (Tensor, Tensor, Tensor, Tensor)")
        lib.impl("exact_vocab_draw", _draw, "CUDA")

        def fake(z, m, r, table):
            rows = z.shape[0]
            e = z.new_empty((rows,), dtype=torch.int64)
            return e, e.new_empty((rows,)), e.new_empty((rows,)), z.new_empty((rows,), dtype=torch.bool)
        torch.library.register_fake("y_lang::exact_vocab_draw", fake, lib=lib)
        _LIB = lib
    return torch.ops.y_lang.exact_vocab_draw


_OP = _register()


def available(device) -> bool:
    return FUSED and _OP is not None and torch.device(device).type == "cuda"


def draw(z, r, table, bad=None):
    """(token, p of it, l) for float32 logits z [rows][V] and random
    integers r [rows] (int64, below 2^62): `rl.draw_rows` with nothing
    blocked. The row maxima are torch's, so they fuse into the kernel that
    makes z; the check of `softmax_ints` (no logit inf or NaN) is the
    kernel's, per row, refused to `bad`."""
    from .train_lm import refuse
    m = (z * LOG2E_Q16).amax(dim=-1)
    tok, p, l, nonfinite = _OP(z, m, r, table)
    refuse(bad, nonfinite, "exact train: a softmax input that is not finite (inf or NaN)")
    return tok, p, l
