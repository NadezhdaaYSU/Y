"""Exact training, T2: a tiny GPT-style character model trained from scratch,
whose weights after any number of steps are the same bits on a CPU and on a
GPU.

The rule is the inference path's (`attention.exact_attention_core`):

  * every SUM is a sum of integers, so its order cannot matter: the linear
    layers' products are int8 x int8 on an integer GEMM (`mm`, Y's kernel,
    cuBLAS, or the CPU); the attention products are int8 values multiplied
    in float32 with every partial sum an integer below 2^24, so fp32 holds
    each one exactly in any order (checked, `_exact_fp32_bmm`); other sums
    (a norm's mean square, a softmax row sum, an embedding's gradient) are
    taken in int64 after a fixed-point rounding with one power-of-two scale
    per sum (`Ops.sum`);
  * everything else is float64 +, -, * and tensor-by-tensor /, elementwise,
    which IEEE 754 rounds correctly, so every conforming device gives the
    same bits; no float reduction, no transcendental function, no fused
    multiply-add. Two things that look like that are not, both measured on
    this repository's machines (2026-11-21): PyTorch's CUDA divide by a
    Python scalar multiplies by its reciprocal (64,925 of 100,000 float64
    quotients differ from the CPU's), so a constant divisor is written as a
    multiply by a reciprocal computed once in Python; and PyTorch's CPU
    float64 `sqrt` on an AVX-512 build is not correctly rounded (226 of
    20,000 differ from `math.sqrt`; CUDA's matched all 20,000), so square
    roots are integer (`det_sqrt`);
  * the softmax's exp is the integer exp2 (`attention.exp2_neg_q16_16`).

Every int8 operand has one power-of-two scale per row along the summed
dimension, so the scales factor out of the sum exactly.

`Ops(exact=False)` is the same model with ordinary float sums and float
softmax: the float twin (the control, and the accuracy baseline), and, in
float64, the reference the hand-written backward is checked against
(PyTorch autograd, `python/tests/test_exact_train_lm.py`).

The loss printed is computed on the host from the probabilities, for
reading only; nothing in training depends on it.

`Trainer` holds a run. On a GPU, `Trainer(..., graph=True, compile=True)`
compiles the step with `torch.compile` and replays it as one CUDA graph,
with the same bits as the eager step and the CPU (the tests, and the
5000-step records in docs/benchmarks/exact_train_lm_2026-11-21/fast/).
Every check is deferred to `Trainer.verify`, one read of the device.
"""

import hashlib
import math
from dataclasses import dataclass

import torch

from .attention import _table, exp2_neg_q16_16

LOG2E_Q16 = math.log2(math.e) * 65536.0
FP32_EXACT = 1 << 24
F64_EXACT = 1 << 53


# ---- exact building blocks ---------------------------------------------------

def refuse(bad, cond: torch.Tensor, msg: str):
    """Fail closed where any element of the bool tensor `cond` is True.

    With `bad` None the device is read at once and OverflowError raised here.
    Otherwise (`cond.any()`, msg) is appended to the list `bad` and nothing is
    read: `Trainer` reads every flag of a step in one transfer and raises
    before any value of that step reaches the host. A refused value goes on
    being computed with until then, as garbage that no later operation can
    turn into a crash, and is never returned."""
    if bad is None:
        if bool(cond.any()):
            raise OverflowError(msg)
    else:
        bad.append((cond, msg))


def reduce_checks(bad):
    """(flags, messages) for the deferred checks in `bad`: one flag per
    distinct message, any of its conditions, so a step's few hundred checks
    are a few reductions."""
    groups = {}
    for cond, msg in bad:
        groups.setdefault(msg, []).append(cond.reshape(-1))
    return torch.stack([torch.cat(c).any() for c in groups.values()]), tuple(groups)


# The quantizers work on the integers inside a float64, not with float64
# arithmetic: a float64 x is mant * 2^E with an integer mant < 2^53, so
# rounding x * 2^-s to an integer is an integer shift with round-half-even,
# and comparing |x| is comparing its bits. That is the same function as
# `torch.round(x * 2^-s)` (python/tests/test_exact_train_lm.py checks it on
# ties, zeros and subnormals), and on a GeForce GPU, whose float64 units run
# at 1/64 of float32's rate, eight times faster (2026-11-21: 16.6 against
# 133 us for a [1024][512] 12-bit quantizer, compiled, RTX 4070 Ti SUPER).
_INF_FIELD = 0x7FF
Q_SCALE_RANGE = 511          # |s| of a quantizer's scale; 2 * 511 <= 1022


def check_exp(k: torch.Tensor, bad=None):
    """Refuse a scale exponent outside float64's normal range [-1022, 1023]."""
    refuse(bad, (k < -1022) | (k > 1023), "exact train: a scale exponent outside float64's normal range")


def pow2(k: torch.Tensor, bad=None, checked=False) -> torch.Tensor:
    """2^k as float64, built from its bits, for integer k in [-1022, 1023]
    (`checked`: the caller has refused any other k)."""
    k = k.to(torch.int64)
    if not checked:
        check_exp(k, bad)
    return torch.bitwise_left_shift(k + 1023, 52).contiguous().view(torch.float64)


# The two float formats the quantizers read: (the integer type of the same
# width, mantissa bits, the all-ones exponent field, frexp's bias, the bias
# of x = mant * 2^E).
_FMT = {torch.float64: (torch.int64, 52, 0x7FF, 1022, 1075),
        torch.float32: (torch.int32, 23, 0xFF, 126, 150)}


def _fields(x: torch.Tensor):
    """(bits of |x|, biased exponent field) of float64 or float32 x, in the
    integer type of the same width."""
    it, mbits = _FMT[x.dtype][:2]
    a = x.view(it) & torch.iinfo(it).max
    return a, a >> mbits


def exponent(x: torch.Tensor, dim: int, bad=None) -> torch.Tensor:
    """e (int64) with max |x| along `dim` below 2^e: frexp's exponent of the
    max (0 for an all-zero slice), read from its bits; float64 or float32 x.
    A subnormal max gives the format's smallest normal exponent, an upper
    bound where frexp's is tight (a float64 caller then refuses the slice, as
    with frexp; a float32 one quantizes it on that coarser scale, the same
    on every device). A slice holding an inf or a NaN is refused: no scale
    represents it."""
    it, mbits, inf, fbias = _FMT[x.dtype][:4]
    a = x.view(it) & torch.iinfo(it).max    # ordered as |x| is, for finite x
    amax = a.amax(dim=dim, keepdim=True)
    field = amax >> mbits
    refuse(bad, field == inf, "exact train: a value that is not finite (inf or NaN)")
    return torch.where(amax == 0, 0, field - fbias).to(torch.int64)


def _round_shift(mant, r, neg, most):
    """round-half-even(mant / 2^r), negated where `neg`, in mant's integer
    type, for mant >= 0 below 2^(most - 8) and shifts clamped to `most`: a
    shift past it rounds to 0 as `most` does."""
    left = mant << torch.clamp(-r, 0, most)             # r <= 0: exact
    rr = torch.clamp(r, 1, most)
    q0 = mant >> rr
    one = torch.ones_like(rr)
    rem = mant & ((one << rr) - 1)
    half = one << (rr - 1)
    q = q0 + ((rem > half) | ((rem == half) & ((q0 & 1) == 1))).to(mant.dtype)
    q = torch.where(r <= 0, left, q)
    return torch.where(neg, -q, q)


def scaled_round_int(x: torch.Tensor, s: torch.Tensor, wide=True) -> torch.Tensor:
    """round-half-even(x * 2^-s) for finite float64 or float32 x and integer s
    (broadcast), computed in integers, exactly: int64, for |x * 2^-s| < 2^62.
    `wide=False` (float32 x only) works in int32, for a result below 2^30
    that needs no left shift (q8's, at 16 bits or fewer). The definition the
    float32 path of `scaled_round` is checked against."""
    it, mbits, _, _, ebias = _FMT[x.dtype]
    a, field = _fields(x)
    mant = (a & ((1 << mbits) - 1)) | ((field > 0).to(it) << mbits)
    E = torch.clamp(field, min=1) - ebias                # x = mant * 2^E
    neg = x.view(it) < 0
    if wide:
        r = s.to(torch.int64) - E.to(torch.int64)        # x * 2^-s = mant / 2^r
        return _round_shift(mant.to(torch.int64), r, neg, 62)
    if x.dtype != torch.float32:
        raise TypeError("exact train: the int32 rounding is for float32 values")
    return _round_shift(mant, s.to(torch.int32) - E, neg, 30)


def _pow2_f32(k: torch.Tensor) -> torch.Tensor:
    """2^k as float32, from its bits, for integer k in [-126, 127]."""
    return torch.bitwise_left_shift(k.to(torch.int32) + 127, 23).view(torch.float32)


# GPUs whose float64 runs at half float32's rate (P100, V100, A100, H100,
# B200): there the float64 formula below is the cheap one. GeForce and T4
# parts run float64 at 1/32 to 1/64 and keep the integers.
FAST_FLOAT64 = {(6, 0), (7, 0), (8, 0), (9, 0), (10, 0)}
_F64_OK = {}
# None: by the GPU. True or False forces it (tests: the two give the same bits).
FORCE_F64_FORMULA = None


def f64_formula_ok(device) -> bool:
    if FORCE_F64_FORMULA is not None:
        return FORCE_F64_FORMULA
    device = torch.device(device)
    key = (device.type, device.index)
    if key not in _F64_OK:
        _F64_OK[key] = device.type == "cuda" and torch.cuda.get_device_capability(device) in FAST_FLOAT64
    return _F64_OK[key]


def _scaled_round_f64(x: torch.Tensor, s: torch.Tensor, wide=True) -> torch.Tensor:
    """`scaled_round` of float64 x in float64: x * 2^a * 2^b, a + b = -s,
    a = clamp(-s, -1022, 1023), then round. The same function as the
    integer one, for the reason the float32 path is (each factor a normal
    power of two; b in [0, 61] for every s the callers make: q8 |s| <= 511,
    sums -s >= -1022, det_sqrt -s <= 1084; an underflow loses bits only of a
    value whose round is 0)."""
    ns = -s
    a = torch.clamp(ns, -1022, 1023)
    b = torch.clamp(ns - a, -1022, 1023)
    y = torch.round((x * pow2(a, checked=True)) * pow2(b, checked=True))
    return y.to(torch.int64 if wide else torch.int32)


def scaled_round(x: torch.Tensor, s: torch.Tensor, wide=True) -> torch.Tensor:
    """round-half-even(x * 2^-s), exactly, for finite x and integer s
    (broadcast): int64 (`wide`, for |x * 2^-s| < 2^62) or int32 (below 2^30).

    float64 x: in integers on the float's bits (`scaled_round_int`), since a
    GeForce GPU's float64 runs at 1/64 rate; on a GPU whose float64 is fast
    (`f64_formula_ok`) in float64, the same function (`_scaled_round_f64`).
    float32 x: in float32, which is
    the same function and five operations rather than thirty: x * 2^a * 2^b
    with a + b = -s, a = clamp(-s, -126, 127) and so b in [0, 59] for every s
    the callers make (q8: -s >= -120; sums: -s >= -126), is exact (each factor
    is a normal power of two, and the product of x and 2^a can only grow by
    2^b) unless it underflows, which loses bits only of a value below 2^-126,
    whose round is 0 either way; `round` of an exact value is the round-half-
    even of x * 2^-s, an integer below 2^62 that converts exactly. Checked
    against `scaled_round_int` on ties, zeros, subnormals and extreme scales
    (python/tests/test_exact_train_lm.py)."""
    if x.dtype != torch.float32:
        if x.is_cuda and f64_formula_ok(x.device):
            return _scaled_round_f64(x, s, wide)
        return scaled_round_int(x, s, wide)
    ns = -s
    a = torch.clamp(ns, -126, 127)
    b = torch.clamp(ns - a, -126, 127)
    y = torch.round((x * _pow2_f32(a)) * _pow2_f32(b))
    return y.to(torch.int64 if wide else torch.int32)


def q8(x: torch.Tensor, dim: int = -1, bits: int = 8, bad=None):
    """Integers q (int32) and exponents s with x ~ q * 2^s, one s per slice
    along `dim` (the summed dimension), |q| <= 2^(bits-1) - 1 (int8: 127):
    q = round-half-even(x * 2^-s), clamped; float64 or float32 x. A slice
    whose scale is outside 2^-511 .. 2^511 is refused, so the scale of a
    product of two quantized operands, 2^(sa + sb), is always a normal
    float64 and needs no check of its own (a slice that small or large is
    not a training value)."""
    top = (1 << (bits - 1)) - 1
    s = exponent(x, dim, bad) - (bits - 1)
    refuse(bad, (s < -Q_SCALE_RANGE) | (s > Q_SCALE_RANGE),
           f"exact train: a quantizer scale outside 2^-{Q_SCALE_RANGE} .. 2^{Q_SCALE_RANGE}")
    if x.dtype == torch.float32 and bits <= 16:
        # Every element's shift is at least 24 - bits >= 8 (its exponent is
        # at most the max's), so the int32 path never shifts left.
        return scaled_round(x, s, wide=False).clamp(-top, top), s
    return scaled_round(x, s).clamp(-top, top).to(torch.int32), s


def isqrt(n: torch.Tensor, bad=None) -> torch.Tensor:
    """floor(sqrt(n)) for int64 0 <= n < 2^62: a float estimate (any
    device's, within 1 of the answer), then integer corrections, so the
    result is the one integer it must be."""
    refuse(bad, (n < 0) | (n >= 1 << 62), "exact train: isqrt takes 0 <= n < 2^62")
    x = torch.sqrt(n.to(torch.float64)).to(torch.int64)
    for _ in range(2):
        x = x - (x * x > n).to(torch.int64)
        x = x + ((x + 1) * (x + 1) <= n).to(torch.int64)
    return x


def det_sqrt(v: torch.Tensor, bad=None) -> torch.Tensor:
    """sqrt of float64 v >= 0, to within 2^-30 relative, the same bits on
    every device: v scaled by an even power of two to an integer in
    [2^60, 2^62), its integer square root, scaled back. Takes 0 and
    2^-962 <= v < 2^1024; below that the scaling leaves float64's normal
    range and `pow2` refuses. A negative v, an inf or a NaN is refused (a NaN
    used to become 0 on CUDA, which converts NaN to the integer 0)."""
    a, field = _fields(v)
    refuse(bad, (field == _INF_FIELD) | (v < 0), "exact train: det_sqrt takes a finite v >= 0")
    k = torch.where(a == 0, 0, field - 1022) - 61       # frexp's exponent - 61
    k = k - (k & 1)                                    # even, so 2^(k/2) is exact
    check_exp(-k, bad)
    n = scaled_round(v, k)                             # exact: a left shift of v's 53 bits
    return isqrt(n, bad).to(torch.float64) * pow2(torch.div(k, 2, rounding_mode="floor"), bad)


def ndigits(bits: int) -> int:
    """How many digits in [-64, 63] hold every |q| <= 2^(bits-1) - 1: one
    (q itself) for 8 bits or fewer, else the fewest n with
    sum_{i<n} 63 * 128^i >= 2^(bits-1) - 1, the largest positive value n
    digits reach. (ceil((bits - 1) / 7), used until 2026-11-21, is one short
    for 14, 15, 21, 22, ... bits, which it refused at run time: two digits
    reach 8,127, not 8,191.)"""
    if bits <= 8:
        return 1
    n, top = 1, (1 << (bits - 1)) - 1
    while 63 * (128 ** n - 1) // 127 < top:
        n += 1
    return n


def digits(q: torch.Tensor, bits: int, bad=None, bounded=False):
    """Integers q (an integer tensor) with |q| < 2^(bits-1) as `ndigits(bits)`
    digits d_i in [-64, 63], q's dtype, with q = sum_i d_i * 128^i (one
    digit, q itself, for 8 bits or fewer). A product of wide operands is then
    a sum of int8 products, each exact on an int8 GEMM. d = r - 128 *
    floor((r + 64) / 128), the floor an arithmetic shift. A q the digits
    cannot hold is refused, unless `bounded` (the caller clamped |q| to
    2^(bits-1) - 1, which `ndigits` guarantees they hold)."""
    if bits <= 8:
        return [q]
    out, r = [], q
    for _ in range(ndigits(bits)):
        hi = (r + 64) >> 7
        out.append(r - (hi << 7))
        r = hi
    if not bounded:
        refuse(bad, r != 0, f"exact train: a value the {len(out)} digits for {bits} bits cannot hold")
    return out


def capacity(bits: int) -> int:
    """The largest |value| `digits(q, bits)` can represent, digit by digit:
    127 for one int8 digit, sum 64 * 128^i for several."""
    if bits <= 8:
        return 127
    return sum(64 * 128 ** i for i in range(ndigits(bits)))


def _ceil_log2(n: int) -> int:
    return max(0, (n - 1).bit_length())


# Each `mm` takes integer tensors a [M][K], b [N][K] with entries in
# [-127, 127] and returns the exact integers sum_k a[i][k] * b[j][k] as an
# integer tensor.

def mm_f64(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """On float64 BLAS (any device): exact, since every partial sum is an
    integer below 2^53."""
    if a.shape[-1] * 127 * 127 >= F64_EXACT:
        raise OverflowError("exact train: K too large for an exact float64 sum")
    return (a.to(torch.float64) @ b.to(torch.float64).transpose(-1, -2)).to(torch.int64)


def mm_y(a, b):
    """On Y's int8 GEMM."""
    from ..int8_gemm import int8_mm
    return int8_mm(a.to(torch.int8).contiguous(), b.to(torch.int8).contiguous())


def mm_torch(a, b):
    """On `torch._int_mm` (cuBLASLt)."""
    return torch._int_mm(a.to(torch.int8).contiguous(), b.to(torch.int8).contiguous().t())


def _exact_fp32_bmm(a, b):
    """Batched sum_k a[..i][k] * b[..j][k] for integers in [-127, 127] in
    float32: exact because every partial sum is an integer below 2^24 (TF32
    off)."""
    if a.shape[-1] * 127 * 127 >= FP32_EXACT:
        raise OverflowError(f"exact train: K = {a.shape[-1]} leaves fp32's exact range")
    if a.is_cuda and torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("exact train: TF32 rounds the products; set "
                           "torch.backends.cuda.matmul.allow_tf32 = False")
    return (a.to(torch.float32) @ b.to(torch.float32).transpose(-1, -2)).to(torch.int64)


_BMM_OK = {}


def int8_bmm_ok(device) -> bool:
    """Whether attention's products run on `int8_bmm` on `device` (CUDA,
    sm_80 or later), cached so a compiled step reads a constant."""
    device = torch.device(device)
    key = (device.type, device.index)
    if key not in _BMM_OK:
        from .int8_bmm import available
        _BMM_OK[key] = available(device)
    return _BMM_OK[key]


def _exact_int8_bmm(a, b):
    """The same product as `_exact_fp32_bmm`, on int8 tensor cores
    (`int8_bmm`): the same integers."""
    from .int8_bmm import int8_bmm
    return int8_bmm(a.to(torch.int8), b.to(torch.int8)).to(torch.int64)


class Ops:
    """The model's sums and softmax, exact (`exact=True`) or ordinary float
    (`exact=False`: the float twin and the autograd reference)."""

    def __init__(self, exact=True, mm=mm_f64, grad_bits=12, bwd_bits=8, act=None, float_mm=None,
                 bmm=None, fused_softmax=None):
        """Widths of the backward pass's product operands: `grad_bits` for the
        gradient, `bwd_bits` for the activation or weight it meets. The
        forward pass is always int8, the format the model is served in. Past
        8 bits an operand is split into int8 digits (`digits`), so every
        product still runs on an int8 GEMM and is still exact. The default,
        12-bit gradients against int8 activations and weights, is the
        narrowest setting measured to train as well as float32: with int8
        gradients the held-out loss stalled at 2.01 against float32's 1.71
        (docs/benchmarks/exact_train_lm_2026-11-21/).

        `act` is the activations' and gradients' type: None keeps the
        parameters' (float64 for the exact runs); `torch.float32` makes every
        elementwise operation an IEEE float32 one, every product's exact
        integer is scaled in float64 and rounded once to float32, and the
        quantizers read 24-bit mantissas in int32. The sums are still exact
        integer sums and the norms' square roots float64 `det_sqrt`; the
        parameters and Adam stay float64. A different model from `act=None`
        (its values round to float32), equally the same on every device.

        `float_mm` (the float twin only) is the type its products run in:
        `torch.bfloat16` is ordinary mixed-precision training (float32
        parameters and elementwise operations, bf16 tensor-core products),
        a speed baseline.

        `bmm` picks attention's products: "int8" (`int8_bmm`, tensor cores),
        "fp32" (`_exact_fp32_bmm`), or None, int8 where `int8_bmm_ok` (a CUDA
        GPU of sm_80 or later) and fp32 elsewhere. The same integers either
        way."""
        if bmm not in (None, "int8", "fp32"):
            raise ValueError(f"exact train: bmm must be None, 'int8' or 'fp32', not {bmm!r}")
        self.bmm = bmm
        # The causal softmax and its quantizer as one kernel (softmax_kernel),
        # for float32 activations on a GPU: None (where it applies) or False.
        self.fused_softmax = fused_softmax
        if act not in (None, torch.float32, torch.float64):
            raise ValueError(f"exact train: act must be None, float32 or float64, not {act}")
        if float_mm is not None and exact:
            raise ValueError("exact train: float_mm is for the float twin (exact=False)")
        self.act, self.float_mm = act, float_mm
        self.exact, self.mm, self.grad_bits, self.bwd_bits = exact, mm, grad_bits, bwd_bits
        self._tables = {}
        # None: each check reads the device and raises at once. A list: the
        # checks are appended to it for one read per step (`refuse`).
        self.bad = None

    def matmul(self, a, b, batched=False, grad=None, qa=None, qb=None):
        """a @ b^T, summed over the last dimension of both. `grad` names the
        backward pass's gradient operand, "a" or "b" (None: forward). `qa`,
        `qb`: a's or b's q8 copy already made, (q, s) as `q8(x, bits)` returns
        them (`b` itself is then not read: `lora` passes a frozen int8 weight
        this way, with s = 0)."""
        if not self.exact:
            if self.float_mm is not None:
                return (a.to(self.float_mm) @ b.to(self.float_mm).transpose(-1, -2)).to(a.dtype)
            return a @ b.transpose(-1, -2)
        bits_a = 8 if grad is None else self.grad_bits if grad == "a" else self.bwd_bits
        bits_b = 8 if grad is None else self.grad_bits if grad == "b" else self.bwd_bits
        # Every partial sum below, of digit products, is at most K times the
        # digits' capacities: below 2^53, so float64 holds it exactly. Known
        # from the widths alone, so refused before anything runs.
        if a.shape[-1] * capacity(bits_a) * capacity(bits_b) >= F64_EXACT:
            raise OverflowError("exact train: a product sum could pass float64's exact integers")
        qa, sa = q8(a, bits=bits_a, bad=self.bad) if qa is None else qa
        qb, sb = q8(b, bits=bits_b, bad=self.bad) if qb is None else qb
        da = digits(qa, bits_a, bounded=True)
        db = digits(qb, bits_b, bounded=True)
        # Every digit of a against every digit of b in ONE product: the
        # digits stacked along the rows of each operand, block (i, j) of the
        # result the product of digit i of a and digit j of b.
        M, N = qa.shape[-2], qb.shape[-2]
        A = da[0] if len(da) == 1 else torch.cat(da, dim=-2)
        B = db[0] if len(db) == 1 else torch.cat(db, dim=-2)
        if batched:
            int8 = self.bmm == "int8" or (self.bmm is None and A.is_cuda and int8_bmm_ok(A.device))
            if int8:
                P = _exact_int8_bmm(A, B)
            elif A.shape[-1] * 127 * 127 >= FP32_EXACT and not A.is_cuda:
                # A sum too long for float32's exact integers (attention's
                # backward over a query group): float64's, on the CPU.
                P = (A.to(torch.float64) @ B.to(torch.float64).transpose(-1, -2))
            else:
                P = _exact_fp32_bmm(A, B)
            P = P.to(torch.int64)
        else:
            P = self.mm(A, B).to(torch.int64)
        c = None
        for i in range(len(da)):
            for j in range(len(db)):
                part = P[..., i * M:(i + 1) * M, j * N:(j + 1) * N]
                part = part << (7 * (i + j)) if i + j else part
                c = part if c is None else c + part
        # c below 2^53 (the bound above): its float64 is exact, and so is the
        # power-of-two scale: |sa|, |sb| <= 511 (q8), so sa + sb is in range.
        # With float32 activations that exact value is rounded once.
        out = c.to(torch.float64) * pow2(sa + sb.transpose(-1, -2), checked=True)
        return out if self.act in (None, torch.float64) else out.to(self.act)

    def sum(self, x, dim):
        """The sum along `dim` (kept), of x rounded to one fixed-point scale
        per sum chosen so the int64 total cannot overflow."""
        if not self.exact:
            return x.sum(dim=dim, keepdim=True)
        s = exponent(x, dim, self.bad) - 61 + _ceil_log2(x.shape[dim])
        check_exp(-s, self.bad)                     # and so s in [-1023, 1022]:
        refuse(self.bad, s == -1023, "exact train: a scale exponent outside float64's normal range")
        q = scaled_round(x, s)
        return q.sum(dim=dim, keepdim=True).to(torch.float64) * pow2(s, checked=True)

    def scatter_rows(self, idx, src, rows):
        """out[idx[t]] += src[t]: an embedding's gradient."""
        if not self.exact:
            return torch.zeros(rows, src.shape[1], dtype=src.dtype, device=src.device).index_add(0, idx, src)
        s = exponent(src, 0, self.bad) - 61 + _ceil_log2(src.shape[0])
        check_exp(-s, self.bad)
        refuse(self.bad, s == -1023, "exact train: a scale exponent outside float64's normal range")
        q = scaled_round(src, s)
        # Integer adds, so the order the device applies them in cannot matter.
        out = torch.zeros(rows, src.shape[1], dtype=torch.int64, device=src.device).scatter_add_(
            0, idx.reshape(-1, 1).expand(-1, src.shape[1]), q)
        return out.to(torch.float64) * pow2(s, checked=True)

    def causal_softmax_q8(self, z):
        """(P, q, s), the causal `softmax(z)` and its `q8(P, bits=8)` from one
        kernel (`softmax_kernel`, the same bits), or None where it does not
        apply: the float twin, float64 activations, a CPU, or rows that are
        not a power of two from 16 to 1024 keys."""
        if not (self.exact and self.fused_softmax is not False and z.dtype == torch.float32
                and z.is_cuda and z.shape[-1] == z.shape[-2]):
            return None
        from . import softmax_kernel
        if not softmax_kernel.supported(z.shape[-1]):
            return None
        P, q, s, flags = softmax_kernel.softmax_q8(z, z.shape[-1])
        refuse(self.bad, (flags & 1) != 0, "exact train: a softmax input that is not finite (inf or NaN)")
        refuse(self.bad, (flags & 2) != 0, "exact train: a value that is not finite (inf or NaN)")
        refuse(self.bad, (flags & 4) != 0,
               f"exact train: a quantizer scale outside 2^-{Q_SCALE_RANGE} .. 2^{Q_SCALE_RANGE}")
        return P, q.to(torch.int32), s

    def softmax(self, z, blocked):
        """softmax over the last dimension, `blocked` entries excluded."""
        if not self.exact:
            return torch.softmax(z.masked_fill(blocked, float("-inf")), dim=-1)
        table = self._tables.get(z.device)
        if table is None:
            table = self._tables[z.device] = _table(z.device)
        refuse(self.bad, _fields(z)[1] == _FMT[z.dtype][2],
               "exact train: a softmax input that is not finite (inf or NaN)")
        logit = z * LOG2E_Q16                                  # log2 units, Q16.16
        m = logit.masked_fill(blocked, float("-inf")).amax(dim=-1, keepdim=True)
        t = torch.round(m - logit).clamp(0, 1 << 30).masked_fill(blocked, 1 << 30)
        p = exp2_neg_q16_16(t, table).masked_fill(blocked, 0)  # Q0.28, int64
        dt = torch.float64 if self.act is None else self.act
        return p.to(dt) / p.sum(dim=-1, keepdim=True).to(dt)


# ---- the model -----------------------------------------------------------------

@dataclass
class Config:
    vocab: int = 96          # characters, padded to a multiple of 32
    d: int = 128
    heads: int = 4
    layers: int = 2
    ff: int = 512
    ctx: int = 64
    eps: float = 1e-5


def init_params(cfg: Config, seed=0, dtype=torch.float64, device="cpu"):
    """Weights drawn on the CPU in float64, so every device and dtype starts
    from the same numbers. Linear weights are [out][in]."""
    g = torch.Generator().manual_seed(seed)

    def w(*shape, std=0.02):
        return torch.randn(*shape, generator=g, dtype=torch.float64) * std

    p = {"tok": w(cfg.vocab, cfg.d), "pos": w(cfg.ctx, cfg.d), "nf": torch.ones(cfg.d, dtype=torch.float64),
         "out": w(cfg.vocab, cfg.d)}
    proj = 0.02 / math.sqrt(2 * cfg.layers)
    for i in range(cfg.layers):
        p[f"n1.{i}"] = torch.ones(cfg.d, dtype=torch.float64)
        p[f"qkv.{i}"] = w(3 * cfg.d, cfg.d)
        p[f"o.{i}"] = w(cfg.d, cfg.d, std=proj)
        p[f"n2.{i}"] = torch.ones(cfg.d, dtype=torch.float64)
        p[f"up.{i}"] = w(cfg.ff, cfg.d)
        p[f"down.{i}"] = w(cfg.d, cfg.ff, std=proj)
    return {k: v.to(dtype).to(device) for k, v in p.items()}


def _norm_fwd(ops, x, g, eps):
    d = x.shape[-1]
    ms = ops.sum(x * x, -1) * (1.0 / d) + eps
    r = torch.ones_like(ms) / (det_sqrt(ms, ops.bad) if ops.exact else torch.sqrt(ms))
    r, g = r.to(x.dtype), g.to(x.dtype)        # no-ops unless act is float32
    xh = x * r
    return xh * g, (xh, r, g)


def _norm_bwd(ops, dy, cache):
    xh, r, g = cache
    d = xh.shape[-1]
    dg = ops.sum(dy * xh, 0).reshape(-1)
    dxh = dy * g
    return r * (dxh - xh * (ops.sum(dxh * xh, -1) * (1.0 / d)).to(dxh.dtype)), dg


def forward(ops, cfg, p, idx, targets=None):
    """Logits [B*T][vocab] for token ids idx [B][T]; with targets, also the
    probabilities and everything `backward` needs."""
    B, T = idx.shape
    H, Dh = cfg.heads, cfg.d // cfg.heads
    scale = 1.0 / math.sqrt(Dh)
    flat = idx.reshape(-1)
    x = p["tok"][flat] + p["pos"][:T].repeat(B, 1)
    if ops.act is not None:
        x = x.to(ops.act)
    causal = torch.ones(T, T, dtype=torch.bool, device=idx.device).triu(1)
    caches = []
    for i in range(cfg.layers):
        hn, c_n1 = _norm_fwd(ops, x, p[f"n1.{i}"], cfg.eps)
        qkv = ops.matmul(hn, p[f"qkv.{i}"])
        q, k, v = (t.reshape(B, T, H, Dh).transpose(1, 2) for t in qkv.split(cfg.d, dim=1))
        S = ops.matmul(q, k, batched=True) * scale
        fused = ops.causal_softmax_q8(S)
        if fused is None:
            P = ops.softmax(S, causal)
            o = ops.matmul(P, v.transpose(-1, -2), batched=True)      # [B][H][T][Dh]
        else:
            P, qP, sP = fused
            o = ops.matmul(P, v.transpose(-1, -2), batched=True, qa=(qP, sP))
        o2 = o.transpose(1, 2).reshape(B * T, cfg.d)
        x = x + ops.matmul(o2, p[f"o.{i}"])
        hn2, c_n2 = _norm_fwd(ops, x, p[f"n2.{i}"], cfg.eps)
        u = ops.matmul(hn2, p[f"up.{i}"])
        a = torch.relu(u)
        x = x + ops.matmul(a, p[f"down.{i}"])
        caches.append((hn, c_n1, q, k, v, P, o2, hn2, c_n2, u, a))
    hf, c_nf = _norm_fwd(ops, x, p["nf"], cfg.eps)
    logits = ops.matmul(hf, p["out"])
    if targets is None:
        return logits
    pad = torch.zeros(cfg.vocab, dtype=torch.bool, device=idx.device)
    probs = ops.softmax(logits, pad.expand_as(logits))
    return logits, probs, (flat, caches, hf, c_nf)


def _linear_bwd(ops, dy, x, w):
    """dX = dy W and dW = dy^T x, each an exact sum."""
    return ops.matmul(dy, w.t(), grad="a"), ops.matmul(dy.t(), x.t(), grad="a")


def backward(ops, cfg, p, probs, targets, saved):
    """Gradients of the mean cross-entropy, by hand."""
    flat, caches, hf, c_nf = saved
    n = targets.numel()
    B, T = targets.shape
    H, Dh = cfg.heads, cfg.d // cfg.heads
    scale = 1.0 / math.sqrt(Dh)
    g = {}
    onehot = torch.zeros_like(probs).scatter_(1, targets.reshape(-1, 1), 1.0)
    dlogits = (probs - onehot) * (1.0 / n)
    dhf, g["out"] = _linear_bwd(ops, dlogits, hf, p["out"])
    dx, g["nf"] = _norm_bwd(ops, dhf, c_nf)
    for i in reversed(range(cfg.layers)):
        hn, c_n1, q, k, v, P, o2, hn2, c_n2, u, a = caches[i]
        da, g[f"down.{i}"] = _linear_bwd(ops, dx, a, p[f"down.{i}"])
        du = da * (u > 0)
        dhn2, g[f"up.{i}"] = _linear_bwd(ops, du, hn2, p[f"up.{i}"])
        dx2, g[f"n2.{i}"] = _norm_bwd(ops, dhn2, c_n2)
        dx = dx + dx2
        do2, g[f"o.{i}"] = _linear_bwd(ops, dx, o2, p[f"o.{i}"])
        do = do2.reshape(B, T, H, Dh).transpose(1, 2)                 # [B][H][T][Dh]
        dP = ops.matmul(do, v, batched=True, grad="a")                 # [B][H][T][Tk]
        dv = ops.matmul(P.transpose(-1, -2), do.transpose(-1, -2), batched=True, grad="b")
        dS = P * (dP - ops.sum(P * dP, -1).to(P.dtype)) * scale
        dq = ops.matmul(dS, k.transpose(-1, -2), batched=True, grad="a")
        dk = ops.matmul(dS.transpose(-1, -2), q.transpose(-1, -2), batched=True, grad="a")
        dqkv = torch.cat([t.transpose(1, 2).reshape(B * T, cfg.d) for t in (dq, dk, dv)], dim=1)
        dhn, g[f"qkv.{i}"] = _linear_bwd(ops, dqkv, hn, p[f"qkv.{i}"])
        dx1, g[f"n1.{i}"] = _norm_bwd(ops, dhn, c_n1)
        dx = dx + dx1
    g["tok"] = ops.scatter_rows(flat, dx, cfg.vocab)
    g["pos"] = ops.sum(dx.reshape(B, T, cfg.d), 0).reshape(T, cfg.d)
    return g


class Adam:
    """Adam with bias correction, elementwise: multiplies, adds, a
    tensor-by-tensor divide and a square root, each its own operation, so no
    device fuses or reorders them. `exact` takes the square root with
    `det_sqrt`; the float twin with `torch.sqrt`.

    All parameters live in one flat buffer, `flat`: the constructor replaces
    each entry of `params` (the caller's dict) with a view of it, and every
    update is in place. Elementwise, that is the same arithmetic as one update
    per tensor, in a few dozen kernels rather than a few hundred, and a CUDA
    graph can replay it."""

    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.99, eps=1e-8, exact=True):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.exact = exact
        self.keys = list(params)
        self.flat = torch.cat([params[k].reshape(-1) for k in self.keys])
        self.layout, o = [], 0
        for k in self.keys:
            n = params[k].numel()
            self.layout.append((k, o, tuple(params[k].shape)))
            o += n
        self.bind(params)
        self.m = torch.zeros_like(self.flat)
        self.v = torch.zeros_like(self.flat)

    def bind(self, params):
        """Point `params[k]` at its slice of `flat`."""
        for k, o, shape in self.layout:
            n = math.prod(shape)
            params[k] = self.flat[o:o + n].view(shape)
        return params

    def corrections(self):
        """Advance the step count; the bias corrections' reciprocals, computed
        on the host (a constant divisor is a multiply, see the module note)."""
        self.t += 1
        return 1.0 / (1.0 - self.b1 ** self.t), 1.0 / (1.0 - self.b2 ** self.t)

    def update(self, grads, r1, r2, bad=None):
        """One update from `grads` (a dict like `params`), with the
        corrections `r1`, `r2` as floats or 0-d tensors of `flat`'s dtype
        (the same products either way)."""
        g = torch.cat([grads[k].reshape(-1).to(self.flat.dtype) for k in self.keys])
        self.m.copy_(self.m * self.b1 + g * (1.0 - self.b1))
        self.v.copy_(self.v * self.b2 + (g * g) * (1.0 - self.b2))
        vr = self.v * r2
        den = (det_sqrt(vr, bad) if self.exact else torch.sqrt(vr)) + self.eps
        self.flat.copy_(self.flat - (self.m * r1) / den * self.lr)

    def step(self, params, grads, bad=None):
        r1, r2 = self.corrections()
        self.update(grads, r1, r2, bad)


def target_probs(probs, targets):
    """p(target) for each token, in token order, as float64 on the device."""
    flat = targets.reshape(-1)
    return probs[torch.arange(flat.numel(), device=probs.device), flat].to(torch.float64)


def nll_on_host(tp) -> float:
    """Mean -log p(target) from `target_probs`, in float64 on the CPU, for
    reading only. The mean is `math.fsum` (correctly rounded in any order),
    so reordering the batch does not move it; `log` is the host's, which
    another machine's libm may round differently, so records compare
    `tp_digest` instead."""
    nll = (-torch.log(tp.detach().cpu().to(torch.float64).clamp(min=1e-30))).tolist()
    return math.fsum(nll) / len(nll)


def tp_digest(tp) -> str:
    """A digest of `target_probs`: the bits of an integer exp's quotient, the
    same on every device."""
    return hashlib.sha256(tp.detach().cpu().to(torch.float64).numpy().tobytes()).hexdigest()[:16]


def cross_entropy_on_host(probs, targets) -> float:
    return nll_on_host(target_probs(probs, targets))


def target_prob_digest(probs, targets) -> str:
    return tp_digest(target_probs(probs, targets))


def digest(params) -> str:
    h = hashlib.sha256()
    for k in sorted(params):
        h.update(k.encode())
        h.update(params[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


# ---- data ----------------------------------------------------------------------

class CharData:
    """A text as character ids (vocabulary padded to `vocab`), with batches
    drawn on the CPU from a seeded generator, so every device sees the same
    batches. The last tenth is held out."""

    def __init__(self, text: str, vocab: int = 96):
        chars = sorted(set(text))
        if len(chars) > vocab:
            raise ValueError(f"{len(chars)} distinct characters, more than vocab = {vocab}")
        self.chars = chars
        self.ids = torch.tensor([chars.index(c) for c in text], dtype=torch.int64)
        cut = len(self.ids) * 9 // 10
        self.train_ids, self.val_ids = self.ids[:cut], self.ids[cut:]

    def batch(self, gen, B, T, split="train"):
        src = self.train_ids if split == "train" else self.val_ids
        starts = torch.randint(0, len(src) - T - 1, (B,), generator=gen)
        x = torch.stack([src[s:s + T] for s in starts.tolist()])
        y = torch.stack([src[s + 1:s + T + 1] for s in starts.tolist()])
        return x, y


class Trainer:
    """One training run: the parameters (views of Adam's flat buffer, in
    `params`), Adam, and `step`.

    `step(x, y)` trains on one batch and returns p(target) for each token,
    on the device, without reading the device: the step's checks (`refuse`)
    are or-ed into flags that `verify` reads in one transfer and raises on.
    Call `verify` before reading anything a step returned.

    `graph=True` (CUDA) captures the step as one CUDA graph on the first call
    and replays it after: the same kernels, launched in one call.
    `compile=True` traces the step with `torch.compile`, which fuses its
    elementwise chains; importing `.model` sets the Inductor flags under
    which it rounds as eager PyTorch does (no FMA, correctly rounded
    divides). Neither is meant to change any bits, and the CPU == GPU tests
    and records check that they do not."""

    def __init__(self, ops, cfg, device="cpu", dtype=torch.float64, seed=0, lr=3e-3,
                 graph=False, compile=False):
        self.ops, self.cfg, self.device = ops, cfg, torch.device(device)
        if (graph or compile) and self.device.type != "cuda":
            raise ValueError("exact train: graph and compile are for a CUDA device")
        self.params = init_params(cfg, seed, dtype, self.device)
        self.opt = Adam(self.params, lr=lr, exact=ops.exact)
        if self.device.type == "cuda":
            int8_bmm_ok(self.device)      # decided before any compile, constants to it
            f64_formula_ok(self.device)
        if compile:
            from . import model  # noqa: F401  (the Inductor rounding flags)
        self._fn = torch.compile(self._step, dynamic=False) if compile else self._step
        self.graph, self.compile = graph, compile
        self._graph, self._acc, self.messages = None, None, None
        if graph or compile:   # the corrections as 0-d tensors: no recompiles
            self._r = torch.ones(2, dtype=dtype, device=self.device)

    def _step(self, x, y, r1, r2):
        bad = []
        self.ops.bad = bad
        params = self.opt.bind({})
        _, probs, saved = forward(self.ops, self.cfg, params, x, y)
        grads = backward(self.ops, self.cfg, params, probs, y, saved)
        self.opt.update(grads, r1, r2, bad)
        self.ops.bad = None
        if not bad:
            bad = [(torch.zeros((), dtype=torch.bool, device=x.device), "(none)")]
        flags, messages = reduce_checks(bad)
        return target_probs(probs, y), flags, messages

    def _note(self, flags, messages):
        if self._acc is None:
            self._acc, self.messages = torch.zeros_like(flags), messages
        elif messages != self.messages:
            raise RuntimeError("exact train: the step's checks changed between steps")

    def step(self, x, y):
        r1, r2 = self.opt.corrections()
        if not (self.graph or self.compile):
            tp, flags, messages = self._fn(x, y, r1, r2)
            self._note(flags, messages)
            self._acc |= flags
            return tp
        self._r[0].fill_(r1)        # a kernel argument: no transfer, no wait
        self._r[1].fill_(r2)
        if not self.graph:
            tp, flags, messages = self._fn(x, y, self._r[0], self._r[1])
            self._note(flags, messages)
            self._acc |= flags
            return tp
        if self._graph is None:
            self._capture(x, y)
        self._x.copy_(x)
        self._y.copy_(y)
        self._graph.replay()
        return self._tp.clone()

    def _capture(self, x, y):
        """Warm up (compiling, and loading every kernel) on a side stream,
        capture, then put back the state the warm-up steps moved."""
        self._x, self._y = x.clone(), y.clone()
        saved = [t.clone() for t in (self.opt.flat, self.opt.m, self.opt.v)]
        r = self._r.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                _, flags, messages = self._fn(self._x, self._y, self._r[0], self._r[1])
        torch.cuda.current_stream().wait_stream(side)
        self._note(flags, messages)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._tp, flags, messages = self._fn(self._x, self._y, self._r[0], self._r[1])
            self._acc |= flags
        self._note(flags, messages)
        for t, c in zip((self.opt.flat, self.opt.m, self.opt.v), saved):
            t.copy_(c)
        self._r.copy_(r)
        self._graph = g

    def verify(self):
        """Raise OverflowError if any check of any step so far failed: one read
        of the device. A failure is kept, so every later call raises too."""
        if self._acc is None:
            return
        hit = self._acc.cpu().tolist()
        if any(hit):
            raise OverflowError("; ".join(sorted({m for m, h in zip(self.messages, hit) if h})))


def train(ops, cfg, data, steps, device, dtype=torch.float64, seed=0, B=16, lr=3e-3,
          eval_every=0, eval_batches=4, log=None, graph=False, compile=False):
    """Trains from `init_params(cfg, seed)`; returns (params, train losses,
    [(step, held-out loss)])."""
    tr = Trainer(ops, cfg, device, dtype, seed, lr, graph=graph, compile=compile)
    gen = torch.Generator().manual_seed(seed + 1)
    losses, evals, pending = [], [], []
    for step in range(steps):
        x, y = data.batch(gen, B, cfg.ctx)
        pending.append(tr.step(x.to(device), y.to(device)))
        at_eval = eval_every and (step + 1) % eval_every == 0
        if at_eval or step + 1 == steps:
            tr.verify()
            losses += [nll_on_host(tp) for tp in pending]
            pending = []
        if at_eval or (eval_every and step + 1 == steps):
            evals.append((step + 1, evaluate(ops, cfg, tr.params, data, device, eval_batches, B)))
            if log:
                log(f"step {step + 1:5d}  train {losses[-1]:.4f}  held-out {evals[-1][1]:.4f}")
    return tr.params, losses, evals


def evaluate(ops, cfg, params, data, device, batches=4, B=16):
    """Mean held-out cross-entropy over fixed batches (seed 12345)."""
    gen = torch.Generator().manual_seed(12345)
    total = 0.0
    with torch.no_grad():
        for _ in range(batches):
            x, y = data.batch(gen, B, cfg.ctx, split="val")
            _, probs, _ = forward(ops, cfg, params, x.to(device), y.to(device))
            total += cross_entropy_on_host(probs, y)
    return total / batches
