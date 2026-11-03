"""Make every reduction in a transformer order-independent, then measure.

`tools/batch_invariance_demo.py` replaced attention only, and that is not
enough: measured on Qwen2.5-0.5B, **99% of output logits change between batch 1
and batch 2** with stock kernels. Attention is one of three batch-dependent
reductions in the network. The others are the linear layers (a GEMM picks its
tiling from the problem size, which changes the summation order) and RMSNorm
(a sum over the hidden dimension).

This module swaps all three for order-independent versions:

  * `ExactLinear`   - W8A8, accumulated in exact int32 via `torch._int_mm`
                      (tensor cores; integer addition, so any tile order gives
                      the same bits). Falls back to float64, which represents
                      every integer below 2^53 exactly, when `_int_mm` will not
                      take the shape, or on a CPU whose int8 GEMM fails
                      `cpu_int_mm_is_exact`.
  * `ExactRMSNorm`  - the sum of squares in float64 over a dimension whose
                      length does not depend on the batch.
  * exact attention - imported from `batch_invariance_demo`.

and, so the same tokens come out of a CPU and a GPU, the two float ops left:

  * `ExactSiLU`     - the sigmoid as integer interpolation in a host table.
  * `ExactRotary`   - RoPE cos/sin from a host-built float64 table.

Accuracy is a separate question from invariance and is reported separately:
W8A8 with per-channel weights and per-token activations is a standard scheme,
but it is still int8, and `tools/attention_real_activations.py` measured that
the quantisation scheme dominates every other error term by ~300,000x.
"""
import math
import os

import torch
import torch.nn as nn

try:
    from w8a8_gemv import w8a8_linear, w8a8_matmul, w8a8_matmul_hilo
except Exception:                                    # noqa: BLE001
    w8a8_linear = w8a8_matmul = w8a8_matmul_hilo = None
# `Y_EXACT_TRITON=0` forces the torch fallback, so the two paths can be A/B'd
# against each other rather than only against the float64 reference.
USE_TRITON = os.environ.get("Y_EXACT_TRITON", "1") != "0"

# Under torch.compile, Inductor rounds differently from eager PyTorch in three
# ways, and each one moved a compiled GPU decode off the CPU's tokens:
#   * a float32 `/` in Triton lowers to `div.full`, an approximate divide,
#     unless `eager_numerics.division_rounding` asks for `div_rn`;
#   * a divide by a CONSTANT becomes a multiply by its reciprocal on every
#     device but the CPU (`torch/_inductor/lowering.py`, `div_prim`), whatever
#     that flag says. `x.new_tensor(lv)` is a constant once traced - see
#     `divisor` below;
#   * a multiply feeding an add in one fused kernel becomes an FMA, rounding
#     once where the CPU rounds twice. Inductor passes Triton
#     `enable_fp_fusion = not emulate_precision_casts`, so TRITON_DEFAULT_FP_FUSION
#     has no effect on it. Measured in exact attention (one layer in 24 on one
#     prompt), from `round(m - logit)` with `logit` a product.
# Measured on 3.67M activations at levels 127: either divide hazard alone moved
# 194 of 4,096 scales off the CPU's; with both fixed, 0. These flags are global
# to the process, which is the point: nothing compiled next to the exact path
# should round differently from it.
import torch._inductor.config as _inductor_config  # noqa: E402

_inductor_config.eager_numerics.division_rounding = True
_inductor_config.emulate_precision_casts = True
_DIVISORS = {}


def divisor(value, like):
    """`value` as a 0-d tensor of `like`'s dtype and device, for dividing by.

    Eager, `x / divisor(v, x)` is `x / x.new_tensor(v)`. Compiled, the tensor
    was created OUTSIDE the traced code, so it is a graph input rather than a
    constant, and the divide stays a divide. Creating it while tracing would
    make it a constant again, so that raises instead: dynamo breaks the graph
    there, runs this call eagerly and traces on with the tensor as an input.
    Measured: without the guard the first compiled call differed from the CPU
    on 1.59M of 3.67M values, with it on none. So a divisor missing at compile
    time costs a graph break, never bits; `convert` creates the ones it can name
    ahead of time. (The flags above are not guarded here - a graph break would
    not stop the rest of the graph using them - `exact_selftest` checks them.)
    """
    key = (float(value), like.device, like.dtype)
    t = _DIVISORS.get(key)
    if t is None:
        if torch.compiler.is_compiling():
            raise RuntimeError(
                f"exact divisor {key} requested while compiling: creating it "
                f"here would make it a constant that Inductor folds into a "
                f"reciprocal multiply")
        t = _DIVISORS[key] = torch.tensor(float(value), dtype=like.dtype,
                                          device=like.device)
    return t


def fail_closed_recompiles():
    """Make dynamo's recompile limit an error, not a silent eager fallback.

    Every model's bound `forward` shares Hugging Face's one code object, so each
    model compiled in a process spends recompiles from the same budget. Past
    `recompile_limit` (8) dynamo silently runs that frame EAGERLY: three compiled
    arms in one process measured one of them uncompiled, under a "compiled"
    label. The limit is raised for the arms the harnesses build, and hitting it
    raises instead of falling back.
    """
    import torch._dynamo
    cfg = torch._dynamo.config
    cfg.recompile_limit = max(cfg.recompile_limit, 64)
    cfg.accumulated_recompile_limit = max(cfg.accumulated_recompile_limit, 1024)
    cfg.fail_on_recompile_limit_hit = True


def compile_forward(model):
    """`torch.compile(model.forward, dynamic=True)`, under `fail_closed_recompiles`."""
    fail_closed_recompiles()
    model.forward = torch.compile(model.forward, dynamic=True)
    return model


def cpu_int_mm_is_exact(mm=None):
    """Does `torch._int_mm` on this CPU accumulate int8 products exactly?

    It is 50-100x faster than the float64 fallback on the CPU (measured at the
    7B MLP shapes, Zen 5), and exact where the hardware is: an int32 sum of
    int8 products. Not everywhere, though. x86 int8 GEMMs without VNNI multiply
    with `vpmaddubsw`, whose 16-bit pair sum SATURATES, and an s8 x s8 GEMM
    built on it shifts one operand by 128 - after which two products of 127s
    pass 32767. So this is measured, on exactly those inputs and on random
    ones, against float64; a mismatch or a refusal keeps the CPU on float64.
    """
    mm = mm or torch._int_mm
    k = 4096
    g = torch.Generator().manual_seed(0)
    full = lambda shape, v: torch.full(shape, v, dtype=torch.int8)
    cases = [(full((4, k), 127), full((k, 8), 127)),
             (full((4, k), -128), full((k, 8), -128)),
             (full((4, k), -127), full((k, 8), 127)),
             (torch.randint(-128, 128, (4, k), generator=g, dtype=torch.int8),
              torch.randint(-128, 128, (k, 8), generator=g, dtype=torch.int8))]
    try:
        return all(torch.equal(mm(a, b).to(torch.float64),
                               a.to(torch.float64) @ b.to(torch.float64))
                   for a, b in cases)
    except Exception:                                # noqa: BLE001
        return False


_CPU_INT_MM_EXACT = cpu_int_mm_is_exact()

# Requested activation width, as `2^n - 1`. 127 is plain int8 (one `tl.dot`);
# anything larger carries the activation as two int8 digits and costs a second
# dot over the same weight tile - see `w8a8_gemv._w8a8_gemm_hilo`.
#
# **511 is where the accuracy stops improving, and that was measured, not
# guessed** (`tools/exact_quant_attribution.py`, wikitext-2 at 60k tokens):
# 127 -> +2.24% over fp32, 255 -> +0.68%, **511 -> -0.04%**, 2047 -> -0.08%,
# 32767 -> -0.12%. So one extra bit past 255 recovers the whole gap and further
# widening buys nothing. The default is left at 127 until the throughput cost is
# measured; `Y_EXACT_ACT_LEVELS=511` turns it on.
ACT_LEVELS = int(os.environ.get("Y_EXACT_ACT_LEVELS", "127"))
_CHECK = os.environ.get("Y_EXACT_CHECK") == "1"   # opt-in measured checks
ACC_BUDGET = 1 << 31                # int32, the `tl.dot` accumulator

# Structural ceiling of the TWO-digit split, and it is not the accumulator that
# sets it. `q = hi*128 + lo` puts `hi = q >> 7` in an int8, so `|q| <= 127*128 +
# 127 = 16383`. Past that `hi` needs a ninth bit and `.to(torch.int8)` **wraps
# silently** - 32767 comes back as 1, with no error and no warning, which is a
# wrong answer rather than an imprecise one. The accumulator budget alone
# permits 262,143 at K=64, so the two limits do NOT imply each other and the
# smaller has to be applied explicitly. A wider activation than this needs three
# digits, which is a different kernel, not a bigger constant.
DIGIT_MAX = 127 * 128 + 127          # 16383


def act_levels_for(k, w_levels=127, requested=None):
    """Widest activation width whose int32 accumulator cannot wrap, capped.

    The naive budget is `k * a_levels * w_levels < 2^31`, and it is **too loose
    by 128 levels**. The kernel accumulates `128*(hi@w)` and `(lo@w)` into one
    int32 interleaved across K blocks, so the worst-case running magnitude is
    the sum of the two parts' bounds, `k * (a_levels + 128) * w_levels` - not
    the bound on their difference. The final value is within `k*a*w`; the
    intermediate is not, and the intermediate is what has to fit. Same class of
    error as counting a bound on a total while a partial sum is what rounds.

    It binds at large K: this model's widest is `down_proj` at K=4864, which
    permits 3,347 and therefore 2047 after rounding down; a 32k-wide MLP would
    permit 387 and therefore 255.

    Rounded down to `2^n - 1` so the digit split is a clean shift: `q = hi*128 +
    lo` needs `lo` to be exactly the low 7 bits.
    """
    # `requested` overrides the module default so a TEST can exercise the wide
    # path while the shipped default is still 127. Without it the self-test's
    # digit-split checks ran at a_levels=127, where `hi` is in [-1, 0] and the
    # split is degenerate - they passed while testing nothing, which is the
    # failure mode this repo keeps finding in its own test files.
    cap = min(ACC_BUDGET // max(k * w_levels, 1) - 128, DIGIT_MAX,
              ACT_LEVELS if requested is None else requested)
    n = max(1, cap.bit_length())
    if (1 << n) - 1 > cap:
        n -= 1
    return max(127, (1 << n) - 1)


def split_digits(q, levels):
    """`q -> (hi, lo)` int8 with `q == hi*128 + lo` exactly, for `|q| <= levels`.

    Two's complement makes this exact for negative `q` as well: `-511 >> 7` is
    `-4` (arithmetic shift) and `-511 & 127` is `1`, and `-4*128 + 1 = -511`.
    Checked over the whole representable range by `exact_selftest`, because
    "obviously exact" bit arithmetic is how the `U32`/`I32` literal bug in the
    PTX backend got in - and this function had exactly that bug: `torch`'s
    `.to(torch.int8)` **wraps rather than raising**, so `|q| > DIGIT_MAX` came
    back as a small wrong number with nothing to notice at all.

    **The bound is checked on `levels`, not on `q`.** Measuring `q.abs().max()`
    is a device-to-host sync, and this runs on every linear of every decode step
    - the same mistake `batch_invariance_demo.assert_bound` exists to correct,
    where three such syncs per attention call were 61% of its time. The declared
    width is also the STRONGER check: it holds for every input, where a
    measurement only ever speaks for the tensor in front of it. `Y_EXACT_CHECK=1`
    adds the measured form beside it so the derivation stays auditable.
    """
    assert levels <= DIGIT_MAX, (
        f"a_levels={levels} is past {DIGIT_MAX}, where `hi = q >> 7` no longer "
        f"fits an int8. `.to(torch.int8)` would WRAP, silently, and the product "
        f"would be wrong rather than imprecise. Two digits top out here; a wider "
        f"activation needs three, which is a different kernel."
    )
    if _CHECK and q.numel():
        m = int(q.abs().max())
        assert m <= levels, (
            f"quantised activation reaches {m} but was declared at {levels} - "
            f"the clamp is not doing what the width says"
        )
    return (q >> 7).to(torch.int8), (q & 127).to(torch.int8)


def _q_per_row(x, levels=127):
    """Symmetric per-row quantisation. Returns (integer-valued tensor, scale).

    `levels` is explicit rather than defaulted-and-forgotten for the reason
    written out in `exact_kv.quantize_rows`: a width written in one place and
    assumed in another has been the cause of six bugs in this code.
    """
    lv = float(levels)
    # Divided by a TENSOR. torch on CUDA computes `tensor / python_float` as a
    # multiply by the reciprocal, which differs from the CPU's IEEE divide on
    # 4.6% of float32 inputs; tensor/tensor is an IEEE divide on both. And not
    # by `x.new_tensor(lv)`, which torch.compile folds back into a reciprocal.
    s = x.abs().amax(dim=-1, keepdim=True) / divisor(lv, x)
    s = torch.where(s == 0, torch.ones_like(s), s)
    return torch.round(x / s).clamp(-lv, lv), s


class ExactLinear(nn.Module):
    """y = (x8 @ w8^T) * sx * sw + b, with the inner product exact.

    The accumulation is integer, so it is associative and commutative: the
    GEMM may tile it however it likes, split K however it likes, and finish its
    CTAs in whatever order - the result is the same bits. That is the whole
    property, and it is why this needs no determinism flag.
    """

    def __init__(self, lin: nn.Linear):
        super().__init__()
        w = lin.weight.data
        # Per-OUTPUT-CHANNEL weight scales: one bad channel must not set the
        # scale for all of them.
        # A tensor divisor, for the reason in `_q_per_row`: with `/ 127.0` the
        # int8 weights built on a GPU differed from the ones built on a CPU,
        # and decoding diverged across the two at token 13.
        sw = w.abs().amax(dim=-1, keepdim=True) / w.new_tensor(127.0)
        sw = torch.where(sw == 0, torch.ones_like(sw), sw)
        w8 = torch.round(w / sw).clamp(-127, 127).to(torch.int8)
        # PRE-TRANSPOSED, once, at construction. The first version of this
        # module called `self.w8.t().contiguous()` inside `forward`, i.e. it
        # re-materialised a 4.3 MB int8 matrix on every call of every layer of
        # every decode step. Measured at M=32: 97.0 us with the transpose in
        # the loop against 8.9 us with it hoisted - an 11x self-inflicted cost
        # that had nothing to do with exactness.
        self.register_buffer("w8t", w8.t().contiguous())
        self.register_buffer("sw", sw.squeeze(-1).float())
        self.register_buffer(
            "bias", lin.bias.data.clone() if lin.bias is not None else None
        )
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        # fp32 carries integers exactly up to 2^24. A dot product of two int8
        # vectors of length K is bounded by K*127^2, and every partial sum is
        # bounded by that same figure, so if it fits the whole matmul is exact
        # in fp32 - no rounding anywhere, hence order-independent.
        self.fp32_exact = lin.in_features * 127 * 127 < (1 << 24)
        # Activation width for THIS layer, derived from its own K. Stored so a
        # layer's bound cannot disagree with the operand it bounds - the rule
        # `exact_kv.quantize_kv` arrived at by getting it wrong six times.
        self.a_levels = act_levels_for(lin.in_features)
        self.split_act = self.a_levels > 127
        divisor(self.a_levels if self.split_act else 127, self.sw)
        # Refuse rather than wrap. Both paths accumulate in int32, and neither
        # said so out loud until `tools/exact_bounds_check.py` asked: the plain
        # int8 path is bounded by `K*127^2` and **silently wraps at K >= 133,153**
        # - unreachable in any model that exists (a 70B MLP is K=28,672) but
        # unstated, which is the same shape as every row in CLAUDE.md's
        # design-rule table. The wide path carries the tighter `(a+128)` bound
        # because its accumulator holds the two digit products interleaved.
        acc_bound = (lin.in_features * (self.a_levels + 128) * 127
                     if self.split_act else lin.in_features * 127 * 127)
        assert acc_bound < ACC_BUDGET, (
            f"in_features={lin.in_features} at a_levels={self.a_levels} can "
            f"reach {acc_bound:.3e} in the int32 accumulator, past 2^31. The "
            f"reduction would wrap - not round - so the result would be wrong "
            f"rather than imprecise, and order-independence would be moot. Max "
            f"K for this width is {ACC_BUDGET // ((self.a_levels + 128) * 127)}."
        )

    def forward(self, x):
        shape = x.shape
        xf = x.reshape(-1, shape[-1])
        if self.split_act:
            return self._forward_split(x, xf, shape)
        x8, sx = _q_per_row(xf)
        x8 = x8.to(torch.int8)
        if (USE_TRITON and w8a8_matmul is not None and x.is_cuda
                and xf.dtype == torch.float32):
            # Only the GEMM is Triton. The quantisation above stays in torch on
            # purpose so inductor can fuse it into the preceding RMSNorm/SiLU -
            # doing it inside the Triton kernel instead was measured at a net
            # LOSS (see `w8a8_matmul`'s docstring). NCU found the CUTLASS int8
            # path running at 36.1% of DRAM peak against stock bf16's 69.5%: it
            # moved half the bytes at half the efficiency, so the two cancelled.
            y = w8a8_matmul(x8.contiguous(), sx.reshape(-1).float(),
                            self.w8t, self.sw, self.bias)
            return y.reshape(*shape[:-1], self.out_features).to(x.dtype)
        m = x8.shape[0]
        acc = None
        if x.is_cuda or _CPU_INT_MM_EXACT:
            # On CUDA `torch._int_mm` accepts only M divisible by 32 (measured: 32, 64,
            # 96, 128, 256 succeed; 1..31, 33..47, 48 and 56 are refused). At
            # decode M is the batch size, so the *original* guard - `M % 8 == 0`
            # - never once fired and every linear layer fell through to float64,
            # which runs at 1/64 rate on a GeForce part. Padding the row count
            # is exact: a zero row of x produces a zero row of the product and
            # rows of a matmul are independent, so the surviving rows are
            # untouched. The CPU takes any M (`cpu_int_mm_is_exact` decides
            # whether it may be used at all).
            pad = (-m) % 32 if x.is_cuda else 0
            try:
                xp = x8 if pad == 0 else torch.nn.functional.pad(x8, (0, 0, 0, pad))
                acc = torch._int_mm(xp, self.w8t)[:m].to(torch.float32)
            except Exception:
                acc = None
        if acc is None and self.fp32_exact:
            acc = torch.matmul(x8.to(torch.float32), self.w8t.to(torch.float32))
        if acc is None:
            # float64 represents every integer below 2^53 exactly, and the
            # largest value here is in_features * 127^2, so no partial sum ever
            # rounds and every summation order agrees.
            acc = torch.matmul(x8.to(torch.float64), self.w8t.to(torch.float64))
            assert float(acc.abs().max()) < 2.0 ** 53, "accumulator left the exact range"
            acc = acc.to(torch.float32)
        y = acc * sx.to(torch.float32) * self.sw.unsqueeze(0)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shape[:-1], self.out_features).to(x.dtype)

    def _forward_split(self, x, xf, shape):
        """Activations wider than int8, carried as two int8 digits.

        The accumulation stays integer end to end - `hi @ w` and `lo @ w` are
        int32 dots and `128*hi_acc + lo_acc` is int32 arithmetic - so the
        order-independence argument is exactly the one the plain path makes. It
        is NOT the same as widening the accumulator: the operands stay int8, so
        the tensor cores are still doing int8 MACs.
        """
        q, sx = _q_per_row(xf, self.a_levels)
        qi = q.to(torch.int32)
        hi, lo = split_digits(qi, self.a_levels)
        if (USE_TRITON and w8a8_matmul_hilo is not None and x.is_cuda
                and xf.dtype == torch.float32):
            y = w8a8_matmul_hilo(hi.contiguous(), lo.contiguous(),
                                 sx.reshape(-1).float(), self.w8t, self.sw,
                                 self.bias)
            return y.reshape(*shape[:-1], self.out_features).to(x.dtype)
        # Fallback: same algebra in torch. `_int_mm` gives int32 directly, which
        # is what the combine needs - the combined value reaches K*a_levels*127
        # (3.16e8 at K=4864), far past fp32's exact range, so combining in float
        # would round even though each half is fine.
        m = hi.shape[0]
        acc = None
        if x.is_cuda or _CPU_INT_MM_EXACT:
            pad = (-m) % 32 if x.is_cuda else 0
            try:
                def mm(t):
                    tp = t if pad == 0 else torch.nn.functional.pad(t, (0, 0, 0, pad))
                    return torch._int_mm(tp, self.w8t)[:m]
                acc = mm(hi) * 128 + mm(lo)
            except Exception:                        # noqa: BLE001
                acc = None
        if acc is None:
            wf = self.w8t.to(torch.float64)
            acc = (torch.matmul(hi.to(torch.float64), wf) * 128.0
                   + torch.matmul(lo.to(torch.float64), wf))
            assert float(acc.abs().max()) < 2.0 ** 53, "accumulator left the exact range"
        y = acc.to(torch.float32) * sx.to(torch.float32) * self.sw.unsqueeze(0)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shape[:-1], self.out_features).to(x.dtype)


class ExactRMSNorm(nn.Module):
    """RMSNorm whose reduction length is fixed by the model, not the batch.

    The sum is over the hidden dimension, so its extent does not change with
    batch size; carrying it in float64 keeps the intermediate wide enough that
    the f32 result is the same however the kernel splits it.
    """

    def __init__(self, norm):
        super().__init__()
        self.register_buffer("weight", norm.weight.data.clone())
        self.eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))
        divisor(norm.weight.numel(), self.weight.to(torch.float64))

    def forward(self, x):
        xd = x.to(torch.float64)
        var = xd.pow(2).sum(dim=-1, keepdim=True) / divisor(xd.shape[-1], xd)
        # sqrt and a tensor/tensor divide are correctly rounded on every IEEE
        # device; `rsqrt` is not specified to the last bit, so x / sqrt(v)
        # gives the same float64 on a CPU and a GPU where x * rsqrt(v) may not.
        y = xd / torch.sqrt(var + self.eps)
        return (y.to(torch.float32) * self.weight).to(x.dtype)


SILU_LIM = 16          # sigmoid is 0 / 1 past +-16, to within 2^-24 of the table
SILU_KNOT = 8         # table knots every 2^-8
SILU_QX = 20          # x quantised to 2^-20 between knots
SILU_QS = 24          # sigmoid in fixed point 2^-24


def silu_table():
    """sigmoid at every knot of [-16, 16], fixed point 2^-24, built on the HOST.

    Built in float64 on the CPU and copied, never computed on the device: a
    table a GPU builds with its own `exp` is the problem this replaces.
    Returns (values, differences to the next knot), both int32.
    """
    n = 2 * SILU_LIM * (1 << SILU_KNOT)
    xs = torch.arange(n + 2, dtype=torch.float64) / (1 << SILU_KNOT) - SILU_LIM
    t = torch.round(torch.sigmoid(xs) * (1 << SILU_QS)).to(torch.int64)
    return t[:-1].to(torch.int32), (t[1:] - t[:-1]).to(torch.int32)


class ExactSiLU(nn.Module):
    """SiLU whose sigmoid is integer interpolation in a host-built table.

    A float32 `exp` differs by an ulp between libm and CUDA, and that alone
    moved a CPU decode off a GPU decode. The first fix took the sigmoid in
    float64 and rounded once, which is only near-certain (two float64 `exp`s
    can still straddle a float32 rounding boundary) and cost 1.46x end to end
    compiled, because float64 runs at 1/64 rate on a GeForce part.

    Here nothing device-dependent is left. `q = round(x * 2^20)` after a clamp
    is exact (a power-of-two scale and round-half-even); the interpolation
    `T[i] + (D[i] * f) >> 12` is int32 and cannot overflow (D <= 2^14, f <
    2^12); the sum is at most 2^24, so its conversion to float32 and the 2^-24
    scale are exact; the one rounding is `x * sigmoid`, an IEEE multiply with no
    add after it for an FMA to absorb. Error against float64 SiLU: <= 2e-6
    absolute, <= 7.3e-4 relative where |SiLU| > 1e-3 (bf16 rounds at 2e-3);
    `exact_selftest.check_silu_table` holds it to that.
    """

    def __init__(self):
        super().__init__()
        t, d = silu_table()
        self.register_buffer("t", t, persistent=False)
        self.register_buffer("d", d, persistent=False)

    def forward(self, x):
        sub = SILU_QX - SILU_KNOT
        # NaN would convert to a device-dependent int32 and index out of range
        # (a trap on a GPU); it is zeroed here and put back by the final multiply.
        xf = torch.nan_to_num(x.to(torch.float32), nan=0.0).clamp(-SILU_LIM, SILU_LIM)
        q = (torch.round(xf * float(1 << SILU_QX)).to(torch.int32)
             + (SILU_LIM << SILU_QX))
        i = (q >> sub).clamp_max(self.t.numel() - 1)
        s = self.t[i] + ((self.d[i] * (q & ((1 << sub) - 1))) >> sub)
        sig = s.to(torch.float32) * (1.0 / (1 << SILU_QS))
        sig = torch.where(xf <= -SILU_LIM, torch.zeros_like(sig), sig)
        sig = torch.where(xf >= SILU_LIM, torch.ones_like(sig), sig)
        return (x.to(torch.float32) * sig).to(x.dtype)


class ExactRotary(nn.Module):
    """RoPE cos/sin read from a table built once, on the host, in float64.

    The stock module evaluates cos/sin in float32 on whatever device runs the
    model, and libm and CUDA differ there by an ulp. The table is computed from
    the module's own float32 `inv_freq`, rounded once to float32, and indexed
    by position, so no device evaluates a transcendental function at all.
    """

    def __init__(self, rot, max_positions):
        super().__init__()
        inv = rot.inv_freq.detach().to("cpu", torch.float64)
        pos = torch.arange(max_positions, dtype=torch.float64)
        emb = torch.cat((pos[:, None] * inv[None, :],) * 2, dim=-1)
        scale = float(getattr(rot, "attention_scaling", 1.0))
        self.register_buffer("cos", (emb.cos() * scale).to(torch.float32), persistent=False)
        self.register_buffer("sin", (emb.sin() * scale).to(torch.float32), persistent=False)

    def forward(self, x, position_ids):
        return self.cos[position_ids].to(x.dtype), self.sin[position_ids].to(x.dtype)


def convert(model, linears=True, norms=True, floats=None):
    """Replace every Linear / RMSNorm in place. Returns counts.

    `floats` (default: follow `norms`) also swaps the SiLU activations and the
    rotary embedding for `ExactSiLU` / `ExactRotary`, the two remaining float
    ops that made a CPU decode differ from a GPU one.
    """
    floats = norms if floats is None else floats
    n_lin = n_norm = 0
    shapes, dev = [], None
    max_pos = getattr(getattr(model, "config", None), "max_position_embeddings", 32768)
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if linears and isinstance(child, nn.Linear):
                new = ExactLinear(child).to(child.weight.device)
                setattr(mod, child_name, new)
                shapes.append((new.out_features, new.in_features))
                dev = child.weight.device
                n_lin += 1
            elif norms and child.__class__.__name__.endswith("RMSNorm"):
                setattr(mod, child_name, ExactRMSNorm(child).to(child.weight.device))
                n_norm += 1
            elif floats and child.__class__.__name__ in ("SiLUActivation", "SiLU"):
                dev_ = next(model.parameters()).device
                setattr(mod, child_name, ExactSiLU().to(dev_))
            elif floats and child.__class__.__name__.endswith("RotaryEmbedding"):
                setattr(mod, child_name,
                        ExactRotary(child, max_pos).to(child.inv_freq.device))
    # The attention's quantiser widths, created now so a compiled forward finds
    # them as graph inputs (see `divisor`; a miss would cost a fallback to eager,
    # not bits). Imported here: exact_kv imports `divisor` from this module.
    cfg = getattr(model, "config", None)
    params = list(model.parameters())
    if cfg is not None and params and hasattr(cfg, "num_attention_heads"):
        import exact_kv
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        like = torch.empty((), dtype=torch.float32, device=params[0].device)
        for lv in (127, exact_kv.k_levels_for(head_dim, 127), exact_kv.V_LEVELS):
            divisor(lv, like)
    # Choose each GEMM's tile now, eagerly, while shapes are concrete ints and
    # nothing is being traced. Doing it lazily on first call put the benchmark
    # inside `torch.compile`, where N and K are SymInts and the kernel cannot
    # even be launched. Five distinct shapes here, a few hundred ms total.
    if (linears and shapes and dev is not None and dev.type == "cuda"
            and USE_TRITON and w8a8_matmul is not None):
        import w8a8_gemv
        w8a8_gemv.prime(shapes, dev)
    return n_lin, n_norm
