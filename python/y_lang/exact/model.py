"""Make every reduction in a transformer order-independent, then measure.

`attention` (once `tools/batch_invariance_demo.py`) replaced attention only, and that is not
enough: measured on Qwen2.5-0.5B, **99% of output logits change between batch 1
and batch 2** with stock kernels. Attention is one of three batch-dependent
reductions in the network. The others are the linear layers (a GEMM picks its
tiling from the problem size, which changes the summation order) and RMSNorm
(a sum over the hidden dimension).

This module swaps all three for order-independent versions:

  * `ExactLinear`   - W8A8, accumulated in exact int32 on the tensor cores:
                      the Triton kernel in `w8a8_gemv`, `torch._int_mm`, or
                      Y's own proved kernel (`Y_EXACT_GEMM`, below). Integer
                      addition, so any tile order gives the same bits, and
                      the three give the same bits. Falls back to float64, which represents
                      every integer below 2^53 exactly, when `_int_mm` will not
                      take the shape, or on a CPU whose int8 GEMM fails
                      `cpu_int_mm_is_exact`.
  * `ExactRMSNorm`  - the sum of squares in float64 over a dimension whose
                      length does not depend on the batch.
  * exact attention - in `attention`.

and, so the same tokens come out of a CPU and a GPU, the two float ops left:

  * `ExactSiLU`     - the sigmoid as integer interpolation in a host table.
  * `ExactRotary`   - RoPE cos/sin from a host-built table, each entry correctly rounded.

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
    from .w8a8_gemv import w8a8_linear, w8a8_matmul, w8a8_matmul_hilo
except Exception:                                    # noqa: BLE001
    w8a8_linear = w8a8_matmul = w8a8_matmul_hilo = None
# `Y_EXACT_TRITON=0` forces the torch fallback, so the two paths can be A/B'd
# against each other rather than only against the float64 reference.
USE_TRITON = os.environ.get("Y_EXACT_TRITON", "1") != "0"

# The kernel that computes `ExactLinear`'s int8 product on a GPU:
#   * "triton" - `w8a8_gemv`'s kernels, the default;
#   * "torch"  - `torch._int_mm`, i.e. cuBLASLt (what `Y_EXACT_TRITON=0` gives);
#   * "y"      - Y's own tensor-core GEMM, `y_lang.int8_gemm.int8_mm`: the
#                kernel whose schedule and value `proofs/Int8Gemm*.v` prove.
#                With it, the GEMM that runs is the GEMM that is proved. It has
#                no fallback: a shape it refuses is an error, not a quiet
#                switch to another kernel.
# All three sums are exact and the epilogue is the same torch code, so the
# choice changes speed and what is trusted, never a bit
# (`exact_selftest.check_y_gemm_agrees`). The CPU path is the same in every mode.
GEMM = os.environ.get("Y_EXACT_GEMM") or ("triton" if USE_TRITON else "torch")
if GEMM not in ("triton", "torch", "y"):
    raise ValueError(f"Y_EXACT_GEMM={GEMM!r}: expected 'triton', 'torch' or 'y'")
_Y_INT8_MM = None


def y_int8_mm():
    """`y_lang.int8_gemm.int8_mm`, from this package, so the op is the one
    built from the same sources as this module."""
    global _Y_INT8_MM
    if _Y_INT8_MM is None:
        from ..int8_gemm import int8_mm
        _Y_INT8_MM = int8_mm
    return _Y_INT8_MM

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

# What the process had before this import, so a baseline that should compile
# the way PyTorch ships can patch them back (exact_throughput's stock arm).
INDUCTOR_DEFAULTS = {
    "eager_numerics.division_rounding":
        _inductor_config.eager_numerics.division_rounding,
    "emulate_precision_casts": _inductor_config.emulate_precision_casts,
}
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
        if GEMM == "y":
            # Y's kernel reads B as [N][K] with K-contiguous rows, which is
            # nn.Linear's own layout. The buffer is still `w8t`, [K][N], as a
            # transposed VIEW of that storage, so every other path, `.to()`
            # (which keeps a dense tensor's strides) and the cross-device
            # weight fingerprint (buffer names and shapes) see what they
            # always did, and `w8t.t()` is the kernel's operand with no copy.
            self.register_buffer("w8t", w8.contiguous().t())
            y_int8_mm()          # imported now, so a missing liby.so fails here
        else:
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

    @classmethod
    def merge(cls, parts):
        """One ExactLinear computing `parts` side by side: output columns in
        order, `cat([p(x) for p in parts], -1)`, bit for bit.

        Nothing is quantised again. A part's weights, scales and bias are the
        rows of the merged ones, the input is quantised per row from `x` alone
        (`_q_per_row`), and each output column is its own integer dot product
        times its own scales. So merging changes how many GEMMs run, never a
        value. Parts that read different widths, or some with a bias and some
        without (a zero bias would turn a -0.0 into +0.0), are refused.
        """
        first = parts[0]
        same = ("in_features", "a_levels", "split_act", "fp32_exact")
        for p in parts[1:]:
            for f in same:
                if getattr(p, f) != getattr(first, f):
                    raise ValueError(f"cannot merge ExactLinears whose {f} differ")
        biases = [p.bias for p in parts]
        if any(b is None for b in biases) and not all(b is None for b in biases):
            raise ValueError("cannot merge ExactLinears with and without a bias")
        rows = torch.cat([p.w8t.t() for p in parts], dim=0)       # [sum N][K]
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        # The layout the parts were built with (`__init__`): Y's GEMM wants
        # the [N][K] matrix contiguous, the others its transpose.
        if first.w8t.t().is_contiguous():
            m.register_buffer("w8t", rows.contiguous().t())
        else:
            m.register_buffer("w8t", rows.t().contiguous())
        m.register_buffer("sw", torch.cat([p.sw for p in parts]))
        m.register_buffer("bias", None if biases[0] is None else torch.cat(biases))
        m.in_features = first.in_features
        m.out_features = sum(p.out_features for p in parts)
        for f in ("fp32_exact", "a_levels", "split_act"):
            setattr(m, f, getattr(first, f))
        return m

    def forward(self, x):
        shape = x.shape
        xf = x.reshape(-1, shape[-1])
        if self.split_act:
            return self._forward_split(x, xf, shape)
        x8, sx = _q_per_row(xf)
        x8 = x8.to(torch.int8)
        if (GEMM == "triton" and w8a8_matmul is not None and x.is_cuda
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
        if GEMM == "y" and x.is_cuda:
            # Y's kernel takes any M (it pads to its own row bucket), and the
            # epilogue below is the one the `_int_mm` path uses.
            acc = y_int8_mm()(x8.contiguous(), self.w8t.t()).to(torch.float32)
        elif x.is_cuda or _CPU_INT_MM_EXACT:
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
        if (GEMM == "triton" and w8a8_matmul_hilo is not None and x.is_cuda
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
        if GEMM == "y" and x.is_cuda:
            # Both digits in ONE launch, rows [hi; lo], so the weights are read
            # once, as the Triton kernel reads them once. int32 throughout: the
            # combine is inside the `(a_levels + 128)` bound asserted above.
            r = y_int8_mm()(torch.cat([hi, lo]), self.w8t.t())
            acc = r[:m] * 128 + r[m:]
        elif x.is_cuda or _CPU_INT_MM_EXACT:
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


# ExactRMSNorm's sum of squares, the same bits in any order (`sum_of_squares`).
#
# A float32 x is m * 2^(p/2 - 150) with m a 24-bit integer (the hidden bit
# set for a normal x; a subnormal has p = 2), so x^2 = m^2 * 2^(p - 300) with
# m^2 < 2^48: exact in float64, but a float64 sum of 3584 of them rounds at
# each add, and its bits depend on the order (a CPU's loop, Inductor's tree,
# a Triton kernel's tree differ). Here each square goes onto a fixed grid,
# 2^-SUMSQ_F of the row's largest square's unit (2^(E - 300), E the row's
# largest p), truncated, and the grid values are added as integers: exact,
# so every order gives the same integer S. The sum is float64(S) * 2^(E - 300
# - SUMSQ_F), one rounding. It is not math.fsum's correctly rounded sum: the
# truncation moves it by less than H * 2^-SUMSQ_F units, 2^-66 of the row's
# largest square at H = 3584, so the two differ only where the exact sum lies
# that close to a float64 rounding boundary. It is closer to the exact sum than
# a float64 loop's ((H - 1) * 2^-53 relative, at worst).
#
# S has up to 48 + SUMSQ_F + 13 bits, so it is carried as two integer limbs of
# SUMSQ_W bits (each limb's sum stays below 2^53 for H <= 8192), and the
# float64 conversion is one correctly rounded add of two exact float64s. A row
# with an infinite or NaN x has no grid; its float64 sum is inf or NaN in any
# order, and that is the sum.
SUMSQ_F, SUMSQ_W, SUMSQ_MAX_H = 32, 40, 8192


def sum_of_squares(x):
    """sum(x^2) over the last dimension of float32 `x`, as float64 [..., 1],
    computed so that every device and every order gives the same bits (see
    SUMSQ_F). float16 / bfloat16 `x` is widened to float32 first (exact)."""
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.to(torch.float32)
    if x.dtype != torch.float32:
        raise TypeError(f"sum_of_squares takes float32 (or 16-bit) values, not {x.dtype}")
    h = x.shape[-1]
    if h > SUMSQ_MAX_H:
        raise ValueError(f"sum_of_squares: rows of {h} > {SUMSQ_MAX_H} values would overflow "
                         f"its {SUMSQ_W}-bit limbs' float64-exact sums")
    xb = x.contiguous().view(torch.int32)
    ex = (xb >> 23) & 255
    frac = (xb & 0x7FFFFF).to(torch.int64)
    m = torch.where(ex == 0, frac, frac | 0x800000)
    p = 2 * ex.clamp(min=1).to(torch.int64)                  # x^2 = m^2 * 2^(p - 300)
    m2 = m * m
    # A zero has p = 2, the smallest, so zeros never raise the maximum (and
    # an all-zero row sums to 0 at any scale).
    e = p.amax(dim=-1, keepdim=True)
    s = (e - p).clamp(min=0)                                 # binades below the largest
    one = torch.ones_like(s)
    # T = floor(m^2 * 2^(F - s)) as hi * 2^W + lo, W = F + 8.
    hi = m2 >> (s + (SUMSQ_W - SUMSQ_F)).clamp(max=63)
    sa = s.clamp(max=SUMSQ_F)
    lo_near = (m2 & ((one << (sa + (SUMSQ_W - SUMSQ_F))) - 1)) << (SUMSQ_F - sa)
    lo_far = (m2 >> (s - SUMSQ_F).clamp(0, 63)) & ((1 << SUMSQ_W) - 1)
    lo = torch.where(s <= SUMSQ_F, lo_near, lo_far)
    big, small = hi.sum(dim=-1, keepdim=True), lo.sum(dim=-1, keepdim=True)
    big = big + (small >> SUMSQ_W)                           # < 2^53: exact in float64
    small = small & ((1 << SUMSQ_W) - 1)
    total = big.to(torch.float64) * float(1 << SUMSQ_W) + small.to(torch.float64)
    # 2^(E - 300 - F) from its bits (normal for every E: E - 332 is in [-330, 176]).
    scale = ((e - (300 + SUMSQ_F) + 1023) << 52).view(torch.float64)
    total = total * scale
    odd = (ex == 255).any(dim=-1, keepdim=True)              # an infinite or NaN x
    return torch.where(odd, x.to(torch.float64).pow(2).sum(dim=-1, keepdim=True), total)


class ExactRMSNorm(nn.Module):
    """RMSNorm whose reduction length is fixed by the model, not the batch.

    The sum is over the hidden dimension, so its extent does not change with
    batch size, and `sum_of_squares` makes it the same bits in any order, so
    on any device and however a kernel splits it.
    """

    def __init__(self, norm):
        super().__init__()
        self.register_buffer("weight", norm.weight.data.clone())
        self.eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))
        divisor(norm.weight.numel(), self.weight.to(torch.float64))

    def forward(self, x):
        if (_NORM_OP is not None and x.is_cuda and x.dtype == torch.float32
                and self.weight.dtype == torch.float32):
            # The same value without a float64 divide per element
            # (`norm_kernel`); one graph node under torch.compile.
            return _NORM_OP(x, self.weight, self.eps)
        xd = x.to(torch.float64)
        var = sum_of_squares(x) / divisor(xd.shape[-1], xd)
        # sqrt and a tensor/tensor divide are correctly rounded on every IEEE
        # device; `rsqrt` is not specified to the last bit, so x / sqrt(v)
        # gives the same float64 on a CPU and a GPU where x * rsqrt(v) may not.
        y = xd / torch.sqrt(var + self.eps)
        return (y.to(torch.float32) * self.weight).to(x.dtype)


# `Y_EXACT_NORM`: "torch" (the default) runs ExactRMSNorm through the formula
# above, "triton" runs it on CUDA through `norm_kernel`. Same bits either way.
# Opt-in until a 7B decode shows it faster in the model: the kernel cannot
# fuse with its neighbours, so the 0.5B forward launches 895 kernels where it
# launched 797 (and persistent reductions were faster alone and 2.2% slower
# in the model, docs/benchmarks/exact_persistent_7b_2026-11-08).
NORM = os.environ.get("Y_EXACT_NORM", "torch")
if NORM not in ("triton", "torch"):
    raise ValueError(f"Y_EXACT_NORM={NORM!r}: expected 'triton' or 'torch'")
# The registration lasts as long as the `Library` object (see attention.py).
_NORM_LIB = globals().get("_NORM_LIB")


def _register_norm_op():
    """`torch.ops.y.exact_rms_norm`, or None where Triton or CUDA is missing
    (then the formula runs, with the same bits)."""
    global _NORM_LIB
    if NORM != "triton" or not torch.cuda.is_available():
        return None
    try:
        from . import norm_kernel
    except ImportError:
        return None
    if not hasattr(torch.ops.y, "exact_rms_norm"):
        lib = torch.library.Library("y", "FRAGMENT")
        lib.define("exact_rms_norm(Tensor x, Tensor weight, float eps) -> Tensor")
        lib.impl("exact_rms_norm", norm_kernel.rms_norm, "CUDA")
        torch.library.register_fake(
            "y::exact_rms_norm",
            lambda x, weight, eps: torch.empty_like(x, dtype=torch.float32), lib=lib)
        _NORM_LIB = lib
    return torch.ops.y.exact_rms_norm


_NORM_OP = _register_norm_op()


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


# A float64 cos or sin is not correctly rounded: glibc, SLEEF (torch's CPU
# kernels) and CUDA each err by up to about an ulp, and differently. Rounding
# one to float32 still gives the correctly rounded float32 unless the true
# value lies within a few float64 ulps of a float32 rounding midpoint; there,
# another host's libm could round the other way. `cos_sin_f32` finds those
# entries (the float64 value's low 29 fraction bits within ROPE_MARGIN of
# 2^28, the pattern of a midpoint, as in `norm_kernel`) and decides them in
# mpmath (ROPE_MP_BITS). Every entry is then the correctly rounded
# float32 of cos(arg) * scale, on any host whose float64 cos/sin errs by less
# than ROPE_LIBM_ULPS ulps (libms claim 1).
ROPE_LIBM_ULPS = 4
ROPE_MARGIN = 2 * ROPE_LIBM_ULPS + 2          # float64 ulps: libm, the scale's rounding
ROPE_MP_BITS = (160, 320, 640, 1280, 2560)   # tried in turn until the rounding is decided


def _rn32_by_mpmath(fn, arg, scale):
    """The correctly rounded float32 of fn(arg) * scale (mpmath's cos or sin,
    float `arg` and `scale`), as a Python float: the value's significand
    scaled to 24 bits (float32's subnormals included) and rounded to nearest,
    at more bits until the value is clearly off the rounding midpoint. cos and
    sin of a nonzero float are never exactly a midpoint, but can come close:
    sin(x) is x - x^3/6 + ..., so for a tiny x on a float32 midpoint it lies
    about 2^-300 below it."""
    import mpmath
    for bits in ROPE_MP_BITS:
        with mpmath.workprec(bits):
            v = fn(mpmath.mpf(arg)) * mpmath.mpf(scale)
            if v == 0:
                return 0.0
            e = max(int(mpmath.frexp(v)[1]) - 1, -126)       # floor(log2 |v|), float32's floor
            if e > 127:
                raise ArithmeticError(f"RoPE table: fn({arg!r}) * {scale!r} overflows float32")
            q = mpmath.ldexp(v, 23 - e)                       # |q| < 2^24
            n = int(mpmath.floor(q))
            r = q - n
            if abs(r - mpmath.mpf(0.5)) > mpmath.ldexp(1, 28 - bits):
                return float(mpmath.ldexp(n + (r > 0.5), e - 23))
    raise ArithmeticError(f"RoPE table: fn({arg!r}) * {scale!r} is too close to a float32 "
                          f"rounding midpoint to decide at {ROPE_MP_BITS[-1]} bits")


def cos_sin_f32(arg, scale=1.0, approx=(torch.cos, torch.sin)):
    """float32 cos(arg) * scale and sin(arg) * scale, each correctly rounded,
    for a float64 tensor `arg` (see ROPE_LIBM_ULPS). `approx` are the float64
    functions that propose each value (the host's; tests pass others)."""
    import mpmath
    if arg.dtype != torch.float64:
        raise TypeError(f"cos_sin_f32 takes float64 arguments, not {arg.dtype}")
    out = []
    for fn, exact in zip(approx, (mpmath.cos, mpmath.sin)):
        y = fn(arg) * scale
        f = y.to(torch.float32)
        bits = y.contiguous().view(torch.int64)
        abits = bits & 0x7FFFFFFFFFFFFFFF
        dist = ((abits & ((1 << 29) - 1)) - (1 << 28)).abs()
        # Near a midpoint, or below 2^-126 (where float32's midpoints have
        # another pattern; this takes sin(0) = 0 too), or not finite.
        unsafe = (dist <= ROPE_MARGIN) | (abits < 0x3810000000000000) | ~torch.isfinite(y)
        idx = unsafe.nonzero(as_tuple=True)
        if idx[0].numel():
            f[idx] = torch.tensor([_rn32_by_mpmath(exact, a, scale) for a in arg[idx].tolist()],
                                  dtype=torch.float32)
        out.append(f)
    return out[0], out[1]


class ExactRotary(nn.Module):
    """RoPE cos/sin read from a table built once, on the host.

    The stock module evaluates cos/sin in float32 on whatever device runs the
    model, and libm and CUDA differ there by an ulp. The table is computed from
    the module's own float32 `inv_freq` at positions 0 .. max_positions - 1:
    each entry the correctly rounded float32 of cos or sin of
    `position * inv_freq` (a float64 product: exact for a float32 `inv_freq`
    below 2^29 positions, and an IEEE multiply anyway) times
    `attention_scaling` (`cos_sin_f32`), so no device evaluates a
    transcendental function and no host's libm decides a bit.
    """

    def __init__(self, rot, max_positions):
        super().__init__()
        inv = rot.inv_freq.detach().to("cpu", torch.float64)
        pos = torch.arange(max_positions, dtype=torch.float64)
        scale = float(getattr(rot, "attention_scaling", 1.0))
        cos, sin = cos_sin_f32(pos[:, None] * inv[None, :], scale)
        self.register_buffer("cos", torch.cat((cos, cos), dim=-1), persistent=False)
        self.register_buffer("sin", torch.cat((sin, sin), dim=-1), persistent=False)

    def forward(self, x, position_ids):
        return self.cos[position_ids].to(x.dtype), self.sin[position_ids].to(x.dtype)


# Merging q/k/v and gate/up replaces Qwen2Attention.forward and Qwen2MLP.forward
# per instance with copies that call one projection where these call several.
# The copies are only valid for the code they were copied from, so these lines
# must be in transformers' forwards; if one is not, nothing is merged.
_ATTENTION_LINES = (
    "query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)",
    "key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)",
    "value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)",
    "query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)",
    "key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)",
    "attn_output = attn_output.reshape(*input_shape, -1).contiguous()",
    "attn_output = self.o_proj(attn_output)",
)
_MLP_LINES = ("down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))",)
# `Y_EXACT_MERGE`: "qkv" (the default) merges q/k/v, "all" q/k/v and gate/up,
# "0" nothing; any other value is refused. Measured at 7B, batch 32, on Y's
# GEMM with the int8 V cache (docs/benchmarks/exact_merged_7b_2026-11-08):
# q/k/v 52.9/53.1 tok/s/seq, nothing 52.1/52.3, both 52.1/52.0. Merging
# gate/up gains nothing there and raises the peak memory of a long prefill:
# one [rows][2 x intermediate] int32 output where there were two halves
# (an 870 MiB allocation in the 7B record's longest prefill, which ran a 16 GB
# card out of memory with a desktop open).
_MERGE_CHOICES = {"0": (), "qkv": ("qkv",), "all": ("qkv", "gate_up")}
if os.environ.get("Y_EXACT_MERGE", "qkv") not in _MERGE_CHOICES:
    raise ValueError(f"Y_EXACT_MERGE={os.environ['Y_EXACT_MERGE']!r}: expected one of "
                     f"{', '.join(_MERGE_CHOICES)}")
MERGE = _MERGE_CHOICES[os.environ.get("Y_EXACT_MERGE", "qkv")]


def _merged_attention_forward(self, hidden_states, position_embeddings, attention_mask,
                              past_key_values=None, **kwargs):
    """transformers' Qwen2Attention.forward, with q/k/v from one projection."""
    import transformers.models.qwen2.modeling_qwen2 as qwen2
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    q, k, v = self.qkv_proj(hidden_states).split(self.qkv_sizes, dim=-1)
    query_states = q.view(hidden_shape).transpose(1, 2)
    key_states = k.view(hidden_shape).transpose(1, 2)
    value_states = v.view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = qwen2.apply_rotary_pos_emb(query_states, key_states, cos, sin)
    attention_interface = qwen2.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, qwen2.eager_attention_forward)
    if past_key_values is not None:
        if _kv.PACK and getattr(attention_interface, "takes_packed_kv", False):
            # The cache keeps rows quantised and packed (kv.PACK): only the
            # new rows are quantised, once, instead of the whole cache every
            # step, and the cache copies 3 bytes a value instead of 8.
            key_states, value_states = _kv.quantize_and_pack(key_states, value_states)
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
    attn_output, attn_weights = attention_interface(
        self, query_states, key_states, value_states, attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling, sliding_window=self.sliding_window, **kwargs)
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _merged_mlp_forward(self, x):
    """transformers' Qwen2MLP.forward, with gate and up from one projection."""
    gate, up = self.gate_up_proj(x).split(self.gate_up_sizes, dim=-1)
    return self.down_proj(self.act_fn(gate) * up)


_MERGED = (("qkv_proj", "qkv_sizes", ("q_proj", "k_proj", "v_proj")),
           ("gate_up_proj", "gate_up_sizes", ("gate_proj", "up_proj")))


def unmerged_buffer_shapes(model):
    """`(name, shape)` of every buffer as the model would have it unmerged.

    A merged ExactLinear's `w8t` ([K][sum N]), `sw` and `bias` ([sum N]) are
    listed as its parts' (`q_proj.w8t` [K][N_q], ...), so anything keyed on
    buffer names and shapes (exact_cross_device's weight fingerprint) is the
    same for a model however it was merged.
    """
    merged = {}
    for path, mod in model.named_modules():
        for fused, sizes, names in _MERGED:
            if isinstance(getattr(mod, fused, None), ExactLinear) and hasattr(mod, sizes):
                key = f"{path}.{fused}" if path else fused
                merged[key] = (path, names, getattr(mod, sizes))
    out = []
    for name, b in model.named_buffers():
        mod_path, _, leaf = name.rpartition(".")
        if mod_path in merged:
            parent, names, sizes = merged[mod_path]
            for part, n in zip(names, sizes):
                shape = (b.shape[0], n) if leaf == "w8t" else (n,)
                out.append((f"{parent}.{part}.{leaf}" if parent else f"{part}.{leaf}",
                            tuple(shape)))
        else:
            out.append((name, tuple(b.shape)))
    return out


def merge_projections(model, groups=("qkv", "gate_up")):
    """Merge each Qwen2 layer's q/k/v and gate/up ExactLinears
    (`ExactLinear.merge`): 4 GEMMs a layer where there were 7, and as many
    fewer activation quantisations and epilogues, with the same bits.
    Returns the number of modules merged; 0, with a message on stderr, if
    transformers' forwards are not the ones the replacements were copied
    from."""
    import inspect
    import sys
    import types
    import transformers.models.qwen2.modeling_qwen2 as qwen2
    attention_src = inspect.getsource(qwen2.Qwen2Attention.forward)
    mlp_src = inspect.getsource(qwen2.Qwen2MLP.forward)
    missing = [line for line in _ATTENTION_LINES if line not in attention_src]
    missing += [line for line in _MLP_LINES if line not in mlp_src]
    if missing:
        sys.stderr.write(
            "y_lang.exact: transformers' Qwen2 forwards are not the ones the merged "
            f"projections were written against ({missing[0]!r} is missing), so q/k/v "
            "and gate/up are left as separate GEMMs (same bits, slower).\n")
        return 0
    merged = 0
    for mod in list(model.modules()):
        names = None
        if isinstance(mod, qwen2.Qwen2Attention) and "qkv" in groups:
            names, fused, sizes, fwd = ("q_proj", "k_proj", "v_proj"), "qkv_proj", "qkv_sizes", _merged_attention_forward
        elif isinstance(mod, qwen2.Qwen2MLP) and "gate_up" in groups:
            names, fused, sizes, fwd = ("gate_proj", "up_proj"), "gate_up_proj", "gate_up_sizes", _merged_mlp_forward
        if names is None or not all(isinstance(getattr(mod, n, None), ExactLinear) for n in names):
            continue
        parts = [getattr(mod, n) for n in names]
        setattr(mod, fused, ExactLinear.merge(parts))
        setattr(mod, sizes, tuple(p.out_features for p in parts))
        for n in names:
            delattr(mod, n)
        mod.forward = types.MethodType(fwd, mod)
        merged += 1
    return merged


def convert(model, linears=True, norms=True, floats=None, merge=None):
    """Replace every Linear / RMSNorm in place. Returns counts.

    `floats` (default: follow `norms`) also swaps the SiLU activations and the
    rotary embedding for `ExactSiLU` / `ExactRotary`, the two remaining float
    ops that made a CPU decode differ from a GPU one.

    `merge` (default: `Y_EXACT_MERGE`, which defaults to q/k/v only) then
    merges projections (`merge_projections`), which changes no bits: True
    (q/k/v and gate/up), False (none), or the groups to merge.
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
    groups = MERGE if merge is None else (("qkv", "gate_up") if merge is True else (merge or ()))
    if groups:
        merge_projections(model, groups)
    exact = [m for m in model.modules() if isinstance(m, ExactLinear)]
    shapes = sorted({(m.out_features, m.in_features) for m in exact})
    if exact:
        dev = exact[0].sw.device
    # The attention's quantiser widths, created now so a compiled forward finds
    # them as graph inputs (see `divisor`; a miss would cost a fallback to eager,
    # not bits). Imported here: exact_kv imports `divisor` from this module.
    cfg = getattr(model, "config", None)
    params = list(model.parameters())
    if cfg is not None and params and hasattr(cfg, "num_attention_heads"):
        from . import kv as exact_kv
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        like = torch.empty((), dtype=torch.float32, device=params[0].device)
        for lv in (127, exact_kv.k_levels_for(head_dim, 127), exact_kv.V_LEVELS):
            divisor(lv, like)
    # Choose each GEMM's tile now, eagerly, while shapes are concrete ints and
    # nothing is being traced. Doing it lazily on first call put the benchmark
    # inside `torch.compile`, where N and K are SymInts and the kernel cannot
    # even be launched. Five distinct shapes here, a few hundred ms total.
    if (shapes and dev is not None and dev.type == "cuda"
            and GEMM == "triton" and w8a8_matmul is not None):
        from . import w8a8_gemv
        w8a8_gemv.prime(shapes, dev)
    return n_lin, n_norm


# Last, because kv imports `divisor` from here: either module can be imported
# first, and `_merged_attention_forward` reads `_kv` only when called.
from . import kv as _kv  # noqa: E402
