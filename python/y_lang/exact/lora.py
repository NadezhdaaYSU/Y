"""Exact LoRA fine-tuning (T3): a model served by `y_lang.exact`, fine-tuned
with integer sums, so the adapter weights after any number of steps are the
same bits on a CPU and on a GPU, and the fine-tuned model is served with the
same tokens on either.

    from y_lang.exact import load, lora
    model, tok = load("Qwen/Qwen2.5-0.5B-Instruct", device)
    ad = lora.attach(model, r=16, alpha=16, seed=0)
    sft = lora.SFT(model, ad, lr=2e-4)
    tp = sft.step(ids, labels)          # labels -100 where nothing is learned
    sft.verify()

**The model trained is the model served.** The forward pass of training is
the served model's own forward: transformers' Qwen2 code over `model`'s exact
modules (`ExactLinear`, `ExactRMSNorm`, `ExactSiLU`, `ExactRotary`,
`attention.exact_attention`), each wrapped here in an autograd Function whose
forward calls the module and whose backward is written out below. With
gradients off the wrappers call the modules and nothing else. So the logits
training computes are the logits generation computes, bit for bit (a test
compares them with a decode over the KV cache), and an RL sampler and its
trainer agree exactly on every log-probability.

**The backward is exact training's rule** (`train_lm`): every sum is an
integer sum, the gradient a 12-bit operand as two int8 digits, the other
operand int8, on an int8 GEMM (Y's on a GPU, float64 BLAS on a CPU, both
exact); the sums over a row (a norm's, the softmax's) are `Ops.sum`'s fixed
point; elementwise work is IEEE float32 +, -, * and tensor-by-tensor /. The
quantizers of the forward pass are passed through as the identity (the
straight-through estimator) and the frozen weights are their int8 values
times their scales. So the gradient is a fixed function of the model, the
adapters and the batch, the same bits on every device, not the float
gradient of the quantized model; tests hold each piece to PyTorch's float64
autograd of the unquantized piece.

The adapters: `y = base(x) + c * (x A^T) B^T`, c = alpha / r a power of two
(exact), A [r][K] and B [N][r] float64 master weights (Adam, `train_lm.Adam`,
float64 with `det_sqrt`), each product of the forward an exact int8 product
on per-row power-of-two scales (`Ops.matmul`). A starts uniform in +-1/sqrt(K)
from a seeded generator on the CPU, B at zero, as LoRA's paper and PEFT do.
"""

import hashlib
import math
import os

import torch
import torch.nn as nn

from . import attention as _attn
from . import kv as _kv
from . import mlp_kernel as _mlp
from . import model as _model
from . import qmm as _qmm
from .train_lm import (F64_EXACT, Adam, Ops, det_sqrt, exact_product, exact_sum, f64_formula_ok, int8_bmm_ok,
                       mm_f64, product_widths, q8, reduce_checks, refuse, softmax_ints)

TARGETS = ("qkv_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "gate_up_proj",
           "q_proj", "k_proj", "v_proj")


# ---- int8 GEMMs of any shape -------------------------------------------------

Y_MAX_K = 131040


def _pad_to(t, dim, mult):
    n = t.shape[dim]
    extra = (-n) % mult
    # Under torch.compile n is symbolic and `if not extra` guards on n mod
    # `mult`: a length that is a multiple of 32 compiled its own version
    # (2026-11-25, a T4: the whole-layer backward 12 times, 3 to 4 minutes
    # each). Compiled, every length is padded, a pad of 0 the same values.
    if not torch.compiler.is_compiling() and not extra:
        return t
    pad = [0, 0] * (t.dim() - 1 - (dim % t.dim())) + [0, extra]
    return torch.nn.functional.pad(t, pad)


def mm_y_any(a, b):
    """`a @ b^T` for integer a [M][K], b [N][K] with entries in [-127, 127],
    exact, on Y's int8 GEMM for any shape: K and N zero-padded to its
    multiples (a zero adds nothing), K past its 131,040 split into pieces
    whose int32 results are added in int64."""
    from ..int8_gemm import int8_mm
    a = a.to(torch.int8)
    b = b.to(torch.int8)
    N = b.shape[0]
    a = _pad_to(a, 1, 32)
    b = _pad_to(_pad_to(b, 1, 32), 0, 8)
    K = a.shape[1]
    if K <= Y_MAX_K:
        # One call. Under `torch.compile` with dynamic shapes this is a
        # guard on K's range; the loop below would fix K to one value, and
        # every new length (a token count, in the adapters' gradients) would
        # compile again until the recompile limit sent the rest eager.
        M = a.shape[0]
        if torch.compiler.is_compiling():
            # The op pads a's rows to its kernel's row count itself, a fill
            # and a copy a call; padded here, the pad fuses into the kernel
            # that makes a. Zero rows, and each output row is its own row's.
            from ..int8_gemm import rows_bucket
            a = torch.nn.functional.pad(a, (0, 0, 0, rows_bucket(M) - M))
        return int8_mm(a.contiguous(), b.contiguous())[:M, :N].to(torch.int64)
    out = None
    for k0 in range(0, K, Y_MAX_K):
        part = int8_mm(a[:, k0:k0 + Y_MAX_K].contiguous(), b[:, k0:k0 + Y_MAX_K].contiguous())
        part = part[:, :N].to(torch.int64)
        out = part if out is None else out + part
    return out


# `Y_EXACT_QMM=0` turns off the fused product (`qmm`) for comparisons.
QMM = os.environ.get("Y_EXACT_QMM", "1") != "0"
# dx's two products in one kernel (`qmm.qmm_sum`, the same bits).
QMM_SUM = os.environ.get("Y_EXACT_QMM_SUM", "1") != "0"


def default_mm(device):
    """Y's GEMM on a GPU, float64 BLAS (exact below 2^53) on a CPU."""
    return mm_y_any if torch.device(device).type == "cuda" else mm_f64


def make_ops(device, mm=None, grad_bits=12, bwd_bits=8, compile=False):
    """The exact `Ops` of the adapters' products; `compile` (a GPU only)
    runs each product compiled (`CompiledOps`)."""
    if compile and torch.device(device).type != "cuda":
        raise ValueError("lora: compile is for a CUDA device")
    cls = CompiledOps if compile else Ops
    return cls(exact=True, mm=mm or default_mm(device), grad_bits=grad_bits, bwd_bits=bwd_bits,
               act=torch.float32)


# ---- compiled products -------------------------------------------------------

def _checked_product(a, b, bits_a, bits_b, mm, batched, int8, act, qa, qb):
    bad = []
    out = exact_product(a, b, bits_a, bits_b, mm, batched, int8, act, qa, qb, bad)
    return out, [c for c, _ in bad], [m for _, m in bad]


_COMPILED = []


def _compiled_product():
    if not _COMPILED:
        _COMPILED.append(torch.compile(_checked_product, dynamic=True))
    return _COMPILED[0]


class CompiledOps(Ops):
    """`Ops` whose exact products on a GPU each run as one compiled function
    (`torch.compile` of `train_lm.exact_product`: the quantizers, the digit
    split, the GEMM, the join and the scale fused into a few kernels), with
    shapes dynamic so a new batch length does not recompile. The same
    function: it is integer and bit operations, products by powers of two,
    one rounding to float32 and the GEMM, none of which Inductor rounds
    differently (`model.py`'s flags cover the operations that it does).

    One hazard it does have, measured 2026-11-22: under `dynamic=True` a
    Python int becomes symbolic, even a closure's constant, and Inductor
    lowers q8's `(1 << (bits - 1)) - 1` to float64 `pow(2.0, bits - 1) - 1`,
    which CUDA's pow puts below 2048, so 12-bit values clamped at 2046, not
    2047 (91 of a step's 935 products differed). So this sets
    `torch._dynamo.config.specialize_int` in every thread
    (`model.set_in_every_thread`: the backward's products compile on
    autograd's own thread, which a plain assignment does not reach), and
    refuses to run in a thread where it is unset; the widths are then
    constants of each graph. Ints specialized cost recompiles, never bits.
    Tests compare every product of a step with the eager one, and a value at
    the clamp."""

    # A training step's products come in about 20 kinds (widths, operand
    # types, a frozen int8 weight or not, transposed operands, batched or
    # not), each one compiled graph with its lengths dynamic. Past PyTorch's
    # default limit of 8 graphs a function runs eagerly, silently: measured
    # 2026-11-24, the backward then launched about 500,000 kernels a step.
    RECOMPILE_LIMIT = 64

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        _model.set_in_every_thread(torch._dynamo.config, "specialize_int", True)
        # Python floats (attention's scale, a norm's eps) as constants too, as
        # the inference path compiles them. Symbolic (torch 2.13's default),
        # each arrived as a 0-d CPU float64 tensor and part of the graph ran
        # as a C++ CPU kernel (found 2026-11-24: its build cannot link from a
        # path with a space in it).
        _model.set_in_every_thread(torch._dynamo.config, "specialize_float", True)
        if torch._dynamo.config.recompile_limit < self.RECOMPILE_LIMIT:
            _model.set_in_every_thread(torch._dynamo.config, "recompile_limit", self.RECOMPILE_LIMIT)

    def product(self, a, b, bits_a, bits_b, batched, int8, qa, qb):
        if not a.is_cuda:
            return super().product(a, b, bits_a, bits_b, batched, int8, qa, qb)
        if not torch._dynamo.config.specialize_int:
            raise RuntimeError("lora: compiled products need torch._dynamo.config.specialize_int "
                               "(a symbolic width moves q8's clamp, see CompiledOps)")
        # Autograd never differentiates through a product (the wrapped
        # modules' backward calls it), so its operands' requires_grad, a
        # guard of the compiled graph, need not double the graphs.
        a, b = a.detach(), b.detach()
        out, conds, msgs = _compiled_product()(a, b, bits_a, bits_b, self.mm, batched, int8,
                                               self.act, qa, qb)
        for cond, msg in zip(conds, msgs):
            refuse(self.bad, cond, msg)
        return out


# ---- the wrapped modules -----------------------------------------------------

def _needs_grad(*ts):
    return torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in ts)


class _FrozenFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mod):
        ctx.mod = mod
        return mod.base(x)

    @staticmethod
    def backward(ctx, dy):
        return (ctx.mod.dx_base(dy) if ctx.needs_input_grad[0] else None), None


class Frozen(nn.Module):
    """An `ExactLinear` that is not trained: dX = dY W, W its int8 values
    times its row scales, as an exact product (the scales folded into dY)."""

    def __init__(self, base, ops):
        super().__init__()
        self.base = base
        self.ops = ops
        self._wk = None

    def forward(self, x):
        if _needs_grad(x):
            return _FrozenFn.apply(x, self)
        return self.base(x)

    def _weight(self):
        w = self.base.w8t                                  # [K][N] int8
        if self._wk is None or self._wk[0].device != w.device:
            wk = w if w.is_contiguous() else w.contiguous()
            self._wk = (wk, torch.zeros(w.shape[0], 1, dtype=torch.int64, device=w.device))
        return self._wk

    def dx_base(self, dy):
        base = self.base
        g = dy.reshape(-1, base.out_features).to(torch.float32) * base.sw
        wk, s0 = self._weight()
        dx = self.ops.matmul(g, wk, grad="a", qb=(wk, s0))
        return dx.reshape(*dy.shape[:-1], base.in_features)


class _LoRAFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, A, B, mod):
        ctx.mod = mod
        y, xf, u = mod._forward(x, A, B)
        ctx.save_for_backward(xf, u)
        ctx.shape = x.shape
        return y

    @staticmethod
    def backward(ctx, dy):
        mod = ctx.mod
        xf, u = ctx.saved_tensors
        A, B = mod.A, mod.B
        wk, s0 = mod._weight()
        dx, dA, dB = _run(mod.ops, _lora_backward, dy.detach(), xf, u, A.detach(), B.detach(),
                          mod.base.sw, wk, s0, mod.scale, ctx.needs_input_grad[0], *_widths(mod.ops))
        if dx is not None:
            dx = dx.reshape(ctx.shape)
        return dx, dA.to(A.dtype), dB.to(B.dtype), None


def _widths(ops):
    return ops.grad_bits, ops.bwd_bits, ops.mm, ops.act


def _exact_mm(a, b, grad, grad_bits, bwd_bits, mm, act, bad, qb=None):
    """`Ops.matmul` of two 2-D operands, its checks appended to `bad`. On a
    GPU with Y's GEMM and float32 results, the quantized operands' product
    runs as one kernel (`qmm`, the same bits as `exact_product`)."""
    bits_a, bits_b = product_widths(a.shape[-1], grad, grad_bits, bwd_bits)
    if (QMM and mm is mm_y_any and act is torch.float32 and bits_a <= 12 and bits_b <= 8
            and a.shape[-1] <= _qmm.MAX_K and _qmm.available(a.device)):
        qa, sa = q8(a, bits=bits_a, bad=bad)
        qbv, sb = q8(b, bits=bits_b, bad=bad) if qb is None else qb
        return _qmm.qmm(qa, sa, qbv, sb, bits_a, bits_b)
    return exact_product(a, b, bits_a, bits_b, mm, False, False, act, None, qb, bad)


def _exact_mm_sum(a, qb, c, d, grad_bits, bwd_bits, mm, act, bad):
    """`_exact_mm(a, qb[0], "a", ..., qb=qb) + _exact_mm(c, d, "a", ...)`,
    the same checks in the same order; where both products take `qmm`, one
    kernel (`qmm.qmm_sum`: both exact products, each rounded to float32 as
    `qmm` rounds it, added in float32) instead of two and an add."""
    bits_a, bits_b = product_widths(a.shape[-1], "a", grad_bits, bwd_bits)
    bits_c, bits_d = product_widths(c.shape[-1], "a", grad_bits, bwd_bits)
    if (QMM_SUM and QMM and mm is mm_y_any and act is torch.float32
            and bits_a <= 12 and bits_b <= 8 and bits_c <= 12 and bits_d <= 8
            and max(a.shape[-1], c.shape[-1]) <= _qmm.MAX_K and _qmm.available(a.device)):
        qa, sa = q8(a, bits=bits_a, bad=bad)
        qc, sc = q8(c, bits=bits_c, bad=bad)
        qdv, sd = q8(d, bits=bits_d, bad=bad)
        return _qmm.qmm_sum(qa, sa, qb[0], qb[1], bits_a, bits_b, qc, sc, qdv, sd, bits_c, bits_d)
    return (_exact_mm(a, qb[0], "a", grad_bits, bwd_bits, mm, act, bad, qb=qb)
            + _exact_mm(c, d, "a", grad_bits, bwd_bits, mm, act, bad))


def _lora_forward(x, A, B, base, scale, grad_bits, bwd_bits, mm, act):
    """`LoRALinear`'s forward: ((y, x as float32 rows, u = x A^T), check
    conditions, their messages)."""
    bad = []
    y = base(x)
    xf = x.reshape(-1, base.in_features).to(torch.float32)
    u = _exact_mm(xf, A, None, grad_bits, bwd_bits, mm, act, bad)              # [M][r]
    z = _exact_mm(u, B, None, grad_bits, bwd_bits, mm, act, bad)               # [M][N]
    y = (y.reshape(-1, base.out_features) + z * scale).reshape(y.shape)
    return (y, xf, u), [c for c, _ in bad], [m for _, m in bad]


def _lora_backward(dy, xf, u, A, B, sw, wk, s0, scale, need_dx, grad_bits, bwd_bits, mm, act):
    """`_LoRAFn`'s backward: ((dx or None, dA, dB), check conditions, their
    messages). dx = dY W (the frozen int8 W, its row scales folded into dY)
    + du A."""
    bad = []
    N = B.shape[0]
    dyc = dy.reshape(-1, N).to(torch.float32) * scale                         # a power of two: exact
    dB = _exact_mm(dyc.t(), u.t(), "a", grad_bits, bwd_bits, mm, act, bad)     # [N][r], over tokens
    du = _exact_mm(dyc, B.t(), "a", grad_bits, bwd_bits, mm, act, bad)         # [M][r]
    dA = _exact_mm(du.t(), xf.t(), "a", grad_bits, bwd_bits, mm, act, bad)     # [r][K], over tokens
    dx = None
    if need_dx:
        g = dy.reshape(-1, N).to(torch.float32) * sw
        dx = _exact_mm_sum(g, (wk, s0), du, A.t(), grad_bits, bwd_bits, mm, act, bad)
    # In the adapters' type (float64) here, where the cast fuses.
    return (dx, dA.to(A.dtype), dB.to(B.dtype)), [c for c, _ in bad], [m for _, m in bad]


def _run(ops, fn, *args):
    """`fn(*args)`, a function of tensors and constants returning (result,
    check conditions, their messages): compiled (one graph per kind of call,
    shapes dynamic) when `ops` is `CompiledOps` on a GPU and nothing is being
    compiled around it, else as it is. Its checks go to `ops.bad` as `Ops`'
    own do."""
    on_gpu = any(isinstance(a, torch.Tensor) and a.is_cuda for a in args)
    if isinstance(ops, CompiledOps) and on_gpu and not torch.compiler.is_compiling():
        if not torch._dynamo.config.specialize_int:
            # The products' widths must be constants of the graph (see
            # CompiledOps: a symbolic width moved q8's clamp).
            raise RuntimeError("lora: compiled steps need torch._dynamo.config.specialize_int")
        fn = _compiled(fn)
    out, conds, msgs = fn(*args)
    for cond, msg in zip(conds, msgs):
        refuse(ops.bad, cond, msg)
    return out


class LoRALinear(Frozen):
    """`base(x) + scale * (x A^T) B^T`, each product exact (`Ops.matmul`)."""

    def __init__(self, base, ops, r, alpha):
        super().__init__(base, ops)
        scale = alpha / r
        if scale <= 0 or math.frexp(scale)[0] != 0.5:
            raise ValueError(f"lora: alpha / r = {scale} must be a power of two, so that "
                             f"scaling by it is exact")
        self.r, self.scale = r, float(scale)
        self.A = self.B = None

    def _forward(self, x, A, B):
        return _run(self.ops, _lora_forward, x.detach(), A.detach(), B.detach(), self.base, self.scale,
                    *_widths(self.ops))

    def forward(self, x):
        if _needs_grad(x, self.A, self.B):
            return _LoRAFn.apply(x, self.A, self.B, self)
        return self._forward(x, self.A, self.B)[0]


class _NormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mod):
        ctx.mod = mod
        ctx.save_for_backward(x)
        return _run(mod.ops, _module_forward, x.detach(), mod.base)

    @staticmethod
    def backward(ctx, dy):
        mod = ctx.mod
        ops, base = mod.ops, mod.base
        (x,) = ctx.saved_tensors
        div = _model.divisor(x.shape[-1], x.new_empty(0, dtype=torch.float64))
        return _run(ops, _norm_backward, x.detach(), dy.detach(), base.weight, base.eps, div), None


def _norm_backward(x, dy, weight, eps, div):
    """`_NormFn`'s backward, a function of tensors and constants (so
    `CompiledOps` users run it compiled, the same bits: integer sums, an
    integer square root, float64 and float32 operations that the process's
    Inductor flags round as eager does, and `div` made outside the graph so
    it stays a divide). Returns (dx, check conditions, their messages)."""
    bad = []
    H = x.shape[-1]
    v = _model.sum_of_squares(x) / div + eps
    r = (torch.ones_like(v) / det_sqrt(v, bad)).to(torch.float32)
    xh = x * r
    dxh = dy.to(torch.float32) * weight
    mean = (exact_sum(dxh * xh, -1, bad) * (1.0 / H)).to(torch.float32)
    return r * (dxh - xh * mean), [c for c, _ in bad], [m for _, m in bad]


_COMPILED_FNS = {}


def _compiled(fn):
    """`fn` under `torch.compile` with dynamic shapes, one per function."""
    if fn not in _COMPILED_FNS:
        _COMPILED_FNS[fn] = torch.compile(fn, dynamic=True)
    return _COMPILED_FNS[fn]


class Norm(nn.Module):
    """An `ExactRMSNorm` (weight frozen): dx = r (dy w - xh mean(dy w xh)),
    r = 1 / sqrt(mean(x^2) + eps) by `det_sqrt`, the mean an exact sum."""

    def __init__(self, base, ops):
        super().__init__()
        self.base, self.ops = base, ops

    def forward(self, x):
        if _needs_grad(x):
            return _NormFn.apply(x, self)
        return _run(self.ops, _module_forward, x, self.base)


class _SiLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mod):
        ctx.mod = mod
        ctx.save_for_backward(x)
        return _run(mod.ops, _silu_forward, x.detach(), mod.base)

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        return _run(ctx.mod.ops, _silu_backward, x.detach(), dy.detach(), ctx.mod.base), None


def _module_forward(x, base):
    """An exact module's own forward (an `ExactRMSNorm`'s), no checks."""
    return base(x), [], []


def _silu_forward(x, base):
    return base(x), [], []


def _silu_backward(x, dy, base):
    """d silu(x) = s (1 + x (1 - s)), s the table's sigmoid."""
    sig = base.sigmoid(x)
    x = x.to(torch.float32)
    return dy * (sig * (1.0 + x * (1.0 - sig))), [], []


class SiLU(nn.Module):
    """An `ExactSiLU`: d silu(x) = s (1 + x (1 - s)), s the table's sigmoid."""

    def __init__(self, base, ops):
        super().__init__()
        self.base, self.ops = base, ops

    def forward(self, x):
        if _needs_grad(x):
            return _SiLUFn.apply(x, self)
        return _run(self.ops, _silu_forward, x, self.base)


class _AttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask, module, scaling, ops):
        _warm_attention(q, k, scaling, ops)
        # The module only reaches a decode cache, which training refuses
        # (`attach`), so None: one compiled graph for every layer.
        out = _run(ops, _attn_forward, q.detach(), k.detach(), v.detach(), mask, scaling)
        ctx.save_for_backward(q, k, v, mask if mask is not None else torch.empty(0))
        ctx.has_mask, ctx.scaling, ctx.ops = mask is not None, scaling, ops
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, mask = ctx.saved_tensors
        mask = mask if ctx.has_mask else None
        ops = ctx.ops
        d = q.shape[-1]
        sc = ctx.scaling if ctx.scaling is not None else 1.0 / math.sqrt(d)
        int8 = _bmm_int8(ops, q)
        _warm_attention(q, k, ctx.scaling, ops)
        dq, dk, dv = _run(ops, _attn_backward, q.detach(), k.detach(), v.detach(), mask, dout.detach(), sc,
                          ops.grad_bits, ops.bwd_bits, int8, ops.act)
        return dq, dk, dv, None, None, None, None


def _bmm_int8(ops, q):
    return ops.bmm == "int8" or (ops.bmm is None and q.is_cuda and int8_bmm_ok(q.device))


_WARM = set()


def _warm_attention(q, k, scaling, ops):
    """Run the attention's forward and backward once, uncompiled, on a tiny
    input of `q` and `k`'s head layout, so that what they make on first use
    exists before a compiled graph reaches it: the exp2 table (a global
    write) and the quantizers' divisors, which `model.divisor` refuses to
    make while compiling (a constant there would become a reciprocal
    multiply), sending the rest of that call eager, with a recompile per
    length (measured 2026-11-24)."""
    if q.is_cuda:
        # The device's two decisions made here, before any compiled call (as
        # `Trainer` makes them): made first inside a traced call, each filled
        # its cache dict there, and the next call failed that guard and
        # compiled the whole layer again (2026-11-25).
        f64_formula_ok(q.device)
        int8_bmm_ok(q.device)
    key = (q.device, q.dtype, q.shape[1], k.shape[1], q.shape[-1], scaling, _kv.PACK)
    if key in _WARM or not isinstance(ops, CompiledOps):
        return
    if q.device not in _attn.TABLE:
        _attn.TABLE[q.device] = _attn._table(q.device)
    nh, nkv, d = q.shape[1], k.shape[1], q.shape[-1]
    sc = scaling if scaling is not None else 1.0 / math.sqrt(d)
    g = torch.Generator(device=q.device).manual_seed(0)

    def r(*shape):
        return torch.randn(*shape, generator=g, device=q.device).to(q.dtype)
    with torch.no_grad():
        _attn_forward(r(1, nh, 2, d), r(1, nkv, 2, d), r(1, nkv, 2, d), None, scaling)
        if _kv.PACK:
            kp, vp = _kv.quantize_and_pack(r(1, nkv, 2, d), r(1, nkv, 2, d))
            _attn_forward(r(1, nh, 1, d), kp, vp, None, scaling)        # decode
            _attn_forward(r(1, nh, 2, d), kp, vp, None, scaling)        # a prefill
        _attn_backward(r(1, nh, 2, d), r(1, nkv, 2, d), r(1, nkv, 2, d), None, r(1, 2, nh, d), sc,
                       ops.grad_bits, ops.bwd_bits, _bmm_int8(ops, q), ops.act)
    _WARM.add(key)


def _attn_forward(q, k, v, mask, scaling):
    """`exact_attention` without a module: (output, no checks)."""
    out, _ = _attn.exact_attention(None, q, k, v, mask, scaling=scaling)
    return out, [], []


def _attn_backward(q, k, v, mask, dout, sc, grad_bits, bwd_bits, int8, act):
    """`_AttnFn`'s backward: ((dq, dk, dv), check conditions, their
    messages), every product exact (`Ops.matmul`, batched), the softmax's
    row sums exact (`exact_sum`), the weights the forward's own integers."""
    bad = []

    def bmm(a, b_, grad):
        bits_a, bits_b = product_widths(a.shape[-1], grad, grad_bits, bwd_bits)
        return exact_product(a, b_, bits_a, bits_b, None, True, int8, act, None, None, bad)

    b, nh, T, d = q.shape
    nkv, tk = k.shape[1], k.shape[-2]
    rep = nh // nkv
    P = attention_weights(q, k, v, mask, sc)                     # [b][nh][T][tk]
    f32 = torch.float32
    Pg = P.reshape(b, nkv, rep * T, tk)
    qg = q.reshape(b, nkv, rep * T, d)
    dOg = dout.transpose(1, 2).reshape(b, nkv, rep * T, d).to(f32)
    dPg = bmm(dOg, v, "a")
    dv = bmm(Pg.transpose(-1, -2), dOg.transpose(-1, -2), "b")
    dSg = Pg * (dPg - exact_sum(Pg * dPg, -1, bad).to(f32)) * sc
    dq = bmm(dSg, k.transpose(-1, -2), "a")
    dk = bmm(dSg.transpose(-1, -2), qg.transpose(-1, -2), "a")
    return (dq.reshape(b, nh, T, d), dk, dv), [c for c, _ in bad], [m for _, m in bad]


def attention_weights(q, k, v, mask, sc):
    """The forward pass's softmax weights, p / l, as float32 (each p, l the
    exact integers of `attention.exact_softmax_weights`)."""
    dev = q.device
    if dev not in _attn.TABLE:
        _attn.TABLE[dev] = _attn._table(dev)
    qi, sq = _attn.quantize_queries(q)
    ki, sk, vi, sv, k_lv, v_lv = _kv.quantize_kv(None, k, v, _attn.Q_LEVELS)
    p, l, _ = _attn.exact_softmax_weights(qi, sq, ki, sk, sv, k_lv, mask, sc, _attn.TABLE[dev])
    return (p.to(torch.float64) / l.to(torch.float64).clamp(min=1.0)).to(torch.float32)


_OPS = {}


def attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    """`attention.exact_attention`, with an exact backward when gradients
    are wanted (Qwen2's eager attention while adapters are attached)."""
    ops = _OPS.get(id(module))
    if ops is not None and _needs_grad(query, key, value):
        return _AttnFn.apply(query, key, value, attention_mask, module, scaling, ops), None
    if ops is not None:
        # No gradients (sampling, a prefill), K/V maybe packed: the module
        # only reaches a decode cache, which `attach` refuses.
        _warm_attention(query, key, scaling, ops)
        return _run(ops, _attn_forward, query, key, value, attention_mask, scaling), None
    return _attn.exact_attention(module, query, key, value, attention_mask, scaling=scaling,
                                 dropout=dropout, **kw)


attention.takes_packed_kv = True


# ---- whole decoder layers (compiled training) --------------------------------
#
# Above, each wrapped module is an autograd Function of its own, one compiled
# call each way: about 18 calls a decoder layer, 3,000 a training step of
# Qwen2.5-0.5B, with Python, Dynamo and autograd between them. That host time
# was the step's critical path (the GPU busy 0.63 of 1.0 s, 2026-11-24).
# `_LayerFn` makes a decoder layer ONE Function: its forward one compiled call
# of `_layer_forward`, its backward one of `_layer_backward`. They are built
# from the same pure functions the wrappers call (`_module_forward`,
# `_lora_forward`, `_attn_forward`, `_silu_forward` and their backwards) and
# transformers' own `apply_rotary_pos_emb`, in the order transformers'
# decoder layer runs them; the glue between them (residual adds, the MLP's
# product, the rotary embedding, views) is written out as autograd computes
# it: an add's gradient passes through, a product's is the gradient times the
# other operand, a tensor used twice gets the sum of its two gradients (an
# IEEE add, the same bits in either order). The one freedom left is the sign
# of a zero where autograd adds a slice's zero padding (the rotary halves):
# every such gradient enters only quantizers (q8: |x| and x / 2^s, so -0.0
# and +0.0 give the same integer 0), never an adapter's or a returned value
# directly. Compiled elementwise work rounds as eager does (`model.py`'s
# Inductor flags: no FMA contraction, divides correctly rounded). Tests: the
# replay of recorded runs (every step's p(target), so every earlier update)
# and the GPU against the CPU, which keeps the per-module path.
#
# Only for compiled training on a GPU, and only for a layer of the shape it
# was written from (merged q/k/v, separate gate and up, every projection with
# adapters); `Y_EXACT_LAYER=0` keeps the per-module path everywhere.

LAYER_FUSED = os.environ.get("Y_EXACT_LAYER", "1") != "0"


def _layer_forward(h, cos, sin, mask, Aq, Bq, Ao, Bo, Ag, Bg, Au, Bu, Ad, Bd,
                   n1, n2, qkv, o, gate, up, down, act, sizes, head_dim, scaling, sq, so, sg, su, sd,
                   grad_bits, bwd_bits, mm, act_dt):
    """A decoder layer's forward (transformers' Qwen2/Llama decoder layer
    over the merged attention forward): ((out, saved), checks, messages)."""
    import transformers.models.qwen2.modeling_qwen2 as qwen2
    conds, msgs = [], []

    def run(fn, *a):
        out, c, m = fn(*a)
        conds.extend(c)
        msgs.extend(m)
        return out
    w = (grad_bits, bwd_bits, mm, act_dt)
    x1 = run(_module_forward, h, n1)
    input_shape = x1.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)
    yq, xf_q, u_q = run(_lora_forward, x1, Aq, Bq, qkv, sq, *w)
    q, k, v = yq.split(sizes, dim=-1)
    q = q.view(hidden_shape).transpose(1, 2)
    k = k.view(hidden_shape).transpose(1, 2)
    v = v.view(hidden_shape).transpose(1, 2)
    q, k = qwen2.apply_rotary_pos_emb(q, k, cos, sin)
    attn = run(_attn_forward, q, k, v, mask, scaling)
    a2 = attn.reshape(*input_shape, -1).contiguous()
    yo, xf_o, u_o = run(_lora_forward, a2, Ao, Bo, o, so, *w)
    h2 = h + yo
    x2 = run(_module_forward, h2, n2)
    if _mlp.usable(gate, up, act, x2, *w):
        # Both projections and their glue in one kernel (`mlp_kernel`): the
        # same bits, without the int32 and float32 intermediates in memory.
        g, yu, a, m, xf_g, u_g, u_u = run(_mlp.gate_up_silu, x2, Ag, Bg, Au, Bu, gate, up, act, sg, su, *w)
        xf_u = xf_g
    else:
        g, xf_g, u_g = run(_lora_forward, x2, Ag, Bg, gate, sg, *w)
        yu, xf_u, u_u = run(_lora_forward, x2, Au, Bu, up, su, *w)
        a = run(_silu_forward, g, act)
        m = a * yu
    yd, xf_d, u_d = run(_lora_forward, m, Ad, Bd, down, sd, *w)
    out = h2 + yd
    saved = (h, xf_q, u_q, q, k, v, xf_o, u_o, h2, xf_g, u_g, xf_u, u_u, g, a, yu, xf_d, u_d)
    return (out, saved), conds, msgs


def _rotary_backward(d, cos, sin):
    """The gradient of `apply_rotary_pos_emb`'s x * cos + rotate_half(x) * sin
    with respect to x: d * cos, plus rotate_half's backward of d * sin (its
    cat's halves back to the slices they came from, the first negated)."""
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    ds = d * s
    half = ds.shape[-1] // 2
    return d * c + torch.cat((ds[..., half:], -ds[..., :half]), dim=-1)


def _layer_backward(dout, h, xf_q, u_q, q, k, v, xf_o, u_o, h2, xf_g, u_g, xf_u, u_u, g, a, yu, xf_d, u_d,
                    cos, sin, mask, Aq, Bq, Ao, Bo, Ag, Bg, Au, Bu, Ad, Bd,
                    wq, wo, wg, wu, wd, n1w, n1eps, n2w, n2eps, div1, div2, act, sc, int8, need_dh,
                    sq, so, sg, su, sd, grad_bits, bwd_bits, mm, act_dt):
    """`_layer_forward`'s backward: ((dh or None, dAq, dBq, ..., dAd, dBd),
    checks, messages). `w*` are each projection's (sw, int8 weight, zero
    scales) as `Frozen._weight` holds them."""
    conds, msgs = [], []

    def run(fn, *a_):
        out, c, m_ = fn(*a_)
        conds.extend(c)
        msgs.extend(m_)
        return out
    w = (grad_bits, bwd_bits, mm, act_dt)
    shape = h.shape
    # out = h2 + yd: both get dout.
    dm, dAd, dBd = run(_lora_backward, dout, xf_d, u_d, Ad, Bd, *wd, sd, True, *w)
    dm = dm.reshape(yu.shape)
    # m = a * yu
    da = dm * yu
    dyu = dm * a
    dg = run(_silu_backward, g, da, act)
    dx2g, dAg, dBg = run(_lora_backward, dg, xf_g, u_g, Ag, Bg, *wg, sg, True, *w)
    dx2u, dAu, dBu = run(_lora_backward, dyu, xf_u, u_u, Au, Bu, *wu, su, True, *w)
    dx2 = dx2g.reshape(shape) + dx2u.reshape(shape)
    dh2 = dout + run(_norm_backward, h2, dx2, n2w, n2eps, div2)
    # h2 = h + yo: both get dh2.
    da2, dAo, dBo = run(_lora_backward, dh2, xf_o, u_o, Ao, Bo, *wo, so, True, *w)
    b_, nh, T, d = q.shape
    dattn = da2.reshape(b_, T, nh, d)
    dq, dk, dv = run(_attn_backward, q, k, v, mask, dattn, sc, grad_bits, bwd_bits, int8, act_dt)
    dq = _rotary_backward(dq, cos, sin)
    dk = _rotary_backward(dk, cos, sin)
    dyq = torch.cat([t.transpose(1, 2).reshape(b_, T, -1) for t in (dq, dk, dv)], dim=-1)
    dx1, dAq, dBq = run(_lora_backward, dyq, xf_q, u_q, Aq, Bq, *wq, sq, need_dh, *w)
    dh = None
    if need_dh:
        dh = dh2 + run(_norm_backward, h, dx1.reshape(shape), n1w, n1eps, div1)
    return (dh, dAq, dBq, dAo, dBo, dAg, dBg, dAu, dBu, dAd, dBd), conds, msgs


class _LayerFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, cos, sin, mask, layer, *params):
        f = layer._y_fused
        ops = f["ops"]
        _warm_attention(f["q_like"], f["k_like"], f["scaling"], ops)
        out, saved = _run(ops, _layer_forward, h.detach(), cos, sin, mask, *[p.detach() for p in params],
                          *f["mods"], f["sizes"], f["head_dim"], f["scaling"], *f["scales"], *_widths(ops))
        ctx.save_for_backward(*saved, cos, sin, mask if mask is not None else torch.empty(0))
        ctx.layer, ctx.has_mask = layer, mask is not None
        ctx.params = params
        return out

    @staticmethod
    def backward(ctx, dout):
        f = ctx.layer._y_fused
        ops = f["ops"]
        *saved, cos, sin, mask = ctx.saved_tensors
        mask = mask if ctx.has_mask else None
        h, q, k, h2 = saved[0], saved[3], saved[4], saved[8]
        like = h.new_empty(0, dtype=torch.float64)
        div1 = _model.divisor(h.shape[-1], like)
        div2 = _model.divisor(h2.shape[-1], like)
        sc = f["scaling"] if f["scaling"] is not None else 1.0 / math.sqrt(q.shape[-1])
        _warm_attention(q, k, f["scaling"], ops)
        n1, n2 = f["norms"]
        weights = [(m.base.sw, *m._weight()) for m in f["loras"]]
        dh, *grads = _run(ops, _layer_backward, dout.detach(), *saved, cos, sin, mask,
                          *[p.detach() for p in ctx.params], *weights, n1.weight, n1.eps, n2.weight, n2.eps,
                          div1, div2, f["act"], sc, _bmm_int8(ops, q), ctx.needs_input_grad[0],
                          *f["scales"], *_widths(ops))
        grads = [gr.to(p.dtype) for gr, p in zip(grads, ctx.params)]
        return (dh, None, None, None, None, *grads)


def _fuse_layer(layer, ops):
    """Give `layer` the one-Function path (`_LayerFn`) for compiled GPU
    training if it is the shape `_layer_forward` was written from; returns
    whether it did."""
    attn, mlp = getattr(layer, "self_attn", None), getattr(layer, "mlp", None)
    if attn is None or mlp is None:
        return False
    names = ((attn, "qkv_proj"), (attn, "o_proj"), (mlp, "gate_proj"), (mlp, "up_proj"), (mlp, "down_proj"))
    loras = [getattr(o, n, None) for o, n in names]
    n1, n2 = getattr(layer, "input_layernorm", None), getattr(layer, "post_attention_layernorm", None)
    act = getattr(mlp, "act_fn", None)
    attn_cls, mlp_cls = _model.attention_and_mlp_classes()
    fwd = vars(attn).get("forward")
    if not (all(isinstance(m, LoRALinear) for m in loras) and isinstance(n1, Norm) and isinstance(n2, Norm)
            and isinstance(act, SiLU) and isinstance(attn, attn_cls) and isinstance(mlp, mlp_cls)
            and getattr(fwd, "__func__", None) is _model._merged_attention_forward
            and "forward" not in vars(mlp)                 # transformers' own: gate and up apart
            and getattr(attn.config, "_attn_implementation", None) == "eager"
            and getattr(attn, "sliding_window", None) is None):
        return False
    nh = attn.config.num_attention_heads
    nkv = getattr(attn.config, "num_key_value_heads", nh)
    dev = loras[0].base.w8t.device
    layer._y_fused = {
        "ops": ops, "loras": loras, "norms": (n1.base, n2.base), "act": act.base,
        "mods": (n1.base, n2.base, *[m.base for m in loras], act.base),
        "sizes": attn.qkv_sizes, "head_dim": attn.head_dim, "scaling": attn.scaling,
        "scales": tuple(m.scale for m in loras),
        "q_like": torch.empty(1, nh, 1, attn.head_dim, device=dev),
        "k_like": torch.empty(1, nkv, 1, attn.head_dim, device=dev),
    }
    plain = layer.forward

    def forward(hidden_states, attention_mask=None, position_ids=None, past_key_values=None,
                use_cache=False, position_embeddings=None, **kwargs):
        params = [t for m in loras for t in (m.A, m.B)]
        if (LAYER_FUSED and past_key_values is None and position_embeddings is not None
                and hidden_states.is_cuda and _needs_grad(hidden_states, *params)):
            cos, sin = position_embeddings
            return _LayerFn.apply(hidden_states, cos, sin, attention_mask, layer, *params)
        return plain(hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                     past_key_values=past_key_values, use_cache=use_cache,
                     position_embeddings=position_embeddings, **kwargs)
    layer.forward = forward
    return True


# ---- attaching ---------------------------------------------------------------

class Adapters:
    """The adapters of one model: `params` (name -> float64 tensor, leaves
    that require grad) and the modules that read them."""

    def __init__(self, model, ops, modules):
        self.model, self.ops, self.modules = model, ops, modules
        self.params = {}
        for name, m in modules.items():
            self.params[f"{name}.A"] = m.A
            self.params[f"{name}.B"] = m.B

    def rebind(self, params):
        """Point every module (and `self.params`) at `params`' tensors."""
        for name, m in self.modules.items():
            m.A, m.B = params[f"{name}.A"], params[f"{name}.B"]
        self.params = dict(params)

    def load(self, values):
        """Copy `values` (name -> tensor, e.g. a saved state) in."""
        with torch.no_grad():
            for k, t in self.params.items():
                t.copy_(values[k].to(t.device, t.dtype))

    def state(self):
        return {k: t.detach().cpu().clone() for k, t in self.params.items()}

    def digest(self) -> str:
        return digest(self.params)


def digest(params) -> str:
    h = hashlib.sha256()
    for k in sorted(params):
        h.update(k.encode())
        h.update(params[k].detach().cpu().to(torch.float64).contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def attach(model, r=16, alpha=16, targets=None, seed=0, mm=None, grad_bits=12, bwd_bits=8,
           compile=False):
    """Adapters on `model`'s exact linear layers named in `targets` (default:
    every projection of every decoder layer), every other exact module given
    its exact backward. `model` must already be converted (`y_lang.exact.load`).
    `compile` (a GPU): the exact products compiled (`CompiledOps`), the same
    bits. Returns the `Adapters`."""
    import transformers.models.qwen2.modeling_qwen2 as qwen2
    if _kv.USE_CACHE:
        raise RuntimeError("lora: Y_EXACT_KVCACHE keeps quantised K/V across calls of the same "
                           "shape, so a training step would read the step before's rows")
    if any(isinstance(m, (Frozen, Norm, SiLU)) for m in model.modules()):
        raise RuntimeError("lora: adapters are already attached to this model")
    targets = TARGETS if targets is None else tuple(targets)
    device = next(model.buffers()).device
    ops = make_ops(device, mm, grad_bits, bwd_bits, compile)
    for p in model.parameters():
        p.requires_grad_(False)
    gen = torch.Generator().manual_seed(seed)
    modules = {}
    for path, mod in list(model.named_modules()):
        for name, child in list(mod.named_children()):
            full = f"{path}.{name}" if path else name
            if isinstance(child, _model.ExactLinear):
                if name in targets and ".layers." in f".{full}":
                    new = LoRALinear(child, ops, r, alpha)
                    K, N = child.in_features, child.out_features
                    bound = 1.0 / math.sqrt(K)
                    A = (torch.rand(r, K, generator=gen, dtype=torch.float64) * 2 - 1) * bound
                    new.A = A.to(device).requires_grad_()
                    new.B = torch.zeros(N, r, dtype=torch.float64, device=device).requires_grad_()
                    modules[full] = new
                else:
                    new = Frozen(child, ops)
                setattr(mod, name, new)
            elif isinstance(child, _model.ExactRMSNorm):
                setattr(mod, name, Norm(child, ops))
            elif isinstance(child, _model.ExactSiLU):
                setattr(mod, name, SiLU(child, ops))
            if isinstance(child, _model.attention_and_mlp_classes()[0]):
                _OPS[id(child)] = ops
    if not modules:
        raise ValueError(f"lora: no exact linear layer named {targets} in a decoder layer")
    if isinstance(ops, CompiledOps):
        for layer in getattr(getattr(model, "model", None), "layers", ()):
            _fuse_layer(layer, ops)
    if qwen2.eager_attention_forward is not _attn.exact_attention and \
            qwen2.eager_attention_forward is not attention:
        raise RuntimeError("lora: the model's attention is not y_lang.exact's")
    qwen2.eager_attention_forward = attention
    return Adapters(model, ops, modules)


# ---- the loss ----------------------------------------------------------------

def softmax_rows(ops, logits, rows=64, targets=None):
    """`Ops.softmax` over the vocabulary, `rows` at a time (its int64
    numerators are 8 bytes a logit). With `targets` (one token per row):
    (softmax, p, l), p each target's integer numerator and l its row's sum
    (`Ops.softmax_ints`)."""
    none = torch.zeros(1, logits.shape[-1], dtype=torch.bool, device=logits.device)
    P, pt, lt = [], [], []
    for i in range(0, logits.shape[0], rows):
        p, l = ops.softmax_ints(logits[i:i + rows], none.expand(min(rows, logits.shape[0] - i), -1))
        P.append(ops.ratio(p, l))
        if targets is not None:
            pt.append(p.gather(1, targets[i:i + rows, None]).reshape(-1))
            lt.append(l.reshape(-1))
    P = torch.cat(P)
    return P if targets is None else (P, torch.cat(pt), torch.cat(lt))


def _vocab_rows(logits, tgt, w, table, act):
    """Rows of `SFT._accumulate`'s loss with per-token weights: ((p(target)
    float, p and l of each target, dlogits = (P - onehot) * w), checks):
    P = p / l (`Ops.ratio`) from the exact softmax's integers."""
    bad = []
    blocked = torch.zeros(1, logits.shape[-1], dtype=torch.bool, device=logits.device).expand(logits.shape[0], -1)
    p, l = softmax_ints(logits, blocked, table, bad)
    dt = torch.float64 if act is None else act
    P = p.to(dt) / l.to(dt)
    col = tgt[:, None]
    onehot = torch.zeros_like(P).scatter_(1, col, 1.0)
    d = (P - onehot) * w.to(P.dtype)[:, None]
    out = (P.gather(1, col).reshape(-1), p.gather(1, col).reshape(-1), l.reshape(-1), d)
    return out, [c for c, _ in bad], [m for _, m in bad]


def target_positions(labels):
    """(flat positions whose next token is learned, those tokens): the
    usual shift, labels[t + 1] predicted at t, -100 ignored."""
    nxt = labels[:, 1:]
    keep = nxt != -100
    B, T = labels.shape
    pos = torch.arange(T - 1, device=labels.device).expand(B, -1) + \
        torch.arange(B, device=labels.device)[:, None] * T
    return pos[keep], nxt[keep]


# ---- supervised fine-tuning --------------------------------------------------

class SFT:
    """Supervised fine-tuning of the adapters: mean cross-entropy over the
    labelled tokens, Adam on the float64 master weights. Every check of a
    step is deferred to `verify`, one read of the device."""

    def __init__(self, model, adapters, lr=2e-4, b1=0.9, b2=0.999, eps=1e-8):
        self.model, self.ad, self.ops = model, adapters, adapters.ops
        params = {k: t.detach() for k, t in adapters.params.items()}
        self.opt = Adam(params, lr=lr, b1=b1, b2=b2, eps=eps, exact=True)
        self._rebind()
        self._acc, self.messages = None, None

    def _rebind(self):
        views = self.opt.bind({})
        self.ad.rebind({k: t.detach().requires_grad_() for k, t in views.items()})

    def forward(self, ids):
        """The hidden states before the head, [B][T][H] (with gradients).

        The causal mask is passed in, 4-D, as transformers would make it for
        eager attention without padding (0 where a key is seen, float32's
        min above the diagonal; the exact attention reads only `mask < -1`):
        transformers returns a 4-D mask as it is, where making one read the
        device twice (a packed-sequence test and the eager mask), and every
        read stops the host from queueing work ahead of the GPU."""
        B, T = ids.shape
        mask = torch.full((T, T), torch.finfo(torch.float32).min, device=ids.device).triu_(1)
        return self.model.model(input_ids=ids, attention_mask=mask.expand(B, 1, T, T),
                                use_cache=False).last_hidden_state

    def logits_at(self, hidden, pos):
        h = hidden.reshape(-1, hidden.shape[-1]).index_select(0, pos)
        return self.model.lm_head(h)

    def step(self, ids, labels):
        """One step; returns p(target) of the labelled tokens (float32, in
        order), from the forward pass before the update."""
        return self.run(ids, labels)[0]

    def run(self, ids, labels, weights=None, check=None):
        """`step`, with what RL needs. Returns (p(target), p, l): p and l
        the exact softmax's integers at each labelled token (its numerator,
        its row's sum). `weights` (float32, one per labelled token in
        `target_positions` order) makes the gradient through the softmax
        weights * (P - onehot) in place of (P - onehot) / n. `check(p, l)`
        returns (a bool tensor, a message), refused with the step's checks."""
        return self.run_chunks([(ids, labels, weights, check)])

    def run_chunks(self, chunks):
        """One update from several batches, each (ids, labels, weights,
        check) as `run` takes them, forward and backward one at a time with
        the gradients added in float64 in this order (so the same bits on
        every device), then one Adam step: a batch too large for the
        device's memory, in pieces. Each piece's products quantize on its
        own scales, so a split is a different function from the whole batch,
        each as exact. Returns `run`'s three tensors, the pieces' in order."""
        bad = []
        self.ops.bad = bad
        outs = []
        try:
            for ids, labels, weights, check, *targets in chunks:
                # A fifth element: (positions, tokens), `target_positions`
                # already computed on the host (`rl.training_batch`).
                outs.append(self._accumulate(ids, labels, weights, check, bad, *targets))
            grads = {}
            for k, t in self.ad.params.items():
                grads[k] = t.grad if t.grad is not None else torch.zeros_like(t)
            r1, r2 = self.opt.corrections()
            with torch.no_grad():
                self.opt.update(grads, r1, r2, bad)
        finally:
            self.ops.bad = None
            # Also after a failed piece: a later step must not add to them.
            for t in self.ad.params.values():
                t.grad = None
        if not bad:
            bad = [(torch.zeros((), dtype=torch.bool, device=chunks[0][0].device), "(none)")]
        flags, messages = reduce_checks(bad)
        if self._acc is None:
            self._acc, self.messages = torch.zeros_like(flags), messages
        if messages != self.messages:
            # A different batch shape can reach a different set of checks;
            # read this step's at once rather than mixing the two lists.
            if bool(flags.any()):
                raise OverflowError("; ".join(m for m, f in zip(messages, flags.tolist()) if f))
        else:
            self._acc |= flags
        if len(outs) == 1:
            return outs[0]
        return tuple(torch.cat([o[i] for o in outs]) for i in range(3))

    def _accumulate(self, ids, labels, weights, check, bad, targets=None):
        """Forward and backward of one batch, its gradients added to the
        adapters' `.grad`; (p(target), p, l)."""
        pos, tgt = targets if targets is not None else target_positions(labels)
        n = tgt.numel()
        if n == 0:
            raise ValueError("lora: a batch with no labelled token")
        if weights is not None and weights.shape != (n,):
            raise ValueError(f"lora: {tuple(weights.shape)} weights for {n} labelled tokens")
        if ids.shape[0] == 1 and isinstance(self.ops, CompiledOps) and ids.is_cuda:
            # A piece of one row runs as two: the row and a copy that no
            # target reads. Compiled with dynamic shapes, a size of 1 is a
            # special case (PyTorch's 0/1 specialization), and so is the
            # causal mask's stride then: every compiled function of the
            # layers compiled again, 432 s on a T4 (2026-11-25: GRPO.step's
            # pieces of 9 rows, the last one of 1). The copy has the row's
            # values, so each tensor's maximum, and so each quantizer's
            # scale, is the row's. It has no target, so its gradients are 0
            # and every integer sum is the same. The same bits (tests: the
            # CPU's, which runs the row alone).
            ids = ids.expand(2, -1).contiguous()
        hidden = self.forward(ids)
        logits = self.logits_at(hidden, pos)
        if weights is not None and isinstance(self.ops, CompiledOps) and logits.is_cuda:
            # The same values as below, `rows` at a time in one compiled
            # call each, without the whole P or a one-hot in memory.
            with torch.no_grad():
                table, lg = self.ops.table(logits.device), logits.detach()
                parts = [_run(self.ops, _vocab_rows, lg[i:i + 64], tgt[i:i + 64], weights[i:i + 64], table,
                              self.ops.act) for i in range(0, n, 64)]
                tp, p_t, l_t, dlogits = (torch.cat([q[j] for q in parts]) for j in range(4))
                if check is not None:
                    refuse(bad, *check(p_t, l_t))
            logits.backward(dlogits)
            return tp, p_t, l_t
        with torch.no_grad():
            P, p_t, l_t = softmax_rows(self.ops, logits.detach(), targets=tgt)
            tp = P[torch.arange(n, device=P.device), tgt]
            onehot = torch.zeros_like(P).scatter_(1, tgt[:, None], 1.0)
            if weights is None:
                dlogits = (P - onehot) * (1.0 / n)
            else:
                dlogits = (P - onehot) * weights.to(P.dtype)[:, None]
            if check is not None:
                refuse(bad, *check(p_t, l_t))
        logits.backward(dlogits)
        return tp, p_t, l_t

    def verify(self):
        """Raise OverflowError if a check of any step so far failed."""
        if self._acc is None:
            return
        hit = self._acc.cpu().tolist()
        if any(hit):
            raise OverflowError("; ".join(sorted({m for m, h in zip(self.messages, hit) if h})))


def nll(tp) -> float:
    """Mean -log p(target), on the host, for reading only (`math.fsum`)."""
    vals = (-torch.log(tp.detach().cpu().to(torch.float64).clamp(min=1e-30))).tolist()
    return math.fsum(vals) / max(len(vals), 1)


def tp_digest(tp) -> str:
    return hashlib.sha256(tp.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()).hexdigest()[:16]
