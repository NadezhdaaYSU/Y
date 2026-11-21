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

def pow2(k: torch.Tensor) -> torch.Tensor:
    """2^k as float64, built from its bits, for integer k in [-1022, 1023]."""
    k = k.to(torch.int64)
    if k.numel() and (int(k.min()) < -1022 or int(k.max()) > 1023):
        raise OverflowError("exact train: a scale exponent outside float64's normal range")
    return torch.bitwise_left_shift(k + 1023, 52).contiguous().view(torch.float64)


def exponent(x: torch.Tensor, dim: int) -> torch.Tensor:
    """e with max |x| along `dim` below 2^e (0 for an all-zero slice); frexp
    is exact."""
    return torch.frexp(x.abs().amax(dim=dim, keepdim=True))[1].to(torch.int64)


def q8(x: torch.Tensor, dim: int = -1, bits: int = 8):
    """Integer-valued float64 q and exponents s with x ~ q * 2^s, one s per
    slice along `dim` (the summed dimension), |q| <= 2^(bits-1) - 1 (int8:
    127)."""
    top = (1 << (bits - 1)) - 1
    s = exponent(x, dim) - (bits - 1)
    return torch.round(x * pow2(-s)).clamp(-top, top), s


def isqrt(n: torch.Tensor) -> torch.Tensor:
    """floor(sqrt(n)) for int64 0 <= n < 2^62: a float estimate (any
    device's, within 1 of the answer), then integer corrections, so the
    result is the one integer it must be."""
    if n.numel() and (int(n.min()) < 0 or int(n.max()) >= 1 << 62):
        raise OverflowError("exact train: isqrt takes 0 <= n < 2^62")
    x = torch.sqrt(n.to(torch.float64)).to(torch.int64)
    for _ in range(2):
        x = x - (x * x > n).to(torch.int64)
        x = x + ((x + 1) * (x + 1) <= n).to(torch.int64)
    return x


def det_sqrt(v: torch.Tensor) -> torch.Tensor:
    """sqrt of float64 v >= 0, to within 2^-30 relative, the same bits on
    every device: v scaled by an even power of two to an integer in
    [2^59, 2^61), its integer square root, scaled back. Takes 0 and
    2^-962 <= v < 2^1024; below that the scaling leaves float64's normal
    range and `pow2` refuses."""
    k = torch.frexp(v)[1].to(torch.int64) - 61
    k = k - (k & 1)                                    # even, so 2^(k/2) is exact
    n = (v * pow2(-k)).to(torch.int64)                 # an integer: v has 53 bits
    return isqrt(n).to(torch.float64) * pow2(torch.div(k, 2, rounding_mode="floor"))


def digits(q: torch.Tensor, bits: int):
    """Integer-valued float64 q with |q| < 2^(bits-1) as int8-valued digits
    d_i in [-64, 63] with q = sum_i d_i * 128^i (one digit, q itself, for 8
    bits or fewer). A product of wide operands is then a sum of int8
    products, each exact on an int8 GEMM. All arithmetic here is on small
    integers, exact in float64."""
    if bits <= 8:
        return [q]
    out, r = [], q
    for _ in range(-(-(bits - 1) // 7)):
        hi = torch.floor((r + 64) * (1.0 / 128))
        out.append(r - hi * 128)
        r = hi
    if bool((r != 0).any()):
        raise OverflowError(f"exact train: a value the {len(out)} digits for {bits} bits cannot hold")
    return out


def _ceil_log2(n: int) -> int:
    return max(0, (n - 1).bit_length())


def mm_f64(a: torch.Tensor, b: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """sum_k a[i][k] * b[j][k] for integer-valued float64 a, b of `bits` bits:
    exact, since every partial sum is an integer below 2^53."""
    top = (1 << (bits - 1)) - 1
    if a.shape[-1] * top * top >= F64_EXACT:
        raise OverflowError("exact train: K too large for an exact float64 sum")
    return a @ b.transpose(-1, -2)


def mm_y(a, b):
    """The same product on Y's int8 GEMM."""
    from ..int8_gemm import int8_mm
    return int8_mm(a.to(torch.int8).contiguous(), b.to(torch.int8).contiguous()).to(torch.float64)


def mm_torch(a, b):
    """The same product on `torch._int_mm` (cuBLASLt)."""
    return torch._int_mm(a.to(torch.int8).contiguous(),
                         b.to(torch.int8).contiguous().t()).to(torch.float64)


def _exact_fp32_bmm(a, b):
    """Batched sum_k a[..i][k] * b[..j][k] for int8-valued a, b in float32:
    exact because every partial sum is an integer below 2^24 (TF32 off)."""
    if a.shape[-1] * 127 * 127 >= FP32_EXACT:
        raise OverflowError(f"exact train: K = {a.shape[-1]} leaves fp32's exact range")
    if a.is_cuda and torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("exact train: TF32 rounds the products; set "
                           "torch.backends.cuda.matmul.allow_tf32 = False")
    return (a.to(torch.float32) @ b.to(torch.float32).transpose(-1, -2)).to(torch.float64)


class Ops:
    """The model's sums and softmax, exact (`exact=True`) or ordinary float
    (`exact=False`: the float twin and the autograd reference)."""

    def __init__(self, exact=True, mm=mm_f64, grad_bits=12, bwd_bits=8):
        """Widths of the backward pass's product operands: `grad_bits` for the
        gradient, `bwd_bits` for the activation or weight it meets. The
        forward pass is always int8, the format the model is served in. Past
        8 bits an operand is split into int8 digits (`digits`), so every
        product still runs on an int8 GEMM and is still exact. The default,
        12-bit gradients against int8 activations and weights, is the
        narrowest setting measured to train as well as float32: with int8
        gradients the held-out loss stalled at 2.01 against float32's 1.71
        (docs/benchmarks/exact_train_lm_2026-11-21/)."""
        self.exact, self.mm, self.grad_bits, self.bwd_bits = exact, mm, grad_bits, bwd_bits
        self._tables = {}

    def matmul(self, a, b, batched=False, grad=None):
        """a @ b^T, summed over the last dimension of both. `grad` names the
        backward pass's gradient operand, "a" or "b" (None: forward)."""
        if not self.exact:
            return a @ b.transpose(-1, -2)
        bits_a = 8 if grad is None else self.grad_bits if grad == "a" else self.bwd_bits
        bits_b = 8 if grad is None else self.grad_bits if grad == "b" else self.bwd_bits
        qa, sa = q8(a, bits=bits_a)
        qb, sb = q8(b, bits=bits_b)
        c = None
        for i, da in enumerate(digits(qa, bits_a)):
            for j, db in enumerate(digits(qb, bits_b)):
                part = _exact_fp32_bmm(da, db) if batched else self.mm(da, db)
                part = part * float(128 ** (i + j)) if i + j else part
                c = part if c is None else c + part      # integers below 2^53: exact
        if (abs(qa).amax() * abs(qb).amax() * a.shape[-1]) >= F64_EXACT:
            raise OverflowError("exact train: a product sum past float64's exact integers")
        return c * pow2(sa + sb.transpose(-1, -2))

    def sum(self, x, dim):
        """The sum along `dim` (kept), of x rounded to one fixed-point scale
        per sum chosen so the int64 total cannot overflow."""
        if not self.exact:
            return x.sum(dim=dim, keepdim=True)
        s = exponent(x, dim) - 61 + _ceil_log2(x.shape[dim])
        q = torch.round(x * pow2(-s)).to(torch.int64)
        return q.sum(dim=dim, keepdim=True).to(torch.float64) * pow2(s)

    def scatter_rows(self, idx, src, rows):
        """out[idx[t]] += src[t]: an embedding's gradient."""
        if not self.exact:
            return torch.zeros(rows, src.shape[1], dtype=src.dtype, device=src.device).index_add(0, idx, src)
        s = exponent(src, 0) - 61 + _ceil_log2(src.shape[0])
        q = torch.round(src * pow2(-s)).to(torch.int64)
        out = torch.zeros(rows, src.shape[1], dtype=torch.int64, device=src.device).index_add(0, idx, q)
        return out.to(torch.float64) * pow2(s)

    def softmax(self, z, blocked):
        """softmax over the last dimension, `blocked` entries excluded."""
        if not self.exact:
            return torch.softmax(z.masked_fill(blocked, float("-inf")), dim=-1)
        table = self._tables.get(z.device)
        if table is None:
            table = self._tables[z.device] = _table(z.device)
        logit = z * LOG2E_Q16                                  # log2 units, Q16.16
        m = logit.masked_fill(blocked, float("-inf")).amax(dim=-1, keepdim=True)
        t = torch.round(m - logit).clamp(0, 1 << 30).masked_fill(blocked, 1 << 30)
        p = exp2_neg_q16_16(t, table).masked_fill(blocked, 0)  # Q0.28, int64
        return p.to(torch.float64) / p.sum(dim=-1, keepdim=True).to(torch.float64)


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
    r = torch.ones_like(ms) / (det_sqrt(ms) if ops.exact else torch.sqrt(ms))
    xh = x * r
    return xh * g, (xh, r, g)


def _norm_bwd(ops, dy, cache):
    xh, r, g = cache
    d = xh.shape[-1]
    dg = ops.sum(dy * xh, 0).reshape(-1)
    dxh = dy * g
    return r * (dxh - xh * (ops.sum(dxh * xh, -1) * (1.0 / d))), dg


def forward(ops, cfg, p, idx, targets=None):
    """Logits [B*T][vocab] for token ids idx [B][T]; with targets, also the
    probabilities and everything `backward` needs."""
    B, T = idx.shape
    H, Dh = cfg.heads, cfg.d // cfg.heads
    scale = 1.0 / math.sqrt(Dh)
    flat = idx.reshape(-1)
    x = p["tok"][flat] + p["pos"][:T].repeat(B, 1)
    causal = torch.ones(T, T, dtype=torch.bool, device=idx.device).triu(1)
    caches = []
    for i in range(cfg.layers):
        hn, c_n1 = _norm_fwd(ops, x, p[f"n1.{i}"], cfg.eps)
        qkv = ops.matmul(hn, p[f"qkv.{i}"])
        q, k, v = (t.reshape(B, T, H, Dh).transpose(1, 2) for t in qkv.split(cfg.d, dim=1))
        P = ops.softmax(ops.matmul(q, k, batched=True) * scale, causal)
        o = ops.matmul(P, v.transpose(-1, -2), batched=True)          # [B][H][T][Dh]
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
    onehot = torch.zeros_like(probs)
    onehot[torch.arange(n, device=probs.device), targets.reshape(-1)] = 1.0
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
        dS = P * (dP - ops.sum(P * dP, -1)) * scale
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
    `det_sqrt`; the float twin with `torch.sqrt`."""

    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.99, eps=1e-8, exact=True):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.sqrt = det_sqrt if exact else torch.sqrt
        self.m = {k: torch.zeros_like(v) for k, v in params.items()}
        self.v = {k: torch.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        r1, r2 = 1.0 / (1.0 - self.b1 ** self.t), 1.0 / (1.0 - self.b2 ** self.t)
        for k in params:
            gk = grads[k]
            self.m[k] = self.m[k] * self.b1 + gk * (1.0 - self.b1)
            self.v[k] = self.v[k] * self.b2 + (gk * gk) * (1.0 - self.b2)
            den = self.sqrt(self.v[k] * r2) + self.eps
            params[k] = params[k] - (self.m[k] * r1) / den * self.lr


def _target_probs(probs, targets):
    return probs.detach().cpu().to(torch.float64)[torch.arange(targets.numel()), targets.reshape(-1).cpu()]


def cross_entropy_on_host(probs, targets) -> float:
    """Mean -log p(target), in float64 on the CPU, for reading only. The mean
    is `math.fsum` (correctly rounded in any order), so reordering the batch
    does not move it; `log` is the host's, which another machine's libm may
    round differently, so records compare `target_prob_digest` instead."""
    nll = (-torch.log(_target_probs(probs, targets).clamp(min=1e-30))).tolist()
    return math.fsum(nll) / len(nll)


def target_prob_digest(probs, targets) -> str:
    """A digest of p(target) for each token, in token order: the bits of an
    integer exp's quotient, the same on every device."""
    return hashlib.sha256(_target_probs(probs, targets).numpy().tobytes()).hexdigest()[:16]


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


def train(ops, cfg, data, steps, device, dtype=torch.float64, seed=0, B=16, lr=3e-3,
          eval_every=0, eval_batches=4, log=None):
    """Trains from `init_params(cfg, seed)`; returns (params, train losses,
    [(step, held-out loss)])."""
    params = init_params(cfg, seed, dtype, device)
    opt = Adam(params, lr=lr, exact=ops.exact)
    gen = torch.Generator().manual_seed(seed + 1)
    losses, evals = [], []
    for step in range(steps):
        x, y = data.batch(gen, B, cfg.ctx)
        x, y = x.to(device), y.to(device)
        _, probs, saved = forward(ops, cfg, params, x, y)
        losses.append(cross_entropy_on_host(probs, y))
        grads = backward(ops, cfg, params, probs, y, saved)
        opt.step(params, grads)
        if eval_every and ((step + 1) % eval_every == 0 or step + 1 == steps):
            evals.append((step + 1, evaluate(ops, cfg, params, data, device, eval_batches, B)))
            if log:
                log(f"step {step + 1:5d}  train {losses[-1]:.4f}  held-out {evals[-1][1]:.4f}")
    return params, losses, evals


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
