"""Exact training, first step (T1): one linear layer trained with SGD, where
every sum is an integer sum, so the weights after any number of steps are the
same bits on any device and in any reduction order.

    from y_lang.exact import train
    run = train.LinearRun.synthetic(n=64, k=256, tokens=512, seed=0)
    losses = run.train(steps=200, mm=train.mm_cpu)
    run.digest()                      # the same on a CPU and on a GPU

The layer is y = x W^T, the loss half the squared error against targets t.
Every value is an integer with a power-of-two scale:

  * the master weights `W` are int64 in Q(F): W_real = W * 2^-F;
  * each product's operands are int8 copies (`quantize_rows`), each row with
    its own shift s, the least that puts the row's largest |v| below 128; a
    power-of-two scale multiplies exactly, so no rounding happens outside the
    quantisers, and they round integers, not floats;
  * the three products (forward x W^T, input gradient g W, weight gradient
    g^T x) are int8 x int8 sums in int32: exact in any order;
  * the update W -= 2^-L * dW is an integer shift and subtract.

So the only place an order could matter, the sums, cannot change the result,
and the one place a device could differ, rounding, is integer arithmetic every
device does alike. The weight gradient sums over tokens; with the tokens split
into chunks (as across data-parallel GPUs, or a split K), the partial sums add
in int64 in any order. Its int8 copies use one scale per output row over all
tokens, so splitting needs that scale first: a max, which is also order-free.

This trades accuracy for exactness: weights and gradients pass through int8
with per-row scales. `FloatRun` is the float32 twin, the control: its sums
round, so its weights depend on the device and the chunk order.
"""

import hashlib
from dataclasses import dataclass, field

import torch

F = 30          # master weights and targets are Q30
EX = -7         # x is int8 with scale 2^-7: x_real in [-127/128, 127/128]
LR_SHIFT = 9    # learning rate 2^-9: below 2 / the largest Hessian eigenvalue at 512 tokens


def bit_length(m: torch.Tensor) -> torch.Tensor:
    """The number of bits of each non-negative int64 in `m` (0 for 0), by
    comparison with every power of two: integer work, the same everywhere."""
    p = torch.bitwise_left_shift(torch.ones(63, dtype=torch.int64, device=m.device),
                                 torch.arange(63, dtype=torch.int64, device=m.device))
    return (m.unsqueeze(-1) >= p).sum(-1)


def _check_bits(v: torch.Tensor, limit: int, what: str):
    top = int(bit_length(v.abs().amax().reshape(1))[0]) if v.numel() else 0
    if top > limit:
        raise OverflowError(f"exact train: {what} needs {top} bits, past the {limit} "
                            "this step licenses; refusing rather than wrapping")


def shift(v: torch.Tensor, e) -> torch.Tensor:
    """v * 2^e for int64 `v` and integer `e` (a tensor broadcast against `v`):
    exact for e >= 0, rounded to nearest with ties toward +inf for e < 0.
    Refuses a left shift that would leave int64."""
    e = torch.as_tensor(e, dtype=torch.int64, device=v.device).expand(v.shape)
    up = e.clamp(min=0)
    if v.numel():
        _check_bits(v, 62 - int(up.max()), "a left shift")
        if int((-e).max()) > 62:
            raise OverflowError("exact train: a right shift past 62 bits")
    down = (-e).clamp(min=0)
    half = torch.bitwise_left_shift(torch.ones_like(down), (down - 1).clamp(min=0)) * (down > 0)
    return torch.where(e >= 0, torch.bitwise_left_shift(v, up),
                       torch.bitwise_right_shift(v + half, down))


def quantize_rows(v: torch.Tensor):
    """`v` [R][C] int64 -> (q [R][C] int8, s [R][1] int64) with q = v / 2^s
    rounded and s >= 0 the least shift putting the row's largest |v| below
    128 (q is then within [-127, 127] after one clamp: the rounding can reach
    128 above and -128 below, proofs/ExactTrainShift.v)."""
    m = v.abs().amax(dim=1, keepdim=True)
    s = (bit_length(m) - 7).clamp(min=0)
    q = shift(v, -s).clamp(-127, 127).to(torch.int8)
    return q, s


def mm_cpu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """sum_k a[i][k] * b[j][k] for int8 a [M][K], b [N][K]: int64 [M][N]."""
    return a.to(torch.int64) @ b.to(torch.int64).t()


def mm_y(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The same product on Y's int8 GEMM (`y_lang.int8_gemm.int8_mm`)."""
    from ..int8_gemm import int8_mm
    return int8_mm(a.contiguous(), b.contiguous()).to(torch.int64)


def mm_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The same product on `torch._int_mm` (cuBLASLt on a GPU)."""
    return torch._int_mm(a.contiguous(), b.t()).to(torch.int64)


def _chunks(tokens: int, chunks: int, order=None):
    if tokens % chunks:
        raise ValueError(f"{tokens} tokens do not split into {chunks} chunks")
    w = tokens // chunks
    idx = list(range(chunks)) if order is None else list(order)
    if sorted(idx) != list(range(chunks)):
        raise ValueError(f"order {order} is not a permutation of {chunks} chunks")
    return [slice(i * w, (i + 1) * w) for i in idx]


def _digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


@dataclass
class LinearRun:
    """One linear layer's exact training state: master weights `W` (int64
    [N][K], Q(F)), inputs `x` (int8 [T][K], scale 2^EX) and targets `t`
    (int64 [T][N], Q(F))."""
    W: torch.Tensor
    x: torch.Tensor
    t: torch.Tensor
    last_dx: torch.Tensor = field(default=None, repr=False)

    @classmethod
    def synthetic(cls, n=64, k=256, tokens=512, seed=0, device="cpu"):
        """Targets t = x W*^T + noise, from a teacher W* with 16-bit weights
        (scale 2^-15) and noise of up to 2^-6, so no int8 student fits them
        exactly: the residual never reaches 0 and every step still moves the
        weights. (With an int8 teacher the student reached it exactly, the
        residual became 0 and every later step was a no-op, which any two
        devices agree on.) The student starts from small random weights.
        Everything is drawn on the CPU and then moved, so every device gets
        the same problem."""
        g = torch.Generator().manual_seed(seed)
        x = torch.randint(-127, 128, (tokens, k), generator=g, dtype=torch.int8)
        teacher = torch.randint(-(1 << 15), 1 << 15, (n, k), generator=g, dtype=torch.int64)
        noise = torch.randint(-(1 << (F - 6)), 1 << (F - 6), (tokens, n), generator=g, dtype=torch.int64)
        t = shift(x.to(torch.int64) @ teacher.t(), F + EX - 15) + noise   # Q(F)
        W = torch.randint(-(1 << (F - 8)), 1 << (F - 8), (n, k), generator=g, dtype=torch.int64)
        return cls(W=W.to(device), x=x.to(device), t=t.to(device))

    def to(self, device):
        return LinearRun(W=self.W.to(device), x=self.x.to(device), t=self.t.to(device))

    def step(self, mm, chunks=1, order=None) -> float:
        """One SGD step; returns the loss before it, 0.5 * sum r^2 in real
        units, computed on the CPU from the integer residual."""
        W, x, t = self.W, self.x, self.t
        wq, sw = quantize_rows(W)                         # [N][K], [N][1]
        y = shift(mm(x, wq), sw.t() + EX)                 # [T][N], Q(F)
        r = y - t                                         # the loss's gradient in y
        # Weight gradient: sum over tokens; one scale per output row n.
        gq, sg = quantize_rows(r.t().contiguous())        # [N][T], [N][1]
        xt = x.t().contiguous()                           # [K][T]
        dw = torch.zeros_like(W)
        for c in _chunks(x.shape[0], chunks, order):
            dw += mm(gq[:, c], xt[:, c])                  # int64 adds: any order
        # dW_real = dw * 2^(sg - F) * 2^EX; the step is 2^-L of it, in Q(F).
        self.W = W - shift(dw, sg + EX - LR_SHIFT)
        # Input gradient (for a layer below; T1 only records it): sum over n,
        # with one scale per token and one per input column of W.
        hq, sh = quantize_rows(r)                         # [T][N], [T][1]
        wtq, su = quantize_rows(W.t().contiguous())       # [K][N], [K][1]
        self.last_dx = shift(mm(hq, wtq), sh + su.t() - F)  # Q(F)
        rr = r.cpu().to(torch.float64) * 2.0 ** -F
        return 0.5 * float((rr * rr).sum())

    def train(self, steps, mm, chunks=1, order=None):
        return [self.step(mm, chunks, order) for _ in range(steps)]

    def digest(self) -> str:
        return _digest(self.W)


@dataclass
class FloatRun:
    """The float32 twin of a `LinearRun`: the same data and start, ordinary
    float sums. Its weights depend on the device and the chunk order."""
    W: torch.Tensor
    x: torch.Tensor
    t: torch.Tensor

    @classmethod
    def like(cls, run: LinearRun, device="cpu"):
        return cls(W=(run.W.cpu().to(torch.float64) * 2.0 ** -F).to(torch.float32).to(device),
                   x=(run.x.cpu().to(torch.float32) * 2.0 ** EX).to(device),
                   t=(run.t.cpu().to(torch.float64) * 2.0 ** -F).to(torch.float32).to(device))

    def step(self, chunks=1, order=None) -> float:
        r = self.x @ self.W.t() - self.t
        dw = torch.zeros_like(self.W)
        for c in _chunks(self.x.shape[0], chunks, order):
            dw += r[c].t() @ self.x[c]
        self.W = self.W - dw * 2.0 ** -LR_SHIFT
        return 0.5 * float((r.double() ** 2).sum())

    def train(self, steps, chunks=1, order=None):
        return [self.step(chunks, order) for _ in range(steps)]

    def digest(self) -> str:
        return _digest(self.W)
