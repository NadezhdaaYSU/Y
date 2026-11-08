"""`ExactRMSNorm`'s float64 divide without dividing, on CUDA (Triton).

`ExactRMSNorm` computes, per row of float32 `x`, `r = sqrt(sum(x^2)/H + eps)`
in float64 and then `float32(x / r) * weight`: a correctly rounded float64
divide per element, rounded to float32. A GeForce GPU runs float64 at 1/64 of
its float32 rate, and that divide was half of the norm's kernel at the 7B
decode shape (`docs/benchmarks/exact_persistent_7b_2026-11-08/`).

This kernel divides once per row (`rinv = 1/r`) and multiplies per element,
`q = x * rinv`, and gets the same float32 as `float32(x / r)` everywhere, by
this argument (u = 2^-53, every rounding to nearest):

* `rinv = (1/r)(1 + d1)` and `q = x * rinv * (1 + d2)` with `|d1|, |d2| <= u`,
  so `|q - x/r| <= (2u + u^2)|x/r|`; and `RN64(x/r)` is within `u|x/r|` of
  `x/r`. Hence `|q - RN64(x/r)| <= (3u + u^2)|x/r| < 3.0001 ulp64(q)`, since
  `ulp64(q) > u|q|` and `|x/r| <= |q| / (1 - 2u - u^2)`.
* Rounding to float32 is constant between consecutive float32 midpoints, and
  each midpoint is a float64. If no midpoint lies within 3 ulp64 of `q`, none
  lies between `q` and `RN64(x/r)` (or on either), so both round to the same
  float32.
* For `|q| >= 2^-126` (float32 normal) the midpoints of `q`'s binade are the
  float64s whose low 29 fraction bits are `2^28`; the nearest midpoints in
  the binades either side are at least 2^27 ulp64 away. So "no midpoint
  within 3 ulp64" is `|(bits(q) & (2^29 - 1)) - 2^28| > 3`.
* `q == 0` exactly when `x == 0`, and then both are `x`'s signed zero.

Anything else (a midpoint within 3 ulp64, `|q| < 2^-126`, not finite) takes
the true divide: when any element of a row needs it, the row is divided
outright. On random activations that never happens; on inputs built to sit on
midpoints the cheap path alone was wrong on 13% of them and the guarded one on
none (`python/y_lang/tests/test_exact_norm_kernel.py`).

The float64 sum of squares is a tree sum here and a different order on the
CPU, exactly as with the Inductor kernel this replaces; each square of a
float32 is exact in float64, so only the additions round.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _quotient_to_f32(xd, r, rinv):
    """float32(xd / r) for float64 `xd`, `r`, with `rinv = RN64(1/r)`: the
    multiply where it provably agrees with the divide, the divide where any
    element of the block is not proved (see the module docstring)."""
    q = xd * rinv
    bits = q.to(tl.int64, bitcast=True)
    dist = tl.abs((bits & 536870911) - 268435456)          # 2^29 - 1, 2^28
    aq = tl.abs(q)
    unsafe = (q != 0.0) & ((aq < 1.1754943508222875e-38)   # 2^-126
                           | (dist <= 3) | (aq > 1.7976931348623157e308) | (q != q))
    if tl.max(unsafe.to(tl.int32), axis=0) > 0:
        q = tl.where(unsafe, xd / r, q)   # float64 `/` is div.rn.f64
    return q.to(tl.float32)


@triton.jit
def _rms_norm_kernel(x_ptr, w_ptr, out_ptr, eps_ptr, H, stride_x, stride_o,
                     BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < H
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0)
    xd = x.to(tl.float64)
    eps = tl.load(eps_ptr)                       # float64: a Python float arg is float32
    var = tl.sum(xd * xd, axis=0) / H.to(tl.float64)
    r = tl.sqrt(var + eps)                       # float64: sqrt.rn.f64
    rinv = 1.0 / r
    y = _quotient_to_f32(xd, r, rinv)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + row * stride_o + offs, y * w, mask=mask)


@triton.jit
def _quotient_kernel(x_ptr, r_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """`_quotient_to_f32` on arrays, for testing it on chosen (x, r) pairs."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    xd = tl.load(x_ptr + offs, mask=mask, other=1.0).to(tl.float64)
    r = tl.load(r_ptr + offs, mask=mask, other=1.0)
    rinv = 1.0 / r
    tl.store(out_ptr + offs, _quotient_to_f32(xd, r, rinv), mask=mask)


def rms_norm(x, weight, eps):
    """`ExactRMSNorm`'s value for float32 CUDA `x` (`[..., H]`, last dim
    contiguous) and float32 `weight`: `float32(x / sqrt(mean(x^2) + eps)) *
    weight`, the mean and root in float64."""
    H = x.shape[-1]
    x2 = x.reshape(-1, H)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    out = torch.empty((x2.shape[0], H), dtype=torch.float32, device=x.device)
    if x2.shape[0]:
        BLOCK = triton.next_power_of_2(H)
        _rms_norm_kernel[(x2.shape[0],)](
            x2, weight, out, _eps_tensor(eps, x.device), H, x2.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=8 if BLOCK >= 2048 else 4, enable_fp_fusion=False)
    return out.view(x.shape)


_EPS = {}


def _eps_tensor(eps, device):
    key = (float(eps), device)
    t = _EPS.get(key)
    if t is None:
        t = _EPS[key] = torch.tensor([float(eps)], dtype=torch.float64, device=device)
    return t


def quotient_to_f32(x, r):
    """`_quotient_to_f32` elementwise: float32 `x`, float64 `r`, both CUDA."""
    out = torch.empty_like(x, dtype=torch.float32)
    n = x.numel()
    _quotient_kernel[(triton.cdiv(n, 1024),)](x, r, out, n, BLOCK=1024,
                                              enable_fp_fusion=False)
    return out
