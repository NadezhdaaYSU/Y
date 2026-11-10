"""`ExactRMSNorm`'s float64 divide without dividing, on CUDA (Triton).

`ExactRMSNorm` computes, per row of float32 `x`, `r = sqrt(sum(x^2)/H + eps)`
in float64, the sum `model.sum_of_squares`'s, and then
`float32(x / r) * weight`: a correctly rounded float64
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

Anything else (a midpoint within 3 ulp64, `|q| < 2^-126`, `|q| >= 2^127`, not
finite) takes the true divide: when any element of a row needs it, the row is
divided outright. On random activations that never happens; on inputs built to
sit on midpoints the cheap path alone was wrong on 13% of them and the guarded
one on none (`python/y_lang/tests/test_exact_norm_kernel.py`).

The rest of the float64 work is done on bits, as integers, where the inputs
allow it (float64 runs at 1/64 rate here, integers at full rate):

* `x` as float64, when every `x` of the row is a normal float32 or zero:
  `x`'s exponent is rebiased by 896 and its fraction shifted up 29 bits. A row
  with a subnormal, infinite or NaN `x` converts as before.
* The sum of squares (`_grid_sum_of_squares`): the 48-bit integer squares
  `m * m` of the 24-bit significands, put on the row's fixed grid and added
  as two 40-bit-limb integer sums, as `model.sum_of_squares` defines it. A
  row with an infinite or NaN `x` sums float64 squares instead (inf or NaN
  in any order).
* `q` to float32, on the guarded path: `q` is then 0 or in `[2^-126, 2^127)`,
  where float32's rounding is `q`'s bits plus 2^28, shifted down 29, exponent
  rebiased; no tie can reach it, since the guard sent every `q` within 3
  ulp64 of a midpoint to the divide.

The sum of squares is exact integer arithmetic, so the tree order of the
reductions here, Inductor's and the CPU's give the same bits
(`python/y_lang/tests/test_exact_norm_kernel.py`).
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
    # The other tests on |q|'s bits, as integers: a GeForce part compares
    # float64s at 1/64 rate, and four float64 compares per element cost more
    # than the norm's own float64 arithmetic. Non-negative float64s order as
    # their bit patterns, so 0 < abits < bits(2^-126) is 0 < |q| < 2^-126,
    # one unsigned compare with abits - 1 wrapping to 2^64 - 1 at q == 0
    # (written as `abits != 0 & ...`, LLVM turns the zero test back into a
    # float64 compare). q == 0 is never unsafe: its dist is 2^28.
    # abits >= bits(2^127) covers infinity and NaN, and keeps float32
    # overflow off the integer rounding below.
    abits = bits & 0x7FFFFFFFFFFFFFFF
    tiny = (abits - 1).to(tl.uint64, bitcast=True) < 0x380FFFFFFFFFFFFF  # bits(2^-126) - 1
    unsafe = tiny | (dist <= 3) | (abits >= 0x47E0000000000000)          # bits(2^127)
    if tl.max(unsafe.to(tl.int32), axis=0) > 0:
        y = tl.where(unsafe, xd / r, q).to(tl.float32)   # float64 `/` is div.rn.f64
    else:
        # q is 0 or in [2^-126, 2^127) and no tie: float32(q) from its bits.
        f = tl.maximum(((abits + 268435456) >> 29) - 7516192768, 0)  # 2^28; 896 << 23
        y = (f | ((bits >> 32) & 2147483648)).to(tl.int32).to(tl.float32, bitcast=True)
    return y


@triton.jit
def _max2(a0, a1, b0, b1):
    return tl.maximum(a0, b0), tl.maximum(a1, b1)


@triton.jit
def _add2(a0, a1, b0, b1):
    return a0 + b0, a1 + b1


@triton.jit
def _grid_sum_of_squares(frac, ex, p, e):
    """`model.sum_of_squares` for one row, from its float32 values' fraction
    bits `frac` and biased exponents `ex` (no infinite or NaN value), with
    `p = 2 * max(ex, 1)` (x^2 = m^2 * 2^(p - 300)) and `e` the row's largest
    p: the squares on the row's fixed grid, summed as integers (any order
    gives the same sum), then one rounding to float64. See model.SUMSQ_F."""
    m = tl.where(ex == 0, frac, frac | 8388608).to(tl.int64)
    m2 = m * m
    s = (e - p).to(tl.int64)                        # >= 0, even
    # T = floor(m^2 * 2^(32 - s)) = hi * 2^40 + lo.
    hi = m2 >> tl.minimum(s + 8, 63)
    sa = tl.minimum(s, 32)
    lo_near = (m2 & (1099511627775 >> (32 - sa))) << (32 - sa)        # 2^40 - 1
    lo_far = (m2 >> tl.minimum(tl.maximum(s - 32, 0), 63)) & 1099511627775
    lo = tl.where(s <= 32, lo_near, lo_far)
    big, small = tl.reduce((hi, lo), 0, _add2)      # one pass for both limbs
    big = big + (small >> 40)                       # < 2^53 for H <= 8192
    small = small & 1099511627775
    total = big.to(tl.float64) * 1099511627776.0 + small.to(tl.float64)  # one rounding
    scale = ((e.to(tl.int64) + 691) << 52).to(tl.float64, bitcast=True)  # 2^(e - 332)
    return total * scale


# `H` unspecialised: Triton makes an integer argument equal to 1 a constant,
# which has no `.to`.
@triton.jit(do_not_specialize=["H"])
def _rms_norm_kernel(x_ptr, w_ptr, out_ptr, eps_ptr, H, stride_x, stride_o, sum_ptr,
                     BLOCK: tl.constexpr, WRITE_SUM: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < H
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0)
    xb = x.to(tl.int32, bitcast=True)
    ex = (xb >> 23) & 255
    frac = xb & 8388607
    # 2: infinite or NaN, 1: subnormal, 0: normal or zero. One reduction
    # gives the row's worst kind and its largest p (a zero's p is 2, the
    # smallest, so zeros never raise it).
    kind = tl.where(ex == 255, 2, tl.where((ex == 0) & (frac != 0), 1, 0))
    p = 2 * tl.maximum(ex, 1)
    e, worst = tl.reduce((p, kind), 0, _max2)
    if worst > 0:
        xd = x.to(tl.float64)
    else:
        # Every x normal or zero (see the module docstring).
        nz = ex != 0
        xbits = ((xb & 2147483647).to(tl.int64) << 29) + tl.where(nz, 4035225266123964416, 0)
        xd = (xbits | ((xb.to(tl.int64) >> 32) << 63)).to(tl.float64, bitcast=True)  # 896 << 52
    if worst == 2:
        sumsq = tl.sum(xd * xd, axis=0)             # inf or NaN, in any order
    else:
        sumsq = _grid_sum_of_squares(frac, ex, p, e)
    if WRITE_SUM:                                # tests read the sum this kernel uses
        tl.store(sum_ptr + row, sumsq)
    eps = tl.load(eps_ptr)                       # float64: a Python float arg is float32
    var = sumsq / H.to(tl.float64)
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


def rms_norm(x, weight, eps, sums=None):
    """`ExactRMSNorm`'s value for float32 CUDA `x` (`[..., H]`, last dim
    contiguous) and float32 `weight`: `float32(x / sqrt(mean(x^2) + eps)) *
    weight`, the mean and root in float64. With `sums` (float64, one per row),
    the kernel also writes there the sum of squares it used (for tests)."""
    H = x.shape[-1]
    if H > 8192:
        raise ValueError(f"exact_rms_norm: rows of {H} > 8192 values would overflow the sum "
                         f"of squares' 40-bit limbs (model.SUMSQ_MAX_H)")
    x2 = x.reshape(-1, H)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    out = torch.empty((x2.shape[0], H), dtype=torch.float32, device=x.device)
    if x2.shape[0]:
        BLOCK = triton.next_power_of_2(H)
        # Every warp repeats the row's float64 sqrt and divide, so warps past
        # a few cost more than they hide: at 32 x 3584 (7B decode) 16 warps
        # took 8.6 us, 8 took 6.5, 4 took 6.0 and 2 took 9.8.
        _rms_norm_kernel[(x2.shape[0],)](
            x2, weight, out, _eps_tensor(eps, x.device), H, x2.stride(0), out.stride(0),
            sums if sums is not None else out, BLOCK=BLOCK,
            WRITE_SUM=sums is not None, num_warps=4 if BLOCK <= 4096 else 8,
            enable_fp_fusion=False)
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



def sum_of_squares(x):
    """The sum of squares `rms_norm`'s kernel computes for float32 CUDA `x`
    [rows, H] (last dim contiguous), read back from that kernel: float64
    [rows, 1], for tests."""
    sums = torch.empty((x.shape[0], 1), dtype=torch.float64, device=x.device)
    rms_norm(x, torch.ones(x.shape[-1], device=x.device), 1.0, sums=sums)
    return sums
