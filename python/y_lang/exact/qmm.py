"""An exact product of quantized operands in one Triton kernel.

`qmm(qa, sa, qb, sb)` is `train_lm.exact_product`'s result for 2-D operands
already quantized (`q8`): float32(float64(sum_k qa[m][k] qb[n][k]) *
2^(sa[m] + sb[n])), with |qa| <= 2047 (up to 12 bits) and |qb| <= 127.

`exact_product` splits qa into int8 digits, stacks them, pads, runs an
int8 GEMM and joins the blocks with shifts: four or five kernels. Here one
kernel does it. Each tile of qa splits in registers into lo in [-64, 63]
and hi = (qa - lo) / 128 (exact: qa - lo is a multiple of 128; |hi| <= 16),
two int8 dots accumulate in int32 (|lo| 64 and |hi| 16 times 127 times K
stay below 2^31 for every K this takes), and the sum is joined in int64,
hi * 128 + lo: the exact integer, whatever split computed it. That integer
is below 2^53 (`Ops.matmul` refuses wider products), so its float64 is exact;
2^(sa + sb) is built from its bits (|sa|, |sb| <= 511, `q8` refuses others),
so the product is exact; the one rounding is float64 to float32,
round-to-nearest-even, as `.to(torch.float32)` rounds.

Tests compare it with `exact_product` bit for bit
(python/tests/test_exact_qmm.py).
"""

import torch

MAX_K = 131040          # 127 * 127 * K below 2^31: a one-digit qa's int32 sums
_LIB = globals().get("_LIB")

try:
    import triton
    import triton.language as tl
except ImportError:                                        # pragma: no cover
    triton = None


if triton is not None:
    @triton.autotune(
        configs=[
            triton.Config({"BM": 64, "BN": 64, "BK": 64}, num_warps=4, num_stages=3),
            triton.Config({"BM": 128, "BN": 64, "BK": 64}, num_warps=4, num_stages=3),
            triton.Config({"BM": 64, "BN": 128, "BK": 64}, num_warps=4, num_stages=3),
            triton.Config({"BM": 128, "BN": 128, "BK": 64}, num_warps=8, num_stages=3),
            triton.Config({"BM": 32, "BN": 64, "BK": 128}, num_warps=4, num_stages=3),
            triton.Config({"BM": 64, "BN": 32, "BK": 128}, num_warps=4, num_stages=3),
            triton.Config({"BM": 32, "BN": 32, "BK": 64}, num_warps=2, num_stages=2),
        ],
        key=["M_BUCKET", "N_BUCKET", "K_BUCKET", "TWO"],
    )
    @triton.jit
    def _qmm_kernel(a_ptr, b_ptr, sa_ptr, sb_ptr, o_ptr, M, N, K,
                    s_am, s_ak, s_bn, s_bk, s_sa, s_sb, s_om, s_on,
                    M_BUCKET, N_BUCKET, K_BUCKET,
                    TWO: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        lo_acc = tl.zeros((BM, BN), dtype=tl.int32)
        hi_acc = tl.zeros((BM, BN), dtype=tl.int32)
        for k0 in range(0, K, BK):
            kk = k0 + rk
            a = tl.load(a_ptr + rm[:, None] * s_am + kk[None, :] * s_ak,
                        mask=(rm[:, None] < M) & (kk[None, :] < K), other=0)
            b = tl.load(b_ptr + rn[:, None] * s_bn + kk[None, :] * s_bk,
                        mask=(rn[:, None] < N) & (kk[None, :] < K), other=0)
            bt = tl.trans(b.to(tl.int8))                       # [BK][BN]
            a = a.to(tl.int32)
            if TWO:
                lo = ((a + 64) & 127) - 64                     # [-64, 63]
                hi = (a - lo) >> 7                             # exact: a - lo = 128 hi
                lo_acc += tl.dot(lo.to(tl.int8), bt, out_dtype=tl.int32)
                hi_acc += tl.dot(hi.to(tl.int8), bt, out_dtype=tl.int32)
            else:
                lo_acc += tl.dot(a.to(tl.int8), bt, out_dtype=tl.int32)
        c = hi_acc.to(tl.int64) * 128 + lo_acc.to(tl.int64)
        ea = tl.load(sa_ptr + rm * s_sa, mask=rm < M, other=0).to(tl.int64)
        eb = tl.load(sb_ptr + rn * s_sb, mask=rn < N, other=0).to(tl.int64)
        e = ea[:, None] + eb[None, :]
        two_e = ((e + 1023) << 52).to(tl.float64, bitcast=True)   # 2^e, exact
        out = (c.to(tl.float64) * two_e).to(tl.float32)
        tl.store(o_ptr + rm[:, None] * s_om + rn[None, :] * s_on, out,
                 mask=(rm[:, None] < M) & (rn[None, :] < N))


def _bucket(n):
    return 1 << max(4, (int(n) - 1).bit_length())


def _qmm(qa, sa, qb, sb, two):
    M, K = qa.shape
    N = qb.shape[0]
    if qb.shape[1] != K:
        raise ValueError(f"qmm: qa is [{M}][{K}], qb [{N}][{qb.shape[1]}]")
    if K > MAX_K:
        raise ValueError(f"qmm: K = {K} > {MAX_K}")
    out = torch.empty((M, N), dtype=torch.float32, device=qa.device)
    if M == 0 or N == 0:
        return out
    grid = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))  # noqa: E731
    _qmm_kernel[grid](qa, qb, sa, sb, out, M, N, K,
                      qa.stride(0), qa.stride(1), qb.stride(0), qb.stride(1),
                      sa.stride(0), sb.stride(0), out.stride(0), out.stride(1),
                      _bucket(M), _bucket(N), _bucket(K), TWO=bool(two))
    return out


def _register():
    global _LIB
    if triton is None or not torch.cuda.is_available():
        return None
    if not hasattr(torch.ops.y_lang, "exact_qmm"):
        lib = torch.library.Library("y_lang", "FRAGMENT")
        lib.define("exact_qmm(Tensor qa, Tensor sa, Tensor qb, Tensor sb, bool two) -> Tensor")
        lib.impl("exact_qmm", _qmm, "CUDA")
        torch.library.register_fake(
            "y_lang::exact_qmm",
            lambda qa, sa, qb, sb, two: qa.new_empty((qa.shape[0], qb.shape[0]), dtype=torch.float32),
            lib=lib)
        _LIB = lib
    return torch.ops.y_lang.exact_qmm


_OP = _register()


def available(device) -> bool:
    """Whether the kernel runs on `device`: CUDA, sm_80 or later. Triton's
    int8 `tl.dot` does not build for sm_75 (a Tesla T4; `model.triton_int8_dot_ok`),
    where `lora._exact_mm` keeps `exact_product` on Y's GEMM: the same bits."""
    device = torch.device(device)
    return (_OP is not None and device.type == "cuda"
            and torch.cuda.get_device_capability(device) >= (8, 0))


def qmm(qa, sa, qb, sb, bits_a, bits_b):
    """float32 [M][N]: qa [M][K] (|qa| < 2^(bits_a - 1), bits_a <= 12),
    qb [N][K] (bits_b <= 8), sa [M] or [M][1], sb [N] or [N][1] integer
    exponents."""
    if bits_a > 12 or bits_b > 8:
        raise ValueError(f"qmm: takes up to 12-bit a and 8-bit b, not {bits_a} and {bits_b}")
    return _OP(qa, sa.reshape(-1), qb, sb.reshape(-1), bits_a > 8)
