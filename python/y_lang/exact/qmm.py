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
round-to-nearest-even, as `.to(torch.float32)` rounds. Where 2^(sa + sb) is
a normal float32 the kernel gets the same bits in float32 (the integer
rounded once, then scaled exactly; see the kernel), and only other tiles
compute in float64, which GeForce cards run at 1/64 speed.

Tests compare it with `exact_product` bit for bit
(python/tests/test_exact_qmm.py).

`qmm_sum` is two such products and their float32 sum in one kernel, for
LoRA's dx (the frozen weight's product plus the adapter's): the same bits
as the two calls and torch's add.
"""

import torch

MAX_K = 131040          # 127 * 127 * K below 2^31: a one-digit qa's int32 sums
_LIB = globals().get("_LIB")

try:
    import triton
    import triton.language as tl
except ImportError:                                        # pragma: no cover
    triton = None


def _fitting(configs, args):
    """The tiles worth timing for a shape (fewer to time on each new shape's
    first call): K-tiles of 32 for K <= 32 and of 64 or more otherwise;
    16-column tiles only for 32 or fewer columns, and no tile wider than
    max(32, columns). Any tile gives the same bits (exact integer sums, the
    same epilogue)."""
    short_k = args["K_BUCKET"] <= 32
    n = args["N_BUCKET"]
    kept = [c for c in configs if (c.kwargs["BK"] == 32) == short_k
            and (c.kwargs["BN"] > 16 or n <= 32) and c.kwargs["BN"] <= max(32, n)]
    return kept or configs


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
            # Decode's adapter products: 16 outputs over a long K (few row
            # blocks), or K = 16 into many outputs.
            triton.Config({"BM": 16, "BN": 16, "BK": 128}, num_warps=2, num_stages=4),
            triton.Config({"BM": 16, "BN": 16, "BK": 256}, num_warps=4, num_stages=3),
            triton.Config({"BM": 32, "BN": 16, "BK": 256}, num_warps=4, num_stages=3),
            triton.Config({"BM": 16, "BN": 32, "BK": 128}, num_warps=2, num_stages=4),
            triton.Config({"BM": 32, "BN": 64, "BK": 32}, num_warps=2, num_stages=2),
            triton.Config({"BM": 64, "BN": 128, "BK": 32}, num_warps=4, num_stages=2),
            triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=8, num_stages=2),
        ],
        key=["M_BUCKET", "N_BUCKET", "K_BUCKET", "TWO"],
        prune_configs_by={"early_config_prune": lambda configs, named_args, **kw: _fitting(configs, named_args)},
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
        # float32 first: c rounded once to 24 bits (round to nearest even),
        # then times 2^e, which is exact while the product is a normal
        # float32 or overflows (rounding to 24 bits commutes with a power of
        # two there; proofs/ExactQmmEpilogue.v), so the bits are
        # float32(c * 2^e). With 2^e a normal
        # float32 (-126 <= e <= 127) and c a nonzero integer, |c 2^e| >=
        # 2^-126, never subnormal; a tile with any other e takes the float64
        # product. GeForce cards run float64 at 1/64 of float32.
        ok = (e >= -126) & (e <= 127)
        two_e32 = ((tl.where(ok, e, 0) + 127) << 23).to(tl.int32).to(tl.float32, bitcast=True)
        out = c.to(tl.float32) * two_e32
        if tl.max((ok == 0).to(tl.int32)) > 0:
            two_e = ((e + 1023) << 52).to(tl.float64, bitcast=True)   # 2^e, exact
            out = tl.where(ok, out, (c.to(tl.float64) * two_e).to(tl.float32))
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


# ---------------------------------------------------------------------------
# Two products and their float32 sum in one kernel.
#
# Exact LoRA training's dx is `qmm(q8(g), W) + qmm(q8(du), q8(A^T))`: the
# frozen weight's product over the layer's outputs plus the adapter's over
# its rank. Run apart, both write a float32 [M][K_in] tensor and the add
# reads both back. Here each output tile forms both exact integer sums, takes
# each through `_qmm_kernel`'s epilogue (the same float32 bits as `qmm`), and
# adds them in float32 (FMA contraction off), so the bits are those of the
# two calls and the add: the add rounds once, as torch's float32 add does.
# ---------------------------------------------------------------------------

if triton is not None:
    @triton.jit
    def _exact_dot(a_ptr, b_ptr, rm, rn, M, N, K, s_am, s_ak, s_bn, s_bk,
                   TWO: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """int64 [BM][BN]: sum_k a[m][k] b[n][k] exactly, as `_qmm_kernel`'s
        main loop forms it."""
        rk = tl.arange(0, BK)
        lo_acc = tl.zeros((BM, BN), dtype=tl.int32)
        hi_acc = tl.zeros((BM, BN), dtype=tl.int32)
        for k0 in range(0, K, BK):
            kk = k0 + rk
            a = tl.load(a_ptr + rm[:, None] * s_am + kk[None, :] * s_ak,
                        mask=(rm[:, None] < M) & (kk[None, :] < K), other=0)
            b = tl.load(b_ptr + rn[:, None] * s_bn + kk[None, :] * s_bk,
                        mask=(rn[:, None] < N) & (kk[None, :] < K), other=0)
            bt = tl.trans(b.to(tl.int8))
            a = a.to(tl.int32)
            if TWO:
                lo = ((a + 64) & 127) - 64
                hi = (a - lo) >> 7
                lo_acc += tl.dot(lo.to(tl.int8), bt, out_dtype=tl.int32)
                hi_acc += tl.dot(hi.to(tl.int8), bt, out_dtype=tl.int32)
            else:
                lo_acc += tl.dot(a.to(tl.int8), bt, out_dtype=tl.int32)
        return hi_acc.to(tl.int64) * 128 + lo_acc.to(tl.int64)

    @triton.jit
    def _scaled(c, sa_ptr, sb_ptr, rm, rn, M, N, s_sa, s_sb):
        """float32(c * 2^(sa[m] + sb[n])): `_qmm_kernel`'s epilogue."""
        ea = tl.load(sa_ptr + rm * s_sa, mask=rm < M, other=0).to(tl.int64)
        eb = tl.load(sb_ptr + rn * s_sb, mask=rn < N, other=0).to(tl.int64)
        e = ea[:, None] + eb[None, :]
        ok = (e >= -126) & (e <= 127)
        two_e32 = ((tl.where(ok, e, 0) + 127) << 23).to(tl.int32).to(tl.float32, bitcast=True)
        out = c.to(tl.float32) * two_e32
        if tl.max((ok == 0).to(tl.int32)) > 0:
            two_e = ((e + 1023) << 52).to(tl.float64, bitcast=True)
            out = tl.where(ok, out, (c.to(tl.float64) * two_e).to(tl.float32))
        return out

    @triton.autotune(
        configs=[
            triton.Config({"BM": 64, "BN": 64, "BK": 64}, num_warps=4, num_stages=3),
            triton.Config({"BM": 128, "BN": 64, "BK": 64}, num_warps=4, num_stages=3),
            triton.Config({"BM": 64, "BN": 128, "BK": 64}, num_warps=4, num_stages=3),
            triton.Config({"BM": 128, "BN": 128, "BK": 64}, num_warps=8, num_stages=3),
            triton.Config({"BM": 32, "BN": 64, "BK": 128}, num_warps=4, num_stages=3),
            triton.Config({"BM": 64, "BN": 32, "BK": 128}, num_warps=4, num_stages=3),
            triton.Config({"BM": 32, "BN": 32, "BK": 64}, num_warps=2, num_stages=2),
            triton.Config({"BM": 64, "BN": 128, "BK": 32}, num_warps=4, num_stages=2),
            triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=8, num_stages=2),
        ],
        key=["M_BUCKET", "N_BUCKET", "K_BUCKET", "R_BUCKET", "TWO", "TWO2"],
        prune_configs_by={"early_config_prune": lambda configs, named_args, **kw: _fitting(configs, named_args)},
    )
    @triton.jit
    def _qmm_sum_kernel(a_ptr, b_ptr, sa_ptr, sb_ptr, c_ptr, d_ptr, sc_ptr, sd_ptr, o_ptr,
                        M, N, K, R,
                        s_am, s_ak, s_bn, s_bk, s_sa, s_sb,
                        s_cm, s_ck, s_dn, s_dk, s_sc, s_sd, s_om, s_on,
                        M_BUCKET, N_BUCKET, K_BUCKET, R_BUCKET,
                        TWO: tl.constexpr, TWO2: tl.constexpr, BR: tl.constexpr,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        rm = tl.program_id(0) * BM + tl.arange(0, BM)
        rn = tl.program_id(1) * BN + tl.arange(0, BN)
        c1 = _exact_dot(a_ptr, b_ptr, rm, rn, M, N, K, s_am, s_ak, s_bn, s_bk, TWO, BM, BN, BK)
        out = _scaled(c1, sa_ptr, sb_ptr, rm, rn, M, N, s_sa, s_sb)
        c2 = _exact_dot(c_ptr, d_ptr, rm, rn, M, N, R, s_cm, s_ck, s_dn, s_dk, TWO2, BM, BN, BR)
        out = out + _scaled(c2, sc_ptr, sd_ptr, rm, rn, M, N, s_sc, s_sd)
        tl.store(o_ptr + rm[:, None] * s_om + rn[None, :] * s_on, out,
                 mask=(rm[:, None] < M) & (rn[None, :] < N))


def _qmm_sum(qa, sa, qb, sb, two, qc, sc, qd, sd, two2):
    M, K = qa.shape
    N = qb.shape[0]
    R = qc.shape[1]
    if qb.shape[1] != K or tuple(qc.shape) != (M, R) or tuple(qd.shape) != (N, R):
        raise ValueError(f"qmm_sum: qa [{M}][{K}], qb {list(qb.shape)}, qc {list(qc.shape)}, "
                         f"qd {list(qd.shape)}")
    if K > MAX_K or R > MAX_K:
        raise ValueError(f"qmm_sum: K = {K}, R = {R}, past {MAX_K}")
    out = torch.empty((M, N), dtype=torch.float32, device=qa.device)
    if M == 0 or N == 0:
        return out
    kw = {} if qa.device.type == "cpu" else {"enable_fp_fusion": False}
    grid = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))  # noqa: E731
    _qmm_sum_kernel[grid](qa, qb, sa, sb, qc, qd, sc, sd, out, M, N, K, R,
                          qa.stride(0), qa.stride(1), qb.stride(0), qb.stride(1),
                          sa.stride(0), sb.stride(0),
                          qc.stride(0), qc.stride(1), qd.stride(0), qd.stride(1),
                          sc.stride(0), sd.stride(0), out.stride(0), out.stride(1),
                          _bucket(M), _bucket(N), _bucket(K), _bucket(R),
                          # The rank's K block: Triton's int8 dot takes K of 32 or more; the
                          # masked loads pad it with zeros, which add nothing to the sums.
                          TWO=bool(two), TWO2=bool(two2), BR=max(32, triton.next_power_of_2(R)), **kw)
    return out


def _register_sum():
    if triton is None or not torch.cuda.is_available():
        return None
    if not hasattr(torch.ops.y_lang, "exact_qmm_sum"):
        lib = torch.library.Library("y_lang", "FRAGMENT")
        lib.define("exact_qmm_sum(Tensor qa, Tensor sa, Tensor qb, Tensor sb, bool two, "
                   "Tensor qc, Tensor sc, Tensor qd, Tensor sd, bool two2) -> Tensor")
        lib.impl("exact_qmm_sum", _qmm_sum, "CUDA")
        torch.library.register_fake(
            "y_lang::exact_qmm_sum",
            lambda qa, sa, qb, sb, two, qc, sc, qd, sd, two2:
                qa.new_empty((qa.shape[0], qb.shape[0]), dtype=torch.float32),
            lib=lib)
        globals()["_LIB_SUM"] = lib
    return torch.ops.y_lang.exact_qmm_sum


_LIB_SUM = globals().get("_LIB_SUM")
_OP_SUM = _register_sum()


def qmm_sum(qa, sa, qb, sb, bits_a, bits_b, qc, sc, qd, sd, bits_c, bits_d):
    """`qmm(qa, sa, qb, sb, bits_a, bits_b) + qmm(qc, sc, qd, sd, bits_c,
    bits_d)` in float32, in one kernel: the same bits. qa [M][K], qb [N][K],
    qc [M][R], qd [N][R]."""
    for x, y in ((bits_a, bits_b), (bits_c, bits_d)):
        if x > 12 or y > 8:
            raise ValueError(f"qmm_sum: takes up to 12-bit a and 8-bit b, not {x} and {y}")
    return _OP_SUM(qa, sa.reshape(-1), qb, sb.reshape(-1), bits_a > 8,
                   qc, sc.reshape(-1), qd, sd.reshape(-1), bits_c > 8)
