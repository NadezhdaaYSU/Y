"""Exact batched int8 products on tensor cores, for exact training's attention.

    from y_lang.exact.int8_bmm import int8_bmm, available
    c = int8_bmm(a, b)    # a [..., M, K], b [..., N, K] int8 -> c [..., M, N] int32 = a @ b^T

Each product of two int8 values is exact, and an int32 sum of K of them is
exact while K * 128 * 128 <= 2^31 - 1 (refused before anything runs
otherwise), so `c` is the integer sums whatever order the tensor cores add
them in: the same integers as the float32 product `train_lm._exact_fp32_bmm`
computes without tensor cores, and as a CPU (python/tests/test_exact_int8_bmm.py).

A Triton kernel (`tl.dot` with an int32 accumulator), registered as the op
`torch.ops.y_lang.int8_bmm` so that torch.compile keeps it in its graph and a
CUDA graph can capture it. sm_80 and later: Triton's int8 dot does not build
for sm_75 (`docs/known_issues.md`, `w8a8_gemv`), where `available` is False
and the caller keeps the float32 product.
"""

import torch
import triton
import triton.language as tl

MAX_K = (2 ** 31 - 1) // (128 * 128)          # 131,071


@triton.jit
def _int8_bmm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                     sa_g, sa_m, sb_g, sb_n, sc_g, sc_m,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    g = tl.program_id(2).to(tl.int64)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        a = tl.load(a_ptr + g * sa_g + rm[:, None] * sa_m + kk[None, :],
                    mask=(rm[:, None] < M) & (kk[None, :] < K), other=0)
        b = tl.load(b_ptr + g * sb_g + rn[:, None] * sb_n + kk[None, :],
                    mask=(rn[:, None] < N) & (kk[None, :] < K), other=0)
        acc += tl.dot(a, tl.trans(b), out_dtype=tl.int32)
    tl.store(c_ptr + g * sc_g + rm[:, None] * sc_m + rn[None, :], acc,
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def check_operands(a, b):
    if a.dtype != torch.int8 or b.dtype != torch.int8 or a.dim() != 3 or b.dim() != 3:
        raise ValueError(f"int8_bmm: a and b must be 3-D int8, not {a.dim()}-D {a.dtype} and "
                         f"{b.dim()}-D {b.dtype}")
    if a.shape[0] != b.shape[0] or a.shape[2] != b.shape[2]:
        raise ValueError(f"int8_bmm: a is {tuple(a.shape)}, so b must be [{a.shape[0]}][N][{a.shape[2]}], "
                         f"not {tuple(b.shape)}")
    if a.shape[2] > MAX_K:
        raise OverflowError(f"int8_bmm: K = {a.shape[2]}; past {MAX_K} an int8 dot product can leave int32")
    if a.stride(2) != 1 or b.stride(2) != 1:
        raise ValueError("int8_bmm: a and b must be contiguous along K")


def launch(a, b, c, BM=64, BN=64, BK=64):
    """c[g] = a[g] @ b[g]^T for 3-D int8 a, b and int32 c, on any device
    Triton runs on (the CPU too, under TRITON_INTERPRET=1)."""
    check_operands(a, b)
    G, M, K = a.shape
    N = b.shape[1]
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN), G)
    _int8_bmm_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                           c.stride(0), c.stride(1), BM=BM, BN=BN, BK=BK)
    return c


def _int8_bmm(a, b):
    check_operands(a, b)
    if not a.is_cuda or a.device != b.device:
        raise ValueError("int8_bmm: a and b must be on one CUDA device")
    c = torch.empty((a.shape[0], a.shape[1], b.shape[1]), dtype=torch.int32, device=a.device)
    with torch.cuda.device(a.device):
        return launch(a, b, c)


def _fake_int8_bmm(a, b):
    return a.new_empty((a.shape[0], a.shape[1], b.shape[1]), dtype=torch.int32)


_LIB = globals().get("_LIB")


def _register_op():
    global _LIB
    if not hasattr(torch.ops.y_lang, "int8_bmm"):
        lib = torch.library.Library("y_lang", "FRAGMENT")
        lib.define("int8_bmm(Tensor a, Tensor b) -> Tensor")
        lib.impl("int8_bmm", _int8_bmm, "CUDA")
        torch.library.register_fake("y_lang::int8_bmm", _fake_int8_bmm, lib=lib)
        _LIB = lib
    return torch.ops.y_lang.int8_bmm


_OP = _register_op()


def available(device) -> bool:
    """Whether the kernel runs on `device`: CUDA, sm_80 or later."""
    device = torch.device(device)
    return device.type == "cuda" and torch.cuda.get_device_capability(device) >= (8, 0)


def int8_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a [..., M, K] @ b[..., N, K]^T for int8 a, b with the same leading
    dimensions: the exact int32 sums, [..., M, N]."""
    lead = a.shape[:-2]
    M, K = a.shape[-2:]
    N = b.shape[-2]
    c = _OP(a.reshape(-1, M, K).contiguous(), b.reshape(-1, N, K).contiguous())
    return c.reshape(*lead, M, N)
