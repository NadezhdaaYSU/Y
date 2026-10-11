"""
Quantized and LLM helpers.

None of these runs a kernel the Y compiler emitted. `fp8_gemm` runs
hand-written CUDA C++ (`tests/y_tensor_core_gemm.cu`, compiled by CuPy); the
rest are PyTorch calls. Y's own FP8 GEMM is the `@tile(M, N, K)` kernel
(`tests/gemm_fp8_*.ysu`).
"""


import math
from typing import Optional
import torch

_FP8_OUT_CACHE = {}

import cupy as cp

_FP8_KERNEL = None
_FP16_KERNEL = None

def _get_kernels():
    global _FP8_KERNEL, _FP16_KERNEL
    if _FP8_KERNEL is None:
        import os
        cuda_src_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "tests", "y_tensor_core_gemm.cu")
        if os.path.exists(cuda_src_path):
            with open(cuda_src_path, "r") as f:
                cuda_src = f.read()
            mod = cp.RawModule(code=cuda_src, options=("-std=c++17", "--use_fast_math"))
            fn_fp8 = mod.get_function("y_fp8_tensor_core_gemm_kernel")
            fn_fp8.max_dynamic_shared_size_bytes = 65536
            _FP8_KERNEL = fn_fp8

            fn_fp16 = mod.get_function("y_tensor_core_gemm_kernel")
            fn_fp16.max_dynamic_shared_size_bytes = 65536
            _FP16_KERNEL = fn_fp16
    return _FP8_KERNEL, _FP16_KERNEL

def fp8_gemm(a: torch.Tensor, b: torch.Tensor, scale_a: float = 1.0, scale_b: float = 1.0, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Multiply with hand-written CUDA C++, not a Y-compiled kernel.

    1-byte CUDA operands are meant to run `tests/y_tensor_core_gemm.cu`'s FP8
    kernel and 2-byte ones its FP16 kernel, compiled by CuPy. Other CUDA
    operands are converted to FP16 for `torch.mm`; CPU operands go to a
    float32 `torch.matmul`.

    When the CuPy path cannot run, this raises. It used to fall back to
    `torch.mm` without saying so, and with CuPy 14 it always did: NVRTC cannot
    find `crt/mma.h` through the pip CUDA headers, and `Stream.from_external`
    no longer takes a raw pointer. Every "FP8 GEMM" this returned on CUDA was
    cuBLAS on FP16 copies of the operands.
    """
    M, K = a.shape[0], a.shape[1]
    N = b.shape[1]

    if a.is_cuda and b.is_cuda and a.element_size() in (1, 2):
        try:
            return _cupy_gemm(a, b, scale_a, scale_b, out, M, N, K)
        except Exception as e:
            raise RuntimeError(
                "fp8_gemm's CUDA C++ kernel (tests/y_tensor_core_gemm.cu, through CuPy) did not run: "
                f"{type(e).__name__}: {e}. Nothing is substituted for it. For a reference product use "
                "torch.mm; the Y compiler's FP8 GEMM is the @tile(M, N, K) kernel "
                "(tests/gemm_fp8_*.ysu).") from e

    if a.is_cuda and b.is_cuda:
        scale = scale_a * scale_b
        if out is None:
            res = torch.mm(a.to(torch.float16), b.to(torch.float16))
            if scale != 1.0:
                res.mul_(scale)
            return res
        else:
            torch.matmul(a.to(torch.float16), b.to(torch.float16), out=out)
            if scale != 1.0:
                out.mul_(scale)
            return out

    scale = scale_a * scale_b
    a_float = a.to(torch.float32)
    b_float = b.to(torch.float32)
    res = torch.matmul(a_float, b_float)
    if scale != 1.0:
        res.mul_(scale)
    if out is not None:
        out.copy_(res.to(a.dtype))
        return out
    return res.to(a.dtype) if a.dtype != torch.float32 else res


def _cupy_gemm(a, b, scale_a, scale_b, out, M, N, K):
    """The CuPy launch `fp8_gemm` makes for 1- and 2-byte CUDA operands."""
    fn_fp8, fn_fp16 = _get_kernels()
    if fn_fp8 is None or fn_fp16 is None:
        raise FileNotFoundError("tests/y_tensor_core_gemm.cu is not beside this package")
    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = ((N + 127) // 128, (M + 127) // 128, 1)
    blk = (256, 1, 1)
    stream = cp.cuda.Stream.from_external(torch.cuda.current_stream().cuda_stream)
    if a.element_size() == 1:
        args = (a.data_ptr(), b.data_ptr(), out.data_ptr(), float(scale_a), float(scale_b), int(M), int(N), int(K))
        fn_fp8(grid, blk, args, shared_mem=65536, stream=stream)
    else:
        args = (a.data_ptr(), b.data_ptr(), out.data_ptr(), int(M), int(N), int(K))
        fn_fp16(grid, blk, args, shared_mem=65536, stream=stream)
        if (scale_a * scale_b) != 1.0:
            out.mul_(scale_a * scale_b)
    return out


def swiglu_fast(x: torch.Tensor, gate: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """`silu(x) * gate` in PyTorch (two calls, not a fused kernel)."""
    # silu(x) = x * sigmoid(x)
    silu_x = torch.sigmoid(x).mul_(x)
    if out is not None:
        out.copy_(silu_x.mul_(gate))
        return out
    return silu_x.mul_(gate)




def fp8_gemm_scaled(a: torch.Tensor, b: torch.Tensor, scale_ab: float = 1.0, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """`torch.matmul(a, b) * scale_ab` in PyTorch."""
    if out is not None:
        torch.matmul(a, b, out=out)
        out.mul_(scale_ab)
        return out
    res = torch.matmul(a, b)
    if scale_ab != 1.0:
        res.mul_(scale_ab)
    return res



def int4_weight_only_gemm(x: torch.Tensor, w_q4: torch.Tensor, scales: torch.Tensor, zeros: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Dequantise 4-bit weights and multiply, in PyTorch.

    The weights are NOT packed: one 4-bit value per byte, `[K, N]`. A packed
    `[K // 2, N]` matrix fails the multiply's shape check.

    Args:
        x: Input activations tensor [M, K] in FP16/BF16.
        w_q4: Unpacked 4-bit weight values [K, N] (uint8/int8), one per byte.
        scales: Per-channel or per-group scaling factors [GroupCount, N].
        zeros: Optional zero-point offsets.
    Returns:
        Output tensor [M, N] in activation precision.
    """
    M, K = x.shape
    N = w_q4.shape[1] if w_q4.ndim == 2 else w_q4.shape[-1]

    # Dequantize weights: (w_q4 - zero) * scale
    if w_q4.dtype in (torch.uint8, torch.int8):
        w_dequant = w_q4.to(x.dtype)
        if zeros is not None:
            w_dequant = w_dequant - zeros.to(x.dtype)
        if scales.ndim == 2 and scales.shape[0] < K:
            # Grouped scaling
            group_size = K // scales.shape[0]
            scales_expanded = scales.repeat_interleave(group_size, dim=0)[:K, :]
            w_dequant = w_dequant * scales_expanded
        else:
            w_dequant = w_dequant * scales
    else:
        w_dequant = w_q4.to(x.dtype) * scales

    return torch.matmul(x, w_dequant)

def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: Optional[float] = None
) -> torch.Tensor:
    """Paged-attention decoding in PyTorch: a Python loop over sequences and blocks.

    Args:
        query: Query tensor [num_seqs, num_heads, head_dim].
        key_cache: Paged key cache buffer [num_blocks, num_heads, head_dim, block_size].
        value_cache: Paged value cache buffer [num_blocks, num_heads, head_dim, block_size].
        block_tables: Physical block table mapping [num_seqs, max_blocks_per_seq].
        seq_lens: Current context sequence lengths [num_seqs].
        scale: Softmax scaling factor (defaults to 1 / sqrt(head_dim)).
    Returns:
        Attention output tensor [num_seqs, num_heads, head_dim].
    """
    num_seqs, num_heads, head_dim = query.shape
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    outputs = []
    for i in range(num_seqs):
        q_i = query[i] * scale # [num_heads, head_dim]
        seq_len = int(seq_lens[i].item())

        # Collect keys and values for sequence i across physical blocks
        k_blocks = []
        v_blocks = []
        blocks = block_tables[i]

        for b in blocks:
            if b.item() < 0:
                break
            b_idx = int(b.item())
            # [num_heads, head_dim, block_size] -> transpose to [num_heads, block_size, head_dim]
            k_b = key_cache[b_idx].permute(0, 2, 1)
            v_b = value_cache[b_idx].permute(0, 2, 1)
            k_blocks.append(k_b)
            v_blocks.append(v_b)

        if not k_blocks:
            outputs.append(torch.zeros_like(q_i))
            continue

        keys = torch.cat(k_blocks, dim=1)[:, :seq_len, :]  # [num_heads, seq_len, head_dim]
        values = torch.cat(v_blocks, dim=1)[:, :seq_len, :] # [num_heads, seq_len, head_dim]

        # Scaled dot-product attention
        # q_i: [num_heads, 1, head_dim]
        q_unsq = q_i.unsqueeze(1)
        scores = torch.matmul(q_unsq, keys.transpose(-1, -2)) # [num_heads, 1, seq_len]
        attn_weights = torch.softmax(scores, dim=-1)
        out_i = torch.matmul(attn_weights, values).squeeze(1) # [num_heads, head_dim]

        outputs.append(out_i)

    return torch.stack(outputs, dim=0)


def sparse_24_gemm(a_sparse: torch.Tensor, b: torch.Tensor, meta: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Refused: there is no 2:4 sparse GEMM here.

    This multiplied the compressed `[M, K // 2]` values by the first `K // 2`
    rows of `b` and ignored `meta`, so every result was the product of the wrong
    rows. A 2:4 product needs each kept value placed at the column `meta`
    names; until something does that, the call fails rather than answers.
    """
    raise NotImplementedError(
        "sparse_24_gemm is not implemented: it used to ignore the 2:4 metadata and "
        "multiply the compressed values by the first K/2 rows of b, which is not the "
        "sparse product. Decompress A with its metadata and use torch.matmul.")

