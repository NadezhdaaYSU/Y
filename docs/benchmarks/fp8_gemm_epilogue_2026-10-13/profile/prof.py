"""ncu --profile-from-start off -o out python3 prof.py 1024

One launch each of Y's @tile FP8 GEMM and the head-to-head's Triton FP8 GEMM,
inside cudaProfilerStart/Stop, for `ncu --profile-from-start off`."""
import re
import sys

import torch
import triton
import triton.language as tl

import os
from pathlib import Path

# The checkout this record lives in, or Y_REPO.
REPO = os.environ.get("Y_REPO", str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, REPO + "/python")
import y_lang  # noqa: E402


@triton.jit
def triton_fp8_gemm(a_ptr, b_ptr, c_ptr, M, N, K, inv_sa, inv_sb, s_ab,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptr + rm[:, None] * K + (k0 + rk)[None, :])
        b = tl.load(b_ptr + (k0 + rk)[:, None] * N + rn[None, :])
        acc = tl.dot((a * inv_sa).to(tl.float8e4nv), (b * inv_sb).to(tl.float8e4nv), acc)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc * s_ab)


def main(n=4096):
    src = f"""
@tile({n}, {n}, {n})
kernel gemm_fp8_{n}(A: GlobalMemory<F32>, B: GlobalMemory<F32>, scale_a: F32, scale_b: F32, C: GlobalMemory<F32>) {{
}}
"""
    torch.zeros(1, device="cuda")   # a CUDA context before the module load
    k = y_lang.TorchKernel(src, kernel_name=f"gemm_fp8_{n}", target_sm="sm_89")
    k._ensure_loaded()
    m = re.search(r"Launch grid required: \((\d+), (\d+), (\d+)\) CTAs, block \((\d+),(\d+),(\d+)\)", k.ptx)
    v = [int(x) for x in m.groups()]
    grid, block = tuple(v[:3]), tuple(v[3:])
    g = torch.Generator(device="cuda").manual_seed(0)
    # Small integers times 2^-3: exact in e4m3, so every implementation agrees exactly.
    A = (torch.randint(-8, 9, (n, n), device="cuda", generator=g).float() * 0.125).contiguous()
    B = (torch.randint(-8, 9, (n, n), device="cuda", generator=g).float() * 0.125).contiguous()
    Cy, Ct = torch.empty(n, n, device="cuda"), torch.empty(n, n, device="cuda")
    y = lambda: k.launch(grid, block, [A, B, 1.0, 1.0, Cy])
    # Pinned to what the recorded runs autotuned to, so contention cannot pick another.
    cfg = dict(BM=128, BN=128, BK=32, num_warps=8, num_stages=3) if n <= 1024 else \
        dict(BM=128, BN=128, BK=32, num_warps=4, num_stages=2)
    t = lambda: triton_fp8_gemm[(n // 128, n // 128)](A, B, Ct, n, n, n, 1.0, 1.0, 1.0, **cfg)
    for _ in range(3):
        y()
        t()
    torch.cuda.synchronize()
    ref = A @ B
    print("Y grid", grid, "block", block, "| Triton", cfg,
          "| Y exact:", torch.equal(Cy, ref), "Triton exact:", torch.equal(Ct, ref), flush=True)
    torch.cuda.profiler.start()
    y()
    t()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4096)
