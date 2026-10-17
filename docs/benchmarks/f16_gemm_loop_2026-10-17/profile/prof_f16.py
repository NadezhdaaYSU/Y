"""ncu --profile-from-start off -o out python3 prof_f16.py 4096

One launch each of Y's @tile F16 GEMM (f16 A, B -> f32 C) and f16_bench.py's
Triton GEMM with the configuration its autotuner picked at that size, inside
cudaProfilerStart/Stop. Run from a directory whose .ysu_hw_profile holds the
measured F16 tiles (the main checkout's)."""
import re
import sys

import torch
import triton
import triton.language as tl

import y_lang


@triton.jit
def triton_gemm(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptr + rm[:, None] * K + (k0 + rk)[None, :])
        b = tl.load(b_ptr + (k0 + rk)[:, None] * N + rn[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)


# What f16_bench.py's autotuner picked (bench1.log, 2026-10-13).
TRITON_CFG = {
    512: dict(BM=64, BN=64, BK=64, num_warps=4, num_stages=3),
    1024: dict(BM=64, BN=128, BK=32, num_warps=4, num_stages=4),
    2048: dict(BM=128, BN=128, BK=32, num_warps=4, num_stages=4),
    4096: dict(BM=128, BN=128, BK=32, num_warps=4, num_stages=4),
    8192: dict(BM=128, BN=256, BK=32, num_warps=8, num_stages=3),
}


def main(n):
    torch.zeros(1, device="cuda")
    src = (f"@tile({n}, {n}, {n})\nkernel gemm_f16_{n}(A: GlobalMemory<F16>, B: GlobalMemory<F16>, "
           f"C: GlobalMemory<F32>) {{\n}}\n")
    k = y_lang.TorchKernel(src, kernel_name=f"gemm_f16_{n}", target_sm="sm_89")
    k._ensure_loaded()
    cm, cn, wm, wn = (int(x) for x in re.search(r"CTA (\d+)x(\d+)x\d+ \| (\d+)x(\d+) warps", k.ptx).groups())
    g = torch.Generator(device="cuda").manual_seed(n)
    a = torch.randint(-4, 5, (n, n), device="cuda", generator=g).half()
    b = torch.randint(-4, 5, (n, n), device="cuda", generator=g).half()
    cy = torch.full((n, n), float("nan"), device="cuda")
    ct = torch.full((n, n), float("nan"), device="cuda")
    cfg = TRITON_CFG[n]
    y = lambda: k.launch((n // cn, n // cm, 1), (wm * wn * 32, 1, 1), [a, b, cy])
    t = lambda: triton_gemm[(n // cfg["BM"], n // cfg["BN"])](a, b, ct, n, n, n, **cfg)
    for _ in range(3):
        y()
        t()
    torch.cuda.synchronize()
    ref = (a.double() @ b.double()).float()
    print(f"Y tile {cm}x{cn} {wm}x{wn} | Triton {cfg} | Y exact: {torch.equal(cy, ref)} "
          f"Triton exact: {torch.equal(ct, ref)}", flush=True)
    torch.cuda.profiler.start()
    y()
    t()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4096)
