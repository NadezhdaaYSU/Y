"""Y's @tile FP32 GEMM against cuBLAS (torch.mm, TF32 off) on whatever GPU this
runs on: every element of Y's C equal bit for bit to an in-order fmaf chain
(the claim), then speed. Interleaved rounds, the minimum per side.

    python3 f32_vs_cublas.py [MxNxK ...]
"""
import re
import sys
import time

import cupy as cp
import numpy as np
import torch

from y_lang.torch_interop import TorchKernel, device_target

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

ORACLE = cp.RawKernel(r'''
extern "C" __global__ void oracle(const float* A, const float* B, float* C, int M, int N, int K) {
  int j = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y; if (j >= N) return;
  float acc = 0.f; for (int k = 0; k < K; k++) acc = fmaf(A[(size_t)i * K + k], B[(size_t)k * N + j], acc);
  C[(size_t)i * N + j] = acc; }''', "oracle")

DEFAULT = ["512x512x512", "1024x1024x1024", "2048x2048x2048", "4096x4096x4096", "1023x1025x1027",
           "1024x4096x4096", "128x4096x4096", "4096x128x4096", "64x8192x2048", "4096x4096x1024"]


def build(m, n, k):
    name = f"y_{m}_{n}_{k}"
    src = (f"@tile({m}, {n}, {k})\nkernel {name}(A: GlobalMemory<F32>, B: GlobalMemory<F32>, "
           f"C: GlobalMemory<F32>) {{\n}}\n\nfn main() {{\n}}\n")
    kern = TorchKernel(src, kernel_name=name, target_sm=device_target())
    ptx = kern.ptx
    tile = re.search(r"tile `(\w+)`", ptx).group(1)
    block = int(re.search(r"Launch: block \((\d+),1,1\)", ptx).group(1))
    g = re.search(r"\((\d+), (\d+), 1\) is one tile per CTA", ptx)
    copies = "plain loads" if "stages of plain loads" in ptx else "cp.async"
    return kern, tile, block, (int(g.group(1)), int(g.group(2)), 1), copies


def timed(fn, reps):
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        fn()
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1) / reps * 1e-3


def main():
    dev = torch.cuda.get_device_name(0)
    print(f"{dev} ({device_target()}), torch {torch.__version__}, cuBLAS TF32 off")
    x = torch.randn(4096, 4096, device="cuda")
    t0 = time.time()
    while time.time() - t0 < 3:                       # clocks up
        x @ x
    torch.cuda.synchronize()
    for spec in sys.argv[1:] or DEFAULT:
        m, n, k = map(int, spec.split("x"))
        g = torch.Generator(device="cuda").manual_seed(m * 31 + n * 7 + k)
        A = torch.randn(m, k, device="cuda", generator=g)
        B = torch.randn(k, n, device="cuda", generator=g)
        C = torch.full((m, n), float("nan"), device="cuda")
        kern, tile, block, grid, copies = build(m, n, k)
        kern.launch(grid, (block, 1, 1), [A, B, C])
        torch.cuda.synchronize()
        want = cp.zeros((m, n), cp.float32)
        ORACLE(((n + 127) // 128, m, 1), (128, 1, 1),
               (cp.asarray(A), cp.asarray(B), want, np.int32(m), np.int32(n), np.int32(k)))
        cp.cuda.Device().synchronize()
        exact = bool((C.view(torch.int32) == torch.as_tensor(want, device="cuda").view(torch.int32)).all())
        cub = torch.mm(A, B)
        cub_same = int((cub.view(torch.int32) == torch.as_tensor(want, device="cuda").view(torch.int32)).sum())
        flops = 2.0 * m * n * k
        reps = max(3, min(200, int(2e11 / flops)))
        ty, tc = [], []
        for r in range(11):
            order = [("y", lambda: kern.launch(grid, (block, 1, 1), [A, B, C])), ("c", lambda: torch.mm(A, B, out=cub))]
            if r % 2:
                order.reverse()
            for side, fn in order:
                (ty if side == "y" else tc).append(timed(fn, reps))
        y, c = min(ty), min(tc)
        print(f"{spec:16s} tile {tile:5s} ({copies}): bit-exact {exact}  (cuBLAS equals the chain on "
              f"{cub_same / (m * n):.0%})  | cuBLAS {flops / c / 1e12:5.2f} TF  Y {flops / y / 1e12:5.2f} TF"
              f"  = {c / y:.3f}x", flush=True)


main()
