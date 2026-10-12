"""The `@tile(M, N, K)` FP8 GEMM, run on the GPU and compared exactly.

Inputs are small integers times 2^-3, exact in e4m3, so every product and
every float32 partial sum is exact and the kernel must equal `A @ B` bit for
bit, whatever its tiling or order. C starts as NaN, so a tile no CTA writes
fails too. The shapes cover both CTA tiers, ragged M and N edges, the
`n % 4 != 0` path that keeps the unpadded layout, and grids that are not a
whole number of launch groups (`FP8_GEMM_GROUP_M`, 8 M-tiles).

Until 2026-10-12 nothing in the test suites ran this kernel on a GPU; the
head-to-head benchmark's exactness check and tests/test_fp8_gemm_correctness.py
(run by hand) were the only checks.
"""

import re
import unittest

import torch

from y_lang.torch_interop import TorchKernel

SHAPES = [
    (64, 64, 64),        # small tier, one CTA
    (256, 256, 256),     # small tier, 4x4 grid
    (17, 12, 64),        # ragged M, N below one warp tile
    (33, 17, 64),        # N % 4 != 0: the unpadded path
    (200, 136, 128),     # ragged, small tier
    (1024, 1024, 1024),  # large tier, one full launch group
    (1300, 520, 192),    # large tier, 11 M-tiles: a group of 8 and one of 3
    (600, 1000, 128),    # 5 M-tiles, fewer than one group
    (1030, 1026, 64),    # N % 4 != 0 on the large tier, grouped order
]


def gpu_has_fp8():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


def fp8_gemm(m, n, k):
    name = f"gemm_fp8_t_{m}_{n}_{k}"
    src = (f"@tile({m}, {n}, {k})\nkernel {name}(A: GlobalMemory<F32>, B: GlobalMemory<F32>, "
           f"scale_a: F32, scale_b: F32, C: GlobalMemory<F32>) {{\n}}\n")
    kern = TorchKernel(src, kernel_name=name, target_sm="sm_89")
    g = re.search(r"Launch grid required: \((\d+), (\d+), (\d+)\) CTAs, block \((\d+),(\d+),(\d+)\)", kern.ptx)
    v = [int(x) for x in g.groups()]
    return kern, tuple(v[:3]), tuple(v[3:])


@unittest.skipUnless(gpu_has_fp8(), "needs an sm_89 or later GPU")
class Fp8GemmIsExact(unittest.TestCase):
    def setUp(self):
        torch.backends.cuda.matmul.allow_tf32 = False

    def run_shape(self, m, n, k, scale_a=1.0, scale_b=1.0):
        gen = torch.Generator(device="cuda").manual_seed(m * 7 + n * 3 + k)
        a = torch.randint(-8, 9, (m, k), device="cuda", generator=gen).float() * 0.125
        b = torch.randint(-8, 9, (k, n), device="cuda", generator=gen).float() * 0.125
        c = torch.full((m, n), float("nan"), device="cuda")
        kern, grid, block = fp8_gemm(m, n, k)
        kern.launch(grid, block, [a * scale_a, b * scale_b, scale_a, scale_b, c])
        torch.cuda.synchronize()
        ref = (a @ b) * (scale_a * scale_b)
        bad = ~(c == ref)
        self.assertFalse(bool(bad.any()), f"{int(bad.sum())} of {m * n} elements differ; first at "
                                          f"{tuple(bad.nonzero()[0].tolist()) if bad.any() else None}")

    def test_every_shape(self):
        for shape in SHAPES:
            with self.subTest(shape=shape):
                self.run_shape(*shape)

    def test_scales_divide_before_quantizing_and_multiply_after(self):
        # A/scale_a and B/scale_b are the small integers again, so this is
        # exact only if the kernel quantizes A/scale_a and scales the result.
        self.run_shape(1024, 1024, 256, scale_a=2.0, scale_b=0.5)
        self.run_shape(200, 136, 128, scale_a=0.25, scale_b=4.0)


if __name__ == "__main__":
    unittest.main()
