"""The `@tile(M, N, K)` FP8 GEMM, run on the GPU and compared exactly.

Inputs are small integers times 2^-3, exact in e4m3, so every product and
every float32 partial sum is exact and the kernel must equal `A @ B` bit for
bit, whatever its tiling or order. C starts as NaN, so a tile no CTA writes
fails too. The shapes cover every tile `fp8_padded_tile` picks, those copied
with `cp.async` and those prefetched into registers, each with ragged M and
N edges, the `n % 4 != 0` path that keeps the unpadded layout,
and grids that are not a whole number of launch groups (`FP8_GEMM_GROUP_M`,
8 M-tiles).

Until 2026-10-12 nothing in the test suites ran this kernel on a GPU; the
head-to-head benchmark's exactness check and tests/test_fp8_gemm_correctness.py
(run by hand) were the only checks.
"""

import os
import re
import shutil
import subprocess
import sys
import unittest

import torch

from y_lang.torch_interop import TorchKernel

SHAPES = [
    (64, 64, 64),        # 64x64 copied with cp.async (M or N <= 512, one wave): one CTA
    (256, 256, 256),     # the same tile, 4x4 grid
    (17, 12, 64),        # ragged M, N below one warp tile
    (33, 17, 64),        # N % 4 != 0: the unpadded path
    (200, 136, 128),     # ragged, 64x64 cp.async
    (512, 1100, 64),     # 64x64 prefetched into registers: 144 CTAs, more than one wave; ragged N
    (300, 1000, 128),    # the same: 5 M-tiles, fewer than one launch group
    (600, 700, 128),     # 64x64 with a 32-wide K step: too few 128x128 tiles
    (1300, 520, 192),    # the same: 21 M-tiles, groups of 8, 8 and 5
    (1024, 1024, 1024),  # 128x128 copied with cp.async: one full launch group
    (1000, 1000, 64),    # the same tile, ragged, two K-tiles
    (1300, 1500, 192),   # the same: 11 M-tiles, a group of 8 and one of 3
    (2100, 2052, 64),    # the same, ragged both ways: 17 M-tiles
    (1030, 1026, 64),    # N % 4 != 0 on the large unpadded tier, grouped order
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
        self.run_shape(2100, 2052, 128, scale_a=4.0, scale_b=0.25)


# Ragged shapes of each tile family, run under compute-sanitizer's memcheck:
# a copy past A's last row or B's last column feeds only outputs the
# epilogue does not store, so the exactness check above cannot see it.
MEMCHECK_SHAPES = [(200, 136, 128), (600, 700, 128), (1000, 1000, 64), (300, 1000, 128)]
MEMCHECK_CHILD = """
import sys, torch
sys.path.insert(0, {tests!r})
from test_fp8_gemm import fp8_gemm
for m, n, k in {shapes!r}:
    a = torch.randint(-8, 9, (m, k), device="cuda").float() * 0.125
    b = torch.randint(-8, 9, (k, n), device="cuda").float() * 0.125
    c = torch.full((m, n), float("nan"), device="cuda")
    kern, grid, block = fp8_gemm(m, n, k)
    kern.launch(grid, block, [a, b, 1.0, 1.0, c])
    torch.cuda.synchronize()
    assert torch.equal(c, a @ b), (m, n, k)
print("RAN", len({shapes!r}))
"""


def compute_sanitizer():
    return shutil.which("compute-sanitizer") or next(
        (p for p in ("/opt/cuda/bin/compute-sanitizer", "/usr/local/cuda/bin/compute-sanitizer") if os.path.exists(p)), None)


@unittest.skipUnless(gpu_has_fp8(), "needs an sm_89 or later GPU")
class Fp8GemmReadsOnlyItsOperands(unittest.TestCase):
    def test_no_access_outside_a_b_or_c(self):
        tool = compute_sanitizer()
        if tool is None:
            self.skipTest("compute-sanitizer not found; the out-of-bounds check did not run")
        child = MEMCHECK_CHILD.format(tests=os.path.dirname(os.path.abspath(__file__)), shapes=MEMCHECK_SHAPES)
        # Without the caching allocator each tensor is its own cudaMalloc, so
        # a read past its end is outside every allocation memcheck knows.
        env = dict(os.environ, PYTORCH_NO_CUDA_MEMORY_CACHING="1")
        out = subprocess.run([tool, "--tool", "memcheck", "--error-exitcode", "3", sys.executable, "-c", child],
                             capture_output=True, text=True, env=env, timeout=600)
        text = out.stdout + out.stderr
        self.assertIn("RAN 4", text, text[-3000:])
        self.assertIn("ERROR SUMMARY: 0 errors", text, text[-3000:])
        self.assertEqual(out.returncode, 0, text[-3000:])


if __name__ == "__main__":
    unittest.main()
