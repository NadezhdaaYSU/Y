"""The `@tile(M, N, K)` F16 GEMM (f16 A and B, f32 C), run on the GPU and
compared exactly.

Inputs are small integers, so every product and every float32 partial sum is
exact and the kernel must equal `A @ B` bit for bit, whatever its tiling or
order. C starts as NaN, so a tile no CTA writes fails too. Each tile is
pinned with Y_CTA_OVERRIDE against test_launch's pinned profile, so the
kernels do not depend on this checkout's `.ysu_hw_profile`.

Two staging paths:
- M, N and K whole multiples of the tile: each thread's `cp.async` addresses
  are computed once, before the K loop, and the per-chunk bound checks are
  one CTA-wide predicate. Where that spreads them (128x128x32 and 128x256x32
  here) the copies are issued between the MMAs. The shapes are not square,
  so a stride taken from the wrong operand (A's K for B's N) changes the
  answer.
- Otherwise every chunk computes its address and checks its bounds.
A CTA past the matrix (a grid larger than M and N) and a copy past an
operand's end change no stored value, so those cases run under
compute-sanitizer's memcheck.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import torch

from test_launch import PINNED_PROFILE
from y_lang.torch_interop import TorchKernel

# (cta_m, cta_n, cta_k, warps_m, warps_n, stages): candidates of
# Autotuner::generate_candidates and the measured tiles, from one warp to 16.
TILES = [
    "64,128,32,2,2,2",    # the measured 1024 tile
    "128,128,32,2,2,2",   # the measured 2048-16384 tile
    "64,64,64,4,2,3",     # three stages: the stage index is a remainder by 3
    "128,256,32,4,4,3",   # 16 warps: one A chunk a thread
    "16,32,32,1,1,4",     # one warp, four stages
    "128,128,128,4,2,2",  # one stage fits: the synchronous path
]


def f16_gemm(m, n, k, tile):
    """Compile the plain F16 GEMM with its tile pinned; return the kernel,
    its grid for M x N, its block and the CTA tile."""
    src = (f"@tile({m}, {n}, {k})\nkernel g(A: GlobalMemory<F16>, B: GlobalMemory<F16>, "
           f"C: GlobalMemory<F32>) {{\n}}\n")
    saved_env, saved_cwd = os.environ.get("Y_CTA_OVERRIDE"), os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, ".ysu_hw_profile"), "w") as f:
            f.write(PINNED_PROFILE)
        os.environ["Y_CTA_OVERRIDE"] = tile
        os.chdir(d)
        try:
            kern = TorchKernel(src, kernel_name="g", target_sm="sm_89")
        finally:
            os.chdir(saved_cwd)
            if saved_env is None:
                del os.environ["Y_CTA_OVERRIDE"]
            else:
                os.environ["Y_CTA_OVERRIDE"] = saved_env
    cm, cn, wm, wn = (int(x) for x in re.search(r"CTA (\d+)x(\d+)x\d+ \| (\d+)x(\d+) warps", kern.ptx).groups())
    assert (cm, cn) == tuple(int(x) for x in tile.split(",")[:2]), (tile, cm, cn)
    return kern, (-(-n // cn), -(-m // cm), 1), (wm * wn * 32, 1, 1), (cm, cn)


def shapes_for(tile):
    """Two divisible, not square shapes and a ragged one (M and N past the
    last whole tile; N and K stay multiples of 8, the 16-byte copy
    granularity). The first divisible one has 24 K-tiles: copies the
    pipeline fails to wait for are a race, and a short loop gives it few
    chances to show. Its N is a power of two, so B's K-tile step (2 * N
    bytes) is a shift; the second's, 3 * cta_n, is not, and is a multiply."""
    cm, cn, ck = (int(x) for x in tile.split(",")[:3])
    return [(3 * cm, 2 * cn, 24 * ck), (3 * cm - 5, 2 * cn + 24, 5 * ck), (2 * cm, 3 * cn, 6 * ck)]


def operands(m, n, k, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randint(-4, 5, (m, k), device="cuda", generator=gen).half()
    b = torch.randint(-4, 5, (k, n), device="cuda", generator=gen).half()
    return a, b


def gpu_has_tensor_cores():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


@unittest.skipUnless(gpu_has_tensor_cores(), "needs an sm_89 or later GPU")
class F16GemmIsExact(unittest.TestCase):
    def run_shape(self, m, n, k, tile, extra_ctas=0):
        a, b = operands(m, n, k, m * 7 + n * 3 + k)
        c = torch.full((m, n), float("nan"), device="cuda")
        kern, grid, block, _ = f16_gemm(m, n, k, tile)
        grid = (grid[0] + extra_ctas, grid[1] + extra_ctas, 1)
        kern.launch(grid, block, [a, b, c])
        torch.cuda.synchronize()
        bad = ~(c.double() == a.double() @ b.double())
        self.assertFalse(bool(bad.any()), f"{int(bad.sum())} of {m * n} elements differ at {(m, n, k)} "
                                          f"with {tile}; first at "
                                          f"{tuple(bad.nonzero()[0].tolist()) if bad.any() else None}")
        return kern.ptx

    def test_every_tile_divisible_and_ragged(self):
        for tile in TILES:
            for shape in shapes_for(tile):
                with self.subTest(tile=tile, shape=shape):
                    self.run_shape(*shape, tile)

    def test_the_staging_path_follows_divisibility(self):
        divisible, ragged, _ = shapes_for("64,128,32,2,2,2")
        self.assertTrue("addresses hoisted" in self.run_shape(*divisible, "64,128,32,2,2,2"),
                        f"{divisible} divides the tile, and its loads still check every chunk")
        ptx = self.run_shape(*ragged, "64,128,32,2,2,2")
        self.assertFalse("addresses hoisted" in ptx, f"{ragged} is ragged and its loads were hoisted")
        # One dimension ragged at a time. A copy past M or N feeds only
        # outputs nobody stores, so this, and memcheck, are what see it.
        for shape in ((187, 256, 160), (192, 280, 160), (192, 256, 168)):
            with self.subTest(shape=shape):
                self.assertFalse("addresses hoisted" in self.run_shape(*shape, "64,128,32,2,2,2"),
                                 f"{shape} is ragged and its loads were hoisted")
        # A K-tile's offset is k0 * step bytes: 2 for A, 2 * N for B. A
        # power-of-two step is a shift, which ptxas makes a LEA pair as fast
        # as a mad.wide (a mul.wide measured 0.8% slower); any other is a
        # mul.wide. The translation validator models both, not mad.wide.
        _, _, third = shapes_for("64,128,32,2,2,2")
        for shape, b_step in ((divisible, rf"shl\.b64 %rd\d+, %rd\d+, {(2 * divisible[1]).bit_length() - 1};"),
                              (third, rf"mul\.wide\.u32 %rd\d+, %r\d+, {2 * third[1]};")):
            with self.subTest(shape=shape):
                hoisted = self.run_shape(*shape, "64,128,32,2,2,2")
                self.assertTrue("addresses hoisted" in hoisted, f"{shape} divides the tile, and was not hoisted")
                self.assertTrue(re.search(r"shl\.b64 %rd\d+, %rd\d+, 1;", hoisted), "A's step (2 bytes) is not a shift")
                self.assertTrue(re.search(b_step, hoisted), f"B's step at {shape} is not {b_step}")
                self.assertFalse("mad.wide" in hoisted, "a mad.wide, which the validator does not model")
        # B's fragments: one x4.trans per 16 columns, on both paths.
        self.assertTrue("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16" in ptx and ".x2.trans" not in ptx,
                        "B's fragments are not one ldmatrix.x4.trans per 16 columns")

    def test_the_pipeline_commits_waits_and_synchronises_in_order(self):
        # cp.async.wait_group waits only for committed groups. A copy issued
        # between the MMAs and left out of the group would race the next
        # iteration's reads: measured, such a kernel was wrong in 3 of 1,200
        # runs, so an exactness check cannot be relied on to see it. A wait
        # makes only this thread's copies complete; the barrier after it
        # makes every thread's visible before any ldmatrix reads them.
        for tile in TILES:
            with self.subTest(tile=tile):
                kern = f16_gemm(*shapes_for(tile)[0], tile)[0]
                uncommitted, unsynchronised = 0, False
                for n, line in enumerate(kern.ptx.splitlines()):
                    if "cp.async.cg.shared.global" in line:
                        uncommitted += 1
                    elif "cp.async.commit_group" in line:
                        uncommitted = 0
                    elif "cp.async.wait_group" in line:
                        self.assertEqual(uncommitted, 0, f"{uncommitted} copies uncommitted at PTX line {n + 1}")
                        unsynchronised = True
                    elif "bar.sync" in line:
                        unsynchronised = False
                    elif "ldmatrix" in line:
                        self.assertFalse(unsynchronised, f"ldmatrix after a wait with no barrier, PTX line {n + 1}")

    def test_ctas_past_the_matrix_store_nothing(self):
        for tile in ("64,128,32,2,2,2", "64,64,64,4,2,3", "128,128,32,2,2,2"):
            with self.subTest(tile=tile):
                self.run_shape(*shapes_for(tile)[0], tile, extra_ctas=1)


# Run under memcheck: a copy past A's last row, B's last column or K, or by a
# CTA past the matrix, feeds only outputs the epilogue does not store.
MEMCHECK_CASES = [
    ((192, 256, 160), "64,128,32,2,2,2", 0),
    ((187, 280, 160), "64,128,32,2,2,2", 0),
    ((187, 256, 160), "64,128,32,2,2,2", 0),
    ((192, 280, 160), "64,128,32,2,2,2", 0),
    ((192, 256, 160), "64,128,32,2,2,2", 1),
    ((192, 128, 320), "64,64,64,4,2,3", 1),
    ((384, 256, 160), "128,128,32,2,2,2", 1),
    ((379, 536, 160), "128,256,32,4,4,3", 0),
]
MEMCHECK_CHILD = """
import sys, torch
sys.path.insert(0, {tests!r})
from test_f16_gemm import f16_gemm, operands
for (m, n, k), tile, extra in {cases!r}:
    a, b = operands(m, n, k, 1)
    c = torch.full((m, n), float("nan"), device="cuda")
    kern, grid, block, _ = f16_gemm(m, n, k, tile)
    kern.launch((grid[0] + extra, grid[1] + extra, 1), block, [a, b, c])
    torch.cuda.synchronize()
    assert torch.equal(c.double(), a.double() @ b.double()), (m, n, k, tile)
print("RAN", len({cases!r}))
"""


def compute_sanitizer():
    return shutil.which("compute-sanitizer") or next(
        (p for p in ("/opt/cuda/bin/compute-sanitizer", "/usr/local/cuda/bin/compute-sanitizer") if os.path.exists(p)), None)


@unittest.skipUnless(gpu_has_tensor_cores(), "needs an sm_89 or later GPU")
class F16GemmReadsOnlyItsOperands(unittest.TestCase):
    def test_no_access_outside_a_b_or_c(self):
        tool = compute_sanitizer()
        if tool is None:
            self.skipTest("compute-sanitizer not found; the out-of-bounds check did not run")
        child = MEMCHECK_CHILD.format(tests=os.path.dirname(os.path.abspath(__file__)), cases=MEMCHECK_CASES)
        # Without the caching allocator each tensor is its own cudaMalloc, so
        # a read past its end is outside every allocation memcheck knows.
        env = dict(os.environ, PYTORCH_NO_CUDA_MEMORY_CACHING="1")
        out = subprocess.run([tool, "--tool", "memcheck", "--error-exitcode", "3", sys.executable, "-c", child],
                             capture_output=True, text=True, env=env, timeout=600)
        text = out.stdout + out.stderr
        self.assertIn(f"RAN {len(MEMCHECK_CASES)}", text, text[-3000:])
        self.assertIn("ERROR SUMMARY: 0 errors", text, text[-3000:])
        self.assertEqual(out.returncode, 0, text[-3000:])


if __name__ == "__main__":
    unittest.main()
