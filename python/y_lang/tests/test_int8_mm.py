"""`y_lang.int8_gemm.int8_mm`: Y's proved int8 GEMM, called from PyTorch.

On the GPU each result is compared with a float64 product of the same int8
values, which is exact (|sum| <= K * 2^14 < 2^53), over the whole int8 range,
at row counts either side of the op's padding buckets. It is also compared with
`torch._int_mm` (cuBLASLt), run under `torch.compile` with no graph break,
and run on a side stream. The refusals are checked on host tensors, so they
run without a GPU, and so does the claim that the PTX depends only on the
target, which two different profiles must agree on.

Compiles run in a temporary directory holding `test_launch.PINNED_PROFILE`,
with a temporary `YSU_CACHE_DIR` and no `Y_INT8_TILE`, so nothing depends on
this checkout's profile or PTX cache.
"""

import contextlib
import math
import os
import re
import tempfile
import unittest

import torch

from test_launch import PINNED_PROFILE
from y_lang import int8_gemm as G
from y_lang.compiler import compile_to_ptx

# Every field a profile carries that could reach the int8 kernel, changed.
OTHER_PROFILE = """SM_VERSION=8.0
COMPUTE_CAPABILITY=8.0
GPU_NAME=SomeOtherCard
SM_COUNT=20
MAX_SMEM_PER_SM_BYTES=167936
"""


@contextlib.contextmanager
def pinned(profile=PINNED_PROFILE):
    """Compile against `profile`, with a private PTX cache and no Y_INT8_TILE."""
    saved_cwd = os.getcwd()
    saved = {v: os.environ.get(v) for v in ("YSU_CACHE_DIR", "Y_INT8_TILE")}
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, ".ysu_hw_profile"), "w") as f:
            f.write(profile)
        os.environ["YSU_CACHE_DIR"] = os.path.join(d, "cache")
        os.environ.pop("Y_INT8_TILE", None)
        os.chdir(d)
        try:
            yield
        finally:
            os.chdir(saved_cwd)
            for v, value in saved.items():
                if value is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = value


def int8(shape, gen, device="cpu"):
    return torch.randint(-128, 128, shape, generator=gen, dtype=torch.int8).to(device)


class TheRowBucket(unittest.TestCase):
    def test_every_row_count_has_a_bucket_at_most_one_step_above_it(self):
        for m in range(1, 2100):
            b = G.rows_bucket(m)
            step = 16 if m <= 128 else 128
            self.assertTrue(b >= m and b % step == 0 and b - m < step, (m, b))
        for m in (16, 32, 48, 128, 256, 1024):
            self.assertEqual(G.rows_bucket(m), m)


class TheSplitPolicy(unittest.TestCase):
    """K is split only where the output tiles leave the card half idle."""

    def test_the_factor_for_the_measured_shapes(self):
        sms = 66
        self.assertEqual(G.split_factor(14, 76, sms), 4)    # 0.5B down, decode: not 5 (70 CTAs)
        self.assertEqual(G.split_factor(8, 56, sms), 7)     # 7B k/v, decode
        self.assertEqual(G.split_factor(14, 14, sms), 1)    # 0.5B q/o: K too short
        self.assertEqual(G.split_factor(2, 14, sms), 1)     # 0.5B k/v: K too short
        self.assertEqual(G.split_factor(56, 296, sms), 1)   # 7B down: tiles fill the card
        self.assertEqual(G.split_factor(76, 14, sms), 1)

    def test_never_more_splits_than_k_tiles_and_never_zero(self):
        for ctas in range(1, 200):
            for k_tiles in range(1, 400, 7):
                z = G.split_factor(ctas, k_tiles, 66)
                self.assertTrue(1 <= z <= max(1, k_tiles // 8), (ctas, k_tiles, z))


class TheRefusals(unittest.TestCase):
    """Each names its limit, before anything is compiled or launched."""

    def refused(self, a, b, needle):
        with self.assertRaises((TypeError, ValueError)) as cm:
            G.check_operands(a, b)
        self.assertIn(needle, str(cm.exception))

    def test_shapes_and_types(self):
        i8 = lambda *s: torch.zeros(s, dtype=torch.int8)
        self.refused(torch.zeros(4, 64), i8(8, 64), "2-D int8")
        self.refused(i8(4, 64), torch.zeros(8, 64, dtype=torch.int32), "2-D int8")
        self.refused(i8(64), i8(8, 64), "2-D int8")
        self.refused(i8(4, 64), [[0] * 64] * 8, "not a torch.Tensor")
        self.refused(i8(4, 64), i8(8, 32), "b must be [N][64]")
        self.refused(i8(4, 48), i8(8, 48), "K = 48")
        self.refused(i8(4, 0), i8(8, 0), "K = 0")
        self.refused(i8(1, 131072), i8(8, 131072), "K = 131072")
        self.refused(i8(4, 64), i8(12, 64), "N = 12")
        self.refused(i8(4, 64), i8(0, 64), "N = 0")

    def test_host_operands_of_a_legal_shape(self):
        i8 = lambda *s: torch.zeros(s, dtype=torch.int8)
        self.refused(i8(4, 64), i8(8, 64), "one CUDA device")
        self.refused(i8(4, G.MAX_K), i8(8, G.MAX_K), "one CUDA device")

    @unittest.skipIf(G._OP is None, "this torch has no torch.library")
    def test_the_registered_op_refuses_with_the_same_messages(self):
        # Host tensors reach the op's own body (it is registered for CPU
        # too), not the dispatcher's "could not run" error.
        i8 = lambda *s: torch.zeros(s, dtype=torch.int8)
        for a, b, needle in ((i8(4, 64), i8(8, 64), "one CUDA device"),
                             (i8(4, 48), i8(8, 48), "K = 48"),
                             (i8(4, 64), i8(8, 32), "b must be [N][64]")):
            with self.assertRaises(ValueError) as cm:
                G._OP(a, b)
            self.assertIn(needle, str(cm.exception))

    @unittest.skipIf(G._OP is None, "this torch has no torch.library")
    def test_a_reload_keeps_the_registration(self):
        # The registration lives as long as its `torch.library.Library`
        # object; a reload that dropped it would remove the op.
        import gc
        import importlib
        lib = G._LIB
        importlib.reload(G)
        gc.collect()
        self.assertIs(G._LIB, lib)
        self.assertTrue(hasattr(torch.ops.y_lang, "int8_mm"))
        i8 = lambda *s: torch.zeros(s, dtype=torch.int8)
        with self.assertRaises(ValueError):
            torch.ops.y_lang.int8_mm(i8(4, 64), i8(8, 64))


class ThePtx(unittest.TestCase):
    """The kernel the op compiles, read without a GPU."""

    def test_the_launch_is_read_from_the_header_and_checked_against_it(self):
        for m, n, k in ((16, 8, 32), (16, 896, 896), (32, 4864, 896), (256, 896, 4864),
                        (4096, 4096, 4096)):
            with pinned():
                ptx = compile_to_ptx(G.source(m, n, k), target_sm="sm_89", name=G.KERNEL_NAME)
            threads, ctas = G.launch_geometry(ptx, m, n, k)
            cm, cn, wm, wn = (int(v) for v in re.search(
                r"\| CTA (\d+)x(\d+)x\d+ \| (\d+)x(\d+) warps", ptx).groups())
            self.assertEqual(threads, 32 * wm * wn, (m, n, k))
            self.assertEqual(ctas, math.ceil(m / cm) * math.ceil(n / cn), (m, n, k))
            with self.assertRaises(RuntimeError):
                G.launch_geometry(ptx, m + 16, n, k)
        with self.assertRaises(RuntimeError):
            G.launch_geometry(".version 7.8\n.target sm_89\n", 16, 8, 32)

    def test_the_ptx_depends_on_the_target_and_shape_only(self):
        src = G.source(128, 4864, 896)
        with pinned():
            a = compile_to_ptx(src, target_sm="sm_89", name=G.KERNEL_NAME)
        with pinned(OTHER_PROFILE):
            b = compile_to_ptx(src, target_sm="sm_89", name=G.KERNEL_NAME)
            c = compile_to_ptx(src, target_sm="sm_80", name=G.KERNEL_NAME)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)   # the control: the target does reach the PTX
        self.assertIn(".target sm_80", c)


def gpu_ok():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0)


@unittest.skipUnless(gpu_ok(), "needs an sm_80 or later GPU")
class OnTheGpu(unittest.TestCase):
    def setUp(self):
        self.gen = torch.Generator().manual_seed(11)

    def check(self, a, b):
        with pinned():
            c = G.int8_mm(a, b)
        torch.cuda.synchronize()
        want = (a.double() @ b.double().t()).cpu()
        self.assertEqual((c.dtype, tuple(c.shape)), (torch.int32, (a.shape[0], b.shape[0])))
        self.assertTrue(torch.equal(c.cpu().double(), want),
                        f"{tuple(a.shape)} x {tuple(b.shape)}: not the exact product")
        return c

    def test_exact_across_the_row_buckets(self):
        for n, k in ((8, 32), (896, 896), (128, 896), (4864, 896), (896, 4864)):
            b = int8((n, k), self.gen, "cuda")
            for m in (1, 15, 16, 17, 128, 129, 300):
                self.check(int8((m, k), self.gen, "cuda"), b)

    def test_exact_at_the_licensed_k_with_full_range_products(self):
        k = G.MAX_K
        for va, vb in ((-128, -128), (-128, 127), (127, 127)):
            a = torch.full((16, k), va, dtype=torch.int8, device="cuda")
            b = torch.full((8, k), vb, dtype=torch.int8, device="cuda")
            c = self.check(a, b)
            self.assertEqual(int(c[0, 0]), va * vb * k)

    def test_a_split_launch_is_exact(self):
        # 8 or 14 output tiles over a long K: the op splits K over gridDim.z
        # and adds the partials into a zeroed C.
        for m, n, k in ((32, 512, 3584), (17, 896, 4864)):
            with pinned():
                _, _, ctas, z = G.kernel_for(G.rows_bucket(m), n, k, "cuda")
            if 2 * ctas <= torch.cuda.get_device_properties(0).multi_processor_count:
                self.assertGreater(z, 1, (m, n, k, ctas))
            self.check(int8((m, k), self.gen, "cuda"), int8((n, k), self.gen, "cuda"))

    def test_the_same_bits_as_cublaslt(self):
        for m, n, k in ((32, 896, 896), (64, 4864, 896), (96, 896, 4864)):
            a, b = int8((m, k), self.gen, "cuda"), int8((n, k), self.gen, "cuda")
            with pinned():
                c = G.int8_mm(a, b)
            self.assertTrue(torch.equal(c, torch._int_mm(a, b.t())), (m, n, k))

    def test_zero_rows(self):
        b = int8((8, 32), self.gen, "cuda")
        c = G.int8_mm(torch.empty((0, 32), dtype=torch.int8, device="cuda"), b)
        self.assertEqual((c.dtype, tuple(c.shape)), (torch.int32, (0, 8)))

    def test_device_layout_refusals(self):
        k = 64
        wide = int8((8, k + 32), self.gen, "cuda")
        with self.assertRaisesRegex(ValueError, "must be contiguous"):
            G.int8_mm(int8((4, k), self.gen, "cuda"), wide[:, :k])
        buf = torch.zeros(4 * k + 8, dtype=torch.int8, device="cuda")
        with self.assertRaisesRegex(ValueError, "16-byte aligned"):
            G.int8_mm(buf[8:].view(4, k), int8((8, k), self.gen, "cuda"))

    def test_on_the_callers_stream(self):
        a, b = int8((17, 896), self.gen, "cuda"), int8((128, 896), self.gen, "cuda")
        side = torch.cuda.Stream()
        with pinned(), torch.cuda.stream(side):
            a2, b2 = a * 1, b * 1      # produced on the side stream
            c = G.int8_mm(a2, b2)
        side.synchronize()
        self.assertTrue(torch.equal(c.cpu().double(), (a.double() @ b.double().t()).cpu()))

    def test_compiled_with_no_graph_break(self):
        import torch._dynamo
        torch._dynamo.reset()
        fn = torch.compile(lambda a, b: G.int8_mm(a, b) + 1, dynamic=True, fullgraph=True)
        b = int8((128, 896), self.gen, "cuda")
        for m in (17, 40):
            a = int8((m, 896), self.gen, "cuda")
            with pinned():
                got = fn(a, b)
            self.assertTrue(torch.equal(got.cpu().double(),
                                        (a.double() @ b.double().t()).cpu() + 1), m)
        torch._dynamo.reset()


if __name__ == "__main__":
    unittest.main()
