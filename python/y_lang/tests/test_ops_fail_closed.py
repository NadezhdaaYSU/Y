"""`y_lang.ops` fails instead of answering with something else.

`fp8_gemm` swallowed any exception from its CuPy path and returned `torch.mm`
on FP16 copies of the operands. With CuPy 14 the path never ran (NVRTC cannot
find `crt/mma.h`; `Stream.from_external` no longer takes a raw pointer), so
every CUDA "FP8 GEMM" it returned was cuBLAS - which is what the head-to-head
benchmark timed as Y. `sparse_24_gemm` ignored its 2:4 metadata and multiplied
the compressed values by the first half of B's rows.
"""

import unittest

import torch

from y_lang.ops import quant


class FailingKernel:
    def __call__(self, *args, **kwargs):
        raise RuntimeError("the CuPy kernel failed")


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class Fp8GemmTests(unittest.TestCase):
    def setUp(self):
        self.saved = quant._get_kernels
        self.a = torch.randn(64, 64, device="cuda").to(torch.float8_e4m3fn)
        self.b = torch.randn(64, 64, device="cuda").to(torch.float8_e4m3fn)

    def tearDown(self):
        quant._get_kernels = self.saved

    def test_a_failing_kernel_raises_rather_than_falling_back(self):
        quant._get_kernels = lambda: (FailingKernel(), FailingKernel())
        for a, b in ((self.a, self.b), (self.a.to(torch.float16), self.b.to(torch.float16))):
            with self.assertRaises(RuntimeError):
                quant.fp8_gemm(a, b)

    def test_a_missing_kernel_raises(self):
        quant._get_kernels = lambda: (None, None)
        with self.assertRaises(RuntimeError):
            quant.fp8_gemm(self.a, self.b)

    def test_whatever_this_machine_does_it_is_not_cublas_in_disguise(self):
        # The real path: either the CuPy kernel runs, or the call raises and
        # says what to use instead. It must not return torch.mm's product.
        try:
            got = quant.fp8_gemm(self.a, self.b)
        except RuntimeError as e:
            self.assertIn("torch.mm", str(e))
            self.assertIn("@tile", str(e))
            return
        cublas = torch.mm(self.a.to(torch.float16), self.b.to(torch.float16))
        self.assertFalse(torch.equal(got, cublas), "fp8_gemm returned torch.mm's product bit for bit")

    def test_other_dtypes_still_multiply(self):
        a, b = torch.randn(32, 16, device="cuda"), torch.randn(16, 8, device="cuda")
        self.assertEqual(quant.fp8_gemm(a, b).shape, (32, 8))


class Sparse24Tests(unittest.TestCase):
    def test_it_is_refused(self):
        with self.assertRaises(NotImplementedError):
            quant.sparse_24_gemm(torch.randn(8, 8), torch.randn(16, 8), torch.zeros(8, 1, dtype=torch.int16))


if __name__ == "__main__":
    unittest.main()
