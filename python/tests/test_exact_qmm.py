"""The fused exact product (`y_lang.exact.qmm`) against `exact_product`, bit
for bit, on a GPU: the shapes exact LoRA training multiplies, 8- and 12-bit
gradients, values at the clamp, zero rows, transposed operands and a frozen
int8 weight; and the adapters' products choosing it."""

import os
import sys
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))

from test_exact_lora import bits_equal  # noqa: E402

GPU = torch.cuda.is_available()


@unittest.skipUnless(GPU, "needs a CUDA GPU")
class FusedProduct(unittest.TestCase):
    def setUp(self):
        from y_lang.exact import lora, qmm, train_lm
        torch.backends.cuda.matmul.allow_tf32 = False
        self.lora, self.qmm, self.T = lora, qmm, train_lm
        self.g = torch.Generator(device="cuda").manual_seed(0)

    def operands(self, M, N, K, a_dtype=torch.float32, b_dtype=torch.float64):
        g = self.g
        a = torch.randn(M, K, device="cuda", generator=g, dtype=torch.float64)
        a = a * torch.exp2(torch.randint(-20, 20, (M, 1), device="cuda", generator=g)).double()
        b = torch.randn(N, K, device="cuda", generator=g, dtype=torch.float64) * 0.03
        a[0] = 0                                   # a zero row
        if M > 2:
            a[1, 0] = a[1].abs().max() * 4         # one value at the top: the clamp
            a[2] = torch.where(torch.arange(K, device="cuda") % 2 == 0, 1.0, -1.0)  # +-max
        return a.to(a_dtype), b.to(b_dtype)

    def check(self, a, b, bits_a, qb=None):
        T, lora = self.T, self.lora
        bits_b = 8
        want = T.exact_product(a, b, bits_a, bits_b, lora.mm_y_any, False, False, torch.float32, None, qb)
        qa, sa = T.q8(a, bits=bits_a)
        qbv, sb = T.q8(b, bits=bits_b) if qb is None else qb
        got = self.qmm.qmm(qa, sa, qbv, sb, bits_a, bits_b)
        self.assertTrue(bits_equal(want, got), f"{tuple(a.shape)} x {tuple(b.shape)}, {bits_a} bits: "
                        f"{int((want != got).sum())} values differ")

    def test_training_shapes(self):
        shapes = [(1, 16, 16), (7, 16, 896), (300, 896, 16), (1024, 16, 896), (513, 4864, 896),
                  (64, 896, 4864), (33, 1152, 1000), (1792, 16, 992), (16, 1792, 992), (5, 3, 40)]
        for M, N, K in shapes:
            for bits_a in (8, 12):
                a, b = self.operands(M, N, K)
                self.check(a, b, bits_a)

    def test_extreme_scales(self):
        """Scales whose product leaves float32's range: results that are
        float32 subnormals (the float64 product rounded once), or infinite,
        with zero rows among them (a product of 0 and a scale float32 cannot
        hold must stay 0)."""
        for shift_a, shift_b in ((-130, -8), (-100, -40), (100, 40), (-150, 10)):
            a, b = self.operands(40, 24, 96)
            a = (a.double() * 2.0 ** shift_a).float() if abs(shift_a) < 120 else a.double() * 2.0 ** shift_a
            b = b * 2.0 ** shift_b
            for bits_a in (8, 12):
                self.check(a, b, bits_a)

    def test_transposed_operands(self):
        """Operands that are transposed views (the adapters' gradients over
        tokens: dyc.t(), u.t(), du.t(), xf.t())."""
        g = self.g
        for M, N, K in ((896, 16, 300), (16, 4864, 512), (4864, 16, 1000)):
            a = torch.randn(K, M, device="cuda", generator=g).t()
            b = (torch.randn(K, N, device="cuda", generator=g, dtype=torch.float64) * 0.03).t()
            self.assertFalse(a.is_contiguous())
            self.check(a, b, 12)
            self.check(a, b, 8)

    def test_a_frozen_int8_weight(self):
        g = self.g
        M, N, K = 77, 896, 4864
        a = torch.randn(M, K, device="cuda", generator=g)
        w = torch.randint(-127, 128, (N, K), device="cuda", generator=g, dtype=torch.int8)
        s0 = torch.zeros(N, 1, dtype=torch.int64, device="cuda")
        self.check(a, w, 12, qb=(w, s0))

    def test_the_adapters_products_use_it(self):
        """`lora._exact_mm` on a GPU takes the fused path (and gives
        `exact_product`'s bits); with `QMM` off it does not."""
        from unittest import mock
        a, b = self.operands(64, 16, 896)
        calls = []
        real = self.qmm.qmm

        def spy(*args):
            calls.append(1)
            return real(*args)
        with mock.patch.object(self.qmm, "qmm", spy):
            got = self.lora._exact_mm(a, b, "a", 12, 8, self.lora.mm_y_any, torch.float32, None)
            self.assertEqual(len(calls), 1)
            with mock.patch.object(self.lora, "QMM", False):
                ref = self.lora._exact_mm(a, b, "a", 12, 8, self.lora.mm_y_any, torch.float32, None)
            self.assertEqual(len(calls), 1)
        self.assertTrue(bits_equal(got, ref))


if __name__ == "__main__":
    unittest.main()
