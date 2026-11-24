"""The fused vocabulary draw (`y_lang.exact.vocab_draw`) against `draw_rows`'
torch code, bit for bit, on a GPU: token, p of it and l, for vocabularies
of any length against the kernel's block, logits over every range (where t
clamps and p is 0), ties, a draw on every kind of block boundary; a logit
that is not finite refused; and `draw_rows` choosing it."""

import os
import sys
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))

GPU = torch.cuda.is_available()


@unittest.skipUnless(GPU, "needs a CUDA GPU")
class FusedDraw(unittest.TestCase):
    def setUp(self):
        from y_lang.exact import rl, train_lm, vocab_draw
        self.rl, self.T, self.VD = rl, train_lm, vocab_draw
        self.ops = train_lm.Ops(act=torch.float32)
        self.g = torch.Generator(device="cuda").manual_seed(0)

    def torch_draw(self, z, r):
        old = self.VD.FUSED
        self.VD.FUSED = False
        try:
            return self.rl.draw_rows(self.ops, z, r, 64)
        finally:
            self.VD.FUSED = old

    def check(self, z, r, what):
        want = self.torch_draw(z, r)
        got = self.rl.draw_rows(self.ops, z, r, 64)
        for name, a, b in zip(("token", "p", "l"), want, got):
            self.assertEqual(a.dtype, b.dtype, f"{what}: {name}")
            self.assertTrue(torch.equal(a, b), f"{what}: {name} differs in {int((a != b).sum())} rows")

    def rand_r(self, rows):
        return torch.randint(0, 2 ** 62, (rows,), device="cuda", generator=self.g, dtype=torch.int64)

    def test_lengths_and_ranges(self):
        B = self.VD.BLOCK
        for V in (1, 5, B - 1, B, B + 1, 3 * B, 128256, 151936):
            for spread in (0.01, 1.0, 4.0, 40.0, 3000.0):
                z = torch.randn(6, V, device="cuda", generator=self.g) * spread
                self.check(z, self.rand_r(6), f"V {V}, spread {spread}")

    def test_ties(self):
        V = 5000
        z = torch.randn(5, V, device="cuda", generator=self.g)
        z[0] = 1.5                                   # all equal: uniform
        z[1, ::7] = z[1].max()                       # the maximum many times
        z[2] = torch.round(z[2] * 2) / 2             # many equal values
        z[3] = -1e30                                 # very negative, equal
        z[4, 1:] = -1e30                             # one logit, the rest p = 0
        for _ in range(4):
            self.check(z, self.rand_r(5), "ties")

    def test_draws_on_block_boundaries(self):
        """r mod l at, below and above every block's running sum and at
        running sums inside blocks, so the first block (and entry) whose
        running sum passes u is decided at each edge."""
        B = self.VD.BLOCK
        rows, V = 4, 6 * B + 17
        z = torch.randn(rows, V, device="cuda", generator=self.g) * 3
        p, l = self.ops.softmax_ints(z, torch.zeros_like(z, dtype=torch.bool))
        pad = torch.nn.functional.pad(p, (0, 7 * B - V))
        block_cum = torch.cumsum(pad.view(rows, 7, B).sum(-1), dim=-1)          # [rows][7]
        inner = torch.cumsum(p, dim=-1)[:, [0, 1, B - 1, B, B + 1, 3 * B + 5, V - 2, V - 1]]
        edges = torch.cat([block_cum, inner, torch.zeros_like(block_cum[:, :1])], dim=1)
        for col in range(edges.shape[1]):
            for d in (-1, 0, 1):
                r = torch.remainder(edges[:, col] + d, l.view(-1))
                self.check(z, r, f"edge {col}, {d:+d}")

    def test_not_finite_is_refused(self):
        z = torch.randn(4, 3000, device="cuda", generator=self.g)
        r = self.rand_r(4)
        bad = []
        self.rl.draw_rows(self.ops, z, r, 64)                       # finite: nothing refused
        for v in (float("inf"), float("-inf"), float("nan")):
            y = z.clone()
            y[2, 2999] = v
            with self.assertRaises(OverflowError, msg=str(v)):
                self.VD.draw(y, r, self.ops.table(y.device), None)
            bad = []
            self.VD.draw(y, r, self.ops.table(y.device), bad)
            self.assertEqual(len(bad), 1)
            self.assertEqual(bad[0][0].reshape(-1).tolist(), [False, False, True, False], str(v))
        bad = []
        self.VD.draw(z, r, self.ops.table(z.device), bad)
        self.assertFalse(any(bool(c.any()) for c, _ in bad))

    def test_draw_rows_takes_it_on_a_gpu(self):
        from unittest import mock
        calls = []
        real = self.VD.draw

        def spy(*a):
            calls.append(1)
            return real(*a)
        z = torch.randn(3, 2000, device="cuda", generator=self.g)
        with mock.patch.object(self.VD, "draw", spy):
            self.rl.draw_rows(self.ops, z, self.rand_r(3), 64)
            self.assertEqual(len(calls), 1)
            self.torch_draw(z, self.rand_r(3))
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
