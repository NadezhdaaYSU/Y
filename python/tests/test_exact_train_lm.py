"""y_lang.exact.train_lm (T2): the tiny character model's hand-written
backward matches autograd; trained with exact sums, its weights do not
depend on the order of the batch's rows or on the device; the float32 twin,
the control, does."""

import unittest

import torch

from y_lang.exact import train_lm as L

CFG = L.Config(vocab=96, d=64, heads=2, layers=1, ff=128, ctx=32)
TEXT = (L.__doc__ or "") * 12


def _cuda_ok():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (7, 5)


def run(ops, device, dtype, steps=12, permute=False, cfg=CFG, B=4):
    """Training as `train_lm.train` does it; with `permute`, each batch's
    rows are reversed, which reorders every sum over tokens."""
    data = L.CharData(TEXT)
    params = L.init_params(cfg, 0, dtype, device)
    opt = L.Adam(params, lr=3e-3, exact=ops.exact)
    gen = torch.Generator().manual_seed(1)
    losses = []
    for _ in range(steps):
        x, y = data.batch(gen, B, cfg.ctx)
        if permute:
            x, y = x.flip(0), y.flip(0)
        x, y = x.to(device), y.to(device)
        _, probs, saved = L.forward(ops, cfg, params, x, y)
        losses.append(L.cross_entropy_on_host(probs, y))
        opt.step(params, L.backward(ops, cfg, params, probs, y, saved))
    return params, losses


class Backward(unittest.TestCase):
    def test_matches_autograd(self):
        cfg = L.Config(vocab=32, d=64, heads=2, layers=2, ff=128, ctx=16)
        ops = L.Ops(exact=False)
        p = L.init_params(cfg, seed=0)
        for v in p.values():
            v.requires_grad_(True)
        g = torch.Generator().manual_seed(5)
        x = torch.randint(0, 32, (3, 16), generator=g)
        y = torch.randint(0, 32, (3, 16), generator=g)
        _, probs, saved = L.forward(ops, cfg, p, x, y)
        loss = -torch.log(probs[torch.arange(48), y.reshape(-1)]).mean()
        auto = dict(zip(p, torch.autograd.grad(loss, list(p.values()))))
        with torch.no_grad():
            man = L.backward(ops, cfg, p, probs.detach(), y, saved)
        self.assertEqual(set(man), set(p))
        for k in p:
            rel = float((man[k] - auto[k]).abs().max() / auto[k].abs().max())
            self.assertLess(rel, 1e-12, k)


class Pieces(unittest.TestCase):
    def test_pow2_is_exact(self):
        k = torch.tensor([-1022, -7, 0, 1, 52, 1023])
        self.assertEqual(L.pow2(k).tolist(), [2.0 ** int(e) for e in k])
        with self.assertRaises(OverflowError):
            L.pow2(torch.tensor([1024]))

    def test_isqrt_is_the_floor_square_root(self):
        import math
        g = torch.Generator().manual_seed(6)
        n = torch.cat([torch.tensor([0, 1, 2, 3, 4, 15, 16, (1 << 62) - 1, (1 << 61) + 12345]),
                       torch.randint(0, 1 << 62, (2000,), generator=g)])
        for v, r in zip(n.tolist(), L.isqrt(n).tolist()):
            self.assertEqual(r, math.isqrt(v))
        with self.assertRaises(OverflowError):
            L.isqrt(torch.tensor([1 << 62]))

    def test_det_sqrt_is_close_and_scaled_exactly(self):
        import math
        v = torch.cat([torch.tensor([0.0, 1e-280, 1e-8, 0.25, 1.0, 2.0, 3.0, 1e300], dtype=torch.float64),
                       torch.rand(2000, generator=torch.Generator().manual_seed(7), dtype=torch.float64) * 1e4])
        s = L.det_sqrt(v)
        for x, r in zip(v.tolist(), s.tolist()):
            self.assertLessEqual(abs(r - math.sqrt(x)), math.sqrt(x) * 2.0 ** -29)
        self.assertEqual(L.det_sqrt(torch.tensor([4.0, 0.25], dtype=torch.float64)).tolist(), [2.0, 0.5])
        with self.assertRaises(OverflowError):      # refused, not wrong
            L.det_sqrt(torch.tensor([1e-300], dtype=torch.float64))

    def test_exact_sum_ignores_order(self):
        g = torch.Generator().manual_seed(2)
        x = torch.randn(7, 1000, generator=g, dtype=torch.float64) * torch.logspace(-8, 8, 1000, dtype=torch.float64)
        ops = L.Ops(exact=True)
        perm = torch.randperm(1000, generator=g)
        self.assertTrue(torch.equal(ops.sum(x, -1), ops.sum(x[:, perm], -1)))
        # And it is a sum: close to the float64 one.
        self.assertTrue(torch.allclose(ops.sum(x, -1), x.sum(-1, keepdim=True), rtol=1e-12, atol=0))

    def test_exact_softmax_rows_sum_to_one(self):
        ops = L.Ops(exact=True)
        z = torch.randn(5, 33, generator=torch.Generator().manual_seed(3), dtype=torch.float64) * 4
        blocked = torch.zeros_like(z, dtype=torch.bool)
        blocked[:, -3:] = True
        p = ops.softmax(z, blocked)
        self.assertTrue(bool((p[:, -3:] == 0).all()))
        self.assertTrue(torch.allclose(p.sum(-1), torch.ones(5, dtype=torch.float64), atol=1e-12))
        ref = torch.softmax(z.masked_fill(blocked, float("-inf")), -1)
        self.assertLess(float((p - ref).abs().max()), 1e-6)

    def test_matmul_is_the_product_of_the_int8_copies(self):
        ops = L.Ops(exact=True)
        g = torch.Generator().manual_seed(4)
        a = torch.randn(8, 64, generator=g, dtype=torch.float64)
        b = torch.randn(16, 64, generator=g, dtype=torch.float64)
        c = ops.matmul(a, b)
        rel = float((c - a @ b.t()).abs().max() / (a @ b.t()).abs().max())
        self.assertLess(rel, 0.05)          # int8 operands: a few percent
        qa, sa = L.q8(a)
        qb, sb = L.q8(b)
        ref = torch.tensor([[sum(int(qa[i, k]) * int(qb[j, k]) for k in range(64)) * 2.0 ** (int(sa[i]) + int(sb[j]))
                             for j in range(16)] for i in range(8)], dtype=torch.float64)
        self.assertTrue(torch.equal(c, ref))


class Training(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params, cls.losses = run(L.Ops(exact=True), "cpu", torch.float64)

    def test_it_trains(self):
        self.assertLess(self.losses[-1], self.losses[0] - 0.5)

    def test_row_order_does_not_change_exact_weights(self):
        params, losses = run(L.Ops(exact=True), "cpu", torch.float64, permute=True)
        self.assertEqual(L.digest(params), L.digest(self.params))
        self.assertEqual(losses, self.losses)

    def test_float_twin_depends_on_row_order(self):
        a, _ = run(L.Ops(exact=False), "cpu", torch.float32)
        b, _ = run(L.Ops(exact=False), "cpu", torch.float32, permute=True)
        self.assertNotEqual(L.digest(a), L.digest(b))


@unittest.skipUnless(_cuda_ok(), "needs a CUDA GPU (sm_75 or later)")
class OnTheGpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.backends.cuda.matmul.allow_tf32 = False
        cls.ref, cls.losses = run(L.Ops(exact=True), "cpu", torch.float64)

    def _same_as_cpu(self, mm):
        params, losses = run(L.Ops(exact=True, mm=mm), "cuda", torch.float64)
        self.assertEqual(losses, self.losses)
        self.assertEqual(L.digest(params), L.digest(self.ref))

    def test_y_gemm_gives_the_cpu_weights(self):
        self._same_as_cpu(L.mm_y)

    def test_cublas_gives_the_cpu_weights(self):
        self._same_as_cpu(L.mm_torch)

    def test_float_twin_differs_across_devices(self):
        a, _ = run(L.Ops(exact=False), "cpu", torch.float32)
        b, _ = run(L.Ops(exact=False), "cuda", torch.float32)
        self.assertNotEqual(L.digest(a), L.digest(b))

    def test_det_sqrt_same_bits_on_both(self):
        # torch.sqrt's float64 differs between this CPU and CUDA (2026-11-21);
        # det_sqrt must not.
        v = torch.rand(100000, generator=torch.Generator().manual_seed(8), dtype=torch.float64) * 10 + 0.01
        self.assertTrue(torch.equal(L.det_sqrt(v), L.det_sqrt(v.cuda()).cpu()))

    def test_scalar_divide_is_written_as_a_multiply(self):
        # Why: PyTorch's CUDA divide by a Python scalar multiplies by its
        # reciprocal, which the CPU does not (2026-11-21).
        import inspect
        import re
        src = inspect.getsource(L)
        self.assertIn("* (1.0 / d)", src)
        self.assertIsNone(re.search(r"(?<!1\.0) / (d|n|c1|c2)\b", src))


if __name__ == "__main__":
    unittest.main()
