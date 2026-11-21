"""y_lang.exact.train (T1): one linear layer trained with integer sums gives
the same weights in any reduction order and on any device; its float32 twin,
the control, does not."""

import unittest

import torch

from y_lang.exact import train as T


def _cuda_ok():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (7, 5)


class Arithmetic(unittest.TestCase):
    def test_bit_length(self):
        m = torch.tensor([0, 1, 2, 3, 127, 128, (1 << 62) - 1, 1 << 62], dtype=torch.int64)
        self.assertEqual(T.bit_length(m).tolist(), [0, 1, 2, 2, 7, 8, 62, 63])

    def test_shift_rounds_to_nearest_ties_up(self):
        v = torch.tensor([5, -5, 6, -6, 7, -7, 3], dtype=torch.int64)
        self.assertEqual(T.shift(v, -1).tolist(), [3, -2, 3, -3, 4, -3, 2])
        self.assertEqual(T.shift(v, 2).tolist(), [20, -20, 24, -24, 28, -28, 12])
        self.assertEqual(T.shift(v, 0).tolist(), v.tolist())

    def test_shift_refuses_to_wrap(self):
        with self.assertRaises(OverflowError):
            T.shift(torch.tensor([1 << 40], dtype=torch.int64), 30)

    def test_quantize_rows_power_of_two_scale(self):
        g = torch.Generator().manual_seed(1)
        v = torch.randint(-(1 << 40), 1 << 40, (16, 64), generator=g, dtype=torch.int64)
        v[3] = 0
        v[4, :] = 127
        q, s = T.quantize_rows(v)
        self.assertEqual(q.dtype, torch.int8)
        self.assertLessEqual(int(q.abs().max()), 127)
        back = torch.bitwise_left_shift(q.to(torch.int64), s)
        # Rounding error at most half a step, except a clamped 128 (one step).
        step = torch.bitwise_left_shift(torch.ones_like(s), s)
        self.assertTrue(bool(((back - v).abs() <= step).all()))
        self.assertEqual(int(s[3]), 0)
        self.assertEqual(int(s[4]), 0)
        self.assertTrue(bool((q[4] == 127).all()))
        # The shift is the least that fits: one less would overflow int8.
        big = v.abs().amax(dim=1)
        for r in range(16):
            if int(s[r]) > 0:
                self.assertGreaterEqual(int(big[r]) >> (int(s[r]) - 1), 128)


class Training(unittest.TestCase):
    STEPS = 120

    @classmethod
    def setUpClass(cls):
        cls.base = T.LinearRun.synthetic(seed=0)
        run = cls.base.to("cpu")
        cls.losses = run.train(cls.STEPS - 2, T.mm_cpu)
        cls.before_last = run.digest()
        cls.losses += run.train(2, T.mm_cpu)
        cls.ref = run

    def test_it_trains_and_keeps_moving(self):
        # Not a run that does nothing: the loss falls by orders of magnitude,
        # never reaches 0 (the targets are not fittable), and the last steps
        # still change the weights, so two runs that agree agree on work.
        self.assertLess(self.losses[-1], self.losses[0] / 1000)
        self.assertGreater(self.losses[-1], 0.0)
        self.assertNotEqual(self.before_last, self.ref.digest())

    def test_chunk_order_does_not_change_exact_weights(self):
        for chunks, order in ((4, [2, 0, 3, 1]), (16, list(range(15, -1, -1))), (8, [5, 1, 7, 0, 3, 6, 2, 4])):
            run = self.base.to("cpu")
            losses = run.train(self.STEPS, T.mm_cpu, chunks, order)
            self.assertEqual(run.digest(), self.ref.digest(), (chunks, order))
            self.assertEqual(losses, self.losses)

    def test_float_twin_depends_on_chunk_order(self):
        # The control: the same split in float32 moves the weights.
        a = T.FloatRun.like(self.base)
        a.train(self.STEPS)
        b = T.FloatRun.like(self.base)
        b.train(self.STEPS, chunks=4, order=[2, 0, 3, 1])
        self.assertNotEqual(a.digest(), b.digest())

    def test_refuses_a_bad_split(self):
        with self.assertRaises(ValueError):
            self.base.to("cpu").step(T.mm_cpu, chunks=3)
        with self.assertRaises(ValueError):
            self.base.to("cpu").step(T.mm_cpu, chunks=4, order=[0, 0, 1, 2])


@unittest.skipUnless(_cuda_ok(), "needs a CUDA GPU (sm_75 or later)")
class OnTheGpu(unittest.TestCase):
    STEPS = 60

    @classmethod
    def setUpClass(cls):
        torch.backends.cuda.matmul.allow_tf32 = False
        cls.base = T.LinearRun.synthetic(seed=3)
        cls.ref = cls.base.to("cpu")
        cls.losses = cls.ref.train(cls.STEPS, T.mm_cpu)

    def _same_as_cpu(self, mm, chunks=1, order=None):
        run = self.base.to("cuda")
        losses = run.train(self.STEPS, mm, chunks, order)
        self.assertEqual(run.digest(), self.ref.digest())
        self.assertEqual(losses, self.losses)
        self.assertTrue(torch.equal(run.last_dx.cpu(), self.ref.last_dx))

    def test_y_gemm_gives_the_cpu_weights(self):
        self._same_as_cpu(T.mm_y)
        self._same_as_cpu(T.mm_y, 4, [3, 1, 0, 2])

    def test_cublas_gives_the_cpu_weights(self):
        self._same_as_cpu(T.mm_torch)
        self._same_as_cpu(T.mm_torch, 8, list(range(7, -1, -1)))

    def test_float_twin_differs_across_devices(self):
        a = T.FloatRun.like(self.base, "cpu")
        a.train(self.STEPS)
        b = T.FloatRun.like(self.base, "cuda")
        b.train(self.STEPS)
        self.assertNotEqual(a.digest(), b.digest())


if __name__ == "__main__":
    unittest.main()
