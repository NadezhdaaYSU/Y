"""y_lang.exact.int8_bmm: exact batched int8 products, the integers a CPU and
the float32 product give; exact training with them gives the CPU's weights."""

import os
import subprocess
import sys
import unittest

import torch

PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

INTERPRETED = r"""
import torch
from y_lang.exact import int8_bmm as I
g = torch.Generator().manual_seed(0)
for G, M, N, K in ((3, 200, 72, 64), (2, 64, 64, 256), (1, 130, 33, 96), (2, 512, 256, 64)):
    a = torch.randint(-128, 128, (G, M, K), dtype=torch.int8, generator=g)
    b = torch.randint(-128, 128, (G, N, K), dtype=torch.int8, generator=g)
    a[0, 0] = -128; b[0, 0] = -128; a[0, 1] = 127; b[0, 1] = -128
    c = torch.empty(G, M, N, dtype=torch.int32)
    I.launch(a, b, c)
    assert torch.equal(c.long(), a.long() @ b.long().transpose(1, 2)), (G, M, N, K)
print("exact")
"""


def _gpu_ok():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0)


class OnTheCpu(unittest.TestCase):
    def test_the_kernel_is_exact_under_the_interpreter(self):
        # Triton's interpreter runs the same kernel on the CPU: tiles past the
        # edges, K over several blocks, -128 * -128 at the corners.
        env = dict(os.environ, TRITON_INTERPRET="1", PYTHONPATH=PY)
        out = subprocess.run([sys.executable, "-c", INTERPRETED], env=env, capture_output=True,
                             text=True, timeout=900)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        self.assertIn("exact", out.stdout)

    def test_a_k_past_int32_is_refused(self):
        from y_lang.exact import int8_bmm as I
        a = torch.zeros(1, 1, I.MAX_K + 1, dtype=torch.int8)
        with self.assertRaises(OverflowError):
            I.check_operands(a, a)
        I.check_operands(a[..., :I.MAX_K], a[..., :I.MAX_K])


@unittest.skipUnless(_gpu_ok(), "needs a CUDA GPU of sm_80 or later")
class OnTheGpu(unittest.TestCase):
    def test_the_same_integers_as_the_float32_product(self):
        from y_lang.exact import train_lm as L
        from y_lang.exact.int8_bmm import int8_bmm
        torch.backends.cuda.matmul.allow_tf32 = False
        g = torch.Generator(device="cuda").manual_seed(1)
        for shape_a, shape_b in (((16, 4, 64, 32), (16, 4, 64, 32)), ((16, 4, 128, 64), (16, 4, 32, 64)),
                                 ((2, 8, 512, 256), (2, 8, 64, 256))):
            a = torch.randint(-127, 128, shape_a, device="cuda", generator=g, dtype=torch.int32)
            b = torch.randint(-127, 128, shape_b, device="cuda", generator=g, dtype=torch.int32)
            got = int8_bmm(a.to(torch.int8), b.to(torch.int8)).long()
            self.assertTrue(torch.equal(got, L._exact_fp32_bmm(a, b)), shape_a)
            self.assertTrue(torch.equal(got.cpu(), a.cpu().long() @ b.cpu().long().transpose(-1, -2)))

    def test_training_with_it_gives_the_cpu_weights(self):
        import test_exact_train_lm as T
        from y_lang.exact import train_lm as L
        torch.backends.cuda.matmul.allow_tf32 = False
        for act in (None, torch.float32):
            ref, losses = T.run(L.Ops(exact=True, act=act), "cpu", torch.float64)
            for bmm in ("int8", "fp32"):
                params, gl = T.run(L.Ops(exact=True, mm=L.mm_y, act=act, bmm=bmm), "cuda", torch.float64)
                self.assertEqual(gl, losses, (act, bmm))
                self.assertEqual(L.digest(params), L.digest(ref), (act, bmm))


if __name__ == "__main__":
    unittest.main()
