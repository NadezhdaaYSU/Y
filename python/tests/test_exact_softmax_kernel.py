"""y_lang.exact.softmax_kernel: the causal softmax and its quantizer in one
kernel give the bits of train_lm's Python path, under Triton's interpreter
on the CPU and on a GPU, and refuse what the Python refuses."""

import os
import subprocess
import sys
import unittest

import torch

PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

COMPARE = r"""
import torch
from y_lang.exact import train_lm as L
from y_lang.exact import softmax_kernel as K
from y_lang.exact.attention import _table

def cases(g):
    yield 2, 2, 64, torch.randn(2, 2, 64, 64, generator=g)
    yield 1, 2, 256, torch.randn(1, 2, 256, 256, generator=g)
    yield 1, 1, 256, torch.randn(1, 1, 256, 256, generator=g) * 400     # m - logit past 2^23
    yield 3, 1, 16, torch.randn(3, 1, 16, 16, generator=g)
    yield 1, 2, 64, torch.round(torch.randn(1, 2, 64, 64, generator=g) * 4) / 4   # ties
    yield 1, 1, 32, torch.randn(1, 1, 32, 32, generator=g) * 1e-30      # tiny scores

def compare(device):
    g = torch.Generator().manual_seed(0)
    table = _table(device)
    for B, H, T, S in cases(g):
        S = S.to(device)
        blocked = torch.ones(T, T, dtype=torch.bool, device=device).triu(1)
        P_ref = L.Ops(exact=True, act=torch.float32).softmax(S, blocked)
        q_ref, s_ref = L.q8(P_ref, bits=8)
        P, q, s, bad = K.launch(S.reshape(-1, T), T, table)
        assert torch.equal(P.view(torch.int32), P_ref.reshape(-1, T).contiguous().view(torch.int32)), (B, H, T)
        assert torch.equal(q.to(torch.int32), q_ref.reshape(-1, T)), (B, H, T)
        assert torch.equal(s, s_ref.reshape(-1, 1)), (B, H, T)
        assert int(bad.abs().sum()) == 0
    S = torch.randn(4, 64, generator=g).to(device)
    S[1, 3] = float("nan")
    S[2, 63] = float("inf")
    _, _, _, bad = K.launch(S, 64, table)
    # Bit 1, the softmax input refused, on exactly the rows holding a NaN or
    # an inf (what the rest of such a row computes is garbage, never read).
    assert (bad.reshape(-1) & 1).tolist() == [0, 1, 1, 0], bad.reshape(-1).tolist()
    print("same bits")

compare(DEVICE)
"""


def _gpu_ok():
    return torch.cuda.is_available()


class OnTheCpu(unittest.TestCase):
    def test_the_kernel_is_the_python_under_the_interpreter(self):
        env = dict(os.environ, TRITON_INTERPRET="1", PYTHONPATH=PY)
        out = subprocess.run([sys.executable, "-c", COMPARE.replace("DEVICE", "'cpu'")], env=env,
                             capture_output=True, text=True, timeout=1500)
        self.assertEqual(out.returncode, 0, out.stderr[-3000:])
        self.assertIn("same bits", out.stdout)

    def test_unsupported_rows_are_refused(self):
        from y_lang.exact import softmax_kernel as K
        self.assertFalse(K.supported(48))
        self.assertFalse(K.supported(2048))
        self.assertTrue(K.supported(256))
        with self.assertRaises(ValueError):
            K.launch(torch.zeros(4, 48), 48, None)


@unittest.skipUnless(_gpu_ok(), "needs a CUDA GPU")
class OnTheGpu(unittest.TestCase):
    def test_the_kernel_is_the_python_on_the_gpu(self):
        env = dict(os.environ, PYTHONPATH=PY)
        out = subprocess.run([sys.executable, "-c", COMPARE.replace("DEVICE", "'cuda'")], env=env,
                             capture_output=True, text=True, timeout=1500)
        self.assertEqual(out.returncode, 0, out.stderr[-3000:])
        self.assertIn("same bits", out.stdout)

    def test_training_uses_it_and_gives_the_cpu_weights(self):
        import test_exact_train_lm as T
        from y_lang.exact import train_lm as L
        torch.backends.cuda.matmul.allow_tf32 = False
        ref, losses = T.run(L.Ops(exact=True, act=torch.float32), "cpu", torch.float64)
        for fused in (None, False):
            params, gl = T.run(L.Ops(exact=True, mm=L.mm_y, act=torch.float32, fused_softmax=fused),
                               "cuda", torch.float64)
            self.assertEqual(gl, losses, fused)
            self.assertEqual(L.digest(params), L.digest(ref), fused)


if __name__ == "__main__":
    unittest.main()
