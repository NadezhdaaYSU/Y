"""y_lang.exact.attn_weights_kernel: the attention softmax's integers p and l,
formed in double float32 with a guard and a float64 fallback, are
`attention.exact_softmax_weights`' integers: exact ties, near ties, blocked
rows and random rows, under Triton's interpreter on the CPU and on a GPU;
the guard fires on near ties; on a GPU exact_softmax_weights gives the same
p, l with the kernel and without it, and compiled training calls it and
gives the CPU's adapters and p(target)s with it and without it. Other
dtypes and mask shapes keep the Python path, and without Triton the module
still imports."""

import os
import subprocess
import sys
import unittest
from unittest import mock

import torch

PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

COMPARE = r"""
import math
import torch
from y_lang.exact import attention as A
from y_lang.exact import attn_weights_kernel as K

dev = DEVICE
table = A._table(dev)

def reference(s, a, sk, mask, Q, NH):
    # exact_softmax_weights' arithmetic on the flattened operands, in torch.
    R, T = s.shape
    logit = s.to(torch.float64) * a[:, None] * sk.to(torch.float64).repeat_interleave(Q, dim=0)
    if mask is not None:
        rows = torch.arange(R, device=s.device)
        blocked = mask[(rows // (NH * Q)) * Q + rows % Q] < -1.0
    else:
        blocked = torch.zeros_like(logit, dtype=torch.bool)
    m = torch.where(blocked, torch.full_like(logit, torch.finfo(torch.float64).min), logit).amax(-1, keepdim=True)
    tq = torch.round(m - logit)
    tq = torch.where(blocked, torch.full_like(tq, 1 << 30), tq).clamp(0, 1 << 30)
    p = A.exp2_neg_q16_16(tq, table)
    p = torch.where(blocked, torch.zeros_like(p), p)
    return p, p.sum(-1)

def check(name, s, a, sk, mask, Q, NH, BR=1, want_fallback=None):
    s, a, sk = s.to(dev), a.to(dev), sk.to(dev)
    mask = None if mask is None else mask.to(dev)
    p, l, fb = K.launch(s, a, sk, mask, table, Q, NH, A.LN2_Q30, A.RECIP6_Q32, BR=BR)
    pr, lr = reference(s, a, sk, mask, Q, NH)
    assert torch.equal(p, pr) and torch.equal(l, lr), name
    p64, l64, fb64 = K.launch(s, a, sk, mask, table, Q, NH, A.LN2_Q30, A.RECIP6_Q32, force64=True, BR=BR)
    assert torch.equal(p64, pr) and torch.equal(l64, lr), name + " (float64 path)"
    assert int(fb64.sum()) == s.shape[0]
    if want_fallback is not None:
        assert (int(fb.sum()) > 0) == want_fallback, (name, int(fb.sum()))

g = torch.Generator().manual_seed(1)
R, T = 32, 16
s = torch.randint(-2000, 2000, (R, T), generator=g).float()
check("exact half-integer ties", s, torch.full((R,), 0.5, dtype=torch.float64), torch.ones(R, T), None, 1, R,
      want_fallback=True)
for e in (2.0 ** -30, -2.0 ** -24, 2.0 ** -22):
    check(f"near ties {e}", s, torch.full((R,), 0.5 * (1 + e), dtype=torch.float64), torch.ones(R, T), None, 1, R,
          want_fallback=True)
Q, NH, T = 4, 2, 6
s = torch.randint(-10 ** 6, 10 ** 6, (2 * NH * Q, T), generator=g).float()
a = torch.rand(2 * NH * Q, generator=g, dtype=torch.float64) * 1e-3
sk = torch.rand(2 * NH, T, generator=g).float() * 0.02
mask = torch.zeros(2 * Q, T)
mask[Q:] = -1e9                                          # the second batch row fully blocked
mask[:Q] = torch.triu(torch.full((Q, T), -1e9), diagonal=T - Q + 1)
check("a causal mask and a fully blocked row", s, a, sk, mask, Q, NH)
s = torch.randint(-10 ** 6, 10 ** 6, (8, 5), generator=g).float()
check("huge logits take the float64 path", s, torch.full((8,), 64.0, dtype=torch.float64),
      torch.ones(8, 5), None, 1, 8, want_fallback=True)
R, T = 512 if dev == "cpu" else 8192, 64
s = torch.randint(-1032256, 1032257, (R, T), generator=g).float()
a = (torch.rand(R, generator=g, dtype=torch.float64) * 0.02 + 1e-5) / 8 * math.log2(math.e) * 65536
sk = torch.rand(R, T, generator=g).float() * 0.03 + 1e-5
check("random rows at the real scales", s, a, sk, None, 1, R, BR=4)
print("same bits")
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

    def test_cpu_tensors_keep_the_python_path(self):
        from y_lang.exact import attn_weights_kernel as K
        self.assertFalse(K.usable(torch.zeros(1, 2, 3, 4), None))

    def test_without_triton_attention_still_imports(self):
        code = ("import sys; sys.modules['triton'] = None\n"
                "import torch\n"
                "from y_lang.exact import attention, attn_weights_kernel as K\n"
                "assert K.triton is None and K._OP is None\n"
                "assert not K.usable(torch.zeros(1, 2, 3, 4), None)\n"
                "print('imports')\n")
        out = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, PYTHONPATH=PY),
                             capture_output=True, text=True, timeout=600)
        self.assertEqual(out.returncode, 0, out.stderr[-3000:])
        self.assertIn("imports", out.stdout)


@unittest.skipUnless(_gpu_ok(), "needs a CUDA GPU")
class OnTheGpu(unittest.TestCase):
    def test_the_kernel_is_the_python_on_the_gpu(self):
        env = dict(os.environ, PYTHONPATH=PY)
        out = subprocess.run([sys.executable, "-c", COMPARE.replace("DEVICE", "'cuda'")], env=env,
                             capture_output=True, text=True, timeout=1500)
        self.assertEqual(out.returncode, 0, out.stderr[-3000:])
        self.assertIn("same bits", out.stdout)

    def test_other_dtypes_and_shapes_keep_the_python_path(self):
        from y_lang.exact import attn_weights_kernel as K
        s = torch.zeros(1, 2, 3, 4, device="cuda")
        m = torch.zeros(1, 1, 3, 4, device="cuda")
        sk = torch.zeros(1, 2, 1, 4, device="cuda")
        self.assertTrue(K.usable(s, m, sk))
        self.assertTrue(K.usable(s, m.half(), sk.bfloat16()))
        self.assertFalse(K.usable(s.double(), m, sk))
        self.assertFalse(K.usable(s, m.double(), sk))
        self.assertFalse(K.usable(s, m, sk.double()))
        self.assertFalse(K.usable(s, m[:, :, :2], sk))
        self.assertFalse(K.usable(s, torch.zeros(1, 2, 3, 4, device="cuda"), sk))
        self.assertFalse(K.usable(torch.zeros(1, 2, 3, K.MAX_T + 1, device="cuda"), None, None))

    def test_exact_softmax_weights_gives_the_same_integers_with_and_without_it(self):
        import math
        from y_lang.exact import attention as A
        from y_lang.exact import attn_weights_kernel as K
        from y_lang.exact import kv as KV
        g = torch.Generator().manual_seed(2)
        b, nh, nkv, q, t, d = 2, 6, 2, 9, 9, 64
        qf = torch.randn(b, nh, q, d, generator=g).cuda()
        kf = torch.randn(b, nkv, t, d, generator=g).cuda()
        vf = torch.randn(b, nkv, t, d, generator=g).cuda()
        qi, sq = A.quantize_queries(qf)
        ki, sk, vi, sv, k_lv, _ = KV.quantize_kv(None, kf, vf, A.Q_LEVELS)
        mask = torch.triu(torch.full((q, t), -1e9, device="cuda"), diagonal=1)[None, None].expand(b, 1, q, t)
        table = A._table("cuda")
        self.assertTrue(K.usable(torch.zeros(b, nh, q, t, device="cuda"), mask))
        sc = 1 / math.sqrt(d)
        p1, l1, v1 = A.exact_softmax_weights(qi, sq, ki, sk, sv, k_lv, mask, sc, table)
        try:
            K.ENABLED = False
            p0, l0, v0 = A.exact_softmax_weights(qi, sq, ki, sk, sv, k_lv, mask, sc, table)
        finally:
            K.ENABLED = True
        self.assertTrue(torch.equal(p1, p0))
        self.assertTrue(torch.equal(l1, l0))
        self.assertTrue(torch.equal(v1, v0))

    def test_training_calls_it_and_is_the_cpus(self):
        # Compiled training (the layer path, where Inductor traces the custom
        # op): its adapters and p(target)s are the CPU's, with the kernel and
        # without it, and a profile shows the kernel ran.
        from test_exact_lora import train_tiny
        from y_lang.exact import attn_weights_kernel as K
        from y_lang.exact import lora
        with mock.patch.object(lora, "LAYER_FUSED", True):
            torch._dynamo.reset()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                on = train_tiny("cuda", steps=3, compile=True, lengths=(30, 27))
            calls = sum(e.count for e in prof.key_averages() if e.key == "y_lang::exact_attn_p")
            self.assertGreater(calls, 0, "training did not call the kernel")
            with mock.patch.object(K, "ENABLED", False):
                torch._dynamo.reset()
                off = train_tiny("cuda", steps=3, compile=True, lengths=(30, 27))
        cpu = train_tiny("cpu", steps=3, lengths=(30, 27))
        self.assertEqual(on[:2], cpu[:2])
        self.assertEqual(off[:2], cpu[:2])



if __name__ == "__main__":
    unittest.main()
