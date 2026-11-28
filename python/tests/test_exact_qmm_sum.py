"""`qmm.qmm_sum`, LoRA's dx in one kernel (the frozen weight's product plus
the adapter's, each rounded to float32 as `qmm` rounds it, added in
float32), against two `qmm` calls and torch's add and against
`exact_product`, bit for bit: the training shapes, 8- and 12-bit operands,
zero rows, and scales that send either product through qmm's float64
branch; under Triton's interpreter on the CPU and on a GPU. Then
`lora._lora_backward` gives the same dx, dA, dB and checks with it and
without it, and compiled training calls it and gives the CPU's adapters and
p(target)s with it and without it."""

import os
import subprocess
import sys
import unittest
from unittest import mock

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, "..")
sys.path.insert(0, HERE)

COMPARE = r"""
import torch
from y_lang.exact import qmm as Q, train_lm as T, lora
dev = DEVICE
g = torch.Generator(device=dev).manual_seed(0)
mm = lora.mm_y_any if dev == "cuda" else lora.default_mm("cpu")

def ops(M, N, K, R, sa_shift, sc_shift):
    a = torch.randn(M, K, generator=g, device=dev, dtype=torch.float64)
    a = a * torch.exp2(torch.randint(-20, 20, (M, 1), generator=g, device=dev)).double() * 2.0 ** sa_shift
    b = torch.randint(-127, 128, (N, K), generator=g, device=dev).to(torch.int8)
    c = torch.randn(M, R, generator=g, device=dev, dtype=torch.float64) * 2.0 ** sc_shift
    d = torch.randn(N, R, generator=g, device=dev, dtype=torch.float64) * 0.03
    a[0] = 0
    c[min(1, M - 1)] = 0
    if M > 2:
        a[2, 0] = a[2].abs().max() * 4             # a value at the clamp
    return a.float(), b, c.float(), d

def bits(t):
    return t.contiguous().view(torch.int32)

cases = CASES
for M, N, K, R, sa_s, sc_s in cases:
    a, b, c, d = ops(M, N, K, R, sa_s, sc_s)
    for bits_a, bits_c in ((12, 12), (8, 12), (12, 8)):
        qa, sa = T.q8(a, bits=bits_a)
        sb = torch.zeros(N, 1, dtype=sa.dtype, device=dev)
        qc, sc = T.q8(c, bits=bits_c)
        qd, sd = T.q8(d, bits=8)
        if dev == "cuda":
            got = Q.qmm_sum(qa, sa, b, sb, bits_a, 8, qc, sc, qd, sd, bits_c, 8)
            sep = Q.qmm(qa, sa, b, sb, bits_a, 8) + Q.qmm(qc, sc, qd, sd, bits_c, 8)
        else:
            got = Q._qmm_sum(qa, sa.reshape(-1), b, sb.reshape(-1), bits_a > 8,
                             qc, sc.reshape(-1), qd, sd.reshape(-1), bits_c > 8)
            sep = (Q._qmm(qa, sa.reshape(-1), b, sb.reshape(-1), bits_a > 8)
                   + Q._qmm(qc, sc.reshape(-1), qd, sd.reshape(-1), bits_c > 8))
        ref = (T.exact_product(a, b, bits_a, 8, mm, False, False, torch.float32, None, (b, sb))
               + T.exact_product(c, d, bits_c, 8, mm, False, False, torch.float32, None, None))
        case = (M, N, K, R, sa_s, sc_s, bits_a, bits_c)
        assert torch.equal(bits(got), bits(sep)), (case, "against the two calls", int((got != sep).sum()))
        assert torch.equal(bits(got), bits(ref)), (case, "against exact_product", int((got != ref).sum()))
print("same bits")
"""

# (M, N, K, R, the first operand's scale shift, the second's): a shift
# of -120, 100 or -140 sends that product's tiles through the float64 branch.
SMALL = [(5, 7, 40, 16, 0, 0), (33, 20, 96, 16, 0, 0), (17, 48, 130, 8, 0, 0),
         (9, 16, 64, 16, -120, 0), (9, 16, 64, 16, 0, -140), (12, 24, 64, 16, 100, 0), (1, 16, 16, 16, 0, 0)]
TRAINING = SMALL + [(400, 4864, 896, 16, 0, 0), (400, 896, 4864, 16, 0, 0), (513, 896, 1152, 16, 0, 0),
                    (300, 1152, 896, 8, 0, 0), (257, 896, 896, 16, -110, -20)]


def _run(device, cases, interpret):
    env = dict(os.environ, PYTHONPATH=PY)
    if interpret:
        env["TRITON_INTERPRET"] = "1"
    code = COMPARE.replace("DEVICE", repr(device)).replace("CASES", repr(cases))
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=2400)


GPU = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0)


class OnTheCpu(unittest.TestCase):
    def test_the_kernel_is_the_two_products_under_the_interpreter(self):
        out = _run("cpu", SMALL, True)
        self.assertEqual(out.returncode, 0, out.stderr[-3000:])
        self.assertIn("same bits", out.stdout)


@unittest.skipUnless(GPU, "needs a CUDA GPU of sm_80 or later")
class OnTheGpu(unittest.TestCase):
    def test_the_kernel_is_the_two_products_on_the_gpu(self):
        out = _run("cuda", TRAINING, False)
        self.assertEqual(out.returncode, 0, out.stderr[-3000:])
        self.assertIn("same bits", out.stdout)

    def test_lora_backward_is_the_same_with_and_without_it(self):
        from y_lang.exact import lora
        torch.backends.cuda.matmul.allow_tf32 = False
        g = torch.Generator(device="cuda").manual_seed(4)
        M, K, N, r = 77, 96, 136, 16
        dy = torch.randn(M, N, generator=g, device="cuda")
        xf = torch.randn(M, K, generator=g, device="cuda")
        u = torch.randn(M, r, generator=g, device="cuda")
        A = torch.randn(r, K, generator=g, device="cuda", dtype=torch.float64) * 0.1
        B = torch.randn(N, r, generator=g, device="cuda", dtype=torch.float64) * 0.1
        sw = torch.rand(N, generator=g, device="cuda") * 0.01 + 1e-3
        wk = torch.randint(-127, 128, (K, N), generator=g, device="cuda").to(torch.int8)
        s0 = torch.zeros(K, 1, dtype=torch.int64, device="cuda")     # as Frozen._weight holds it
        args = (dy, xf, u, A, B, sw, wk, s0, 0.5, True, 12, 8, lora.mm_y_any, torch.float32)
        with mock.patch.object(lora, "QMM_SUM", True), \
                mock.patch.object(lora._qmm, "qmm_sum", wraps=lora._qmm.qmm_sum) as spy:
            on = lora._lora_backward(*args)
            self.assertEqual(spy.call_count, 1, "the backward did not take qmm_sum")
        with mock.patch.object(lora, "QMM_SUM", False):
            off = lora._lora_backward(*args)
        (dx1, dA1, dB1), c1, m1 = on
        (dx0, dA0, dB0), c0, m0 = off
        for x, y in ((dx1, dx0), (dA1, dA0), (dB1, dB0)):
            self.assertTrue(x.dtype == y.dtype and torch.equal(x, y))
        self.assertEqual(m1, m0)
        self.assertEqual(len(c1), len(c0))
        for x, y in zip(c1, c0):
            self.assertTrue(torch.equal(torch.as_tensor(x), torch.as_tensor(y)))

    def test_training_calls_it_and_is_the_cpus(self):
        from test_exact_lora import train_tiny
        from y_lang.exact import lora
        with mock.patch.object(lora, "LAYER_FUSED", True):
            torch._dynamo.reset()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                on = train_tiny("cuda", steps=3, compile=True, lengths=(30, 27))
            calls = sum(e.count for e in prof.key_averages() if e.key == "y_lang::exact_qmm_sum")
            self.assertGreater(calls, 0, "training did not call the kernel")
            with mock.patch.object(lora, "QMM_SUM", False):
                torch._dynamo.reset()
                off = train_tiny("cuda", steps=3, compile=True, lengths=(30, 27))
        cpu = train_tiny("cpu", steps=3, lengths=(30, 27))
        self.assertEqual(on[:2], cpu[:2])
        self.assertEqual(off[:2], cpu[:2])


if __name__ == "__main__":
    unittest.main()
