"""`mlp_kernel.gate_up_silu`, the MLP's gate and up projections and their glue
in one kernel, against the separate path it replaces (`lora._lora_forward`
for each projection, `ExactSiLU`, a * yu): every output bit for bit, the
adapters' intermediates and the checks equal, on shapes that do not divide
the tiles and on values that reach each branch (an all-zero row, inputs past
the sigmoid table's +-16 clamp, adapter scales whose product leaves float32's
normal range, so qmm's float64 branch runs). Then training: the layer path
calls the kernel, and its adapters and p(target)s are the CPU's, with the
kernel and without it."""

import os
import sys
import unittest
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _bits(t):
    return t.contiguous().view(torch.int32) if t.dtype == torch.float32 else t


def _same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(_bits(a), _bits(b))


class TheKernelsAssumptions(unittest.TestCase):
    """What `proofs/ExactMlpKernel.v` assumes of the code: the built sigmoid
    table is within its interpolation bounds (0 <= t <= 2^24, 0 <= d <= 2^14),
    has the TN >= 1 entries the index bound needs and the constants the
    kernel hard-codes; and `usable` refuses a K past the int32 bound."""

    def test_the_silu_table_meets_the_bounds(self):
        from y_lang.exact import mlp_kernel, model as M
        t, d = M.silu_table()
        self.assertEqual(t.numel(), d.numel())
        self.assertGreaterEqual(t.numel(), 1)
        self.assertEqual(t.numel(), (2 * M.SILU_LIM << M.SILU_KNOT) + 1)   # every index q >> 12 reaches
        self.assertTrue(bool((t >= 0).all()) and bool((t <= 1 << 24).all()))
        self.assertTrue(bool((d >= 0).all()) and bool((d <= 1 << 14).all()))
        mlp_kernel._check_constants()

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0),
                         "usable is False off such a GPU whatever K is")
    def test_k_past_the_bound_takes_the_separate_path(self):
        from y_lang.exact import lora, mlp_kernel, model as M
        w = (12, 8, lora.mm_y_any, torch.float32)
        x = torch.zeros(1, 1, 8, device="cuda")
        act = M.ExactSiLU()

        class Lin:                                    # only the fields `usable` reads
            out_features, split_act, bias = 8, False, None

            def __init__(self, k):
                self.in_features = k
        with mock.patch.object(M, "ExactLinear", Lin):
            # The control: the same modules at a K inside the bound are taken.
            self.assertTrue(mlp_kernel.usable(Lin(4096), Lin(4096), act, x, *w))
            self.assertFalse(mlp_kernel.usable(Lin(140000), Lin(140000), act, x, *w))


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0),
                     "needs a CUDA GPU with Triton's int8 dot (sm_80 or later)")
class GateUpSiLU(unittest.TestCase):
    def setUp(self):
        from y_lang.exact import lora, mlp_kernel, model as M
        self.lora, self.mk, self.M = lora, mlp_kernel, M
        self.w = (12, 8, lora.mm_y_any, torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = False

    def case(self, rows, K, N, R, scale_g, scale_u, b_scale=0.05, seed=0):
        M, lora, mk = self.M, self.lora, self.mk
        g_ = torch.Generator().manual_seed(seed)
        gate = M.ExactLinear(torch.nn.Linear(K, N, bias=False)).cuda()
        up = M.ExactLinear(torch.nn.Linear(K, N, bias=False)).cuda()
        act = M.ExactSiLU().cuda()
        x = torch.randn(rows, K, generator=g_) * 1.7
        if rows > 3:
            x[1] = 0.0                                   # sx = 1, u = 0
            x[2] = torch.randn(K, generator=g_) * 60.0    # g far past +-16
        x = x.reshape(1, rows, K).cuda()
        Ag, Au = (torch.randn(R, K, generator=g_, dtype=torch.float64).cuda() * 0.02 for _ in range(2))
        Bg, Bu = (torch.randn(N, R, generator=g_, dtype=torch.float64).cuda() * b_scale for _ in range(2))
        self.assertTrue(mk.usable(gate, up, act, x, *self.w))
        (g, xf, ug), c1, m1 = lora._lora_forward(x, Ag, Bg, gate, scale_g, *self.w)
        (yu, _, uu), c2, m2 = lora._lora_forward(x, Au, Bu, up, scale_u, *self.w)
        a = act(g)
        m = a * yu
        (G, YU, A, Mm, XF, UG, UU), c, msgs = mk.gate_up_silu(x, Ag, Bg, Au, Bu, gate, up, act,
                                                               scale_g, scale_u, *self.w)
        for name, got, want in (("g", G, g), ("yu", YU, yu), ("a", A, a), ("m", Mm, m),
                                ("xf", XF, xf), ("u_g", UG, ug), ("u_u", UU, uu)):
            self.assertTrue(_same(got, want), f"{name} differs at rows={rows} K={K} N={N} R={R}")
        self.assertEqual(msgs, m1 + m2)
        self.assertEqual([bool(t.any()) for t in c], [bool(t.any()) for t in c1 + c2])
        return g, ug, Bg

    def test_shapes_off_the_tiles(self):
        for rows, K, N, R, sg, su in ((1, 64, 48, 8, 1.0, 1.0), (17, 64, 200, 16, 0.5, 2.0),
                                      (130, 96, 130, 33, 2.0, 1.0), (990, 896, 520, 16, 1.0, 1.0)):
            self.case(rows, K, N, R, sg, su)

    def test_the_sigmoid_clamps_are_reached(self):
        g = self.case(40, 64, 96, 8, 1.0, 1.0, seed=3)[0]
        self.assertTrue(bool((g.abs() > 16).any()) and bool((g.abs() < 16).any()))

    def test_the_float64_branch_of_the_adapter_product(self):
        # B ~ 2^-125: the scales' sum su + sB falls below -126 (checked), so
        # those tiles take qmm's float64 product.
        from y_lang.exact.train_lm import q8
        _, ug, Bg = self.case(33, 64, 80, 16, 1.0, 1.0, b_scale=2.0 ** -125, seed=5)
        e = q8(ug, bits=8)[1] + q8(Bg, bits=8)[1].reshape(1, -1)
        self.assertLess(int(e[ug.abs().amax(-1) > 0].min()), -126)

    def test_the_layer_calls_it_and_training_is_the_cpus(self):
        from test_exact_lora import train_tiny
        lora, mk = self.lora, self.mk
        with mock.patch.object(lora, "LAYER_FUSED", True):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                fused = train_tiny("cuda", steps=3, compile=True, lengths=(30, 27))
            calls = sum(e.count for e in prof.key_averages() if e.key == "y_lang::exact_mlp_gate_up")
            self.assertGreater(calls, 0, "the layer path did not call the fused kernel")
            with mock.patch.object(mk, "FUSED", False):
                torch._dynamo.reset()
                separate = train_tiny("cuda", steps=3, compile=True, lengths=(30, 27))
        cpu = train_tiny("cpu", steps=3, lengths=(30, 27))
        self.assertEqual(fused[:2], cpu[:2])
        self.assertEqual(separate[:2], cpu[:2])


if __name__ == "__main__":
    unittest.main()
