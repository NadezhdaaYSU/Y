"""`y_lang.exact.model.convert` keeps the Triton int8 GEMM off a card whose
Triton cannot build it, and the layers then give the same bits.

On a Tesla T4 (sm_75, Kaggle, 2026-11-11) Triton 3.7 failed to compile
`w8a8_gemv`'s int8 `tl.dot`, so the exact path, which picks that kernel by
default, died before its first token. Below sm_80 the default now takes
`torch._int_mm` (or float64 where that is refused); a `Y_EXACT_GEMM=triton`
named explicitly is refused there. The card is this machine's; `convert`'s
rule (`triton_int8_dot_ok`) is patched, never the capability torch reports,
which Triton would compile for. (The first version patched the capability; on
the T4 itself Triton then built every later kernel in the process for sm_89.)
GPU only.
"""

import copy
import unittest
from unittest import mock

import torch

CUDA = torch.cuda.is_available()


def tiny_qwen2():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      tie_word_embeddings=False, attn_implementation="eager")
    return Qwen2ForCausalLM(cfg).eval().cuda()


@unittest.skipUnless(CUDA, "needs a CUDA device")
class TestGemmChoice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from y_lang.exact import model as em
        cls.em = em
        if em.GEMM != "triton" or em.w8a8_matmul is None:
            raise unittest.SkipTest(f"the default GEMM here is {em.GEMM!r}, not Triton")
        cls.float_model = tiny_qwen2()
        g = torch.Generator(device="cuda").manual_seed(1)
        cls.ids = torch.randint(0, 256, (3, 17), device="cuda", generator=g)
        cls.real_ok = em.triton_int8_dot_ok("cuda")

    def converted(self, triton_ok, explicit=False):
        from y_lang.exact import w8a8_gemv
        m = copy.deepcopy(self.float_model)
        primed = []
        real_prime = w8a8_gemv.prime
        with mock.patch.object(self.em, "triton_int8_dot_ok", lambda device: triton_ok), \
                mock.patch.object(self.em, "GEMM_EXPLICIT", explicit), \
                mock.patch.object(w8a8_gemv, "prime",
                                  lambda *a, **k: primed.append(a) or real_prime(*a, **k)):
            self.em.convert(m)
        layers = [x for x in m.modules() if isinstance(x, self.em.ExactLinear)]
        self.assertTrue(layers)
        return m, layers, primed

    def logits(self, model, triton_must_run, int_mm=True):
        calls = []
        real = self.em.w8a8_matmul

        def spy(*a, **k):
            calls.append(1)
            return real(*a, **k)

        def refused(*a, **k):
            raise RuntimeError("torch._int_mm refused (test)")
        with mock.patch.object(self.em, "w8a8_matmul", spy), \
                mock.patch.object(torch, "_int_mm", torch._int_mm if int_mm else refused), \
                torch.no_grad():
            out = model(self.ids).logits
        self.assertEqual(bool(calls), triton_must_run, f"Triton GEMM called {len(calls)} times")
        return out

    def same_bits(self, got, want, what):
        diff = int((got.view(torch.int32) != want.view(torch.int32)).sum())
        self.assertEqual(diff, 0, f"{what}: {diff} of {want.numel()} logits differ")

    def test_below_sm80_the_default_skips_triton_and_keeps_the_bits(self):
        old, old_layers, old_primed = self.converted(False)      # a card like a T4
        self.assertTrue(not any(x.triton_ok for x in old_layers) and not old_primed)
        got = self.logits(old, triton_must_run=False)
        # The fallbacks agree with each other on any card: _int_mm and float64.
        self.same_bits(got, self.logits(old, triton_must_run=False, int_mm=False), "_int_mm vs float64")
        if not self.real_ok:
            return                                  # no Triton int8 GEMM on this card to compare with
        new, new_layers, new_primed = self.converted(True)
        self.assertTrue(all(x.triton_ok for x in new_layers) and new_primed)
        self.same_bits(got, self.logits(new, triton_must_run=True), "Triton vs _int_mm")

    def test_a_triton_gemm_named_explicitly_is_refused_below_sm80(self):
        with self.assertRaisesRegex(ValueError, r"Y_EXACT_GEMM=triton.*sm_\d+.*sm_80 or later"):
            self.converted(False, explicit=True)
        if self.real_ok:
            _, layers, primed = self.converted(True, explicit=True)    # where it can, it is taken
            self.assertTrue(all(x.triton_ok for x in layers) and primed)

    def test_on_this_card_convert_follows_the_rule(self):
        m = copy.deepcopy(self.float_model)
        self.em.convert(m)
        layers = [x for x in m.modules() if isinstance(x, self.em.ExactLinear)]
        self.assertEqual({x.triton_ok for x in layers}, {self.real_ok})

    def test_the_rule_reads_the_device(self):
        self.assertFalse(self.em.triton_int8_dot_ok("cpu"))
        with mock.patch.object(torch.cuda, "get_device_capability", lambda *a, **k: (7, 5)):
            self.assertFalse(self.em.triton_int8_dot_ok("cuda"))
        with mock.patch.object(torch.cuda, "get_device_capability", lambda *a, **k: (8, 0)):
            self.assertTrue(self.em.triton_int8_dot_ok("cuda"))


if __name__ == "__main__":
    unittest.main()
