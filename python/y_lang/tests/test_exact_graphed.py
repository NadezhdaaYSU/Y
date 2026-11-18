"""`y_lang.exact.graphed.GraphedGreedy` decodes the tokens `greedy.greedy`
decodes, bit for bit, on the exact path: with the cache rows as floats, as
packed int8 rows, and as packed rows read by Y's attention kernel; with every
row the same length and with left padding. Two controls must change the
tokens: a mask that lets the step see the padding, and positions that do not
advance. GPU only (CUDA graphs); the Y-kernel case also needs liby.so.

The switches are module globals read when the code runs (`kv.PACK`,
`kv.INPLACE`, `kv.V_INT8`, `attention.ATTN`), so they are patched here; the
model is a small random Qwen2.
"""

import copy
import unittest
from unittest import mock

import torch

CUDA = torch.cuda.is_available()


def tiny_qwen2(attn="eager"):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      tie_word_embeddings=False, attn_implementation=attn)
    return Qwen2ForCausalLM(cfg).eval().cuda()


@unittest.skipUnless(CUDA, "needs a CUDA device")
class TestGraphedGreedy(unittest.TestCase):
    NEW = 12

    @classmethod
    def setUpClass(cls):
        from y_lang.exact import attention as A
        from y_lang.exact import kv as KV
        from y_lang.exact import model as EM
        from y_lang.exact.graphed import GraphedGreedy
        from y_lang.exact.greedy import greedy
        import transformers.models.qwen2.modeling_qwen2 as qwen2
        cls.A, cls.KV, cls.EM, cls.G, cls.qwen2 = A, KV, EM, GraphedGreedy, qwen2
        cls.greedy = staticmethod(greedy)
        m = tiny_qwen2()
        EM.convert(m)
        cls.exact = m
        g = torch.Generator(device="cuda").manual_seed(3)
        cls.ids = torch.randint(0, 256, (4, 9), device="cuda", generator=g)
        cls.att = torch.ones_like(cls.ids)
        cls.ragged_att = cls.att.clone()
        for r, k in enumerate((0, 1, 3, 5)):              # left padding
            cls.ragged_att[r, :k] = 0

    def switches(self, pack=False, inplace=False, v_int8=False, attn="torch"):
        return [mock.patch.object(self.KV, "PACK", pack), mock.patch.object(self.KV, "INPLACE", inplace),
                mock.patch.object(self.KV, "V_INT8", v_int8), mock.patch.object(self.A, "ATTN", attn)]

    def reference(self, model, att):
        prev = self.qwen2.eager_attention_forward
        self.qwen2.eager_attention_forward = self.A.exact_attention
        try:
            with torch.no_grad():
                return self.greedy(model, self.ids, att, self.NEW)
        finally:
            self.qwen2.eager_attention_forward = prev

    def graphed(self, model, att, cls=None):
        run = (cls or self.G)(model, self.ids.shape[0], self.ids.shape[1], self.NEW)
        return run(self.ids, att), run

    def check(self, **sw):
        patches = self.switches(**sw)
        for p in patches:
            p.start()
        try:
            for att in (self.att, self.ragged_att):
                want = self.reference(self.exact, att)
                got, run = self.graphed(self.exact, att)
                diff = (got != want).any(dim=1).nonzero().flatten().tolist()
                self.assertEqual(diff, [], f"{sw}, padding {att.eq(0).sum(1).tolist()}: rows {diff} differ")
                # The recorded graph again, on the same inputs: the same tokens.
                self.assertTrue(torch.equal(run(self.ids, att), want))
        finally:
            for p in patches:
                p.stop()

    def test_float_rows(self):
        self.check()

    def test_packed_rows(self):
        self.check(pack=True, inplace=True)

    def test_packed_rows_on_y_attention(self):
        try:
            self.A._attn_kernel(torch.device("cuda"))
        except RuntimeError as e:
            self.skipTest(f"Y's attention kernel cannot be built here: {e}")
        self.check(pack=True, inplace=True, v_int8=True, attn="y")

    def test_controls_change_the_tokens(self):
        want = self.reference(self.exact, self.ragged_att)

        class SeesPadding(self.G):
            def _load(self, *a):
                super()._load(*a)
                self.base.fill_(True)

        class StuckPositions(self.G):
            def _advance(self):
                pos = self.pos.clone()
                super()._advance()
                self.pos.copy_(pos)

        for cls in (SeesPadding, StuckPositions):
            got, _ = self.graphed(self.exact, self.ragged_att, cls)
            self.assertFalse(torch.equal(got, want), f"control {cls.__name__} did not change the tokens")

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "recorded for inputs"):
            run = self.G(self.exact, 4, 9, 2)
            run(self.ids[:, :8], self.att[:, :8])
        with self.assertRaisesRegex(ValueError, "CUDA"):
            self.G(copy.deepcopy(self.exact).cpu(), 4, 9, 2)
        odd = copy.deepcopy(self.exact)
        odd.config._attn_implementation = "flex_attention"
        with self.assertRaisesRegex(ValueError, "eager' and 'sdpa'"):
            self.G(odd, 4, 9, 2)
        with self.assertRaisesRegex(ValueError, "past the model"):
            self.G(self.exact, 4, 250, 10)
        split = copy.deepcopy(self.exact)
        split.lm_head.cpu()
        with self.assertRaisesRegex(ValueError, "tensors on cpu"):
            self.G(split, 4, 9, 2)


if __name__ == "__main__":
    unittest.main()
