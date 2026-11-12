"""`tools/exact_cross_device.py`'s weight fingerprint: a property of the
weights, not of where a build put them.

Qwen2.5-3B ties its `lm_head` to the embedding. The low-memory build on a
GPU keeps the embedding on the host and gives the head its own copy on the
device, so `named_parameters` listed the head as well, and a T4's stock
record fingerprinted `4548a1af4d9f1c10` against the CPU's `de4101d908728218`
for the same weights (2026-11-12): the comparison refused a control it had.
A small random Qwen2 stands in for it here. Controls: an untied head that
differs must change the fingerprint, and so must a tied model whose copy no
longer equals the embedding."""

import importlib.util
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "exact_cross_device.py"

try:
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
except ImportError:                                           # pragma: no cover
    torch = None


def load():
    spec = importlib.util.spec_from_file_location("exact_cross_device_fp", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tiny(tied):
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=tied)
    return Qwen2ForCausalLM(cfg).eval()


def untie(model):
    """What the low-memory GPU build does: the head becomes its own copy."""
    model.lm_head.weight = torch.nn.Parameter(model.model.embed_tokens.weight.detach().clone())
    return model


@unittest.skipIf(torch is None, "needs torch and transformers")
class TestWeightFingerprint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X = load()

    def test_a_tied_head_counts_once_wherever_it_lives(self):
        m = tiny(tied=True)
        self.assertIs(m.lm_head.weight, m.model.embed_tokens.weight)
        before = self.X._weight_fingerprint(m)
        untie(m)
        self.assertIsNot(m.lm_head.weight, m.model.embed_tokens.weight)
        self.assertEqual(len(list(m.named_parameters())),
                         len(list(tiny(tied=True).named_parameters())) + 1,
                         "the copy must be listed, or this does not reproduce the T4's build")
        self.assertEqual(self.X._weight_fingerprint(m), before)

    def test_a_copy_that_no_longer_equals_the_embedding_shows(self):
        m = untie(tiny(tied=True))
        before = self.X._weight_fingerprint(m)
        with torch.no_grad():
            m.lm_head.weight[0, 0] += 1.0
        self.assertNotEqual(self.X._weight_fingerprint(m), before)

    def test_an_untied_head_is_still_its_own_weight(self):
        m = tiny(tied=False)
        before = self.X._weight_fingerprint(m)
        with torch.no_grad():
            m.lm_head.weight[0, 0] += 1.0
        self.assertNotEqual(self.X._weight_fingerprint(m), before)


if __name__ == "__main__":
    unittest.main()
