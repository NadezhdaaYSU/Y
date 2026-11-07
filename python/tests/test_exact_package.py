"""`y_lang.exact`: the exact-inference modules, importable from the package and
by their old names from `tools/`, and the public `load` / `generate`.

CPU only. The end-to-end test builds a tiny random Qwen2 model and borrows the
Qwen2.5 tokenizer from the local Hugging Face cache; it is skipped, saying so,
where that is not cached.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"

# old name in tools/ -> y_lang.exact submodule
ALIASES = {
    "exact_model": "model",
    "exact_kv": "kv",
    "batch_invariance_demo": "attention",
    "w8a8_gemv": "w8a8_gemv",
}

ALIAS_CHECK = """
import importlib, json, sys
sys.path.insert(0, sys.argv[1])
out = {}
for old, new in json.loads(sys.argv[2]).items():
    try:
        a = importlib.import_module(old)
    except ImportError as e:            # w8a8_gemv needs triton
        out[old] = "import failed: " + str(e)
        continue
    out[old] = a is importlib.import_module("y_lang.exact." + new)
import exact_model
import y_lang.exact.model as m
was = m.GEMM
exact_model.GEMM = "torch-set-through-the-old-name"
out["assignment_reaches_the_package"] = m.GEMM == "torch-set-through-the-old-name"
m.GEMM = was
try:
    import w8a8_gemv
    out["exact_gemm_profile"] = w8a8_gemv._PROFILE
except ImportError:
    out["exact_gemm_profile"] = None
import exact_large_model
import y_lang.exact.large_model as lm
out["large_model_builders"] = exact_large_model.build_exact is lm.build_exact and callable(exact_large_model.check)
print(json.dumps(out))
"""


def _env():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONPATH=str(ROOT / "python"))
    return env


class TestOldNames(unittest.TestCase):
    def test_tools_names_are_the_package_modules(self):
        ran = subprocess.run([sys.executable, "-c", ALIAS_CHECK, str(TOOLS), json.dumps(ALIASES)],
                             cwd=TOOLS, env=_env(), capture_output=True, text=True, timeout=600)
        self.assertEqual(ran.returncode, 0, ran.stdout[-2000:] + ran.stderr[-3000:])
        out = json.loads(ran.stdout.strip().splitlines()[-1])
        for old in ALIASES:
            if old == "w8a8_gemv" and isinstance(out[old], str) and "triton" in out[old]:
                continue
            self.assertIs(out[old], True, f"{old}: {out[old]}")
        self.assertIs(out["assignment_reaches_the_package"], True)
        self.assertIs(out["large_model_builders"], True)
        if out["exact_gemm_profile"] is not None:
            # In a checkout the tile choices stay where they were before the move.
            self.assertEqual(out["exact_gemm_profile"], str(ROOT / ".ysu_exact_gemm"))

    def test_the_exact_pv_source_is_the_tests_one(self):
        from y_lang.exact import attention
        self.assertEqual(Path(attention.__file__).with_name("exact_pv.ysu").read_bytes(),
                         (ROOT / "tests" / "exact_pv.ysu").read_bytes())


def _have(module):
    try:
        __import__(module)
        return True
    except ImportError:
        return False


@unittest.skipUnless(_have("transformers"), "transformers is not installed")
class TestLoad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        cls.tmp = Path(tempfile.mkdtemp(prefix="y-exact-"))

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmp, True)

    def test_refuses_another_architecture(self):
        from y_lang.exact import load
        d = self.tmp / "llama"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({
            "model_type": "llama", "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 1, "num_attention_heads": 4, "vocab_size": 128}))
        with self.assertRaisesRegex(ValueError, "'llama' model; y_lang.exact supports qwen2 only"):
            load(str(d))

    def test_a_tiny_qwen2_is_converted_close_and_batch_invariant(self):
        import torch
        from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
        try:
            tok = AutoTokenizer.from_pretrained(TOKENIZER)
        except Exception as e:                      # not in the local cache
            self.skipTest(f"{TOKENIZER}'s tokenizer is not cached: {e}")

        d = self.tmp / "tiny_qwen2"
        torch.manual_seed(0)
        cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          max_position_embeddings=512, tie_word_embeddings=True)
        reference = Qwen2ForCausalLM(cfg).eval()
        reference.save_pretrained(d)
        tok.save_pretrained(d)

        import transformers.models.qwen2.modeling_qwen2 as qwen2
        from y_lang.exact import generate, load
        from y_lang.exact import attention, model as em

        model, tok2 = load(str(d), "cpu")
        self.assertIs(qwen2.eager_attention_forward, attention.exact_attention)
        linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        exact = [m for m in model.modules() if isinstance(m, em.ExactLinear)]
        self.assertEqual(linears, [], "every Linear must be converted")
        self.assertEqual(len(exact), 2 * 7 + 1)     # q, k, v, o, gate, up, down per layer, and lm_head

        # Not a degenerate model: the exact logits follow the float model's.
        ids = tok("The capital of France is", return_tensors="pt").input_ids
        with torch.no_grad():
            want = reference(ids).logits.float()
            got = model(ids).logits.float()
        rel = ((got - want).norm() / want.norm()).item()
        self.assertLess(rel, 0.05, f"exact logits are {rel:.3f} away from the float model's")

        prompt = "A suspension bridge carries the weight of the deck"
        alone = generate(model, tok2, prompt, max_new_tokens=12)
        in_batch = attention.decode(model, tok2, prompt, 5, 12, torch.device("cpu"))
        self.assertEqual(len(alone), 12)
        self.assertEqual(alone, in_batch)


if __name__ == "__main__":
    unittest.main()
