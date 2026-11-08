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

    def test_the_kernel_sources_are_the_tests_ones(self):
        from y_lang.exact import attention
        for name in ("exact_pv.ysu", "exact_decode_attention.ysu"):
            self.assertEqual(Path(attention.__file__).with_name(name).read_bytes(),
                             (ROOT / "tests" / name).read_bytes(), name)


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
        # q/k/v merged by default: qkv, o, gate, up, down per layer, and lm_head.
        self.assertEqual(len(exact), 2 * 5 + 1)
        self.assertEqual(model.model.layers[0].self_attn.qkv_sizes, (64, 32, 32))

        # Merging changes no bits: the same model converted without it, and
        # with gate/up merged too.
        from transformers import AutoModelForCausalLM
        unmerged = AutoModelForCausalLM.from_pretrained(str(d), dtype=torch.float32,
                                                        attn_implementation="eager").eval()
        em.convert(unmerged, merge=False)
        self.assertEqual(sum(isinstance(m, em.ExactLinear) for m in unmerged.modules()), 2 * 7 + 1)
        all_merged = AutoModelForCausalLM.from_pretrained(str(d), dtype=torch.float32,
                                                          attn_implementation="eager").eval()
        em.convert(all_merged, merge=True)
        self.assertEqual(sum(isinstance(m, em.ExactLinear) for m in all_merged.modules()), 2 * 4 + 1)
        self.assertEqual(all_merged.model.layers[0].mlp.gate_up_sizes, (128, 128))
        # Buffer names and shapes as the unmerged model has them, whatever
        # was merged (exact_cross_device's weight fingerprint is keyed on them).
        want = sorted((n, tuple(b.shape)) for n, b in unmerged.named_buffers())
        for m in (model, all_merged, unmerged):
            self.assertEqual(sorted(em.unmerged_buffer_shapes(m)), want)
        self.assertNotEqual(sorted((n, tuple(b.shape)) for n, b in model.named_buffers()), want)

        # Not a degenerate model: the exact logits follow the float model's.
        ids = tok("The capital of France is", return_tensors="pt").input_ids
        with torch.no_grad():
            want = reference(ids).logits.float()
            got = model(ids).logits.float()
        rel = ((got - want).norm() / want.norm()).item()
        self.assertLess(rel, 0.05, f"exact logits are {rel:.3f} away from the float model's")
        with torch.no_grad():
            got_unmerged = unmerged(ids).logits.float()
        self.assertTrue(torch.equal(got, got_unmerged), "merging q/k/v changed the logits")
        with torch.no_grad():
            self.assertTrue(torch.equal(all_merged(ids).logits.float(), got_unmerged),
                            "merging q/k/v and gate/up changed the logits")

        prompt = "A suspension bridge carries the weight of the deck"
        alone = generate(model, tok2, prompt, max_new_tokens=12)
        in_batch = attention.decode(model, tok2, prompt, 5, 12, torch.device("cpu"))
        self.assertEqual(len(alone), 12)
        self.assertEqual(alone, in_batch)
        self.assertEqual(alone, generate(unmerged, tok2, prompt, max_new_tokens=12))

        # `greedy` picks generate's tokens, on a left-padded ragged batch too.
        from y_lang.exact.greedy import greedy
        tok2.padding_side = "left"
        batch = tok2([prompt, "Hi", "The capital of France is"], return_tensors="pt",
                     padding=True)
        self.assertEqual(int((batch.attention_mask == 0).sum()) > 0, True, "not ragged")
        # A tiny random model's argmax hardly depends on positions, so the
        # tokens alone let wrong position ids through (two mutants did): every
        # forward must also get generate's inputs.
        calls = {"generate": [], "greedy": []}

        def recorder(key):
            def hook(module, args, kwargs):
                calls[key].append({k: kwargs.get(k) for k in
                                   ("input_ids", "position_ids", "attention_mask", "logits_to_keep")})
            return hook

        handle = model.register_forward_pre_hook(recorder("generate"), with_kwargs=True)
        with torch.no_grad():
            want = model.generate(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                                  max_new_tokens=10, do_sample=False, num_beams=1,
                                  repetition_penalty=1.0, pad_token_id=tok2.eos_token_id)
        handle.remove()
        handle = model.register_forward_pre_hook(recorder("greedy"), with_kwargs=True)
        got = greedy(model, batch.input_ids, batch.attention_mask, 10)
        handle.remove()
        self.assertEqual(got.tolist(), want[:, batch.input_ids.shape[1]:].tolist())
        self.assertEqual(len(calls["greedy"]), 10)
        self.assertEqual(len(calls["generate"]), len(calls["greedy"]))
        for step, (g, y) in enumerate(zip(calls["generate"], calls["greedy"])):
            for k in g:
                if torch.is_tensor(g[k]):
                    self.assertTrue(torch.is_tensor(y[k]) and torch.equal(g[k], y[k]),
                                    f"forward {step}: {k} is {y[k]}, generate's is {g[k]}")
                else:
                    self.assertEqual(g[k], y[k], f"forward {step}: {k}")
        with self.assertRaisesRegex(ValueError, "at least 1"):
            greedy(model, batch.input_ids, batch.attention_mask, 0)

        # kv.PACK: the cache keeps each row quantised once and packed. The
        # logits at every step and greedy's tokens are the float cache's, and
        # the cache really holds packed int8 rows (so the comparison is not
        # between two unpacked runs).
        from transformers import DynamicCache
        from y_lang.exact import kv

        def decode_logits(n):
            cache = DynamicCache(config=model.config)
            att = batch.attention_mask
            pos = (att.long().cumsum(-1) - 1).masked_fill(att == 0, 0)
            logits = []
            with torch.no_grad():
                out = model(input_ids=batch.input_ids, attention_mask=att, position_ids=pos,
                            past_key_values=cache, use_cache=True)
                logits.append(out.logits[:, -1])
                for _ in range(n):
                    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
                    att = torch.cat([att, att.new_ones((att.shape[0], 1))], dim=-1)
                    pos = pos[:, -1:] + 1
                    out = model(input_ids=nxt, attention_mask=att, position_ids=pos,
                                past_key_values=cache, use_cache=True)
                    logits.append(out.logits[:, -1])
            return torch.stack(logits), cache

        saved = kv.PACK
        try:
            kv.PACK = False
            want_logits, float_cache = decode_logits(6)
            kv.PACK = True
            got_logits, packed_cache = decode_logits(6)
            packed_tokens = greedy(model, batch.input_ids, batch.attention_mask, 10)
        finally:
            kv.PACK = saved
        self.assertEqual(float_cache.layers[0].keys.dtype, torch.float32)
        self.assertEqual(packed_cache.layers[0].keys.dtype, torch.int8)
        self.assertEqual(packed_cache.layers[0].keys.shape[-1], 2 * (16 + kv.K_PAD))
        self.assertEqual(packed_cache.layers[0].values.dtype, torch.int8)
        self.assertTrue(torch.equal(got_logits, want_logits), "packing the cache changed logits")
        self.assertEqual(packed_tokens.tolist(), got.tolist())

    def test_packed_rows_round_trip_bit_for_bit(self):
        import torch
        from y_lang.exact import kv
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for dev in devices:
            g = torch.Generator(device=dev).manual_seed(0)
            k = torch.randn(3, 2, 9, 32, device=dev, generator=g)
            v = torch.randn(3, 2, 9, 32, device=dev, generator=g)
            k = k * torch.exp(torch.randn(3, 2, 9, 1, device=dev, generator=g) * 4)
            v = v * torch.exp(torch.randn(3, 2, 9, 1, device=dev, generator=g) * 4)
            k[0, 0, 0] = 0.0                       # a zero row: scale 1
            k_lv = kv.k_levels_for(32, 127)
            ki, sk = kv.quantize_rows(k, k_lv)
            vi, sv = kv.quantize_rows(v, kv.V_LEVELS)
            self.assertEqual(float(ki.abs().max()), k_lv)       # the extremes are in
            kp, vp = kv.quantize_and_pack(k, v)
            self.assertEqual((kp.dtype, kp.shape[-1]), (torch.int8, 2 * (32 + kv.K_PAD)))
            self.assertEqual((vp.dtype, vp.shape[-1]), (torch.int8, 32 + kv.V_PAD))
            ki2, sk2 = kv.unpack_k(kp, 32)
            vi2, sv2 = kv.unpack_v(vp, 32)
            self.assertTrue(torch.equal(ki2, ki), dev)
            self.assertTrue(torch.equal(vi2.to(torch.float32), vi), dev)
            for a, b in ((sk2, sk), (sv2, sv)):
                self.assertTrue(torch.equal(a.view(torch.int32), b.view(torch.int32)), dev)
            # Every scale bit pattern survives, sign and NaN patterns included.
            bits = torch.randint(-2 ** 31, 2 ** 31, (4096, 1), generator=torch.Generator().manual_seed(1),
                                 dtype=torch.int64).to(torch.int32).to(dev)
            bits[:4, 0] = torch.tensor([0, -1, -2 ** 31, 2 ** 31 - 1], dtype=torch.int32)
            s = bits.view(torch.float32)
            z = torch.zeros((4096, 32), device=dev)
            self.assertTrue(torch.equal(kv.unpack_k(kv.pack_k(z, s), 32)[1].view(torch.int32), bits))
            self.assertTrue(torch.equal(kv.unpack_v(kv.pack_v(z, s), 32)[1].view(torch.int32), bits))
            if dev == "cuda":
                # The model runs these compiled: Inductor's lowering of the
                # shifts and bitcasts must give the same rows.
                packed = torch.compile(kv.quantize_and_pack, dynamic=False)(k, v)
                self.assertTrue(torch.equal(packed[0], kp) and torch.equal(packed[1], vp))
                un = torch.compile(lambda a, b: (*kv.unpack_k(a, 32), *kv.unpack_v(b, 32)),
                                   dynamic=False)(kp, vp)
                self.assertTrue(torch.equal(un[1].view(torch.int32), sk.view(torch.int32)))
                self.assertTrue(torch.equal(un[3].view(torch.int32), sv.view(torch.int32)))


if __name__ == "__main__":
    unittest.main()
