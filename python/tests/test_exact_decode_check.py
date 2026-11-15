"""tools/exact_decode_check.py: the decode-step proof's model against the code.

Two parts:
  * the check's transcriptions of the Coq integer functions return what the
    Coq definitions return, on the samples in proofs/ExactDecodeConstants.v
    (which Coq itself checks against the code); no model needed;
  * the check itself, on the CPU, on the pinned Qwen2.5-0.5B-Instruct, at
    two layers and one decode step, with the bound-stress stage. Skipped,
    saying so, when that snapshot is not in the local Hugging Face cache.
"""
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "exact_decode_check.py"
CONSTS = ROOT / "proofs" / "ExactDecodeConstants.v"
SNAPSHOT = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots"
            / "7ae557604adf67be50417f59c2c2f167def9a775" / "model.safetensors")


def samples(name):
    text = CONSTS.read_text()
    body = re.search(rf"Definition {name} : list \(Z \* Z( \* Z)?\) :=\s*\[(.*?)\]\.", text, re.S).group(2)
    return [tuple(int(v) for v in t.split(",")) for t in re.findall(r"\(([-\d, ]+)\)", body)]


class Transcriptions(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(TOOL.parent))
        import exact_decode_check as chk
        chk.C = chk.coq_constants(str(CONSTS))
        self.chk = chk

    def tearDown(self):
        sys.path.remove(str(TOOL.parent))

    def test_widths(self):
        c = self.chk
        rows = samples("ACT_LEVELS_SAMPLES")
        self.assertGreater(len(rows), 100)
        for k, r, a in rows:
            self.assertEqual(c.act_levels_for(k, c.C["W_LEVELS"], r), a, (k, r))
        for d, q, kl in samples("K_LEVELS_SAMPLES"):
            self.assertEqual(c.k_levels_for(d, q), kl, (d, q))
        for q, hi, lo in samples("SPLIT_SAMPLES"):
            self.assertEqual((c.split_hi(q), c.split_lo(q)), (hi, lo), q)

    def test_exp2(self):
        rows = samples("EXP2_SAMPLES")
        self.assertGreater(len(rows), 400)
        for t, v in rows:
            self.assertEqual(self.chk.exp2_neg_q16_16(t), v, t)

    def test_a_wrong_transcription_is_noticed(self):
        """The comparison above can fail: an exp2 off by one at one argument."""
        rows = samples("EXP2_SAMPLES")
        t, v = next((t, v) for t, v in rows if v > 0)
        self.assertNotEqual(self.chk.exp2_neg_q16_16(t) + 1, v)


@unittest.skipUnless(SNAPSHOT.exists(), "Qwen2.5-0.5B-Instruct@7ae5576 is not in the local "
                     "Hugging Face cache; the end-to-end check needs it")
class EndToEnd(unittest.TestCase):
    def test_the_model_is_the_code_on_the_cpu(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("Y_EXACT_")}
        env.update(PYTHONPATH=str(ROOT / "python"), CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1")
        r = subprocess.run([sys.executable, str(TOOL), "--layers", "0", "23", "--steps", "1", "--ragged"],
                           env=env, capture_output=True, text=True, timeout=1200)
        tail = r.stdout + r.stderr[-2000:]
        self.assertEqual(r.returncode, 0, tail)
        m = re.search(r"(\d+) of (\d+) compared tensors bit-identical", r.stdout)
        self.assertIsNotNone(m, tail)
        self.assertEqual(m.group(1), m.group(2))
        self.assertGreaterEqual(int(m.group(2)), 20)


if __name__ == "__main__":
    unittest.main()
