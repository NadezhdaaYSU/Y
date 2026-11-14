"""proofs/ExactDecodeConstants.v is what python/y_lang/exact generates now.

The Coq proofs of the exact decode path take their constants from that file
and check their width functions against the samples in it. If a constant or a
width function changes in the Python, the generated text changes and this
fails until the file is regenerated and the proofs compile against it:

    PYTHONPATH=python python3 tools/exact_decode_constants.py --write
"""
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
COMMITTED = ROOT / "proofs" / "ExactDecodeConstants.v"
TOOL = ROOT / "tools" / "exact_decode_constants.py"


def generated():
    env = {k: v for k, v in os.environ.items()
           if k not in ("Y_EXACT_ACT_LEVELS", "Y_EXACT_K_LEVELS", "Y_EXACT_V_LEVELS")}
    env["PYTHONPATH"] = str(ROOT / "python")
    env["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run([sys.executable, str(TOOL)], env=env, capture_output=True,
                          text=True, timeout=600)


class ExactDecodeConstants(unittest.TestCase):
    def test_committed_file_is_what_the_python_generates(self):
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as e:
            self.skipTest(f"y_lang.exact needs torch and transformers: {e}")
        run = generated()
        self.assertEqual(run.returncode, 0, run.stderr[-2000:])
        self.assertEqual(run.stdout, COMMITTED.read_text(),
                         "python/y_lang/exact no longer generates the committed "
                         "proofs/ExactDecodeConstants.v")

    def test_the_file_holds_the_constants_the_bounds_read(self):
        """A file with the constants deleted would compare equal to nothing
        only if the generator also lost them; this pins the names."""
        text = COMMITTED.read_text()
        for name in ("ACC_BUDGET", "DIGIT_MAX", "W_LEVELS", "Q_LEVELS", "V_LEVELS",
                     "K_LEVELS", "SCORE_BUDGET", "P_BITS", "SUMSQ_MAX_H", "LN2_Q30",
                     "RECIP6_Q32", "EXP_TABLE", "ACT_LEVELS_SAMPLES", "K_LEVELS_SAMPLES",
                     "DIGIT_WIDTH_SAMPLES", "SPLIT_SAMPLES", "EXP2_SAMPLES"):
            self.assertIn(f"Definition {name} ", text)
        self.assertIn("Definition ACC_BUDGET : Z := 2147483648.", text)


if __name__ == "__main__":
    unittest.main()
