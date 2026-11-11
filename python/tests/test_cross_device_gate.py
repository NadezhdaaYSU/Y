"""`tools/exact_cross_device.py --check-gate`: every refusal of `--compare`
on synthetic artifacts, run as a test (it was a manual self-check only).

Scenarios 10 and 11 came from a Kaggle run (2026-11-11): every CPU was
`cpu:x86_64`, so an Intel Xeon's record and an AMD Zen 5's were "the same
device" and refused. CPUs are told apart by `cpu_model` now, and a record
without one is still refused. The control: the old identity (no model) must
make the gate report a wrong scenario."""

import contextlib
import importlib.util
import io
import unittest
from pathlib import Path
from unittest import mock

TOOL = Path(__file__).resolve().parents[2] / "tools" / "exact_cross_device.py"


def load():
    spec = importlib.util.spec_from_file_location("exact_cross_device_under_test", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestCrossDeviceGate(unittest.TestCase):
    def run_gate(self, mod):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = mod.check_gate_logic()
        return code, out.getvalue()

    def test_every_scenario_behaves(self):
        code, out = self.run_gate(load())
        self.assertEqual(code, 0, out)
        self.assertIn("all 11 scenarios", out)

    def test_cpus_by_architecture_alone_fail_it(self):
        mod = load()
        old = lambda dev: (dev.get("gpu_name"), dev.get("capability"))     # noqa: E731
        with mock.patch.object(mod, "device_identity", old), \
                mock.patch.object(mod, "_same_device", lambda a, b: a == b):
            code, out = self.run_gate(mod)
        self.assertEqual(code, 1, out)
        self.assertIn("WRONG two CPUs of different models", out)


if __name__ == "__main__":
    unittest.main()
