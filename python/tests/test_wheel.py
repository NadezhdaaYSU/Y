"""The wheel carries the compiler library and the GPU probe, and refuses a
missing or stale one.

Before `python/setup.py` the wheel was pure Python. It installed cleanly and
failed at the first compile: an installed package finds `liby.so` beside
itself (`compiler._find_liby`), and nothing put it there. Nor was the probe
there, which the compiler runs from beside `liby.so` when the working
directory has no `.ysu_hw_profile`; without it every kernel targeted `sm_80`
whatever the card.

Each build runs on a copy of `python/` inside a scratch cargo layout
(`Cargo.toml`, `src/lib.rs`, `target/release/{liby.so,ysu_gpu_probe}`), so
the staleness check can be driven by setting mtimes, and nothing is written
into the repository. The probe is a script that prints a fake card, so no
GPU is needed.
"""

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
LIB = ROOT / "target" / "release" / "liby.so"

VEC_ADD = """
kernel vec_add(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, N: I32) {
    let idx: I32 = global_thread_id() * 4;
    if idx < N {
        vec_add_v4(A[idx], B[idx], C[idx]);
    }
}
"""

INSTALLED_CHECK = """
import os, sys
import y_lang
from y_lang import compiler
site, target, expected = sys.argv[1:4]
assert y_lang.__file__.startswith(site), y_lang.__file__
lib = compiler._find_liby()
assert lib == os.path.join(site, "y_lang", "liby.so"), lib
ptx = y_lang.compile_to_ptx(sys.stdin.read(), target_sm=target)
targets = [l.split()[1] for l in ptx.splitlines() if l.startswith(".target")]
print("ENTRY" if ".visible .entry vec_add(" in ptx else "NO ENTRY", *targets)
"""

FAKE_PROBE = """#!/bin/sh
echo GPU_NAME=FakeProbeGPU
echo GPU_VENDOR=NVIDIA
echo SM_VERSION=8.6
echo COMPUTE_CAPABILITY=8.6
echo SM_COUNT=12
"""


def _setuptools_too_old():
    try:
        import setuptools
    except ImportError:
        return "setuptools is not installed"
    if int(setuptools.__version__.split(".")[0]) < 77:
        return f"setuptools {setuptools.__version__} is older than 77"
    return None


@unittest.skipIf(_setuptools_too_old(), _setuptools_too_old() or "")
class TestWheel(unittest.TestCase):
    def setUp(self):
        self.assertTrue(LIB.is_file(), f"{LIB} is missing: run `cargo build --release` first")
        self.tmp = Path(tempfile.mkdtemp(prefix="y-wheel-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = self.tmp / "root"
        shutil.copytree(PKG, self.root / "python", symlinks=False,
                        ignore=shutil.ignore_patterns("build", "*.egg-info", "__pycache__",
                                                      "tests", "examples"))
        (self.root / "Cargo.toml").write_text("[package]\nname = \"fake\"\n", encoding="utf-8")
        (self.root / "src").mkdir()
        self.source = self.root / "src" / "lib.rs"
        self.source.write_text("// a compiler source\n", encoding="utf-8")
        self.lib = self.root / "target" / "release" / "liby.so"
        self.lib.parent.mkdir(parents=True)
        shutil.copyfile(LIB, self.lib)
        self.probe = self.lib.parent / "ysu_gpu_probe"
        self.probe.write_text(FAKE_PROBE, encoding="utf-8")
        self.probe.chmod(0o755)
        self.out = self.tmp / "dist"
        # Older than every library below, so each test's verdict is src/lib.rs's.
        self.age(self.root / "Cargo.toml", -100)

    def age(self, path, seconds):
        t = 1_700_000_000 + seconds
        os.utime(path, (t, t))

    def build(self, **env_extra):
        env = {k: v for k, v in os.environ.items() if k not in ("Y_WHEEL_LIBY", "Y_LIB_PATH")}
        env.update(env_extra)
        return subprocess.run(
            [sys.executable, "-m", "pip", "wheel", str(self.root / "python"), "--no-deps",
             "--no-build-isolation", "--no-cache-dir", "-q", "-w", str(self.out)],
            env=env, capture_output=True, text=True, timeout=300)

    def install(self):
        """Build (sources older than both files), install outside the
        repository, and return (site, work, run) where run(target) compiles
        VEC_ADD from `work` with the installed package."""
        self.age(self.source, 0)
        self.age(self.lib, 10)
        self.age(self.probe, 10)
        built = self.build()
        self.assertEqual(built.returncode, 0, built.stderr[-3000:])
        site, work = self.tmp / "site", self.tmp / "work"
        installed = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--no-compile", "-q",
             "--target", str(site), str(self.wheel())], capture_output=True, text=True, timeout=300)
        self.assertEqual(installed.returncode, 0, installed.stderr[-3000:])
        work.mkdir()
        env = {k: v for k, v in os.environ.items() if k not in ("Y_LIB_PATH", "PYTHONPATH")}
        env["PYTHONPATH"] = str(site)
        env["YSU_CACHE_DIR"] = str(self.tmp / "cache")

        def run(target):
            ran = subprocess.run([sys.executable, "-c", INSTALLED_CHECK, str(site), target, ""],
                                 input=VEC_ADD, cwd=work, env=env, capture_output=True,
                                 text=True, timeout=300)
            self.assertEqual(ran.returncode, 0, ran.stdout[-2000:] + ran.stderr[-3000:])
            return ran.stdout.strip().splitlines()[-1]
        return site, work, run

    def wheel(self):
        wheels = list(self.out.glob("*.whl"))
        self.assertEqual(len(wheels), 1, wheels)
        return wheels[0]

    def test_carries_the_library_and_the_probe(self):
        site, work, run = self.install()
        whl = self.wheel()
        version = re.search(r'__version__ = "([^"]+)"',
                            (PKG / "y_lang" / "__init__.py").read_text(encoding="utf-8")).group(1)
        self.assertRegex(whl.name, rf"^y_lang-{re.escape(version)}-py3-none-linux_[a-z0-9_]+\.whl$")
        with zipfile.ZipFile(whl) as z:
            names = z.namelist()
            self.assertIn("y_lang/liby.so", names, "the library must sit in the package, not under .data/")
            self.assertEqual(z.read("y_lang/liby.so"), LIB.read_bytes())
            self.assertEqual(z.read("y_lang/ysu_gpu_probe"), FAKE_PROBE.encode())
            self.assertTrue((z.getinfo("y_lang/ysu_gpu_probe").external_attr >> 16) & 0o100,
                            "the probe must be executable")
            info = f"y_lang-{version}.dist-info"
            self.assertEqual(z.read(f"{info}/licenses/LICENSE"), (ROOT / "LICENSE").read_bytes())
            self.assertEqual(z.read(f"{info}/licenses/NOTICE"), (ROOT / "NOTICE").read_bytes())
            meta = z.read(f"{info}/METADATA").decode()
            self.assertIn("License-Expression: BUSL-1.1", meta)
            self.assertIn("Root-Is-Purelib: false", z.read(f"{info}/WHEEL").decode())

        # A pinned profile: the compile neither probes nor needs a GPU.
        (work / ".ysu_hw_profile").write_text("SM_VERSION=8.9\nSM_COUNT=66\n", encoding="utf-8")
        self.assertEqual(run("sm_89"), "ENTRY sm_89")

    def test_installed_package_runs_its_own_probe(self):
        # No profile in the working directory, and no probe there: the one
        # found must be the one beside the installed liby.so. Without it the
        # profile says SM_VERSION=0.0 and the kernel targets sm_80.
        site, work, run = self.install()
        self.assertEqual(run("auto"), "ENTRY sm_86")
        profile = (work / ".ysu_hw_profile").read_text(encoding="utf-8")
        self.assertIn("GPU_NAME=FakeProbeGPU", profile)

    def test_refuses_a_library_older_than_a_source(self):
        self.age(self.lib, 0)
        self.age(self.source, 10)
        built = self.build()
        self.assertNotEqual(built.returncode, 0)
        self.assertIn("is older than src/lib.rs", built.stdout + built.stderr)
        self.assertEqual(list(self.out.glob("*.whl")) if self.out.exists() else [], [])

    def test_refuses_a_missing_library(self):
        self.lib.unlink()
        built = self.build()
        self.assertNotEqual(built.returncode, 0)
        self.assertIn(f"no compiler library at {self.lib}", built.stdout + built.stderr)

    def test_refuses_a_missing_probe(self):
        self.age(self.source, 0)
        self.age(self.lib, 10)
        self.probe.unlink()
        built = self.build()
        self.assertNotEqual(built.returncode, 0)
        self.assertIn(f"no GPU probe at {self.probe}", built.stdout + built.stderr)

    def test_named_library_is_the_one_carried_and_is_checked_too(self):
        named = self.tmp / "elsewhere" / "liby.so"
        named.parent.mkdir()
        named.write_bytes(b"a library built elsewhere")
        self.age(self.source, 0)
        self.age(named, 10)
        self.age(self.probe, 10)
        self.assertEqual(self.build(Y_WHEEL_LIBY=str(named)).returncode, 0)
        with zipfile.ZipFile(self.wheel()) as z:
            self.assertEqual(z.read("y_lang/liby.so"), b"a library built elsewhere")
        shutil.rmtree(self.out)
        self.age(named, -10)
        built = self.build(Y_WHEEL_LIBY=str(named))
        self.assertNotEqual(built.returncode, 0)
        self.assertIn("is older than src/lib.rs", built.stdout + built.stderr)


if __name__ == "__main__":
    unittest.main()
