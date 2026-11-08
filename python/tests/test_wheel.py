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

import json
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
import json, os, sys
import y_lang
from y_lang import compiler
site, target = sys.argv[1:3]
assert y_lang.__file__.startswith(site), y_lang.__file__
lib = compiler._find_liby()
assert lib == os.path.join(site, "y_lang", "liby.so"), lib
src = sys.stdin.read()
seen = []
for _ in range(3):
    ptx = y_lang.compile_to_ptx(src, target_sm=target)
    seen.append([l.split()[1] for l in ptx.splitlines() if l.startswith(".target")]
                + (["ENTRY"] if ".visible .entry vec_add(" in ptx else []))
stats = compiler.get_cache_stats()
try:
    from y_lang.exact import w8a8_gemv
    exact_gemm = w8a8_gemv._PROFILE
except ImportError:                      # no triton
    exact_gemm = None
print("RESULT " + json.dumps({
    "ptx": seen[0], "same": all(s == seen[0] for s in seen),
    "profile": str(compiler.YCompilerLib.get_instance().hw_profile_path()),
    "cache_dir": stats["cache_dir"],
    "hits": stats["mem_hits"] + stats["disk_hits"], "misses": stats["misses"],
    "exact_gemm": exact_gemm,
}))
"""

FAKE_PROBE = """#!/bin/sh
echo GPU_NAME=FakeProbeGPU
echo GPU_VENDOR=NVIDIA
echo SM_VERSION=8.6
echo COMPUTE_CAPABILITY=8.6
echo SM_COUNT=12
"""


def rust_std_notice():
    sysroot = subprocess.run(["rustc", "--print", "sysroot"], check=True,
                             capture_output=True, text=True).stdout.strip()
    return Path(sysroot) / "share" / "doc" / "rust" / "COPYRIGHT-library.html"


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
        # The user cache is a scratch directory: these runs must never write
        # into the real ~/.cache, least of all a stub probe's fake card.
        env = {k: v for k, v in os.environ.items()
               if k not in ("Y_LIB_PATH", "PYTHONPATH", "YSU_CACHE_DIR", "YSU_HW_PROFILE")}
        env["PYTHONPATH"] = str(site)
        env["XDG_CACHE_HOME"] = str(self.tmp / "xdg")

        def run(target, **extra):
            ran = subprocess.run([sys.executable, "-c", INSTALLED_CHECK, str(site), target],
                                 input=VEC_ADD, cwd=work, env={**env, **extra},
                                 capture_output=True, text=True, timeout=300)
            self.assertEqual(ran.returncode, 0, ran.stdout[-2000:] + ran.stderr[-3000:])
            line = [l for l in ran.stdout.splitlines() if l.startswith("RESULT ")][-1]
            result = json.loads(line[len("RESULT "):])
            self.assertTrue(result["same"], result)
            return result
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
            self.assertIn("y_lang/PROFILE_IN_USER_CACHE", names)
            for module in ("__init__", "model", "kv", "attention", "w8a8_gemv", "large_model",
                           "greedy", "norm_kernel"):
                self.assertIn(f"y_lang/exact/{module}.py", names)
            # The links' targets, not the links: the wheel must carry the sources.
            for src in ("exact_pv.ysu", "exact_decode_attention.ysu"):
                self.assertEqual(z.read(f"y_lang/exact/{src}"), (ROOT / "tests" / src).read_bytes(), src)
            self.assertEqual(z.read("y_lang/liby.so"), LIB.read_bytes())
            self.assertEqual(z.read("y_lang/ysu_gpu_probe"), FAKE_PROBE.encode())
            self.assertTrue((z.getinfo("y_lang/ysu_gpu_probe").external_attr >> 16) & 0o100,
                            "the probe must be executable")
            info = f"y_lang-{version}.dist-info"
            self.assertEqual(z.read(f"{info}/licenses/LICENSE"), (ROOT / "LICENSE").read_bytes())
            self.assertEqual(z.read(f"{info}/licenses/NOTICE"), (ROOT / "NOTICE").read_bytes())
            self.assertEqual(z.read(f"{info}/licenses/THIRD_PARTY_NOTICES"),
                             (PKG / "THIRD_PARTY_NOTICES").read_bytes())
            self.assertEqual(z.read("y_lang/RUST_STD_COPYRIGHT.html"), rust_std_notice().read_bytes())
            meta = z.read(f"{info}/METADATA").decode()
            self.assertIn("License-Expression: BUSL-1.1", meta)
            self.assertIn("Root-Is-Purelib: false", z.read(f"{info}/WHEEL").decode())

        # A profile in the working directory is still the one used: the
        # compile neither probes nor needs a GPU, and nothing is written to
        # the user cache's profile.
        (work / ".ysu_hw_profile").write_text("SM_VERSION=8.9\nSM_COUNT=66\n", encoding="utf-8")
        result = run("auto")
        self.assertEqual(result["ptx"], ["sm_89", "ENTRY"])
        self.assertEqual(result["profile"], str(work / ".ysu_hw_profile"))
        self.assertFalse((self.tmp / "xdg" / "y-lang" / "hw_profile").exists())

    def test_installed_package_probes_into_the_user_cache(self):
        # No profile in the working directory, and no probe there: the one
        # run must be the one beside the installed liby.so (without it the
        # profile says SM_VERSION=0.0 and the kernel targets sm_80), and the
        # profile and the PTX cache go to the user cache, not the working
        # directory.
        site, work, run = self.install()
        cache = self.tmp / "xdg" / "y-lang"
        result = run("auto")
        self.assertEqual(result["ptx"], ["sm_86", "ENTRY"])
        self.assertEqual(result["profile"], str(cache / "hw_profile"))
        self.assertIn("GPU_NAME=FakeProbeGPU", (cache / "hw_profile").read_text(encoding="utf-8"))
        self.assertEqual(result["cache_dir"], str(cache / "ptx"))
        self.assertEqual(sorted(os.listdir(work)), [], "the working directory must stay empty")
        # The first compile probed, so it was not cached; the second was
        # keyed on the new profile and stored; the third is a hit.
        self.assertEqual((result["misses"], result["hits"]), (2, 1), result)
        self.assertTrue(list((cache / "ptx").glob("*.ptx")))
        if result["exact_gemm"] is not None:
            self.assertEqual(result["exact_gemm"], str(cache / "exact_gemm"))

    def test_ysu_hw_profile_names_the_profile(self):
        site, work, run = self.install()
        (work / ".ysu_hw_profile").write_text("SM_VERSION=8.9\nSM_COUNT=66\n", encoding="utf-8")
        named = self.tmp / "named_profile"
        named.write_text("SM_VERSION=8.0\nSM_COUNT=10\n", encoding="utf-8")
        result = run("auto", YSU_HW_PROFILE=str(named))
        self.assertEqual(result["ptx"], ["sm_80", "ENTRY"])
        self.assertEqual(result["profile"], str(named))

    def test_ysu_cache_dir_holds_the_profile_too(self):
        site, work, run = self.install()
        cache = self.tmp / "ysu_cache"
        result = run("auto", YSU_CACHE_DIR=str(cache))
        self.assertEqual(result["ptx"], ["sm_86", "ENTRY"])
        self.assertEqual(result["profile"], str(cache / "hw_profile"))
        self.assertEqual(result["cache_dir"], str(cache))
        if result["exact_gemm"] is not None:
            self.assertEqual(result["exact_gemm"], str(cache / "exact_gemm"))
        self.assertEqual(sorted(os.listdir(work)), [])

    def test_refuses_a_library_older_than_a_source(self):
        self.age(self.lib, 0)
        self.age(self.source, 10)
        built = self.build()
        self.assertNotEqual(built.returncode, 0)
        self.assertIn("is older than src/lib.rs", built.stdout + built.stderr)
        self.assertEqual(list(self.out.glob("*.whl")) if self.out.exists() else [], [])

    def dep_info(self, artifact, *sources):
        """Cargo's dep-info for `artifact`, naming `sources`, with the
        escaped spaces cargo writes."""
        esc = lambda p: str(p).replace(" ", "\\ ")
        artifact.with_suffix(".d").write_text(
            f"{esc(artifact)}: {' '.join(esc(s) for s in sources)}\n", encoding="utf-8")

    def test_dep_info_limits_the_sources_to_the_ones_used(self):
        # A file only another binary is built from (here src/main.rs) is newer
        # than the library; cargo would not rebuild the library for it, so it
        # must not make the library stale. lib.rs, which the library is built
        # from, must.
        main = self.root / "src" / "main.rs"
        main.write_text("// only the executable\n", encoding="utf-8")
        self.dep_info(self.lib, self.root / "Cargo.toml", self.source)
        self.dep_info(self.probe, self.root / "Cargo.toml", self.source)
        self.age(self.source, 0)
        self.age(self.lib, 10)
        self.age(self.probe, 10)
        self.age(main, 20)
        built = self.build()
        self.assertEqual(built.returncode, 0, built.stderr[-3000:])
        shutil.rmtree(self.out)
        self.age(self.source, 30)
        built = self.build()
        self.assertNotEqual(built.returncode, 0)
        self.assertIn("is older than src/lib.rs", built.stdout + built.stderr)
        self.assertNotIn("main.rs", built.stdout + built.stderr)

    def test_dep_info_naming_a_missing_file_falls_back_to_every_source(self):
        # As a build in a container mounted elsewhere writes it: then every
        # file under src/ counts, main.rs included.
        main = self.root / "src" / "main.rs"
        main.write_text("// only the executable\n", encoding="utf-8")
        self.dep_info(self.lib, Path("/no/such/checkout/src/lib.rs"))
        self.age(self.source, 0)
        self.age(self.lib, 10)
        self.age(self.probe, 10)
        self.age(main, 20)
        built = self.build()
        self.assertNotEqual(built.returncode, 0)
        self.assertIn("is older than src/main.rs", built.stdout + built.stderr)

    def test_refuses_a_missing_library(self):
        self.lib.unlink()
        built = self.build()
        self.assertNotEqual(built.returncode, 0)
        self.assertIn(f"no compiler library at {self.lib}", built.stdout + built.stderr)

    def test_refuses_without_the_rust_std_notice(self):
        self.age(self.source, 0)
        self.age(self.lib, 10)
        self.age(self.probe, 10)
        missing = self.tmp / "no-such-notice.html"
        built = self.build(Y_WHEEL_RUST_NOTICE=str(missing))
        self.assertNotEqual(built.returncode, 0)
        self.assertIn(f"no Rust standard library notice at {missing}", built.stdout + built.stderr)

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


@unittest.skipIf(shutil.which("cargo") is None, "cargo is not installed")
class TestThirdPartyNotices(unittest.TestCase):
    def test_notices_match_the_linked_crates(self):
        # Fails when Cargo.lock changes the crates in the binaries and the
        # notices were not regenerated.
        ran = subprocess.run([sys.executable, str(ROOT / "tools" / "third_party_notices.py"), "--check"],
                             capture_output=True, text=True, timeout=300)
        self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)

    def test_every_linked_crate_has_its_licence_text(self):
        text = (PKG / "THIRD_PARTY_NOTICES").read_text(encoding="utf-8")
        sections = text.split("=" * 78)[1:]
        self.assertGreaterEqual(len(sections) // 2, 1)
        for head, body in zip(sections[0::2], sections[1::2]):
            name = head.strip().splitlines()[0]
            self.assertTrue("Permission is hereby granted" in body or "Apache License" in body,
                            f"{name} has no licence text")


if __name__ == "__main__":
    unittest.main()
