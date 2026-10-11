"""The GPU probe beside liby.so is found from any working directory.

With no `.ysu_hw_profile` in the working directory the compiler probes the
hardware and writes one there. It looked for `ysu_gpu_probe` in the working
directory and then beside `current_exe()` - which inside Python is the
interpreter - so a compile through the C API never found the probe built next
to the library. Measured on a machine with an RTX 4070 Ti SUPER: the profile
it wrote said `GPU_NAME=Unknown GPU`, `SM_VERSION=0.0`, the PTX targeted
`sm_80`, and every later compile from that directory read that profile.

Each run copies the built library to `<root>/target/release/liby.so`, puts a
stub probe beside it that records that it ran and reports a GPU no real card
is named, and compiles from an unrelated, profile-less directory. The control
is the same copy without the stub.
"""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from y_lang import compiler

PROGRAM = '''
import y_lang
src = """kernel k(A: GlobalMemory<F32>) {
    st_global_v4_f32(A, 1.0);
}
"""
ptx = y_lang.compile_to_ptx(src)
print("TARGET", [l for l in ptx.splitlines() if l.startswith(".target")][0])
'''


@unittest.skipUnless(sys.platform.startswith("linux"), "the stub probe is a POSIX shell script")
class TestGpuProbeBesideTheLibrary(unittest.TestCase):
    def setUp(self):
        try:
            self.library = Path(compiler._find_liby()).resolve()
        except (OSError, RuntimeError) as error:
            self.skipTest(f"no built liby.so to copy: {error}")
        self.directory = tempfile.TemporaryDirectory(prefix="y-probe-lookup-")
        self.addCleanup(self.directory.cleanup)
        self.scratch = Path(self.directory.name)

    def compile_beside(self, tag, with_stub):
        root = self.scratch / tag
        lib_dir = root / "target" / "release"
        lib_dir.mkdir(parents=True)
        library = shutil.copy(self.library, lib_dir / "liby.so")
        marker = root / "stub_ran"
        if with_stub:
            stub = lib_dir / "ysu_gpu_probe"
            stub.write_text(
                f"#!/bin/sh\n: > '{marker}'\n"
                "echo GPU_NAME=StubBesideTheLibrary\necho SM_VERSION=8.6\necho COMPUTE_CAPABILITY=8.6\n"
            )
            stub.chmod(0o755)
        cwd = self.scratch / f"{tag}_cwd"
        cwd.mkdir()
        env = {
            "PYTHONPATH": str(Path(compiler.__file__).resolve().parents[1]),
            "Y_LIB_PATH": str(library),
            "PATH": "/usr/bin:/bin",
            "YSU_CACHE_DIR": str(cwd / "cache"),
        }
        result = subprocess.run(
            [sys.executable, "-c", PROGRAM], cwd=cwd, env=env, capture_output=True, text=True, timeout=600
        )
        profile = (cwd / ".ysu_hw_profile").read_text() if (cwd / ".ysu_hw_profile").exists() else ""
        return result, marker.exists(), profile

    def test_the_probe_beside_the_library_runs(self):
        result, ran, profile = self.compile_beside("stub", with_stub=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(ran, "the probe beside liby.so was not run:\n" + result.stdout)
        self.assertIn("GPU_NAME=StubBesideTheLibrary", profile)
        self.assertIn("TARGET .target sm_86", result.stdout)

    def test_without_a_probe_the_profile_names_no_gpu(self):
        # The control: the stub's answer above came from the stub, not from
        # anything else this machine has.
        result, ran, profile = self.compile_beside("bare", with_stub=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(ran)
        self.assertNotIn("StubBesideTheLibrary", profile)
        self.assertNotIn("TARGET .target sm_86", result.stdout)


if __name__ == "__main__":
    unittest.main()
