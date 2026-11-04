"""Debug compiles from Python: Y_PTX_DEBUG / Y_PTX_LINEINFO, through the real
liby.so (`y_compile_to_ptx_debug`); no GPU needed.

* `-g` and `--lineinfo` PTX describe the source file y_lang writes, and their
  instructions are the plain compile's.
* The cache keeps the three apart.
* A library without the entry point refuses a debug compile rather than
  compiling without the debug information.
* The CPU interpreter runs a line-table or `-g` module as it runs the plain
  one. It used to DROP the instruction after every `.loc` - the directive has
  no `;`, so the two fused and were read as the directive - which nothing
  could see until Python could ask for a line table.
"""

import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from y_lang import compiler

SCALE = """kernel scale(X: GlobalMemory<F32>, Out: GlobalMemory<F32>, N: I32) {
    let i: I32 = thread_idx_x();
    if i < N {
        let v: F32 = X[i];
        Out[i] = v * 3.0;
    }
}
"""

# No branch: an instruction the interpreter drops is a wrong value, not a
# missing label.
STRAIGHT = """kernel straight(X: GlobalMemory<F32>, Out: GlobalMemory<F32>) {
    let i: I32 = thread_idx_x();
    let a: F32 = X[i];
    let b: F32 = a * 2.0;
    let c: F32 = b + 1.0;
    Out[i] = c;
}
"""


def instructions(ptx):
    """The module without what debug information adds: the labels `-g`
    names, its `debug`, its sections, the line table. Written here, not
    shared with tools/ydb/ydwarf.py."""
    out, in_section = [], False
    for line in ptx.splitlines():
        t = line.strip()
        if in_section:
            in_section = t != "}"
            continue
        if t.startswith(".section"):
            in_section = True
            continue
        if re.fullmatch(r"\$L__ydbg\d+:", t) or t.startswith(".loc ") or t.startswith(".file "):
            continue
        out.append(line.replace(", debug", "") if t.startswith(".target ") else line)
    return "\n".join(out)


class TestPtxDebug(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="y-ptx-debug-")
        self.addCleanup(self.directory.cleanup)
        env = mock.patch.dict(os.environ, {"YSU_CACHE_DIR": self.directory.name})
        env.start()
        self.addCleanup(env.stop)
        for var in ("Y_PTX_DEBUG", "Y_PTX_LINEINFO"):
            os.environ.pop(var, None)
        saved = compiler._JIT_CACHE.copy()
        compiler._JIT_CACHE.clear()
        self.addCleanup(lambda: (compiler._JIT_CACHE.clear(), compiler._JIT_CACHE.update(saved)))

    def test_the_environment_chooses_the_debug_information(self):
        self.assertIsNone(compiler.debug_mode())
        with mock.patch.dict(os.environ, {"Y_PTX_LINEINFO": "1"}):
            self.assertEqual(compiler.debug_mode(), "lineinfo")
        with mock.patch.dict(os.environ, {"Y_PTX_DEBUG": "1", "Y_PTX_LINEINFO": "1"}):
            self.assertEqual(compiler.debug_mode(), "g")
            self.assertIsNone(compiler.debug_mode(False), "debug=False must override the environment")
        # Y_DEBUG selects the CPU interpreter; it is not this.
        with mock.patch.dict(os.environ, {"Y_DEBUG": "1"}):
            self.assertIsNone(compiler.debug_mode())
        with self.assertRaises(ValueError):
            compiler.debug_mode("full")

    def test_a_g_compile_describes_the_file_it_writes(self):
        ptx = compiler.compile_to_ptx(SCALE, debug="g")
        path = compiler.debug_source_path(SCALE)
        self.assertEqual(path.name, "scale.ysu")
        self.assertEqual(path.read_text(), SCALE)
        target = [l for l in ptx.splitlines() if l.startswith(".target ")]
        self.assertEqual(len(target), 1)
        self.assertTrue(target[0].endswith(", debug"), target)
        self.assertIn(".section\t.debug_info", ptx)
        files = re.findall(r'^\.file\s+1\s+"([^"]+)"', ptx, re.M)
        self.assertEqual([Path(f).resolve() for f in files], [path.resolve()])
        # `ydb`'s `verify` reads the PTX beside the source.
        self.assertEqual(path.with_suffix(".ptx").read_text(), ptx)

    def test_a_lineinfo_compile_has_lines_and_nothing_else(self):
        ptx = compiler.compile_to_ptx(SCALE, debug="lineinfo")
        self.assertIn(".loc 1 ", ptx)
        self.assertNotIn(".debug_info", ptx)
        self.assertNotIn(", debug", ptx)
        self.assertFalse(compiler.debug_source_path(SCALE).with_suffix(".ptx").exists())

    def test_the_instructions_are_the_plain_compiles(self):
        plain = compiler.compile_to_ptx(SCALE)
        self.assertNotIn(".loc", plain)
        for mode in ("g", "lineinfo"):
            self.assertEqual(instructions(compiler.compile_to_ptx(SCALE, debug=mode)), instructions(plain), mode)

    def test_the_cache_keeps_the_modes_apart(self):
        g = compiler.compile_to_ptx(SCALE, debug="g")
        self.assertNotIn(".debug_info", compiler.compile_to_ptx(SCALE))
        self.assertNotIn(".debug_info", compiler.compile_to_ptx(SCALE, debug="lineinfo"))
        self.assertEqual(compiler.compile_to_ptx(SCALE, debug="g"), g)
        # A disk hit writes the PTX beside the source again.
        compiler._JIT_CACHE.clear()
        beside = compiler.debug_source_path(SCALE).with_suffix(".ptx")
        beside.unlink()
        self.assertEqual(compiler.compile_to_ptx(SCALE, debug="g"), g)
        self.assertEqual(beside.read_text(), g)

    def test_a_library_without_the_entry_point_refuses(self):
        stale = SimpleNamespace(lib=SimpleNamespace(y_compile_to_ptx=None), lib_path="/old/liby.so",
                                compiler_identity="old")
        with mock.patch.object(compiler.YCompilerLib, "get_instance", return_value=stale):
            with self.assertRaisesRegex(RuntimeError, "has no y_compile_to_ptx_debug.*rebuild"):
                compiler.compile_to_ptx(SCALE, debug="g")

    def test_the_c_api_refuses_what_it_cannot_describe(self):
        import ctypes
        lib = compiler.YCompilerLib.get_instance().lib
        for path, level, why in [(b"/tmp/k.ysu", 3, b"debug level must be 0"),
                                 (None, 2, b"needs the path of the source file"),
                                 (b"", 1, b"needs the path of the source file")]:
            err = ctypes.c_char_p()
            out = lib.y_compile_to_ptx_debug(SCALE.encode(), b"auto", path, level, ctypes.byref(err))
            self.assertIsNone(out, (path, level))
            self.assertIn(why, err.value, (path, level))
            lib.y_free_string(err)
        # Level 0 is the plain compile, path or not.
        err = ctypes.c_char_p()
        out = lib.y_compile_to_ptx_debug(SCALE.encode(), b"auto", None, 0, ctypes.byref(err))
        self.assertIsNotNone(out)
        text = ctypes.cast(out, ctypes.c_char_p).value.decode()
        lib.y_free_string(out)
        self.assertNotIn(".loc", text)
        self.assertIn(".entry scale", text)

    def test_the_interpreter_runs_debug_ptx_as_the_plain_ptx(self):
        try:
            import torch
        except ImportError:
            self.skipTest("no torch")
        from y_lang.interpreter import CPUInterpreter
        cases = [
            ("scale", SCALE, lambda x: [3 * v if i < 6 else 0.0 for i, v in enumerate(x)], 6),
            ("straight", STRAIGHT, lambda x: [2 * v + 1 for v in x], None),
        ]
        for name, src, expect, n in cases:
            for mode in (False, "lineinfo", "g"):
                x = torch.arange(8, dtype=torch.float32)
                out = torch.zeros(8)
                args = [x, out] + ([n] if n is not None else [])
                CPUInterpreter(name, ptx=compiler.compile_to_ptx(src, debug=mode)).run((1, 1, 1), (8, 1, 1), args)
                self.assertEqual(out.tolist(), expect(x.tolist()), f"{name} with debug={mode}")


if __name__ == "__main__":
    unittest.main()
