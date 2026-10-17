"""The native launch path (`y_lang/_launch.c`) against the ctypes one it
replaces.

`pack` returns the parameter slots `launch` passes, so the conversion of
every argument is compared with the ctypes path's (`_fill_slots`) byte for
byte, errors included, without a GPU. On the GPU: a kernel with more
parameters than the C path keeps on its stack, a launch the driver refuses,
and, in child processes (the choice is made once a process), the fallback
when the module cannot be compiled and the settings of `Y_LAUNCHER`.
test_launch.py runs its stream and shared-memory tests through both paths.
"""

import ctypes
import os
import subprocess
import sys
import tempfile
import unittest

import torch

from y_lang import torch_interop
from y_lang.torch_interop import TorchKernel


def native_module():
    module, why = torch_interop._build_native_launcher()
    return module, why


class Ptr:
    def __init__(self, v):
        self.v = v

    def data_ptr(self):
        return self.v


class Index:
    def __index__(self):
        return 42


class NotCallable:
    data_ptr = 5


class IntWithAttribute(int):
    data_ptr = 3


class IntWithPointer(int):
    def data_ptr(self):
        return 99


class FloatWithPointer(float):
    def data_ptr(self):
        return 12


class FloatSubclass(float):
    pass


class RaisesInLookup:
    @property
    def data_ptr(self):
        raise KeyError("lookup")


class MissingByProperty:
    @property
    def data_ptr(self):
        raise AttributeError("missing")


class RaisesInCall:
    def data_ptr(self):
        raise ValueError("call")


# Each is one kernel argument: ints past 2**64 and below 0 (taken modulo
# 2**64), the float32 overflow boundary (3.4028235677973366e38 is the first
# double struct.pack refuses), subnormals and both NaN signs, a `data_ptr`
# that wins over an int or float value, attributes that are not callable or
# whose lookup raises, and types neither path accepts.
VALUES = [
    0, 1, -1, 2**31, 2**63, 2**64 - 1, 2**64, 2**64 + 5, -(2**64) - 3, True, False,
    0.0, -0.0, 1.5, float("inf"), float("-inf"), float("nan"), -float("nan"),
    3.4028234663852886e38, 3.4028235677973362e38, 3.4028235677973366e38, 3.5e38, -3.5e38, 1e308,
    1e-46, 1.4e-45, 7e-46, 1e-39,
    FloatSubclass(2.25), FloatWithPointer(1.0), IntWithAttribute(7), IntWithPointer(8),
    Ptr(0x7F0000001000), Ptr(-1), Ptr(2**64 + 1), Ptr(Index()), Ptr(True), Ptr(1.5), Ptr(None),
    "x", None, b"", [], 1j, Index(), NotCallable(), RaisesInLookup(), MissingByProperty(), RaisesInCall(),
]


def through_ctypes(args):
    slots = [ctypes.c_uint64(0xDEADBEEF) for _ in args]
    torch_interop._fill_slots(slots, args)
    return b"".join(bytes(s) for s in slots)


def outcome(fill, args):
    try:
        return "ok", fill(args)
    except Exception as e:
        return type(e).__name__, str(e)


class TheNativeSlotsAreTheCtypesSlots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module, why = native_module()
        if cls.module is None:
            raise unittest.SkipTest(f"the native launcher could not be built: {why}")

    def test_each_argument_alone(self):
        for v in VALUES:
            with self.subTest(value=repr(v)[:40]):
                self.assertEqual(outcome(self.module.pack, [v]), outcome(through_ctypes, [v]))

    def test_every_argument_at_once_on_the_heap_and_on_the_stack(self):
        # 33 or more slots are allocated; 32 fit on the C path's stack.
        good = [v for v in VALUES if outcome(through_ctypes, [v])[0] == "ok"]
        self.assertGreater(len(good), 32)
        for args in (good, good[:32], good[:1], [], tuple(good)):
            with self.subTest(n=len(args)):
                self.assertEqual(outcome(self.module.pack, args), outcome(through_ctypes, args))

    def test_the_first_bad_argument_decides_the_error(self):
        args = [1, 2.5, "x", None]
        self.assertEqual(outcome(self.module.pack, args), outcome(through_ctypes, args))

    def test_a_changed_source_is_a_different_module(self):
        # A module cached from an older `_launch.c` must not be loaded for a
        # newer one.
        saved = os.environ.get("YSU_CACHE_DIR")
        with tempfile.TemporaryDirectory() as d:
            os.environ["YSU_CACHE_DIR"] = d
            try:
                copy = os.path.join(d, "changed.c")
                with open(os.path.join(os.path.dirname(torch_interop.__file__), "_launch.c")) as f:
                    source = f.read()
                with open(copy, "w") as f:
                    f.write(source + "\n/* changed */\n")
                original, why = torch_interop._build_native_launcher()
                self.assertIsNotNone(original, why)
                changed, why = torch_interop._build_native_launcher(copy)
                self.assertIsNotNone(changed, why)
                self.assertNotEqual(changed.__file__, original.__file__)
            finally:
                if saved is None:
                    os.environ.pop("YSU_CACHE_DIR")
                else:
                    os.environ["YSU_CACHE_DIR"] = saved


# What `launch` hands the driver, checked without a GPU: the module's
# cuLaunchKernel is replaced by a ctypes callback that records its
# arguments. Run in this process on a private copy of the module (the real
# one keeps its driver), and in a child under AddressSanitizer, which is what
# sees a slot written past the C path's stack arrays: those bytes land in
# memory nothing reads.
DRIVER_CHECK = """
import ctypes, struct
def check(module):
    calls = []
    proto = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, *[ctypes.c_uint] * 7, ctypes.c_void_p,
                             ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p)
    def driver(function, gx, gy, gz, bx, by, bz, shared, stream, params, extra):
        n = calls[-1]
        slots = [ctypes.cast(params[i], ctypes.POINTER(ctypes.c_uint64))[0] for i in range(n)]
        calls.append((function or 0, (gx, gy, gz, bx, by, bz), shared, stream or 0, slots, extra))
        return n % 7
    callback = proto(driver)
    module.set_driver(ctypes.cast(callback, ctypes.c_void_p).value)
    checked = 0
    for n in (0, 1, 31, 32, 33, 40, 64, 65, 200):
        args = [(i * 0x9E3779B97F4A7C15) % 2**64 if i % 3 else i / 7.0 for i in range(n)]
        expected = list(struct.unpack(f"<{n}Q", module.pack(args)))
        for function, stream in ((0x1234, 0x5678), (None, None)):
            calls.append(n)
            result = module.launch(function, 3, 5, 7, 11, 13, 1, 4096 + n, stream, args)
            got = calls.pop()
            calls.pop()
            want = (function or 0, (3, 5, 7, 11, 13, 1), 4096 + n, stream or 0, expected, None)
            assert got == want, (n, got[:4], want[:4])
            assert result == n % 7, (n, result)
            checked += 1
    return checked
"""


def isolated_module(directory):
    """A copy of `_ylaunch` with its own driver pointer, built in
    `directory` from a copy of the source."""
    copy = os.path.join(directory, "isolated.c")
    with open(os.path.join(os.path.dirname(torch_interop.__file__), "_launch.c")) as f:
        source = f.read()
    with open(copy, "w") as f:
        f.write(source + "\n/* isolated copy for TheDriverGetsWhatTheCallerPassed */\n")
    saved = os.environ.get("YSU_CACHE_DIR")
    os.environ["YSU_CACHE_DIR"] = directory
    try:
        return torch_interop._build_native_launcher(copy)
    finally:
        if saved is None:
            os.environ.pop("YSU_CACHE_DIR")
        else:
            os.environ["YSU_CACHE_DIR"] = saved


ASAN_CHILD = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("_ylaunch", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
exec(sys.argv[2])
print("CHECKED", check(module))
"""


class TheDriverGetsWhatTheCallerPassed(unittest.TestCase):
    def test_function_dimensions_shared_memory_stream_and_every_slot(self):
        with tempfile.TemporaryDirectory() as d:
            module, why = isolated_module(d)
            if module is None:
                self.skipTest(f"the native launcher could not be built: {why}")
            namespace = {}
            exec(DRIVER_CHECK, namespace)
            self.assertEqual(namespace["check"](module), 18)

    def test_no_slot_is_written_out_of_bounds(self):
        import shutil
        import sysconfig
        cc = shutil.which("gcc")
        if cc is None:
            self.skipTest("no gcc for an AddressSanitizer build; the bounds check did not run")
        asan = subprocess.run([cc, "-print-file-name=libasan.so"], capture_output=True, text=True).stdout.strip()
        if not os.path.isabs(asan) or not os.path.exists(asan):
            self.skipTest("gcc has no libasan; the bounds check did not run")
        with tempfile.TemporaryDirectory() as d:
            built = os.path.join(d, "_ylaunch" + sysconfig.get_config_var("EXT_SUFFIX"))
            compiled = subprocess.run([cc, "-O1", "-g", "-fsanitize=address", "-fno-omit-frame-pointer", "-shared",
                                       "-fPIC", "-I", sysconfig.get_paths()["include"],
                                       os.path.join(os.path.dirname(torch_interop.__file__), "_launch.c"),
                                       "-o", built], capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr[-2000:])
            env = dict(os.environ, LD_PRELOAD=asan, ASAN_OPTIONS="detect_leaks=0")
            out = subprocess.run([sys.executable, "-c", ASAN_CHILD, built, DRIVER_CHECK], capture_output=True,
                                 text=True, env=env, cwd=d, timeout=300)
        text = out.stdout + out.stderr
        self.assertNotIn("AddressSanitizer", text, text[-3000:])
        self.assertIn("CHECKED 18", text, text[-3000:])
        self.assertEqual(out.returncode, 0, text[-3000:])

VEC_ADD = """
kernel vec_add(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, N: I32) {
    let idx: I32 = global_thread_id() * 4;
    if idx < N {
        vec_add_v4(A[idx], B[idx], C[idx]);
    }
}
"""


def many_parameters_kernel(n):
    """A kernel storing each of its `n` I32 parameters and one F32: n + 3
    parameters in all."""
    params = ", ".join(f"p{i}: I32" for i in range(n))
    stores = "\n".join(f"        Out[{i}] = p{i};" for i in range(n))
    return TorchKernel(f"""
kernel many(Out: GlobalMemory<I32>, OutF: GlobalMemory<F32>, x: F32, {params}) {{
    if global_thread_id() == 0 {{
{stores}
        OutF[0] = x;
    }}
}}
""", kernel_name="many", target_sm="sm_80")


class WithLauncher:
    """Runs a test class's kernels through the launcher `LAUNCHER` names."""
    LAUNCHER = "native"

    def setUp(self):
        saved = os.environ.get("Y_LAUNCHER")
        os.environ["Y_LAUNCHER"] = self.LAUNCHER
        self.addCleanup(lambda: os.environ.pop("Y_LAUNCHER") if saved is None
                        else os.environ.__setitem__("Y_LAUNCHER", saved))
        if self.LAUNCHER == "native" and native_module()[0] is None:
            self.skipTest(f"the native launcher could not be built: {native_module()[1]}")
        super().setUp()


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class LaunchesThroughTheNativePath(WithLauncher, unittest.TestCase):
    def setUp(self):
        super().setUp()
        # A kernel loads into the context PyTorch makes current with its
        # first allocation.
        torch.zeros(1, device="cuda")

    def loaded(self, k):
        k._ensure_loaded()
        self.assertEqual(k._native_launch is not None, self.LAUNCHER == "native")
        return k

    def test_more_parameters_than_the_stack_holds(self):
        for n in (29, 30, 37):  # 32, 33 and 40 parameters
            with self.subTest(parameters=n + 3):
                k = self.loaded(many_parameters_kernel(n))
                out = torch.full((n,), -7, dtype=torch.int32, device="cuda")
                out_f = torch.zeros(1, device="cuda")
                values = [(i * 7919) % 100003 - 50000 for i in range(n)]
                k.launch((1, 1, 1), (32, 1, 1), [out, out_f, -2.5] + values)
                torch.cuda.synchronize()
                self.assertEqual(out.tolist(), values)
                self.assertEqual(out_f.item(), -2.5)

    def test_a_launch_the_driver_refuses_raises(self):
        k = self.loaded(TorchKernel(VEC_ADD, kernel_name="vec_add", target_sm="sm_80"))
        a = torch.zeros(1024, device="cuda")
        # 2048 threads in a block: CUDA_ERROR_INVALID_VALUE.
        with self.assertRaisesRegex(RuntimeError, "cuLaunchKernel failed with CUDA error code 1$"):
            k.launch((1, 1, 1), (2048, 1, 1), [a, a, a, 1024])
        # Refused at launch, nothing ran: the context still works.
        b = torch.ones(1024, device="cuda")
        c = torch.zeros(1024, device="cuda")
        k.launch((2, 1, 1), (128, 1, 1), [b, b, c, 1024])
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(c, b + b))

    def test_an_unsupported_argument_launches_nothing(self):
        k = self.loaded(TorchKernel(VEC_ADD, kernel_name="vec_add", target_sm="sm_80"))
        a = torch.ones(1024, device="cuda")
        c = torch.zeros(1024, device="cuda")
        with self.assertRaisesRegex(TypeError, "Unsupported argument type: <class 'str'>"):
            k.launch((2, 1, 1), (128, 1, 1), [a, a, c, "1024"])
        torch.cuda.synchronize()
        self.assertEqual(int(c.count_nonzero()), 0)


class LaunchesThroughCtypes(LaunchesThroughTheNativePath):
    LAUNCHER = "ctypes"


CHILD = """
import os, sys, torch
from y_lang import torch_interop
from y_lang.torch_interop import TorchKernel
k = TorchKernel({src!r}, kernel_name="vec_add", target_sm="sm_80")
a = torch.arange(1024, device="cuda", dtype=torch.float32)
c = torch.zeros_like(a)
try:
    k.launch((2, 1, 1), (128, 1, 1), [a, a, c, 1024])
except Exception as e:
    print("RAISED", type(e).__name__, e)
    sys.exit(0)
torch.cuda.synchronize()
print("EXACT", torch.equal(c, a + a))
print("STATUS", torch_interop.launcher_status())
"""


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class TheLauncherIsChosenOncePerProcess(unittest.TestCase):
    def child(self, cache=None, **env):
        with tempfile.TemporaryDirectory() as fresh:
            cache = cache or fresh
            full = dict(os.environ, YSU_CACHE_DIR=cache)
            full.pop("Y_LAUNCHER", None)
            full.update(env)
            out = subprocess.run([sys.executable, "-c", CHILD.format(src=VEC_ADD)], capture_output=True,
                                 text=True, env=full, timeout=300)
            built = {f: os.stat(os.path.join(cache, f)).st_mtime_ns
                     for f in os.listdir(cache) if f.startswith("_ylaunch_")}
        lines = [ln for ln in out.stdout.splitlines() if ln.startswith(("RAISED", "EXACT", "STATUS"))]
        self.assertTrue(lines, out.stdout[-2000:] + out.stderr[-2000:])
        return lines, built

    def test_a_compiler_that_fails_falls_back_to_ctypes(self):
        lines, built = self.child(CC="false")
        self.assertEqual(lines[0], "EXACT True")
        self.assertRegex(lines[1], r"^STATUS ctypes: false could not compile .*_launch\.c")
        self.assertEqual(built, {})

    def test_native_is_refused_where_it_cannot_be_built(self):
        lines, _ = self.child(CC="false", Y_LAUNCHER="native")
        self.assertRegex(lines[0], r"^RAISED RuntimeError Y_LAUNCHER=native, and the native launcher is unavailable")

    def test_a_misspelt_setting_is_refused(self):
        lines, _ = self.child(Y_LAUNCHER="Native")
        self.assertRegex(lines[0], r"^RAISED ValueError Y_LAUNCHER='Native'")

    def test_ctypes_builds_nothing(self):
        lines, built = self.child(Y_LAUNCHER="ctypes")
        self.assertEqual(lines, ["EXACT True", "STATUS None"])
        self.assertEqual(built, {})

    def test_the_default_builds_once_and_launches_natively(self):
        if native_module()[0] is None:
            self.skipTest(f"the native launcher could not be built: {native_module()[1]}")
        with tempfile.TemporaryDirectory() as cache:
            first, built = self.child(cache)
            self.assertEqual(first[0], "EXACT True")
            self.assertRegex(first[1], r"^STATUS native: .*_ylaunch_[0-9a-f]{24}\.")
            self.assertEqual(len(built), 1, built)
            # A second process loads the module the first built.
            second, again = self.child(cache)
            self.assertEqual(second, first)
            self.assertEqual(again, built)

if __name__ == "__main__":
    unittest.main()
