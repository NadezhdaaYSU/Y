"""`TorchKernel.launch`: what it launches, with how much shared memory, on
which stream.

The launch passed no dynamic shared memory, so every kernel staging through
`.extern .shared` (Y's F16 tile GEMMs) died with an illegal address when
launched from Python; a kernel name the module lacked ran an entry named
`main` instead, if there was one; and each launch built a `Stream` object
and asked twice whether CUDA was available, 6.6 us a call against Triton's
5.7; it was 2.9 after that (docs/benchmarks/launch_2026-10-12/). The GPU
tests run through both launch paths, the native one (`_launch.c`) and
ctypes; test_native_launch.py compares the two.
"""

import ctypes
import os
import re
import tempfile
import unittest

import torch

from test_native_launch import WithLauncher
from y_lang.compiler import is_interpreter_enabled
from y_lang.torch_interop import TorchKernel, _get_cuda_driver

VEC_ADD = """
kernel vec_add(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, N: I32) {
    let idx: I32 = global_thread_id() * 4;
    if idx < N {
        vec_add_v4(A[idx], B[idx], C[idx]);
    }
}
"""


# The profile the GEMMs are compiled against, as tests/common/pinned.rs pins
# one: an sm_89 card with its 100 KB of shared memory per SM (a probe-less
# profile says 48 KB, and the GEMM's stages are clamped to fit it), and the
# recorded @ZeroDrift costs, so that nothing is measured.
PINNED_PROFILE = """SM_VERSION=8.9
COMPUTE_CAPABILITY=8.9
GPU_NAME=PinnedByTest
SM_COUNT=66
MAX_SMEM_PER_SM_BYTES=102400
DRIFT_ACC_Q16.16_PinnedByTest=1790.000
DRIFT_ACC_Q32.32_PinnedByTest=1922.000
DRIFT_ACC_I64_PinnedByTest=2106.000
DRIFT_ACC_F64_PinnedByTest=17726.000
"""


def f16_tile_gemm(n, tile):
    """The F16 tile GEMM at `n`, its CTA tile pinned by Y_CTA_OVERRIDE
    (cta_m,cta_n,cta_k,warps_m,warps_n,stages), which the autotuner reads
    before the profile's measured tiles or its in-process cache, and compiled
    against PINNED_PROFILE: otherwise the tile, and so the shared memory,
    would depend on this checkout's `.ysu_hw_profile`."""
    src = (f"@tile({n}, {n}, {n})\nkernel g(A: GlobalMemory<F16>, B: GlobalMemory<F16>, "
           f"Bias: GlobalMemory<F32>, C: GlobalMemory<F32>) {{\n}}\n")
    saved_env, saved_cwd = os.environ.get("Y_CTA_OVERRIDE"), os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, ".ysu_hw_profile"), "w") as f:
            f.write(PINNED_PROFILE)
        os.environ["Y_CTA_OVERRIDE"] = tile
        os.chdir(d)
        try:
            k = TorchKernel(src, kernel_name="g", target_sm="sm_89")
        finally:
            os.chdir(saved_cwd)
            if saved_env is None:
                del os.environ["Y_CTA_OVERRIDE"]
            else:
                os.environ["Y_CTA_OVERRIDE"] = saved_env
    cm, cn, wm, wn = (int(x) for x in re.search(r"CTA (\d+)x(\d+)x\d+ \| (\d+)x(\d+) warps", k.ptx).groups())
    shared = int(re.search(r"Dynamic shared memory required: (\d+) bytes", k.ptx).group(1))
    return k, (n // cn, n // cm, 1), (wm * wn * 32, 1, 1), shared


def gpu_has_tensor_cores():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


@unittest.skipUnless(gpu_has_tensor_cores(), "needs an sm_89 or later GPU")
class LaunchesWhatThePtxNeeds(WithLauncher, unittest.TestCase):
    def run_gemm(self, n, tile):
        k, grid, block, shared = f16_tile_gemm(n, tile)
        gen = torch.Generator(device="cuda").manual_seed(n)
        # Small integers: every product and sum is exact in f16 x f16 -> f32.
        a = torch.randint(-4, 5, (n, n), device="cuda", generator=gen).half()
        b = torch.randint(-4, 5, (n, n), device="cuda", generator=gen).half()
        bias = torch.randint(-4, 5, (n,), device="cuda", generator=gen).float()
        c = torch.full((n, n), float("nan"), device="cuda")
        k.launch(grid, block, [a, b, bias, c])
        torch.cuda.synchronize()
        ref = torch.relu(a.double() @ b.double() + bias.double())
        wrong = int((c.double() != ref).sum())
        self.assertEqual(wrong, 0, f"{wrong} of {n * n} elements wrong at {n}")
        return shared

    def test_an_f16_tile_gemm_gets_its_dynamic_shared_memory(self):
        # 64x128x32 on 2x2 warps, 2 stages: 27,648 bytes.
        self.assertLessEqual(self.run_gemm(1024, "64,128,32,2,2,2"), 48 * 1024)

    def test_one_needing_more_than_48_kb_opts_in(self):
        # 64x64x64 on 4x2 warps, 4 stages of swizzled 16 KB tiles: 65,536
        # bytes. (Three stages were 55,296 bytes of padded tiles, and are
        # exactly 48 KB swizzled, which needs no opt-in.)
        self.assertGreater(self.run_gemm(512, "64,64,64,4,2,4"), 48 * 1024)


class LaunchesWhatThePtxNeedsThroughCtypes(LaunchesWhatThePtxNeeds):
    LAUNCHER = "ctypes"


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class LaunchesTheNamedKernelOnTheCallersStream(WithLauncher, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.n = 1 << 20
        self.a = torch.randn(self.n, device="cuda")
        self.b = torch.randn(self.n, device="cuda")
        self.grid, self.block = ((self.n // 4 + 127) // 128, 1, 1), (128, 1, 1)
        # A NON-BLOCKING stream. PyTorch's own streams are blocking, so a
        # kernel launched on the legacy default stream by mistake still waits
        # for them and computes the right answer; on this one it does not.
        cuda = _get_cuda_driver()
        handle = ctypes.c_void_p()
        self.assertEqual(cuda.cuStreamCreate(ctypes.byref(handle), 1), 0)
        self.side = torch.cuda.ExternalStream(handle.value)
        self.addCleanup(cuda.cuStreamDestroy_v2, handle)

    def loaded_vec_add(self):
        # Launched once first: loading the module waits for the GPU to go
        # idle, which would hide a launch on the wrong stream.
        k = TorchKernel(VEC_ADD, kernel_name="vec_add", target_sm="sm_80")
        k.launch(self.grid, self.block, [self.a, self.b, torch.zeros_like(self.a), self.n])
        torch.cuda.synchronize()
        return k

    def test_a_name_the_module_lacks_is_refused(self):
        # The module also holds `main`, which a missing name used to run.
        src = VEC_ADD + VEC_ADD.replace("kernel vec_add", "kernel main")
        k = TorchKernel(src, kernel_name="vec_adds", target_sm="sm_80")
        c = torch.zeros_like(self.a)
        with self.assertRaisesRegex(RuntimeError, "no kernel 'vec_adds'"):
            k.launch(self.grid, self.block, [self.a, self.b, c, self.n])

    def test_the_launch_follows_the_current_stream(self):
        k = self.loaded_vec_add()
        a, c = torch.zeros_like(self.a), torch.zeros_like(self.a)
        torch.cuda.synchronize()
        with torch.cuda.stream(self.side):
            # A kernel queued on another stream would run before this fill.
            torch.cuda._sleep(50_000_000)
            a.copy_(self.a)
            k.launch(self.grid, self.block, [a, self.b, c, self.n])
        self.side.synchronize()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(c, self.a + self.b))

    def test_an_explicit_stream_is_used(self):
        k = self.loaded_vec_add()
        a, c = torch.zeros_like(self.a), torch.zeros_like(self.a)
        torch.cuda.synchronize()
        with torch.cuda.stream(self.side):
            torch.cuda._sleep(50_000_000)
            a.copy_(self.a)
        k.launch(self.grid, self.block, [a, self.b, c, self.n], stream=self.side)
        self.side.synchronize()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(c, self.a + self.b))


class LaunchesTheNamedKernelOnTheCallersStreamThroughCtypes(LaunchesTheNamedKernelOnTheCallersStream):
    LAUNCHER = "ctypes"


class TheInterpreterSwitchIsReadEachTime(unittest.TestCase):
    def test_set_unset_and_false_values(self):
        saved = {v: os.environ.get(v) for v in ("Y_INTERPRETER", "Y_DEBUG")}
        try:
            for v in saved:
                os.environ.pop(v, None)
            self.assertFalse(is_interpreter_enabled())
            os.environ["Y_INTERPRETER"] = "1"
            self.assertTrue(is_interpreter_enabled())
            os.environ["Y_INTERPRETER"] = "no"
            self.assertFalse(is_interpreter_enabled())
            os.environ["Y_DEBUG"] = "TRUE"
            self.assertTrue(is_interpreter_enabled())
            del os.environ["Y_DEBUG"]
            del os.environ["Y_INTERPRETER"]
            self.assertFalse(is_interpreter_enabled())
        finally:
            for v, value in saved.items():
                if value is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = value


if __name__ == "__main__":
    unittest.main()
