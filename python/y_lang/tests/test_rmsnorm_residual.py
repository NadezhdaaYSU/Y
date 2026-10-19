"""`rmsnorm_residual_<d>` (fused residual add and RMSNorm, f16 in and out),
run on the GPU.

h = X + Residual in float32; NewResidual = f16(h), which must be exact;
Out = f16(h * rsqrt(mean(h^2) + 1e-5) * Weight), within 2^-10 of float64
(the f16 rounding plus `rsqrt.approx`). Outputs start as NaN, so an element
no thread writes fails, and three launches must agree bit for bit: an
earlier unrolled version of this kernel gave non-deterministic output.

The hidden sizes cover what the emitter's warp rule gives - 1 to 32 warps,
1 or 2 groups of 8 elements a thread, a last group past the row's end
(896, 3072, 5120, 12032), and at 65536 the runtime-loop path (8 groups a
thread on 1024 threads) - and the row counts are odd, so the grid is not a
power of two. A launch with the wrong block size must trap, and a hidden
size that is not a multiple of 8 must be refused.
"""

import os
import re
import subprocess
import sys
import tempfile
import unittest

import torch

from test_launch import PINNED_PROFILE
from y_lang.torch_interop import TorchKernel

# hidden size -> (warps, groups of 8 a thread) the emitter should pick
SHAPES = {256: (1, 1), 896: (4, 1), 2048: (8, 1), 3072: (16, 1), 4096: (16, 1), 5120: (16, 2),
          8192: (16, 2), 12032: (32, 2), 16384: (32, 2), 65536: (32, 8)}
ROWS = (1, 5, 67)


def compile_kernel(d):
    src = (f"kernel rmsnorm_residual_{d}(X: GlobalMemory<F16>, Residual: GlobalMemory<F16>, "
           f"Weight: GlobalMemory<F16>, Out: GlobalMemory<F16>, NewResidual: GlobalMemory<F16>) {{\n}}\n")
    saved = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, ".ysu_hw_profile"), "w") as f:
            f.write(PINNED_PROFILE)
        os.chdir(tmp)
        try:
            kern = TorchKernel(src, kernel_name=f"rmsnorm_residual_{d}", target_sm="sm_89")
        finally:
            os.chdir(saved)
    m = re.search(r"\[Y FUSED ADD\+RMSNORM\] hidden_dim=(\d+) \| (\d+) warps \((\d+) threads\)", kern.ptx)
    assert m and int(m.group(1)) == d, kern.ptx[:2000]
    return kern, int(m.group(2)), int(m.group(3))


def gpu_ok():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0)


@unittest.skipUnless(gpu_ok(), "needs an sm_80 or later GPU")
class RmsnormResidual(unittest.TestCase):
    def test_every_shape_is_exact_where_it_must_be_and_close_elsewhere(self):
        gen = torch.Generator(device="cuda").manual_seed(23)
        for d, (warps, groups) in SHAPES.items():
            kern, got_warps, threads = compile_kernel(d)
            self.assertEqual((got_warps, -(-(d // 8) // threads)), (warps, groups), d)
            w = (torch.rand(d, device="cuda", generator=gen) + 0.5).half()
            for rows in ROWS:
                with self.subTest(d=d, rows=rows):
                    x = torch.randn(rows, d, device="cuda", generator=gen).half()
                    r = torch.randn(rows, d, device="cuda", generator=gen).half()
                    h64 = x.double() + r.double()
                    ref = h64 * torch.rsqrt((h64 * h64).mean(dim=1, keepdim=True) + 1e-5) * w.double()
                    want_h = (x.float() + r.float()).half()
                    seen = None
                    for _ in range(3):
                        out = torch.full_like(x, float("nan"))
                        new_res = torch.full_like(x, float("nan"))
                        kern.launch((rows, 1, 1), (threads, 1, 1), [x, r, w, out, new_res])
                        torch.cuda.synchronize()
                        bits = (out.view(torch.int16).clone(), new_res.view(torch.int16).clone())
                        if seen is None:
                            seen = bits
                            self.assertTrue(torch.equal(new_res.view(torch.int16), want_h.view(torch.int16)),
                                            f"NewResidual is not f16(X + Residual) at d={d}, rows={rows}")
                            err = (out.double() - ref).abs() - (2.0 ** -10 * ref.abs() + 2.0 ** -23)
                            self.assertFalse(bool(torch.isnan(out).any()), f"Out has unwritten elements at d={d}, rows={rows}")
                            self.assertLessEqual(err.max().item(), 0.0, f"Out is off by more than 2^-10 at d={d}, rows={rows}")
                        else:
                            self.assertTrue(torch.equal(bits[0], seen[0]) and torch.equal(bits[1], seen[1]),
                                            f"two launches differ at d={d}, rows={rows}")

    def test_a_launch_with_another_block_size_traps(self):
        # A thread's groups are fixed at compile time for the block size, so
        # 256 threads on the 512-thread kernel would write half of each row
        # and report nothing. A trap poisons the CUDA context, so in a child.
        child = (
            "import sys, torch\n"
            f"sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})\n"
            "from test_rmsnorm_residual import compile_kernel\n"
            "kern, warps, threads = compile_kernel(4096)\n"
            "x = torch.randn(3, 4096, device='cuda').half(); w = torch.ones(4096, device='cuda').half()\n"
            "out = torch.full_like(x, float('nan')); nr = torch.full_like(x, float('nan'))\n"
            "kern.launch((3, 1, 1), (threads, 1, 1), [x, x, w, out, nr]); torch.cuda.synchronize()\n"
            "print('RIGHT', threads, bool(torch.isnan(out).any()), flush=True)\n"
            "kern.launch((3, 1, 1), (threads // 2, 1, 1), [x, x, w, out, nr]); torch.cuda.synchronize()\n"
            "print('WRONG SIZE RAN', flush=True)\n")
        run = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=300)
        text = run.stdout + run.stderr
        self.assertIn("RIGHT 512 False", text, text[-2000:])
        self.assertNotIn("WRONG SIZE RAN", text, text[-2000:])
        self.assertNotEqual(run.returncode, 0, text[-2000:])

    def test_a_width_not_a_multiple_of_8_is_refused(self):
        with self.assertRaises(Exception) as cm:
            compile_kernel(100)
        self.assertIn("not a multiple of 8", str(cm.exception))

    def test_the_rows_are_unrolled_with_every_load_first(self):
        # Each thread's groups are unrolled and every global load comes before
        # any arithmetic, with h kept in registers. As runtime loops over the
        # thread index, which ptxas keeps rolled, with h staged through shared
        # memory, the kernel took 1.08-1.10x Triton's time at 17-34 MB.
        for d in (4096, 5120, 12032):
            with self.subTest(d=d):
                ptx = compile_kernel(d)[0].ptx
                body = ptx[ptx.index("[Y FUSED ADD+RMSNORM]"):]
                self.assertFalse("bra " in body, "the group loop is a runtime loop")
                self.assertFalse("smem_h" in body, "h is staged through shared memory")
                lines = body.splitlines()
                loads = [i for i, l in enumerate(lines) if "ld.global" in l]
                first_math = min(i for i, l in enumerate(lines) if re.search(r"\b(add|fma\.rn|mul)\.f32\b", l))
                groups = SHAPES[d][1]
                self.assertEqual(len(loads), 3 * groups, f"{len(loads)} global loads for {groups} groups")
                self.assertLess(max(loads), first_math, "a global load comes after the arithmetic starts")


if __name__ == "__main__":
    unittest.main()
