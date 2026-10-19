"""`rope_<head_dim>(X, Positions, Out, Rows)` (fused RoPE, interleaved pairs,
base 10000, f16 in and out), run on the GPU.

A CTA is 256 threads covering several rows (16 at head_dim 64, else 8), so
the grid is ceil(Rows / rows_per_cta) CTAs and the last one is partial.
Outputs start as NaN, with NaN rows past `Rows`: a row no thread writes, or
a write past the end, fails. Every row is checked against float64, three
launches must agree bit for bit, and a row's bits must not depend on where
in a CTA it falls. The head dims cover each chunking: one 64-bit load a
lane (64, 128), three 32-bit loads (192), one 128-bit load (256) and two
(512).

A block of another size, a grid that does not reach `Rows`, and a negative
`Rows` must trap; a larger grid is allowed. The three-parameter form, and
head dims that are not a multiple of 64, must be refused: before the
refusal, rope_96 compiled and left a third of every row unwritten, and
rope_32 wrote nothing.
"""

import math
import os
import re
import subprocess
import sys
import tempfile
import unittest

import torch

from test_launch import PINNED_PROFILE
from y_lang.torch_interop import TorchKernel

# head_dim -> (rows a CTA, 32-bit words a lane, words a load)
SHAPES = {64: (16, 2, 2), 128: (8, 2, 2), 192: (8, 3, 1), 256: (8, 4, 4), 512: (8, 8, 4)}
ROWS = (1, 7, 8, 9, 17, 1003)


def source(hd, rows_param=True):
    extra = ", Rows: I32" if rows_param else ""
    return (f"kernel rope_{hd}(X: GlobalMemory<F16>, Positions: GlobalMemory<I32>, "
            f"Out: GlobalMemory<F16>{extra}) {{\n}}\n")


def compile_kernel(hd, rows_param=True):
    saved = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, ".ysu_hw_profile"), "w") as f:
            f.write(PINNED_PROFILE)
        os.chdir(tmp)
        try:
            kern = TorchKernel(source(hd, rows_param), kernel_name=f"rope_{hd}", target_sm="sm_89")
        finally:
            os.chdir(saved)
    m = re.search(r"\[Y FUSED ROPE\] head_dim=(\d+) \| (\d+) threads a CTA: (\d+) rows of (\d+) lanes", kern.ptx)
    assert m and int(m.group(1)) == hd, kern.ptx[:2000]
    return kern, int(m.group(2)), int(m.group(3))


def reference(x, pos):
    rows, hd = x.shape
    inv = 10000.0 ** (-torch.arange(0, hd, 2, device=x.device, dtype=torch.float64) / hd)
    th = pos.double()[:, None] * inv[None, :]
    xv = x.double().view(rows, hd // 2, 2)
    return torch.stack([xv[..., 0] * th.cos() - xv[..., 1] * th.sin(),
                        xv[..., 0] * th.sin() + xv[..., 1] * th.cos()], -1).reshape(rows, hd)


def gpu_ok():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0)


PAD = 40   # NaN rows past `Rows`


@unittest.skipUnless(gpu_ok(), "needs an sm_80 or later GPU")
class Rope(unittest.TestCase):
    def test_every_row_is_rotated_once_and_nothing_past_the_last(self):
        gen = torch.Generator(device="cuda").manual_seed(31)
        for hd, (per_cta, _, _) in SHAPES.items():
            kern, threads, got_per_cta = compile_kernel(hd)
            self.assertEqual((threads, got_per_cta), (256, per_cta), hd)
            # The first 17 rows are shared by every row count, to compare
            # their bits across launches that place them differently.
            x_all = torch.randn(max(ROWS), hd, device="cuda", generator=gen).half()
            pos_all = torch.randint(0, 4096, (max(ROWS),), device="cuda", generator=gen, dtype=torch.int32)
            first = None
            for rows in sorted(ROWS, reverse=True):
                with self.subTest(hd=hd, rows=rows):
                    x, pos = x_all[:rows].contiguous(), pos_all[:rows].contiguous()
                    ref = reference(x, pos)
                    grid = (math.ceil(rows / per_cta), 1, 1)
                    seen = None
                    for _ in range(3):
                        out = torch.full((rows + PAD, hd), float("nan"), device="cuda", dtype=torch.half)
                        kern.launch(grid, (threads, 1, 1), [x, pos, out, rows])
                        torch.cuda.synchronize()
                        bits = out.view(torch.int16).clone()
                        if seen is None:
                            seen = bits
                            self.assertFalse(bool(torch.isnan(out[:rows]).any()), f"unwritten elements at hd={hd}, rows={rows}")
                            self.assertTrue(bool(torch.isnan(out[rows:]).all()), f"a write past Rows at hd={hd}, rows={rows}")
                            err = (out[:rows].double() - ref).abs().max().item()
                            self.assertLessEqual(err, 4e-3 * ref.abs().max().item(), f"hd={hd}, rows={rows}")
                        else:
                            self.assertTrue(torch.equal(bits, seen), f"two launches differ at hd={hd}, rows={rows}")
                    shared = min(rows, 17)
                    if first is None:
                        first = seen[:17]
                    else:
                        self.assertTrue(torch.equal(seen[:shared], first[:shared]),
                                        f"a row's bits depend on the launch at hd={hd}, rows={rows}")

    def _child(self, launch):
        # A trap poisons the CUDA context, so each wrong launch runs in its
        # own process, after a right one in the same process.
        child = (
            "import math, sys, torch\n"
            f"sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})\n"
            "from test_rope import compile_kernel\n"
            "kern, threads, per_cta = compile_kernel(128)\n"
            "rows = 37\n"
            "x = torch.randn(rows, 128, device='cuda').half()\n"
            "pos = torch.randint(0, 4096, (rows,), device='cuda', dtype=torch.int32)\n"
            "out = torch.full((rows + 64, 128), float('nan'), device='cuda', dtype=torch.half)\n"
            "kern.launch((math.ceil(rows / per_cta), 1, 1), (threads, 1, 1), [x, pos, out, rows]); torch.cuda.synchronize()\n"
            "print('RIGHT', threads, per_cta, bool(torch.isnan(out[:rows]).any()), bool(torch.isnan(out[rows:]).all()), flush=True)\n"
            "out.fill_(float('nan'))\n"
            f"{launch}\n"
            "torch.cuda.synchronize()\n"
            "print('RAN', bool(torch.isnan(out[:rows]).any()), bool(torch.isnan(out[rows:]).all()), flush=True)\n")
        run = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=300)
        text = run.stdout + run.stderr
        self.assertIn("RIGHT 256 8 False True", text, text[-2000:])
        return run, text

    def test_wrong_launches_trap(self):
        cases = {
            "half the block": "kern.launch((math.ceil(rows / per_cta), 1, 1), (threads // 2, 1, 1), [x, pos, out, rows])",
            "one CTA short": "kern.launch((math.ceil(rows / per_cta) - 1, 1, 1), (threads, 1, 1), [x, pos, out, rows])",
            "a negative Rows": "kern.launch((math.ceil(rows / per_cta), 1, 1), (threads, 1, 1), [x, pos, out, 0xFFFFFFFF])",
        }
        for what, launch in cases.items():
            with self.subTest(what=what):
                run, text = self._child(launch)
                self.assertNotIn("RAN", text, f"{what} ran: {text[-2000:]}")
                self.assertNotEqual(run.returncode, 0, text[-2000:])

    def test_a_larger_grid_is_allowed_and_writes_nothing_more(self):
        run, text = self._child(
            "kern.launch((3 * math.ceil(rows / per_cta), 1, 1), (threads, 1, 1), [x, pos, out, rows])")
        self.assertIn("RAN False True", text, text[-2000:])
        self.assertEqual(run.returncode, 0, text[-2000:])

    def test_the_old_signature_and_other_widths_are_refused(self):
        with self.assertRaises(Exception) as cm:
            compile_kernel(128, rows_param=False)
        self.assertIn("takes a fourth parameter, `Rows: I32`", str(cm.exception))
        for hd in (32, 96, 160):
            with self.subTest(hd=hd):
                with self.assertRaises(Exception) as cm:
                    compile_kernel(hd)
                self.assertIn("is not a multiple of 64", str(cm.exception))

    def test_a_lane_loads_its_pairs_first_in_vectors(self):
        # One 32-thread CTA a row ran at half occupancy, bound by how fast
        # CTAs launch; 4 bytes a lane at head_dim 64 measured 1.16-1.19x
        # slower than 8. Every load comes before the first sin.approx.
        for hd, (_, words, per_load) in SHAPES.items():
            with self.subTest(hd=hd):
                ptx = compile_kernel(hd)[0].ptx
                body = ptx[ptx.index("[Y FUSED ROPE]"):]
                self.assertNotIn("bra ", body)
                lines = body.splitlines()
                width = {1: "u32", 2: "v2.u32", 4: "v4.u32"}[per_load]
                loads = [l for l in lines if "ld.global" in l]
                # The position, then the row's pairs, `per_load` words at a time.
                self.assertEqual(len(loads), 1 + words // per_load, body)
                self.assertEqual(sum(f"ld.global.{width} " in l for l in loads[1:]), words // per_load, body)
                first_trig = min(i for i, l in enumerate(lines) if "sin.approx" in l)
                self.assertLess(max(i for i, l in enumerate(lines) if "ld.global" in l), first_trig)
                self.assertEqual(sum("sin.approx" in l for l in lines), words, body)


if __name__ == "__main__":
    unittest.main()
