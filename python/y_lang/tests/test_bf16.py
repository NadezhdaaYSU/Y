"""`BF16` (bfloat16) as a buffer element and the BF16 tile GEMM, against torch.

The conversions are compared with torch's on CUDA, bit for bit:
- every one of the 65,536 patterns widened;
- 2^24 float32 patterns and the special values narrowed;
- f64 values at bf16 midpoints and one ulp either side, against the nearest
  bf16 found by search, which torch cannot give: its double-to-bfloat16 goes
  through float and rounds twice. The control counts how often that happens.

The GEMM is compared with cuBLAS's bf16 GEMM with f32 output
(`torch.mm(A, B, out_dtype=torch.float32)`). Above 32 rows, where both chain
the tensor core's accumulator, the docs say every output is the same bits;
this keeps that claim checked.

`tests/ptx_bf16.rs` checks the same against the format's definition, and the
refusals.
"""

import os
import re
import tempfile
import unittest

import numpy as np
import torch

from test_launch import PINNED_PROFILE
from y_lang.torch_interop import TorchKernel

CONVERT = """
kernel widen(A: GlobalMemory<BF16>, C: GlobalMemory<F32>, n: U32) {
    let i: U32 = global_thread_id();
    if i < n {
        C[i] = A[i];
    }
}
kernel narrow(X: GlobalMemory<F32>, B: GlobalMemory<BF16>, n: U32) {
    let i: U32 = global_thread_id();
    if i < n {
        B[i] = X[i];
    }
}
kernel narrow64(X: GlobalMemory<F64>, B: GlobalMemory<BF16>, n: U32) {
    let i: U32 = global_thread_id();
    if i < n {
        B[i] = X[i];
    }
}
"""


def compiled(src, name):
    """`src`'s `name`, compiled against PINNED_PROFILE in a temporary dir."""
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, ".ysu_hw_profile"), "w") as f:
            f.write(PINNED_PROFILE)
        saved = {v: os.environ.pop(v, None) for v in ("Y_CTA_OVERRIDE", "Y_F16_ACCUMULATE")}
        os.chdir(d)
        try:
            return TorchKernel(src, kernel_name=name, target_sm="sm_89")
        finally:
            os.chdir(cwd)
            for v, value in saved.items():
                if value is not None:
                    os.environ[v] = value


def launch(kern, x, y):
    n = x.numel()
    kern.launch(((n + 255) // 256, 1, 1), (256, 1, 1), [x, y, n])
    torch.cuda.synchronize()


def nearest_bf16(x):
    """The nearest bf16 pattern of each float64, ties to even, found by search
    over every finite bf16; the subtractions are exact for the two nearest
    candidates (Sterbenz), so a tie is seen as one."""
    pat = np.arange(0x7F80, dtype=np.uint32)
    vals = (pat << 16).view(np.float32).astype(np.float64)
    top, over = vals[-1], vals[-1] + (vals[-1] - vals[-2]) / 2
    a = np.abs(x)
    i = np.clip(np.searchsorted(vals, a, side="right") - 1, 0, len(vals) - 2)
    lo, hi = vals[i], vals[i + 1]
    pick = np.where(a - lo < hi - a, i, np.where(hi - a < a - lo, i + 1, np.where(i % 2 == 0, i, i + 1)))
    pick = np.where(a >= over, 0x7F80, np.where(a > top, 0x7F7F, pick))
    return (pick | np.where(np.signbit(x), 0x8000, 0)).astype(np.uint16)


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0),
                     "needs an sm_80 or later GPU")
class Conversions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.k = {n: compiled(CONVERT, n) for n in ("widen", "narrow", "narrow64")}

    def test_every_pattern_widens_as_torch_widens_it(self):
        bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).cuda()
        out = torch.empty(65536, dtype=torch.float32, device="cuda")
        launch(self.k["widen"], bits.view(torch.bfloat16), out)
        self.assertTrue(torch.equal(out.view(torch.int32), bits.view(torch.bfloat16).float().view(torch.int32)))

    def test_float32_narrows_as_torch_narrows_it(self):
        g = torch.Generator(device="cuda").manual_seed(0)
        special = torch.tensor([0.0, -0.0, 1.0, 1.00390625, 1.01171875, 3.3961776e38, 3.3961778e38,
                                3.4028235e38, float("inf"), float("-inf"), float("nan"), 1e-40, -1e-40],
                               device="cuda")
        rand = torch.randint(-(1 << 31), (1 << 31) - 1, (1 << 24,), device="cuda", generator=g,
                             dtype=torch.int64).to(torch.int32).view(torch.float32)
        x = torch.cat([special, rand])
        y = torch.empty(x.numel(), dtype=torch.bfloat16, device="cuda")
        launch(self.k["narrow"], x, y)
        self.assertTrue(torch.equal(y.view(torch.int16), x.to(torch.bfloat16).view(torch.int16)),
                        "differs from torch's float32 -> bfloat16 on CUDA, NaN included")

    def test_float64_rounds_once(self):
        pat = np.arange(1, 0x7F80, 53, dtype=np.uint32)
        lo = (pat << 16).view(np.float32).astype(np.float64)
        hi = ((pat + 1) << 16).view(np.float32).astype(np.float64)
        mid = (lo + hi) / 2
        x = np.concatenate([mid, np.nextafter(mid, 0), np.nextafter(mid, np.inf)])
        x = np.concatenate([x, -x, [1e300, -1e300, 1e-300, 0.0]])
        y = torch.empty(x.size, dtype=torch.bfloat16, device="cuda")
        launch(self.k["narrow64"], torch.from_numpy(x).cuda(), y)
        want = nearest_bf16(x)
        got = y.view(torch.int16).cpu().numpy().view(np.uint16)
        twice = int((torch.from_numpy(x).float().to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16) != want).sum())
        self.assertGreater(twice, 100, "too few inputs where rounding through float32 differs; the test is weak")
        self.assertEqual(int((got != want).sum()), 0)


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0),
                     "needs an sm_80 or later GPU")
class TheGemm(unittest.TestCase):
    def gemm(self, m, n, k, a, b):
        kern = compiled(f"@tile({m}, {n}, {k})\nkernel g(A: GlobalMemory<BF16>, B: GlobalMemory<BF16>, "
                        f"C: GlobalMemory<F32>) {{\n}}\n", "g")
        cm, cn, wm, wn = (int(v) for v in re.search(r"CTA (\d+)x(\d+)x\d+ \| (\d+)x(\d+) warps", kern.ptx).groups())
        c = torch.full((m, n), float("nan"), device="cuda")
        kern.launch(((n + cn - 1) // cn, (m + cm - 1) // cm, 1), (wm * wn * 32, 1, 1), [a, b, c])
        torch.cuda.synchronize()
        return c

    def test_the_same_bits_as_cublas_above_32_rows(self):
        g = torch.Generator(device="cuda").manual_seed(1)
        for m, n, k in ((512, 512, 512), (1024, 768, 1024)):
            a = torch.randn((m, k), device="cuda", generator=g).bfloat16()
            b = torch.randn((k, n), device="cuda", generator=g).bfloat16()
            ours = self.gemm(m, n, k, a, b)
            theirs = torch.mm(a, b, out_dtype=torch.float32)
            self.assertTrue(torch.equal(ours.view(torch.int32), theirs.view(torch.int32)), (m, n, k))

    def test_at_32_rows_the_compensated_sum_is_closer(self):
        g = torch.Generator(device="cuda").manual_seed(2)
        m, n, k = 32, 1024, 4096
        a = torch.randn((m, k), device="cuda", generator=g).abs().bfloat16()
        b = torch.randn((k, n), device="cuda", generator=g).abs().bfloat16()
        exact = a.double() @ b.double()
        ours = (self.gemm(m, n, k, a, b).double() - exact).abs().max()
        theirs = (torch.mm(a, b, out_dtype=torch.float32).double() - exact).abs().max()
        self.assertLess(float(ours) * 10, float(theirs), (float(ours), float(theirs)))


if __name__ == "__main__":
    unittest.main()
