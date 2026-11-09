"""`y_lang.exact.norm_kernel`: ExactRMSNorm's value without a float64 divide
per element. GPU only.

The kernel's claim is bit equality with `float32(x / r)`, so every check here
compares bits, against torch's float64 divide on the same GPU, and each has a
control showing the comparison can fail.
"""

import unittest

import numpy as np
import torch

CUDA = torch.cuda.is_available()
try:
    import triton  # noqa: F401
    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False


def _bits_differ(a, b):
    return int((a.view(torch.int32) != b.view(torch.int32)).sum())


def _pairs(kind, n, rng):
    """float32 x and float64 r; "midpoint" pairs put x / r within about a
    float64 ulp of a float32 rounding midpoint, "subnormal" pairs within about
    a float64 ulp of one below 2^-126, where float32's spacing is 2^-149 and
    the midpoints, (2k + 1) * 2^-150, do not have the bit pattern the kernel's
    normal-range test looks for: only its |q| < 2^-126 test catches them;
    "big" pairs put x / r at 2^127 and past float32's largest value, which
    the kernel's integer rounding must not see."""
    if kind == "random":
        x = (rng.standard_normal(n) * np.exp(rng.uniform(-8, 8, n))).astype(np.float32)
        r = np.sqrt(np.exp(rng.uniform(-14, 14, n)) + 1e-6)
    elif kind == "big":
        x = (rng.uniform(0.5, 60, n) * np.sign(rng.standard_normal(n))).astype(np.float32)
        r = np.ldexp(rng.uniform(1, 2, n), rng.integers(-134, -120, n))
    elif kind == "subnormal":
        x = (rng.uniform(0.5, 60, n) * np.sign(rng.standard_normal(n))).astype(np.float32)
        m = np.ldexp((2 * rng.integers(0, 1 << 23, n) + 1).astype(np.float64), -150)
        r = np.abs(x.astype(np.float64) / m)
    else:
        x = (rng.uniform(0.5, 60, n) * np.sign(rng.standard_normal(n))).astype(np.float32)
        c = rng.uniform(1, 2, n).astype(np.float32).astype(np.float64)
        m = np.ldexp(c + np.ldexp(1.0, -24), rng.integers(-12, 6, n))
        r = np.abs(x.astype(np.float64) / m)
    return torch.from_numpy(x).cuda(), torch.from_numpy(r).cuda()


@unittest.skipUnless(CUDA and HAVE_TRITON, "needs CUDA and Triton")
class TestNormKernel(unittest.TestCase):
    def test_quotient_equals_the_divide(self):
        from y_lang.exact import norm_kernel as nk
        rng = np.random.default_rng(0)
        for kind in ("random", "midpoint", "subnormal", "big"):
            x, r = _pairs(kind, 2_000_000, rng)
            want = (x.double() / r).float()
            self.assertEqual(_bits_differ(nk.quotient_to_f32(x, r), want), 0, kind)
            if kind == "big":
                # Both sides of float32's largest value are there.
                self.assertGreater(int(torch.isinf(want).sum()), 100_000)
                self.assertGreater(int((want.abs() >= 2.0 ** 127).sum()
                                       - torch.isinf(want).sum()), 100_000)
            elif kind != "random":
                # Control: the multiply alone is wrong here, so the guard is
                # what makes the result right.
                cheap = (x.double() * (1.0 / r)).float()
                self.assertGreater(_bits_differ(cheap, want), 10_000)

    def test_norm_equals_the_formula(self):
        import y_lang.exact.model as em
        from y_lang.exact import norm_kernel as nk

        class Norm(torch.nn.Module):
            def __init__(self, h, g):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(h, device="cuda", generator=g) + 0.5)
                self.variance_epsilon = 1e-6

        g = torch.Generator(device="cuda").manual_seed(0)
        for h in (896, 3584, 4736):
            norm = em.ExactRMSNorm(Norm(h, g)).cuda()
            for rows in (1, 32, 900):
                x = torch.randn(rows, h, device="cuda", generator=g) * 0.5
                x[:, ::97] *= 60.0
                x[0, :5] = 0.0
                x[-1, 5:9] = torch.tensor([1e-30, -1e-38, 3e-39, -0.0], device="cuda")
                xd = x.double()
                var = xd.pow(2).sum(-1, keepdim=True) / xd.new_tensor(float(h))
                want = ((xd / torch.sqrt(var + norm.eps)).float() * norm.weight)
                got = nk.rms_norm(x, norm.weight, norm.eps)
                self.assertEqual(_bits_differ(got, want), 0, (h, rows))
                # ExactRMSNorm itself takes the kernel on CUDA float32.
                if em._NORM_OP is not None:
                    self.assertEqual(_bits_differ(norm(x), want), 0, (h, rows))
                # No control here: on activations like these even a bare
                # reciprocal multiply never differs (0 of 100M on the CPU), so
                # the guard is exercised by the midpoint pairs above instead.
                # A weight-free control still shows this comparison can fail:
                self.assertGreater(_bits_differ(got, (xd / torch.sqrt(var + 1e-5)).float()
                                                * norm.weight), 0)

    def test_rows_off_the_integer_path(self):
        """Rows the kernel converts on bits (normal or zero x) beside rows it
        cannot (subnormal, infinite, NaN x), with eps = 0 so that the
        squares of tiny values and of zeros decide r."""
        from y_lang.exact import norm_kernel as nk
        g = torch.Generator(device="cuda").manual_seed(1)
        h = 3584
        w = torch.rand(h, device="cuda", generator=g) + 0.5
        x = torch.randn(8, h, device="cuda", generator=g)
        x[0] *= 1e-37                                  # squares near 2^-250
        x[0, ::3] = 0.0
        x[1] = torch.exp2(torch.randint(-125, 127, (h,), device="cuda", generator=g)
                          .float()) * x[1].sign()     # every float32 binade
        x[2, 7] = float("inf")
        x[3, 9] = float("nan")
        x[4, 11] = 1e-40                               # subnormal
        x[5] = 0.0
        x[6, :100] = -0.0
        for eps in (0.0, 1e-6):
            xd = x.double()
            var = xd.pow(2).sum(-1, keepdim=True) / xd.new_tensor(float(h))
            want = (xd / torch.sqrt(var + eps)).float() * w
            got = nk.rms_norm(x, w, eps)
            nan = torch.isnan(want)
            self.assertTrue(torch.equal(nan, torch.isnan(got)))
            self.assertEqual(_bits_differ(got.masked_fill(nan, 0), want.masked_fill(nan, 0)), 0,
                             eps)
        # At eps = 1e-6: the NaN row, and inf / inf.
        self.assertTrue(nan[3].all() and nan[2, 7] and int(nan.sum()) == h + 1)

    def test_no_approximate_instructions(self):
        import re
        from y_lang.exact import norm_kernel as nk
        x = torch.randn(4, 3584, device="cuda")
        w = torch.rand(3584, device="cuda")
        out = torch.empty_like(x)
        k = nk._rms_norm_kernel[(4,)](x, w, out, nk._eps_tensor(1e-6, x.device), 3584,
                                      x.stride(0), out.stride(0), BLOCK=4096, num_warps=8,
                                      enable_fp_fusion=False)
        ptx = k.asm["ptx"]
        self.assertNotIn("approx", ptx)
        self.assertNotRegex(ptx, r"\bfma\.[a-z.]*f64")
        self.assertRegex(ptx, r"\bdiv\.rn\.f64")
        self.assertRegex(ptx, r"\bsqrt\.rn\.f64")
        self.assertTrue(re.search(r"\bmul\.rn\.f64", ptx))
        # Speed, not bits: the guard tests q's bits as integers, because a
        # float64 compare runs at 1/64 rate on a GeForce part (four per
        # element made the kernel 11.4 us at the 7B shape, none 8.1 us).
        self.assertNotRegex(ptx, r"\bsetp\.\w+\.f64")


if __name__ == "__main__":
    unittest.main()
