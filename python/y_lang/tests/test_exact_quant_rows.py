"""`y_lang.exact.w8a8_gemv._quant_rows` (the quantiser of `w8a8_linear`, for
benchmarks and the self-test) computes the model's per-row quantiser,
`model._q_per_row`: the scale a correctly rounded `amax / 127` (a tensor
divide), the integers `round(x / s)`, subnormal scales kept. GPU only.
"""

import re
import unittest

import torch

CUDA = torch.cuda.is_available()
try:
    import triton
    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False


def quant_rows(x):
    from y_lang.exact import w8a8_gemv as w
    m, k = x.shape
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    s = torch.empty((m,), dtype=torch.float32, device=x.device)
    w._quant_rows[(m,)](x, q, s, m, k, x.stride(0), BLOCK_K=min(1024, triton.next_power_of_2(k)))
    return q, s


@unittest.skipUnless(CUDA and HAVE_TRITON, "needs CUDA and Triton")
class TestQuantRows(unittest.TestCase):
    def check(self, x, what):
        from y_lang.exact.model import _q_per_row
        want_q, want_s = _q_per_row(x, 127)
        q, s = quant_rows(x)
        self.assertTrue(torch.equal(s.view(torch.int32), want_s.squeeze(-1).view(torch.int32)),
                        f"{what}: scales differ in {int((s != want_s.squeeze(-1)).sum())} rows")
        self.assertTrue(torch.equal(q, want_q.to(torch.int8)),
                        f"{what}: integers differ in {int((q != want_q.to(torch.int8)).sum())}")
        return s

    def test_the_models_quantiser_at_volume(self):
        g = torch.Generator(device="cuda").manual_seed(0)
        for m, k in ((2048, 896), (2048, 4864), (512, 3584)):
            x = torch.randn(m, k, device="cuda", generator=g)
            x *= torch.exp2(torch.randint(-30, 30, (m, 1), device="cuda", generator=g).float())
            self.check(x, (m, k))

    def test_subnormal_scales_and_values(self):
        g = torch.Generator(device="cuda").manual_seed(1)
        x = torch.randn(64, 896, device="cuda", generator=g)
        # Largest |x| from 2^-149 up to 2^-120: scales below float32's normal
        # range (amax < 127 * 2^-126) in most of these rows.
        x *= torch.exp2(-torch.arange(120, 152, device="cuda").float().repeat(2))[:, None] / 4
        x[0] = 0.0                                        # scale 1, integers 0
        x[1, :] = 0.0
        x[1, 7] = -1e-45                                  # the smallest subnormal
        s = self.check(x, "subnormal")
        self.assertGreater(int((s < 2.0 ** -126).sum()), 30)       # the rows are there
        y = torch.randn(32, 896, device="cuda", generator=g)
        y[:, ::3] = torch.ldexp(torch.randint(1, 1 << 23, (32, 299), device="cuda",
                                              generator=g).float(),
                                torch.tensor(-149, device="cuda"))
        self.check(y, "subnormal values beside normal ones")

    def test_both_divides_are_ieee_without_flush(self):
        x = torch.randn(4, 896, device="cuda")
        from y_lang.exact import w8a8_gemv as w
        q = torch.empty((4, 896), dtype=torch.int8, device="cuda")
        s = torch.empty((4,), dtype=torch.float32, device="cuda")
        k = w._quant_rows[(4,)](x, q, s, 4, 896, x.stride(0), BLOCK_K=1024)
        ptx = k.asm["ptx"]
        divs = re.findall(r"\bdiv\.[a-z0-9.]*f32", ptx)
        self.assertTrue(divs, "no f32 divide in the PTX")
        self.assertTrue(all(d == "div.rn.f32" for d in divs), divs)
        self.assertNotRegex(ptx, r"\brcp\.approx|\bdiv\.full|\bdiv\.approx")


if __name__ == "__main__":
    unittest.main()
