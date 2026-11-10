"""`y_lang.exact.model.sum_of_squares`, ExactRMSNorm's sum of x^2: the same
bits in any order and on every device, because the squares are added as
integers on a fixed grid (model.SUMSQ_F).

Every implementation (torch on the CPU, torch on CUDA eager and under
torch.compile, the Triton norm kernel's) is compared, bit for bit, with an
oracle written here with Python integers, on the rows as given and permuted.
The control: a float64 sum of the same squares changes bits when the row is
permuted.
"""

import math
import struct
import unittest

import numpy as np
import torch

CUDA = torch.cuda.is_available()
try:
    import triton  # noqa: F401
    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

F = 32          # model.SUMSQ_F, restated: the oracle does not import it


def oracle(row):
    """The definition, with Python integers: each x^2 = m^2 * 2^(p - 300)
    (m the 24-bit significand, p = 2 * max(biased exponent, 1)), floored to
    the grid 2^(E - 300 - F), E the row's largest p, summed exactly, rounded
    once (`float(int)` rounds to nearest even)."""
    terms, inf, nan = [], False, False
    for v in row:
        b = struct.unpack("<I", struct.pack("<f", v))[0]
        ex, frac = (b >> 23) & 255, b & 0x7FFFFF
        if ex == 255:
            inf, nan = True, nan or frac != 0
            continue
        m = frac if ex == 0 else frac | (1 << 23)
        terms.append((m * m, 2 * max(ex, 1)))
    if inf:
        return math.nan if nan else math.inf
    ps = [p for m2, p in terms if m2]
    if not ps:
        return 0.0
    e = max(ps)
    total = sum(m2 << (F - (e - p)) if e - p <= F else m2 >> (e - p - F) for m2, p in terms)
    return math.ldexp(float(total), e - 300 - F)


def _rows(kind, n, h, rng):
    if kind == "activations":
        x = rng.standard_normal((n, h)) * 0.5
        x[:, ::97] *= 60.0
    elif kind == "every binade":
        x = np.ldexp(rng.uniform(1, 2, (n, h)), rng.integers(-149, 127, (n, h)))
        x *= np.sign(rng.standard_normal((n, h)))
    elif kind == "subnormal":
        x = np.ldexp(rng.integers(0, 1 << 23, (n, h)).astype(np.float64), -149)
        x[:, ::500] = rng.standard_normal((n, len(range(0, h, 500)))) * 1e-30
    else:                         # "two clusters": half the squares 2^-40 down, truncated
        x = rng.standard_normal((n, h))
        x[:, ::2] *= 2.0 ** -20
    x = torch.from_numpy(x.astype(np.float32))
    x[0, :7] = 0.0
    x[1] = 0.0
    x[2, 5] = -0.0
    return x


def _bits(t):
    return t.contiguous().view(torch.int64)


class TestSumOfSquares(unittest.TestCase):
    def implementations(self):
        from y_lang.exact import model as em
        impls = {"torch cpu": em.sum_of_squares}
        if CUDA:
            compiled = torch.compile(em.sum_of_squares, dynamic=False)
            impls["torch cuda"] = lambda x: em.sum_of_squares(x.cuda()).cpu()
            impls["torch cuda, compiled"] = lambda x: compiled(x.cuda()).cpu()
            if HAVE_TRITON:
                from y_lang.exact import norm_kernel as nk
                impls["triton kernel"] = lambda x: nk.sum_of_squares(x.cuda()).cpu()
        return impls

    def test_every_order_gives_the_oracles_bits(self):
        rng = np.random.default_rng(0)
        impls = self.implementations()
        control = 0
        for kind in ("activations", "every binade", "subnormal", "two clusters"):
            for h in (896, 3584):
                x = _rows(kind, 24, h, rng)
                want = torch.tensor([[oracle(r.tolist())] for r in x], dtype=torch.float64)
                for k in range(3):
                    xp = x if k == 0 else x[:, torch.from_numpy(rng.permutation(h))]
                    for name, f in impls.items():
                        self.assertTrue(torch.equal(_bits(f(xp)), _bits(want)), (kind, h, k, name))
                    if k:
                        d = xp.double()
                        control += int((_bits(d.pow(2).sum(-1, keepdim=True))
                                        != _bits(x.double().pow(2).sum(-1, keepdim=True))).sum())
        self.assertGreater(control, 50)    # a float64 sum's bits do depend on the order

    def test_the_largest_values_and_every_subnormal(self):
        impls = self.implementations()
        big = torch.full((2, 8192), 3.4028234663852886e38)          # float32's largest
        big[1, ::2] = -big[1, ::2]
        sub = torch.from_numpy(np.ldexp(np.arange(1, 8193, dtype=np.float64), -149)
                               .astype(np.float32)).view(1, -1)      # 2^-149 .. 8192 * 2^-149
        mixed = torch.cat([sub, big[:1]], 0)
        for x in (big, sub, mixed):
            want = torch.tensor([[oracle(r.tolist())] for r in x], dtype=torch.float64)
            self.assertTrue(torch.isfinite(want).all())
            for name, f in impls.items():
                self.assertTrue(torch.equal(_bits(f(x)), _bits(want)), name)

    def test_not_fsum_where_the_grid_drops_a_square(self):
        """1 + 2 * (2^-27)^2 is exactly the float64 midpoint 1 + 2^-53, which
        rounds to even (1.0); (2^-60)^2 = 2^-120 lies below the grid and is
        dropped, where math.fsum keeps it and rounds up. The definition says
        1.0, and every implementation must say so."""
        row = torch.zeros(1, 64)
        row[0, :4] = torch.tensor([1.0, 2.0 ** -27, 2.0 ** -27, 2.0 ** -60])
        self.assertEqual(oracle(row[0].tolist()), 1.0)
        self.assertEqual(math.fsum(float(v) ** 2 for v in row[0].tolist()), 1.0 + 2.0 ** -52)
        for name, f in self.implementations().items():
            self.assertEqual(float(f(row)), 1.0, name)
        # Without the tiny square the grid keeps every bit, and agrees with fsum.
        row[0, 3] = 0.0
        for name, f in self.implementations().items():
            self.assertEqual(float(f(row)), math.fsum(float(v) ** 2 for v in row[0].tolist()), name)

    def test_infinite_and_nan_rows(self):
        x = torch.randn(4, 300)
        x[0, 3] = float("inf")
        x[1, 4] = float("-inf")
        x[1, 9] = float("inf")
        x[2, 5] = float("nan")
        x[3, 6] = float("nan")
        x[3, 7] = float("inf")
        for name, f in self.implementations().items():
            got = f(x).view(-1)
            self.assertTrue(math.isinf(got[0]) and got[0] > 0 and math.isinf(got[1]) and got[1] > 0,
                            name)
            self.assertTrue(math.isnan(got[2]) and math.isnan(got[3]), name)

    def test_refuses_what_it_does_not_define(self):
        from y_lang.exact import model as em
        with self.assertRaisesRegex(ValueError, "8192"):
            em.sum_of_squares(torch.zeros(1, 8193))
        with self.assertRaisesRegex(TypeError, "float32"):
            em.sum_of_squares(torch.zeros(1, 8, dtype=torch.float64))
        # 16-bit values widen exactly.
        h = torch.randn(3, 100).to(torch.bfloat16)
        self.assertTrue(torch.equal(_bits(em.sum_of_squares(h)),
                                    _bits(em.sum_of_squares(h.to(torch.float32)))))


if __name__ == "__main__":
    unittest.main()
