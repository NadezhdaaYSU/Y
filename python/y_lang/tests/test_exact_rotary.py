"""`y_lang.exact.model.cos_sin_f32` and `ExactRotary`: every RoPE table entry
is the correctly rounded float32 of cos or sin (times the module's scale),
whatever the host's float64 cos/sin return within `ROPE_LIBM_ULPS` ulps.

The oracle is mpmath at 256 bits with its own rounding to float32. The hard
arguments are built, not searched for: for a float32 rounding midpoint M,
acos(M) or asin(M) rounded to float64 is an argument whose cos or sin lies
within about a float64 ulp of M, where rounding a libm's float64 result can
go either way. A host with another libm is simulated by moving torch's
float64 results a few ulps.
"""

import random
import unittest

import numpy as np
import torch

try:
    import mpmath
    HAVE_MPMATH = True
except ImportError:
    HAVE_MPMATH = False


def oracle(name, arg, scale=1.0):
    """The float32 nearest fn(arg) * scale, decided at 1024 bits among the
    three float32s around float32(float64(value)), which is at most one step
    off (a method of its own: the code under test scales the significand)."""
    with mpmath.workprec(1024):
        fn = mpmath.cos if name == "cos" else mpmath.sin
        v = fn(mpmath.mpf(arg)) * mpmath.mpf(scale)
        c = np.float32(float(v))
        cands = [np.nextafter(c, np.float32(-np.inf)), c, np.nextafter(c, np.float32(np.inf))]
        d = [abs(v - mpmath.mpf(float(x))) for x in cands]
        best = min(range(3), key=lambda i: d[i])
        assert sorted(d)[0] < sorted(d)[1], "a tie at 1024 bits"
        return float(cands[best])


def midpoint_args(n, seed, scale=1.0):
    """n float64 args whose cos(arg) * scale or sin(arg) * scale is within 2
    float64 ulps of a float32 rounding midpoint (checked at 256 bits): (name,
    arg) pairs. For a midpoint M, acos(M) or asin(M) rounded to float64 lands
    that close where the function's slope times the argument's ulp is below
    M's ulp; candidates that land farther are dropped."""
    rng = random.Random(seed)
    out = []
    with mpmath.workprec(256):
        while len(out) < n:
            name = rng.choice(("cos", "sin"))
            if name == "sin" and rng.random() < 0.15:
                # below 2^-126: float32's midpoints there are (2k + 1) * 2^-150
                m = mpmath.ldexp(2 * rng.randrange(1, 1 << 23) + 1, -150)
                if rng.random() < 0.5:
                    m = -m      # sin(-x) lies above -x: rounding at too few bits goes the wrong way
            else:
                e = rng.randrange(-20, 0) if name == "sin" else 0
                m = mpmath.ldexp(2 * rng.randrange(1 << 23, 1 << 24) + 1, e - 25)  # in [2^(e-1), 2^e)
                if rng.random() < 0.5:
                    m = -m
            t = m / mpmath.mpf(scale)
            if abs(t) >= 1:
                continue
            arg = float(mpmath.acos(t) if name == "cos" else mpmath.asin(t))
            fn = mpmath.cos if name == "cos" else mpmath.sin
            v = fn(mpmath.mpf(arg)) * mpmath.mpf(scale)
            ulp = mpmath.ldexp(1, int(mpmath.frexp(m)[1]) - 53)
            if abs(v - m) <= 2 * ulp:
                out.append((name, arg))
    return out


def shifted(fn, k):
    """`fn` with every float64 result moved k ulps (another host's libm)."""
    def g(x):
        y = fn(x)
        b = y.view(torch.int64)
        return torch.where(torch.isfinite(y) & (y != 0), (b + k * torch.sign(y).long()), b).view(
            torch.float64)
    return g


@unittest.skipUnless(HAVE_MPMATH, "needs mpmath")
class TestRopeTable(unittest.TestCase):
    def compute(self, pairs, scale=1.0, approx=(torch.cos, torch.sin)):
        from y_lang.exact import model as em
        args = torch.tensor([a for _, a in pairs], dtype=torch.float64)
        c, s = em.cos_sin_f32(args, scale, approx)
        return [float(c[i]) if name == "cos" else float(s[i]) for i, (name, _) in enumerate(pairs)]

    def test_midpoint_arguments_round_correctly_under_any_libm(self):
        from y_lang.exact import model as em
        for scale in (1.0, 1.25, 0.70710678118654757):
            pairs = midpoint_args(160, seed=int(scale * 1000), scale=scale)
            want = [oracle(n, a, scale) for n, a in pairs]
            naive_wrong = 0
            for k in range(-em.ROPE_LIBM_ULPS, em.ROPE_LIBM_ULPS + 1):
                approx = (shifted(torch.cos, k), shifted(torch.sin, k))
                self.assertEqual(self.compute(pairs, scale, approx), want, (scale, k))
                args = torch.tensor([a for _, a in pairs], dtype=torch.float64)
                ys = [approx[0](args) * scale, approx[1](args) * scale]
                naive = [float(ys[0 if n == "cos" else 1][i].float()) for i, (n, _) in
                         enumerate(pairs)]
                naive_wrong += sum(x != y for x, y in zip(naive, want))
            # Control: rounding the float64 results directly is wrong on many of
            # these arguments for some of the simulated libms.
            self.assertGreater(naive_wrong, 50, scale)

    def test_the_hard_path_is_taken(self):
        from y_lang.exact import model as em
        pairs = midpoint_args(80, seed=7)
        calls = []
        orig = em._rn32_by_mpmath
        try:
            em._rn32_by_mpmath = lambda *a: calls.append(a) or orig(*a)
            self.compute(pairs)
        finally:
            em._rn32_by_mpmath = orig
        # Each argument sends its own function's value (and maybe the other's) to mpmath.
        self.assertGreaterEqual(len(calls), len(pairs))

    def test_a_qwen_table_against_the_oracle(self):
        from transformers import Qwen2Config
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
        from y_lang.exact import model as em
        cfg = Qwen2Config(hidden_size=3584, num_attention_heads=28, num_key_value_heads=4,
                          rope_theta=1000000.0, max_position_embeddings=32768)   # Qwen2.5-7B's
        rot = Qwen2RotaryEmbedding(cfg)
        table = em.ExactRotary(rot, cfg.max_position_embeddings)
        half = table.cos.shape[1] // 2
        self.assertTrue(torch.equal(table.cos[:, :half], table.cos[:, half:]))
        self.assertTrue(torch.equal(table.sin[:, :half], table.sin[:, half:]))
        inv = rot.inv_freq.to(torch.float64)
        rng = random.Random(0)
        cells = {(0, j) for j in range(half)} | {(8312, 39)}    # sin(0) = 0; cos 10.2 ulp64 off a midpoint
        cells |= {(rng.randrange(cfg.max_position_embeddings), rng.randrange(half))
                  for _ in range(3000)}
        for p, j in sorted(cells):
            a = float(p * inv[j])
            self.assertEqual(float(table.cos[p, j]), oracle("cos", a), (p, j))
            self.assertEqual(float(table.sin[p, j]), oracle("sin", a), (p, j))
        pos = torch.tensor([[0, 5, 8312]])
        c, s = table(torch.zeros(1, dtype=torch.bfloat16), pos)
        self.assertEqual((c.dtype, tuple(c.shape)), (torch.bfloat16, (1, 3, 2 * half)))
        self.assertTrue(torch.equal(c, table.cos[pos].to(torch.bfloat16)))

    def test_this_hosts_float64_cos_sin_meet_the_bound(self):
        """The rounding is correct on hosts whose float64 cos/sin err by less
        than ROPE_LIBM_ULPS; check this host's, on the arguments RoPE uses."""
        from y_lang.exact import model as em
        rng = random.Random(1)
        args = [rng.uniform(0, 32768) for _ in range(2000)] + [rng.uniform(0, 1) for _ in range(500)]
        x = torch.tensor(args, dtype=torch.float64)
        for name, fn in (("cos", torch.cos), ("sin", torch.sin)):
            y = fn(x).tolist()
            with mpmath.workprec(256):
                f = mpmath.cos if name == "cos" else mpmath.sin
                for a, yi in zip(args, y):
                    v = f(mpmath.mpf(a))
                    ulp = mpmath.ldexp(1, int(mpmath.floor(mpmath.log(abs(v), 2))) - 52)
                    self.assertLess(abs(mpmath.mpf(yi) - v) / ulp, em.ROPE_LIBM_ULPS, (name, a))

    def test_estimates_a_float32_step_or_two_off(self):
        """A float64 estimate one or two float32 steps (2^29, 2^30 float64
        ulps) off, on a midpoint argument, lands near another midpoint and so
        is decided in mpmath: the entry is still the right one. (An estimate
        that far off and not near a midpoint is trusted: ROPE_LIBM_ULPS is
        the assumption, and test_this_hosts_float64_cos_sin_meet_the_bound
        checks it here.)"""
        from y_lang.exact import model as em
        pairs = [p for p in midpoint_args(60, seed=3) if p[0] == "cos"][:12]
        args = torch.tensor([a for _, a in pairs], dtype=torch.float64)
        want = [oracle("cos", a) for _, a in pairs]
        for k in (1 << 29, -(1 << 29), 1 << 30):
            got = em.cos_sin_f32(args, 1.0, (shifted(torch.cos, k), torch.sin))[0]
            self.assertEqual(got.tolist(), want, k)
        with self.assertRaises(TypeError):
            em.cos_sin_f32(torch.zeros(3, dtype=torch.float32))

if __name__ == "__main__":
    unittest.main()
