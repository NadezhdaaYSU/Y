"""Y's `exact_decode_attention` kernel gives `exact_attention`'s bits.

`y_lang.exact.attention.exact_attention` with `ATTN = "y"` runs each decode
step through tests/exact_decode_attention.ysu; with `ATTN = "torch"` through
the torch ops it replaces. Both are called on the same float operands and
must return identical tensors: same shapes, same bits.

GPU only. The cases cover both Qwen2.5 head sizes (64, 128), key counts below,
at and past multiples of 32 and a single key, left-padding masks, a row with
every key masked, no mask at all, key counts either side of 256 (where the
kernel stops keeping scores in registers), and value ranges wide enough that
many keys get a zero weight and the exp argument reaches its clamp.
"""

import math
import unittest

import torch

CUDA = torch.cuda.is_available()


def _case(b, nh, nkv, t, d, spread, seed, mask="pad"):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(b, nh, 1, d, device="cuda", generator=g)
    k = torch.randn(b, nkv, t, d, device="cuda", generator=g)
    v = torch.randn(b, nkv, t, d, device="cuda", generator=g)
    # Per-row magnitudes over `spread` decades: some rows dominate, and the
    # rest get exp arguments past the table (weight 0) or at the clamp.
    q = q * torch.exp(torch.rand(b, nh, 1, 1, device="cuda", generator=g) * spread)
    k = k * torch.exp(torch.rand(b, nkv, t, 1, device="cuda", generator=g) * spread)
    v = v * torch.exp(torch.rand(b, nkv, t, 1, device="cuda", generator=g) * spread)
    if mask is None:
        am = None
    else:
        am = torch.zeros(b, 1, 1, t, device="cuda")
        neg = torch.finfo(torch.float32).min
        for i in range(b):
            pad = (i * 7) % max(t, 1) if mask == "pad" else 0
            am[i, :, :, :pad] = neg
        if mask == "all" or (mask == "pad" and b > 2):
            am[1, :, :, :] = neg                  # one row with every key masked
    return q, k, v, am, 1.0 / math.sqrt(d)


@unittest.skipUnless(CUDA, "needs a CUDA device")
class TestExactDecodeAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from y_lang.exact import attention, kv
        cls.attention, cls.kv = attention, kv
        cls.saved = (attention.ATTN, kv.V_INT8)
        kv.V_INT8 = True                          # the kernel reads int8 V

    @classmethod
    def tearDownClass(cls):
        cls.attention.ATTN, cls.kv.V_INT8 = cls.saved

    def run_both(self, q, k, v, am, sc):
        A = self.attention
        out = {}
        for mode in ("torch", "y"):
            A.ATTN = mode
            with torch.no_grad():
                out[mode] = A.exact_attention(None, q, k, v, am, scaling=sc)[0]
        return out["torch"], out["y"]

    def test_bits_equal_the_torch_ops(self):
        cases = [
            (3, 14, 2, 37, 64, 2.0, 0, "pad"),      # Qwen2.5-0.5B shape
            (4, 28, 4, 70, 128, 2.0, 1, "pad"),     # Qwen2.5-7B shape
            (2, 28, 4, 32, 128, 1.0, 2, "pad"),     # exactly one chunk
            (2, 28, 4, 33, 128, 1.0, 3, "pad"),
            (2, 28, 4, 1, 128, 1.0, 4, None),       # a single key
            (2, 14, 2, 5, 64, 1.0, 5, None),
            (3, 28, 4, 129, 128, 6.0, 6, "pad"),    # wide ranges: zero weights, clamps
            (3, 14, 2, 64, 64, 6.0, 7, "all"),
            (1, 7, 1, 200, 128, 3.0, 8, None),
            # The kernel keeps scores in registers up to 256 keys and
            # recomputes them past that: both sides of the line.
            (2, 28, 4, 256, 128, 2.0, 11, "pad"),
            (2, 14, 2, 257, 64, 2.0, 12, "pad"),
            (1, 28, 4, 300, 128, 6.0, 13, None),
        ]
        for c in cases:
            q, k, v, am, sc = _case(*c)
            want, got = self.run_both(q, k, v, am, sc)
            self.assertEqual(got.shape, want.shape, c)
            self.assertEqual(got.dtype, want.dtype, c)
            diff = (got.view(torch.int32) != want.view(torch.int32)).sum().item()
            self.assertEqual(diff, 0, f"{c}: {diff} of {want.numel()} values differ")

    def test_the_comparison_can_fail(self):
        # Control: the same comparison on a different temperature differs, so
        # equal bits above are not a comparison that cannot see a change.
        q, k, v, am, sc = _case(2, 28, 4, 40, 128, 2.0, 9, "pad")
        want, _ = self.run_both(q, k, v, am, sc)
        _, other = self.run_both(q, k, v, am, sc * 1.0001)
        self.assertGreater((other.view(torch.int32) != want.view(torch.int32)).sum().item(), 0)

    def core_both(self, s0, s1, sq_bits, sk_bits, sv_bits=None, v_dim0=None):
        """`exact_attention_core` on one query and two keys with scores s0 > s1
        (q = (127, 1, 0, ...), so a key (x, y, 0, ...) scores 127x + y), the
        query scale `sq_bits` and both keys' scale `sk_bits` (float32 bits);
        both V rows' scale `sv_bits` (default 0.01) and their dimension-0
        values `v_dim0` (default random)."""
        import struct
        A = self.attention
        f32 = lambda bits: struct.unpack("<f", struct.pack("<I", bits))[0]
        d = 32
        qi = torch.zeros(1, 1, 1, d, device="cuda")
        qi[..., 0], qi[..., 1] = 127.0, 1.0
        ki = torch.zeros(1, 1, 2, d, device="cuda")
        for t, s in enumerate((s0, s1)):
            ki[0, 0, t, 0], ki[0, 0, t, 1] = float(s // 127), float(s - 127 * (s // 127))
        sq = torch.full((1, 1, 1, 1), f32(sq_bits), dtype=torch.float32, device="cuda").double()
        sk = torch.full((1, 1, 2, 1), f32(sk_bits), dtype=torch.float32, device="cuda")
        g = torch.Generator(device="cuda").manual_seed(0)
        vi = torch.randint(-127, 128, (1, 1, 2, d), generator=g, device="cuda").to(torch.int8)
        if v_dim0 is not None:
            vi[0, 0, 0, 0], vi[0, 0, 1, 0] = v_dim0
        sv = torch.full((1, 1, 2, 1), 0.01 if sv_bits is None else f32(sv_bits), device="cuda")
        table = A._table(torch.device("cuda"))
        out = {}
        for mode in ("torch", "y"):
            A.ATTN = mode
            with torch.no_grad():
                out[mode] = A.exact_attention_core(qi, sq, ki, sk, vi, sv, 1023, 127, None,
                                                   1.0 / math.sqrt(d), table, torch.float32)
        a = ((sq.cpu() * (1.0 / math.sqrt(d))) * math.log2(math.e)) * 65536.0
        skd = sk[0, 0, 0, 0].double().cpu()
        return out["torch"], out["y"], a.reshape(()), skd

    def test_the_logit_is_formed_in_pythons_order(self):
        # (s * a) * sk, not (s * sk) * a: found by search, these operands give
        # exp arguments one apart under the two orders, and different weights.
        s0, s1 = 126359, 126357
        want, got, a, sk = self.core_both(s0, s1, 0x3FC23156, 0x3EADEC92)
        x = ((s0 * a) * sk) - ((s1 * a) * sk)
        x_other = ((s0 * sk) * a) - ((s1 * sk) * a)
        tq, tq_other = torch.round(x), torch.round(x_other)
        self.assertNotEqual(tq.item(), tq_other.item(), "the case no longer discriminates")
        p = self.attention.exp2_neg_q16_16(torch.stack([tq, tq_other]), self.attention._table("cpu"))
        self.assertNotEqual(p[0].item(), p[1].item())
        self.assertTrue(torch.equal(got.view(torch.int32), want.view(torch.int32)), (got, want))

    def test_the_exp_argument_rounds_half_to_even(self):
        # m - logit is exactly k + 0.5 with k even: half-to-even and half-up
        # give different exp arguments and different weights.
        s0, s1 = 128484, 128482
        want, got, a, sk = self.core_both(s0, s1, 0x3FFA0E4D, 0x3E193D2A)
        x = ((s0 * a) * sk) - ((s1 * a) * sk)
        self.assertEqual(x.item() - math.floor(x.item()), 0.5, "the case is no longer a tie")
        self.assertEqual(math.floor(x.item()) % 2, 0)
        p = self.attention.exp2_neg_q16_16(
            torch.tensor([math.floor(x.item()), math.floor(x.item()) + 1.0], dtype=torch.float64),
            self.attention._table("cpu"))
        self.assertNotEqual(p[0].item(), p[1].item())
        self.assertTrue(torch.equal(got.view(torch.int32), want.view(torch.int32)), (got, want))

    def test_packed_cache_rows_give_the_same_bits(self):
        # `kv.PACK`: the cache holds rows quantised once and packed; given
        # those, exact_attention must return what it returns given the float
        # K/V it quantises itself - on Y's kernel, on the torch ops with an
        # int8 V and with a float V, decoding and in a prefill.
        A, kv = self.attention, self.kv
        cases = [(3, 14, 2, 37, 64, 2.0, 0, "pad"), (2, 28, 4, 300, 128, 6.0, 13, None),
                 (3, 14, 2, 64, 64, 6.0, 7, "all")]
        try:
            for c in cases:
                for q_len in (1, 3):
                    q, k, v, am, sc = _case(*c)
                    if q_len > 1:
                        q = torch.randn(c[0], c[1], q_len, c[4], device="cuda")
                        am = None
                    kp, vp = kv.quantize_and_pack(k, v)
                    for mode, v_int8 in (("y", True), ("torch", True), ("torch", False)):
                        A.ATTN, kv.V_INT8 = mode, v_int8
                        with torch.no_grad():
                            want = A.exact_attention(None, q, k, v, am, scaling=sc)[0]
                            got = A.exact_attention(None, q, kp, vp, am, scaling=sc)[0]
                        self.assertTrue(torch.equal(got.view(torch.int32), want.view(torch.int32)),
                                        (c, q_len, mode, v_int8))
        finally:
            kv.V_INT8 = True

    def test_the_output_is_a_correctly_rounded_divide(self):
        # out = float32((acc * wscale) / l). Found by search: these operands
        # put dimension 0's quotient so close to a float32 rounding midpoint
        # that multiplying by 1/l instead gives a different float32.
        import struct
        f32 = lambda bits: struct.unpack("<f", struct.pack("<I", bits))[0]
        s0, s1, sq_b, sk_b, sv_b, v = 100370, 100347, 0x3F58533F, 0x3EEACC02, 0x3F5AE2AC, -72
        want, got, a, sk = self.core_both(s0, s1, sq_b, sk_b, sv_b, (v, v))
        tq = torch.round(((s0 * a) * sk) - ((s1 * a) * sk))
        p = self.attention.exp2_neg_q16_16(torch.stack([tq * 0, tq]), self.attention._table("cpu"))
        sv = torch.full((1, 1, 2), f32(sv_b), dtype=torch.float32).double()
        W, wscale = self.kv.fold_scale_into_weights(p.reshape(1, 1, 2), sv, 28)
        xs = float((W * v).sum()) * float(wscale.reshape(()))
        l = float(p.sum())
        div = torch.tensor(xs / l, dtype=torch.float64).float()
        mul = torch.tensor(xs * (1.0 / l), dtype=torch.float64).float()
        self.assertNotEqual(div.view(torch.int32).item(), mul.view(torch.int32).item(),
                            "the case no longer discriminates")
        self.assertEqual(want.reshape(-1)[0].view(torch.int32).item(), div.view(torch.int32).item())
        self.assertTrue(torch.equal(got.view(torch.int32), want.view(torch.int32)), (got, want))

    def test_subnormal_outputs(self):
        # V rows near 1e-40 put the outputs in float32's subnormal range,
        # where the kernel's reciprocal guard always divides. Exercised, not
        # discriminating: a multiply and a divide differ there about once in
        # 10^10 values, too rarely to find a case that would fail without it.
        q, k, v, am, sc = _case(2, 14, 2, 40, 64, 1.0, 14, "pad")
        v = v * 1e-40
        want, got = self.run_both(q, k, v, am, sc)
        self.assertGreater(int(((want != 0) & (want.abs() < 2.0 ** -126)).sum()), 1000)
        self.assertEqual((got.view(torch.int32) != want.view(torch.int32)).sum().item(), 0)

    def test_prefill_stays_on_the_torch_ops(self):
        q, k, v, am, sc = _case(2, 14, 2, 9, 64, 1.0, 10, None)
        q = torch.randn(2, 14, 3, 64, device="cuda")        # three query positions
        want, got = self.run_both(q, k, v, None, sc)
        self.assertTrue(torch.equal(got, want))


if __name__ == "__main__":
    unittest.main()
