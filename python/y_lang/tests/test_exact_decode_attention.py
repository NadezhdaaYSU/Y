"""Y's `exact_decode_attention` kernel gives `exact_attention`'s bits.

`y_lang.exact.attention.exact_attention` with `ATTN = "y"` runs each decode
step through tests/exact_decode_attention.ysu; with `ATTN = "torch"` through
the torch ops it replaces. Both are called on the same float operands and
must return identical tensors: same shapes, same bits.

GPU only. The cases cover both Qwen2.5 head sizes (64, 128), key counts below,
at and past multiples of 32 and a single key, left-padding masks, a row with
every key masked, no mask at all, and value ranges wide enough that many keys
get a zero weight and the exp argument reaches its clamp.
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

    def test_prefill_stays_on_the_torch_ops(self):
        q, k, v, am, sc = _case(2, 14, 2, 9, 64, 1.0, 10, None)
        q = torch.randn(2, 14, 3, 64, device="cuda")        # three query positions
        want, got = self.run_both(q, k, v, None, sc)
        self.assertTrue(torch.equal(got, want))


if __name__ == "__main__":
    unittest.main()
