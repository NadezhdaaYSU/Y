"""y_lang.exact.train_lm (T2): the tiny character model's hand-written
backward matches autograd; trained with exact sums, its weights do not
depend on the order of the batch's rows or on the device; the float32 twin,
the control, does."""

import unittest

import torch

from y_lang.exact import train_lm as L

CFG = L.Config(vocab=96, d=64, heads=2, layers=1, ff=128, ctx=32)
TEXT = (L.__doc__ or "") * 12


def _cuda_ok():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (7, 5)


def unguarded_reloads(source):
    """(kernel, pointer) for every Triton kernel in Inductor's generated
    `source` that loads a pointer it stored earlier with no
    `tl.debug_barrier()` in between: the store-then-load race of torch
    2.13.0 (y_lang.exact.model)."""
    import re
    found = []
    for kern in re.split(r"@triton\.jit\s*\n", source)[1:]:
        body = kern.split("''', device_str")[0]
        name = re.match(r"\s*def (\w+)", body).group(1)
        guarded = {}                      # pointer -> a barrier since its last store
        for line in body.splitlines():
            s = line.strip()
            if s.startswith("tl.debug_barrier()"):
                guarded = {p: True for p in guarded}
            for p in re.findall(r"tl\.load\((\w+)", s):
                if guarded.get(p) is False:
                    found.append((name, p))
            m = re.match(r"tl\.store\((\w+)", s)
            if m:
                guarded[m.group(1)] = False
    return found


def run(ops, device, dtype, steps=12, permute=False, cfg=CFG, B=4):
    """Training as `train_lm.train` does it; with `permute`, each batch's
    rows are reversed, which reorders every sum over tokens."""
    data = L.CharData(TEXT)
    params = L.init_params(cfg, 0, dtype, device)
    opt = L.Adam(params, lr=3e-3, exact=ops.exact)
    gen = torch.Generator().manual_seed(1)
    losses = []
    for _ in range(steps):
        x, y = data.batch(gen, B, cfg.ctx)
        if permute:
            x, y = x.flip(0), y.flip(0)
        x, y = x.to(device), y.to(device)
        _, probs, saved = L.forward(ops, cfg, params, x, y)
        losses.append(L.cross_entropy_on_host(probs, y))
        opt.step(params, L.backward(ops, cfg, params, probs, y, saved))
    return params, losses


class Backward(unittest.TestCase):
    def test_matches_autograd(self):
        cfg = L.Config(vocab=32, d=64, heads=2, layers=2, ff=128, ctx=16)
        ops = L.Ops(exact=False)
        p = L.init_params(cfg, seed=0)
        for v in p.values():
            v.requires_grad_(True)
        g = torch.Generator().manual_seed(5)
        x = torch.randint(0, 32, (3, 16), generator=g)
        y = torch.randint(0, 32, (3, 16), generator=g)
        _, probs, saved = L.forward(ops, cfg, p, x, y)
        loss = -torch.log(probs[torch.arange(48), y.reshape(-1)]).mean()
        auto = dict(zip(p, torch.autograd.grad(loss, list(p.values()))))
        with torch.no_grad():
            man = L.backward(ops, cfg, p, probs.detach(), y, saved)
        self.assertEqual(set(man), set(p))
        for k in p:
            rel = float((man[k] - auto[k]).abs().max() / auto[k].abs().max())
            self.assertLess(rel, 1e-12, k)


class Pieces(unittest.TestCase):
    def test_pow2_is_exact(self):
        k = torch.tensor([-1022, -7, 0, 1, 52, 1023])
        self.assertEqual(L.pow2(k).tolist(), [2.0 ** int(e) for e in k])
        with self.assertRaises(OverflowError):
            L.pow2(torch.tensor([1024]))

    def test_isqrt_is_the_floor_square_root(self):
        import math
        g = torch.Generator().manual_seed(6)
        n = torch.cat([torch.tensor([0, 1, 2, 3, 4, 15, 16, (1 << 62) - 1, (1 << 61) + 12345]),
                       torch.randint(0, 1 << 62, (2000,), generator=g)])
        for v, r in zip(n.tolist(), L.isqrt(n).tolist()):
            self.assertEqual(r, math.isqrt(v))
        with self.assertRaises(OverflowError):
            L.isqrt(torch.tensor([1 << 62]))

    def test_det_sqrt_is_close_and_scaled_exactly(self):
        import math
        v = torch.cat([torch.tensor([0.0, 1e-280, 1e-8, 0.25, 1.0, 2.0, 3.0, 1e300], dtype=torch.float64),
                       torch.rand(2000, generator=torch.Generator().manual_seed(7), dtype=torch.float64) * 1e4])
        s = L.det_sqrt(v)
        for x, r in zip(v.tolist(), s.tolist()):
            self.assertLessEqual(abs(r - math.sqrt(x)), math.sqrt(x) * 2.0 ** -29)
        self.assertEqual(L.det_sqrt(torch.tensor([4.0, 0.25], dtype=torch.float64)).tolist(), [2.0, 0.5])
        with self.assertRaises(OverflowError):      # refused, not wrong
            L.det_sqrt(torch.tensor([1e-300], dtype=torch.float64))

    def test_digits_rebuild_the_value(self):
        g = torch.Generator().manual_seed(9)
        for bits in (8, 12, 13, 14, 15, 16, 21, 22):
            top = (1 << (bits - 1)) - 1
            q = torch.randint(-top, top + 1, (5000,), generator=g, dtype=torch.int32)
            q[:3] = torch.tensor([-top, top, 0], dtype=torch.int32)
            ds = L.digits(q, bits)
            self.assertEqual(len(ds), L.ndigits(bits))
            back = sum(d * 128 ** i for i, d in enumerate(ds))
            self.assertTrue(torch.equal(back, q))
            if bits > 8:
                for d in ds:
                    self.assertGreaterEqual(int(d.min()), -64)
                    self.assertLessEqual(int(d.max()), 63)
            self.assertGreaterEqual(L.capacity(bits), top)
        with self.assertRaises(OverflowError):
            L.digits(torch.tensor([20000], dtype=torch.int32), 12)   # two digits hold |q| <= 8256

    def test_ndigits_is_the_proofs(self):
        # proofs/ExactTrainArith.v proves its transcription of ndigits enough
        # and the fewest for 9 to 64 bits, and states this table
        # (`ndigits_table`): the Python must return it.
        bits = (8, 9, 12, 13, 14, 15, 16, 20, 21, 22, 28, 29, 64)
        self.assertEqual([L.ndigits(b) for b in bits], [1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 10])
        self.assertEqual([L.capacity(b) for b in (12, 16, 22)], [8256, 1056832, 135274560])

    def test_integer_quantizer_is_round_half_even(self):
        # q8 works on the integers inside the floats; it must be the float64
        # formula round(x * 2^-s), clamped, exactly. Ties at .5 of both
        # parities and signs, zeros of both signs, subnormal entries, rows
        # spanning many binades.
        def by_floats(x, bits):
            top = (1 << (bits - 1)) - 1
            e = torch.frexp(x.abs().amax(-1, keepdim=True))[1].to(torch.int64)
            s = e - (bits - 1)
            return torch.round(x * L.pow2(-s)).clamp(-top, top), s
        g = torch.Generator().manual_seed(11)
        rows = [torch.randn(64, 300, generator=g, dtype=torch.float64)
                * torch.logspace(-30, 30, 300, dtype=torch.float64)]
        ties = (torch.arange(-200, 200, dtype=torch.float64) + 0.5).reshape(2, 200)
        rows.append(torch.cat([ties, torch.full((2, 100), 0.0, dtype=torch.float64)], 1))
        z = torch.zeros(3, 300, dtype=torch.float64)
        z[0, :5] = torch.tensor([-0.0, 0.0, 5e-324, -2.0 ** -1060, 1.0], dtype=torch.float64)
        z[1, 0] = 1e150                  # scales stay within 2^+-511
        z[1, 1] = -1e-300
        rows.append(z)
        x = torch.cat(rows)
        for bits in (8, 12, 16):
            for scale in (1.0, 2.0 ** -7, 2.0 ** -3):
                xs = x * scale
                q, s = L.q8(xs, bits=bits)
                rq, rs = by_floats(xs, bits)
                self.assertTrue(torch.equal(s, rs))
                self.assertTrue(torch.equal(q.to(torch.float64), rq), (bits, scale))
        # The wide rounding behind the exact sums, including left shifts.
        for sh in (-70, -61, -20, 0, 9):
            s = torch.full((x.shape[0], 1), sh, dtype=torch.int64) + L.exponent(x, -1)
            ok = (x * L.pow2(-s)).abs() < 2.0 ** 62
            ref = torch.round(x * L.pow2(-s)).to(torch.int64)
            self.assertTrue(torch.equal(L.scaled_round(x, s)[ok], ref[ok]), sh)

    def test_float32_quantizer_is_round_half_even(self):
        # The int32 path for float32 values: round(x * 2^-s), clamped, done
        # exactly in float64, for the s q8 chose; and that s is frexp's for a
        # row whose max is normal.
        g = torch.Generator().manual_seed(12)
        rows = [torch.randn(64, 300, generator=g) * torch.logspace(-12, 12, 300)]
        ties = (torch.arange(-200, 200, dtype=torch.float32) + 0.5).reshape(2, 200)
        rows.append(torch.cat([ties, torch.zeros(2, 100)], 1))
        z = torch.zeros(3, 300)
        z[0, :5] = torch.tensor([-0.0, 0.0, 1e-45, -3e-39, 1.0])
        z[1, :3] = torch.tensor([3e-39, -1e-40, 2e-44])          # all subnormal
        rows.append(z)
        x = torch.cat(rows)
        for bits in (8, 12, 16):
            for scale in (1.0, 2.0 ** -7, 2.0 ** -3):
                xs = x * scale
                q, s = L.q8(xs, bits=bits)
                self.assertEqual(q.dtype, torch.int32)
                top = (1 << (bits - 1)) - 1
                ref = torch.round(xs.double() * L.pow2(-s)).clamp(-top, top)
                self.assertTrue(torch.equal(q.double(), ref), (bits, scale))
                normal = xs.abs().amax(-1) >= 2.0 ** -126
                fr = torch.frexp(xs.abs().amax(-1, keepdim=True).double())[1] - (bits - 1)
                self.assertTrue(torch.equal(s[normal], fr[normal].to(torch.int64)))
        # The wide rounding behind the exact sums, float32 values, left shifts too.
        for sh in (-61, -40, -20, 0, 9):
            s = torch.full((x.shape[0], 1), sh, dtype=torch.int64) + L.exponent(x, -1)
            y = x.double() * L.pow2(-s)
            ok = y.abs() < 2.0 ** 62
            self.assertTrue(torch.equal(L.scaled_round(x, s)[ok], torch.round(y).to(torch.int64)[ok]), sh)

    def test_float32_formula_is_the_integer_definition(self):
        # scaled_round's float32 path (x * 2^a * 2^b, then round) against the
        # integer definition on the floats' bits: rows over the whole float32
        # range and down into the subnormals, ties of both parities, zeros of
        # both signs, the largest and smallest values; q8's scales and the
        # exact sums' (which shift left).
        g = torch.Generator().manual_seed(13)
        m = torch.randint(-149, 124, (400, 1), generator=g).float()
        x = torch.randn(400, 256, generator=g) * torch.exp2(m.clamp(-126, 123))
        x = x * torch.where(m < -126, torch.exp2(m + 126), torch.ones_like(m))
        x[:, :8] = torch.tensor([0.0, -0.0, 1e-45, -1e-45, 1.17549435e-38, -3e-39, 3.4e38, -3.4e38])
        t = (torch.arange(-2048, 2048).float() + 0.5).reshape(16, 256)
        for v in (x, t, t * 2.0 ** -20, t * 2.0 ** -130, torch.randn(8, 256, generator=g) * 5e37):
            for bits in (8, 12, 16):
                s = L.exponent(v, -1) - (bits - 1)
                ok = (s.abs() <= L.Q_SCALE_RANGE).reshape(-1)
                self.assertTrue(torch.equal(L.scaled_round(v, s, wide=False)[ok],
                                            L.scaled_round_int(v, s, wide=False)[ok]), bits)
            for c in (0, 8, 12):
                s = L.exponent(v, -1) - 61 + c
                self.assertTrue(torch.equal(L.scaled_round(v, s), L.scaled_round_int(v, s)), c)

    def test_non_finite_values_are_refused(self):
        # A NaN used to pass the quantizer, and det_sqrt(NaN) gave 0 on CUDA.
        for bad in (float("nan"), float("inf"), -float("inf")):
            x = torch.ones(2, 8, dtype=torch.float64)
            x[1, 3] = bad
            with self.assertRaises(OverflowError):
                L.q8(x)
            with self.assertRaises(OverflowError):
                L.Ops(exact=True).sum(x, -1)
            with self.assertRaises(OverflowError):
                L.det_sqrt(x.reshape(-1))
            with self.assertRaises(OverflowError):
                L.Ops(exact=True).softmax(x, torch.zeros_like(x, dtype=torch.bool))
            with self.assertRaises(OverflowError):      # and float32 activations
                L.q8(x.float())
            with self.assertRaises(OverflowError):
                L.Ops(exact=True, act=torch.float32).softmax(x.float(), torch.zeros_like(x, dtype=torch.bool))
        with self.assertRaises(OverflowError):
            L.det_sqrt(torch.tensor([-1.0], dtype=torch.float64))
        # A quantizer scale past 2^+-511 is refused, so a product's scale
        # 2^(sa + sb) is always a normal float64.
        for big in (1e160, 1e-160):
            with self.assertRaises(OverflowError):
                L.q8(torch.tensor([[big, 1e-170]], dtype=torch.float64))
        L.q8(torch.tensor([[1e150, 1e-140]], dtype=torch.float64))
        self.assertEqual(L.det_sqrt(torch.tensor([-0.0, 0.0], dtype=torch.float64)).tolist(), [0.0, 0.0])

    def test_wide_matmul_is_the_exact_product(self):
        g = torch.Generator().manual_seed(10)
        a = torch.randn(16, 96, generator=g, dtype=torch.float64)
        b = torch.randn(24, 96, generator=g, dtype=torch.float64)
        ops = L.Ops(exact=True, grad_bits=12, bwd_bits=8)
        c = ops.matmul(a, b, grad="a")
        qa, sa = L.q8(a, bits=12)
        qb, sb = L.q8(b, bits=8)
        ref = (qa.double() @ qb.double().t()) * L.pow2(sa + sb.t())     # exact: integers far below 2^53
        self.assertTrue(torch.equal(c, ref))
        # Wider gradients are closer to the float product.
        f = a @ b.t()
        narrow = L.Ops(exact=True, grad_bits=8).matmul(a, b, grad="a")
        self.assertLess(float((c - f).abs().max()), float((narrow - f).abs().max()))

    def test_exact_sum_ignores_order(self):
        g = torch.Generator().manual_seed(2)
        x = torch.randn(7, 1000, generator=g, dtype=torch.float64) * torch.logspace(-8, 8, 1000, dtype=torch.float64)
        ops = L.Ops(exact=True)
        perm = torch.randperm(1000, generator=g)
        self.assertTrue(torch.equal(ops.sum(x, -1), ops.sum(x[:, perm], -1)))
        # And it is a sum: close to the float64 one.
        self.assertTrue(torch.allclose(ops.sum(x, -1), x.sum(-1, keepdim=True), rtol=1e-12, atol=0))

    def test_exact_softmax_rows_sum_to_one(self):
        ops = L.Ops(exact=True)
        z = torch.randn(5, 33, generator=torch.Generator().manual_seed(3), dtype=torch.float64) * 4
        blocked = torch.zeros_like(z, dtype=torch.bool)
        blocked[:, -3:] = True
        p = ops.softmax(z, blocked)
        self.assertTrue(bool((p[:, -3:] == 0).all()))
        self.assertTrue(torch.allclose(p.sum(-1), torch.ones(5, dtype=torch.float64), atol=1e-12))
        ref = torch.softmax(z.masked_fill(blocked, float("-inf")), -1)
        self.assertLess(float((p - ref).abs().max()), 1e-6)

    def test_matmul_is_the_product_of_the_int8_copies(self):
        ops = L.Ops(exact=True)
        g = torch.Generator().manual_seed(4)
        a = torch.randn(8, 64, generator=g, dtype=torch.float64)
        b = torch.randn(16, 64, generator=g, dtype=torch.float64)
        c = ops.matmul(a, b)
        rel = float((c - a @ b.t()).abs().max() / (a @ b.t()).abs().max())
        self.assertLess(rel, 0.05)          # int8 operands: a few percent
        qa, sa = L.q8(a)
        qb, sb = L.q8(b)
        ref = torch.tensor([[sum(int(qa[i, k]) * int(qb[j, k]) for k in range(64)) * 2.0 ** (int(sa[i]) + int(sb[j]))
                             for j in range(16)] for i in range(8)], dtype=torch.float64)
        self.assertTrue(torch.equal(c, ref))


class Training(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params, cls.losses = run(L.Ops(exact=True), "cpu", torch.float64)

    def test_it_trains(self):
        self.assertLess(self.losses[-1], self.losses[0] - 0.5)

    def test_row_order_does_not_change_exact_weights(self):
        params, losses = run(L.Ops(exact=True), "cpu", torch.float64, permute=True)
        self.assertEqual(L.digest(params), L.digest(self.params))
        self.assertEqual(losses, self.losses)

    def test_trainer_gives_the_same_weights(self):
        # The Trainer (flat Adam, deferred checks) against the dict API above.
        data = L.CharData(TEXT)
        tr = L.Trainer(L.Ops(exact=True), CFG, "cpu", torch.float64)
        gen = torch.Generator().manual_seed(1)
        tps = [tr.step(*data.batch(gen, 4, CFG.ctx)) for _ in range(12)]
        tr.verify()
        self.assertEqual(L.digest(tr.params), L.digest(self.params))
        self.assertEqual([L.nll_on_host(t) for t in tps], self.losses)

    def test_float32_activations_train_and_ignore_row_order(self):
        ops32 = L.Ops(exact=True, act=torch.float32)
        a, la = run(ops32, "cpu", torch.float64)
        b, lb = run(L.Ops(exact=True, act=torch.float32), "cpu", torch.float64, permute=True)
        self.assertLess(la[-1], la[0] - 0.5)
        self.assertEqual(L.digest(a), L.digest(b))
        self.assertEqual(la, lb)
        self.assertNotEqual(L.digest(a), L.digest(self.params))   # a different model

    def test_a_deferred_refusal_still_raises(self):
        data = L.CharData(TEXT)
        tr = L.Trainer(L.Ops(exact=True), CFG, "cpu", torch.float64)
        tr.params["up.0"][3, 5] = float("nan")       # a view of the flat buffer
        tr.step(*data.batch(torch.Generator().manual_seed(1), 4, CFG.ctx))
        with self.assertRaises(OverflowError):
            tr.verify()

    def test_float_twin_depends_on_row_order(self):
        a, _ = run(L.Ops(exact=False), "cpu", torch.float32)
        b, _ = run(L.Ops(exact=False), "cpu", torch.float32, permute=True)
        self.assertNotEqual(L.digest(a), L.digest(b))


@unittest.skipUnless(_cuda_ok(), "needs a CUDA GPU (sm_75 or later)")
class OnTheGpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.backends.cuda.matmul.allow_tf32 = False
        cls.ref, cls.losses = run(L.Ops(exact=True), "cpu", torch.float64)

    def _same_as_cpu(self, mm):
        params, losses = run(L.Ops(exact=True, mm=mm), "cuda", torch.float64)
        self.assertEqual(losses, self.losses)
        self.assertEqual(L.digest(params), L.digest(self.ref))

    def test_y_gemm_gives_the_cpu_weights(self):
        self._same_as_cpu(L.mm_y)

    def test_cublas_gives_the_cpu_weights(self):
        self._same_as_cpu(L.mm_torch)

    def test_float_twin_differs_across_devices(self):
        a, _ = run(L.Ops(exact=False), "cpu", torch.float32)
        b, _ = run(L.Ops(exact=False), "cuda", torch.float32)
        self.assertNotEqual(L.digest(a), L.digest(b))

    def test_int8_gradients_on_y_gemm_give_the_cpu_weights(self):
        ref, losses = run(L.Ops(exact=True, grad_bits=8), "cpu", torch.float64)
        params, gl = run(L.Ops(exact=True, mm=L.mm_y, grad_bits=8), "cuda", torch.float64)
        self.assertEqual(gl, losses)
        self.assertEqual(L.digest(params), L.digest(ref))
        self.assertNotEqual(L.digest(ref), L.digest(self.ref))   # the width did change training

    def test_wide_operands_on_both_sides_give_the_cpu_weights(self):
        ref, losses = run(L.Ops(exact=True, grad_bits=16, bwd_bits=12), "cpu", torch.float64)
        params, gl = run(L.Ops(exact=True, mm=L.mm_torch, grad_bits=16, bwd_bits=12), "cuda", torch.float64)
        self.assertEqual(gl, losses)
        self.assertEqual(L.digest(params), L.digest(ref))

    def test_compiled_graph_gives_the_cpu_weights(self):
        # torch.compile fuses the step and one CUDA graph replays it: the
        # fast path, which must still give the CPU's bits.
        data = L.CharData(TEXT)
        gen = torch.Generator().manual_seed(1)
        bs = [data.batch(gen, 4, CFG.ctx) for _ in range(12)]
        for graph, comp in ((True, False), (True, True)):
            tr = L.Trainer(L.Ops(exact=True, mm=L.mm_y), CFG, "cuda", torch.float64,
                           graph=graph, compile=comp)
            tps = [tr.step(x.cuda(), y.cuda()) for x, y in bs]
            tr.verify()
            self.assertEqual([L.nll_on_host(t) for t in tps], self.losses, (graph, comp))
            self.assertEqual(L.digest(tr.params), L.digest(self.ref), (graph, comp))

    def test_compiled_kernels_reload_only_after_a_barrier(self):
        # Inductor's multi-loop kernels store a buffer and load it back in a
        # later loop; Triton may give the two different thread layouts, so
        # without a barrier a thread loads what another has not stored yet
        # (it corrupted a third of one gradient at width 512). Every such
        # reload in the compiled step must follow a barrier; the control,
        # with the fix removed, must find one that does not.
        import torch._inductor.config as ic
        from torch._inductor.codegen import triton as tc
        from torch._inductor.utils import run_and_get_code
        data = L.CharData(TEXT)
        x, y = (t.cuda() for t in data.batch(torch.Generator().manual_seed(1), 4, CFG.ctx))
        fixed = tc.TritonKernel.load
        self.assertTrue(getattr(fixed, "_y_barrier", False))
        unfixed = fixed.__closure__[0].cell_contents
        found, barriers = {}, {}
        try:
            for name, load in (("fixed", fixed), ("unfixed", unfixed)):
                tc.TritonKernel.load = load
                torch._dynamo.reset()
                with ic.patch(fx_graph_cache=False):
                    tr = L.Trainer(L.Ops(exact=True, mm=L.mm_y), CFG, "cuda", torch.float64, compile=True)
                    _, code = run_and_get_code(tr.step, x, y)
                tr.verify()
                src = "\n".join(code)
                found[name], barriers[name] = unguarded_reloads(src), src.count("tl.debug_barrier()")
        finally:
            tc.TritonKernel.load = fixed
            torch._dynamo.reset()
        self.assertEqual(found["fixed"], [])
        self.assertGreater(barriers["fixed"], 0)          # there was a reload: not vacuous
        self.assertNotEqual(found["unfixed"], [])         # the control

    def test_compiled_step_at_width_512_gives_the_cpu_weights(self):
        # The size at which the race showed (4 layers, width 512, 4096
        # tokens): one compiled step against the CPU's. About two minutes,
        # most of it compiling.
        import torch._inductor.config as ic
        cfg = L.Config(d=512, heads=8, layers=4, ff=2048, ctx=256)
        data = L.CharData(TEXT * 40)
        x, y = data.batch(torch.Generator().manual_seed(1), 16, cfg.ctx)
        ref = L.Trainer(L.Ops(exact=True), cfg, "cpu", torch.float64)
        ref_tp = ref.step(x, y)
        ref.verify()
        torch._dynamo.reset()
        with ic.patch(fx_graph_cache=False):
            tr = L.Trainer(L.Ops(exact=True, mm=L.mm_y), cfg, "cuda", torch.float64, compile=True)
            tp = tr.step(x.cuda(), y.cuda())
        tr.verify()
        torch._dynamo.reset()
        self.assertEqual(L.tp_digest(tp), L.tp_digest(ref_tp))
        self.assertEqual(L.digest(tr.params), L.digest(ref.params))
        self.assertTrue(torch.equal(tr.opt.v.cpu(), ref.opt.v))

    def test_float32_activations_give_the_cpu_weights(self):
        ref, losses = run(L.Ops(exact=True, act=torch.float32), "cpu", torch.float64)
        params, gl = run(L.Ops(exact=True, mm=L.mm_y, act=torch.float32), "cuda", torch.float64)
        self.assertEqual(gl, losses)
        self.assertEqual(L.digest(params), L.digest(ref))
        data = L.CharData(TEXT)
        gen = torch.Generator().manual_seed(1)
        bs = [data.batch(gen, 4, CFG.ctx) for _ in range(12)]
        tr = L.Trainer(L.Ops(exact=True, mm=L.mm_y, act=torch.float32), CFG, "cuda", torch.float64,
                       graph=True, compile=True)
        tps = [tr.step(x.cuda(), y.cuda()) for x, y in bs]
        tr.verify()
        torch._dynamo.reset()
        self.assertEqual([L.nll_on_host(t) for t in tps], losses)
        self.assertEqual(L.digest(tr.params), L.digest(ref))

    def test_a_deferred_refusal_raises_from_a_graph(self):
        data = L.CharData(TEXT)
        gen = torch.Generator().manual_seed(1)
        tr = L.Trainer(L.Ops(exact=True, mm=L.mm_y), CFG, "cuda", torch.float64, graph=True)
        x, y = (t.cuda() for t in data.batch(gen, 4, CFG.ctx))
        tr.step(x, y)
        tr.verify()                                   # clean so far
        tr.params["up.0"][3, 5] = float("nan")
        tr.step(x, y)
        with self.assertRaises(OverflowError):
            tr.verify()

    def test_det_sqrt_same_bits_on_both(self):
        # torch.sqrt's float64 differs between this CPU and CUDA (2026-11-21);
        # det_sqrt must not.
        v = torch.rand(100000, generator=torch.Generator().manual_seed(8), dtype=torch.float64) * 10 + 0.01
        self.assertTrue(torch.equal(L.det_sqrt(v), L.det_sqrt(v.cuda()).cpu()))

    def test_scalar_divide_is_written_as_a_multiply(self):
        # Why: PyTorch's CUDA divide by a Python scalar multiplies by its
        # reciprocal, which the CPU does not (2026-11-21).
        import inspect
        import re
        src = inspect.getsource(L)
        self.assertIn("* (1.0 / d)", src)
        self.assertIsNone(re.search(r"(?<!1\.0) / (d|n|c1|c2)\b", src))


if __name__ == "__main__":
    unittest.main()
