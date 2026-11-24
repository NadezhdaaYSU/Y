"""y_lang.exact.lora: exact LoRA fine-tuning of a model served by
y_lang.exact. The forward pass of training is the served model's (its logits
are a cached decode's, bit for bit), each backward piece is the float64
gradient of the unquantized piece to within its quantization, and the
adapters after training are the same bits on a CPU and a GPU."""

import math
import os
import unittest

import torch
import torch.nn as nn

PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
F64 = torch.float64


def rel(a, b):
    a, b = a.detach().double().flatten(), b.detach().double().flatten()
    return float((a - b).norm() / b.norm())


def tiny_model(seed=0, layers=2, arch="qwen2"):
    """A random Qwen2 (vocab 512, width 64, 4 query heads over 2 K/V heads),
    or the same as a Llama with Llama 3's scaled rotary embedding, converted
    to the exact form."""
    import transformers.models.qwen2.modeling_qwen2 as qwen2
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM
    from y_lang.exact import attention as A, model as M
    torch.manual_seed(seed)
    shape = dict(vocab_size=512, hidden_size=64, intermediate_size=128,
                 num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                 max_position_embeddings=128, tie_word_embeddings=True, attn_implementation="eager")
    if arch == "llama":
        rope = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                "original_max_position_embeddings": 32, "rope_theta": 10000.0}
        m = LlamaForCausalLM(LlamaConfig(**shape, rope_parameters=rope)).float().eval()
    else:
        m = Qwen2ForCausalLM(Qwen2Config(**shape)).float().eval()
    qwen2.eager_attention_forward = A.exact_attention
    M.convert(m)
    return m


def randomize_b(ad, seed=1, std=0.02):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for k, t in ad.params.items():
            if k.endswith(".B"):
                t.copy_((torch.randn(t.shape, generator=g, dtype=F64) * std).to(t.device))


def batch(seed=0, B=3, T=30, prompt=10, vocab=512):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (B, T), generator=g)
    labels = ids.clone()
    labels[:, :prompt] = -100
    return ids, labels


def bits_equal(a, b):
    """The same structure, and every tensor in it the same dtype, shape and
    bits (floats compared as integers of their width, so a NaN or -0.0
    counts)."""
    if isinstance(a, (tuple, list)):
        return isinstance(b, (tuple, list)) and len(a) == len(b) and all(map(bits_equal, a, b))
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape:
            return False
        if a.dtype.is_floating_point:
            it = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
            a, b = a.contiguous().view(it), b.contiguous().view(it)
        return torch.equal(a, b)
    return a == b


def train_tiny(device, steps=4, mm=None, compile=False, lengths=(30,), arch="qwen2"):
    from y_lang.exact import lora
    m = tiny_model(arch=arch).to(device)
    ad = lora.attach(m, r=8, alpha=8, mm=mm, compile=compile)
    sft = lora.SFT(m, ad, lr=3e-3)
    tps = []
    for s in range(steps):
        ids, labels = batch(seed=s, T=lengths[s % len(lengths)])
        tps.append(lora.tp_digest(sft.step(ids.to(device), labels.to(device))))
    sft.verify()
    return ad.digest(), tps, ad


class Pieces(unittest.TestCase):
    """Each backward against float64 autograd of the unquantized piece."""

    def setUp(self):
        from y_lang.exact import lora
        torch.manual_seed(0)
        self.lora = lora
        self.ops = lora.make_ops("cpu")

    def test_frozen_linear(self):
        from y_lang.exact import model as M
        lin = nn.Linear(96, 80)
        ex = M.ExactLinear(lin)
        x = torch.randn(10, 96, requires_grad=True)
        dy = torch.randn(10, 80)
        self.lora.Frozen(ex, self.ops)(x).backward(dy)
        W = ex.w8t.t().double() * ex.sw.double()[:, None]
        self.assertLess(rel(x.grad, dy.double() @ W), 2e-3)     # 12-bit dY

    def test_lora_linear(self):
        from y_lang.exact import model as M
        lin = nn.Linear(96, 80)
        lo = self.lora.LoRALinear(M.ExactLinear(lin), self.ops, 8, 16)
        A = (torch.rand(8, 96, dtype=F64) * 2 - 1) * 0.1
        B = torch.randn(80, 8, dtype=F64) * 0.05
        lo.A, lo.B = A.clone().requires_grad_(), B.clone().requires_grad_()
        x = torch.randn(10, 96, requires_grad=True)
        dy = torch.randn(10, 80)
        lo(x).backward(dy)
        xr, Ar, Br = (t.detach().double().requires_grad_() for t in (x, A, B))
        W = lo.base.w8t.t().double() * lo.base.sw.double()[:, None]
        (xr @ W.t() + 2.0 * (xr @ Ar.t()) @ Br.t()).backward(dy.double())
        self.assertLess(rel(x.grad, xr.grad), 1e-2)
        self.assertLess(rel(lo.A.grad, Ar.grad), 3e-2)           # int8 operands
        self.assertLess(rel(lo.B.grad, Br.grad), 3e-2)
        with self.assertRaises(ValueError):
            self.lora.LoRALinear(M.ExactLinear(lin), self.ops, 8, 12)   # 1.5: not exact

    def test_norm(self):
        from y_lang.exact import model as M

        class Stock:
            pass
        s = Stock()
        s.weight, s.variance_epsilon = torch.randn(64).abs() + 0.5, 1e-6
        norm = self.lora.Norm(M.ExactRMSNorm(s), self.ops)
        x = (torch.randn(6, 64) * 3).requires_grad_()
        dy = torch.randn(6, 64)
        norm(x).backward(dy)
        xr = x.detach().double().requires_grad_()
        (xr / torch.sqrt((xr * xr).mean(-1, keepdim=True) + 1e-6) * s.weight.double()).backward(dy.double())
        self.assertLess(rel(x.grad, xr.grad), 1e-6)

    def test_silu(self):
        from y_lang.exact import model as M
        x = (torch.randn(2000) * 5).requires_grad_()
        dy = torch.randn(2000)
        self.lora.SiLU(M.ExactSiLU(), self.ops)(x).backward(dy)
        xr = x.detach().double().requires_grad_()
        (xr * torch.sigmoid(xr)).backward(dy.double())
        self.assertLess(rel(x.grad, xr.grad), 1e-5)               # the table's sigmoid

    def test_attention_with_grouped_heads(self):
        b, nh, nkv, T, d = 2, 4, 2, 20, 16
        q = torch.randn(b, nh, T, d, requires_grad=True)
        k = torch.randn(b, nkv, T, d, requires_grad=True)
        v = torch.randn(b, nkv, T, d, requires_grad=True)
        mask = torch.full((T, T), torch.finfo(torch.float32).min).triu(1)[None, None].expand(b, 1, T, T)
        sc = d ** -0.5
        out = self.lora._AttnFn.apply(q, k, v, mask, None, sc, self.ops)
        dout = torch.randn_like(out)
        out.backward(dout)
        qr, kr, vr = (t.detach().double().requires_grad_() for t in (q, k, v))
        kk, vv = (t.repeat_interleave(nh // nkv, 1) for t in (kr, vr))
        o = torch.softmax((qr @ kk.transpose(-1, -2)) * sc + mask.double(), -1) @ vv
        o.transpose(1, 2).backward(dout.double())
        self.assertLess(rel(out, o.transpose(1, 2)), 1e-2)
        for got, ref in ((q.grad, qr.grad), (k.grad, kr.grad), (v.grad, vr.grad)):
            self.assertLess(rel(got, ref), 3e-2)

    def test_attention_backward_reads_the_forwards_weights(self):
        # The weights the backward uses are the forward's p / l: their
        # product with V (in float64) is the forward's output to within V's
        # int8 quantization, and every row sums to 1.
        b, nh, nkv, T, d = 1, 4, 2, 12, 16
        q, k, v = torch.randn(b, nh, T, d), torch.randn(b, nkv, T, d), torch.randn(b, nkv, T, d)
        P = self.lora.attention_weights(q, k, v, None, d ** -0.5)
        self.assertTrue(torch.allclose(P.double().sum(-1), torch.ones(b, nh, T, dtype=F64), atol=1e-6))
        out = self.lora._AttnFn.apply(q, k, v, None, None, d ** -0.5, self.ops)
        ref = P.double() @ v.double().repeat_interleave(2, 1)
        self.assertLess(rel(out, ref.transpose(1, 2)), 1e-2)


class TinyModel(unittest.TestCase):
    def test_zero_b_is_the_base_model_and_gradients_change_nothing_forward(self):
        from y_lang.exact import lora
        m = tiny_model()
        ids, _ = batch()
        with torch.no_grad():
            base = m(input_ids=ids).logits
        ad = lora.attach(m, r=8, alpha=8)
        with torch.no_grad():
            self.assertTrue(torch.equal(m(input_ids=ids).logits, base))
        randomize_b(ad)
        with torch.no_grad():
            served = m(input_ids=ids).logits
        self.assertFalse(torch.equal(served, base))                  # the adapters act
        self.assertTrue(torch.equal(m(input_ids=ids).logits.detach(), served))
        with self.assertRaises(RuntimeError):
            lora.attach(m)

    def test_trainer_logits_are_a_cached_decodes(self):
        """What an RL trainer computes is what its sampler computed: the
        training forward's logits at every position are the logits of a
        left-padded batch decoded token by token over the KV cache."""
        from y_lang.exact import kv, lora
        m = tiny_model()
        ad = lora.attach(m, r=8, alpha=8)
        sft = lora.SFT(m, ad, lr=3e-3)
        for s in range(3):
            sft.step(*batch(seed=s))
        g = torch.Generator().manual_seed(7)
        seqs = [torch.randint(0, 512, (17,), generator=g), torch.randint(0, 512, (25,), generator=g)]
        train = [sft.logits_at(sft.forward(s[None]), torch.arange(s.numel())).detach() for s in seqs]
        prompts = [5, 8]
        Lp = max(prompts)
        with torch.no_grad():
            inp = torch.stack([torch.cat([torch.zeros(Lp - p, dtype=torch.long), s[:p]])
                               for s, p in zip(seqs, prompts)])
            am = torch.stack([torch.cat([torch.zeros(Lp - p, dtype=torch.long),
                                         torch.ones(p, dtype=torch.long)]) for p in prompts])
            cache = kv.new_cache(m.config)
            pos = (am.cumsum(-1) - 1).masked_fill(am == 0, 0)
            out = m(input_ids=inp, attention_mask=am, position_ids=pos, past_key_values=cache,
                    use_cache=True)
            got = [[out.logits[i, -1]] for i in range(2)]
            for t in range(max(s.numel() - p for s, p in zip(seqs, prompts))):
                nxt = torch.stack([s[p + t] if p + t < s.numel() else torch.tensor(0)
                                   for s, p in zip(seqs, prompts)])
                am = torch.cat([am, torch.ones(2, 1, dtype=torch.long)], -1)
                pos = pos[:, -1:] + 1
                out = m(input_ids=nxt[:, None], attention_mask=am, position_ids=pos,
                        past_key_values=cache, use_cache=True)
                for i in range(2):
                    got[i].append(out.logits[i, -1])
        for i, (s, p) in enumerate(zip(seqs, prompts)):
            dec = torch.stack(got[i][: s.numel() - p + 1])
            self.assertTrue(torch.equal(dec, train[i][p - 1:s.numel()]), i)

    def test_training_learns_and_repeats(self):
        from y_lang.exact import lora
        m = tiny_model()
        ad = lora.attach(m, r=8, alpha=8)
        sft = lora.SFT(m, ad, lr=3e-3)
        ids, labels = batch()
        first = lora.nll(sft.step(ids, labels))
        for _ in range(8):
            last = lora.nll(sft.step(ids, labels))
        sft.verify()
        self.assertLess(last, first - 0.05)
        d1, t1, _ = train_tiny("cpu")
        d2, t2, _ = train_tiny("cpu")
        self.assertEqual((d1, t1), (d2, t2))

    def test_whole_model_gradient_is_the_float_ones(self):
        """The adapters' gradients against float64 autograd of the stock
        (unquantized) model with the same adapters: 0.028 apart at worst
        (the quantized forward moves them). A coarse check: of six backward
        mutants it catches three (row scales not folded, the norm's mean
        term, SiLU's derivative); `Pieces` catches all six."""
        import transformers.models.qwen2.modeling_qwen2 as qwen2
        from y_lang.exact import attention as A, lora
        from transformers import Qwen2Config, Qwen2ForCausalLM
        m = tiny_model()
        ad = lora.attach(m, r=8, alpha=8)
        randomize_b(ad)
        ids, labels = batch()
        sft = lora.SFT(m, ad, lr=0.0)
        pos, tgt = lora.target_positions(labels)
        logits = sft.logits_at(sft.forward(ids), pos)
        with torch.no_grad():
            P = lora.softmax_rows(sft.ops, logits.detach())
            d = (P - torch.zeros_like(P).scatter_(1, tgt[:, None], 1.0)) / tgt.numel()
        logits.backward(d)
        # The stock model in float64, from the same seed, with the same adapters.
        torch.manual_seed(0)
        cfg = m.config
        saved = qwen2.eager_attention_forward
        qwen2.eager_attention_forward = A._orig
        try:
            ref = Qwen2ForCausalLM(Qwen2Config(**{k: getattr(cfg, k) for k in (
                "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "max_position_embeddings",
                "tie_word_embeddings")}, attn_implementation="eager")).double().eval()
            leaves = {}
            # q/k/v are merged in the exact model: split its adapter's B by rows.
            hooks = []
            for name, mod in ad.modules.items():
                Ar = mod.A.detach().clone().requires_grad_()
                Br = mod.B.detach().clone().requires_grad_()
                leaves[name] = (Ar, Br)
                parent = ref.get_submodule(name.rsplit(".", 1)[0])
                leaf = name.rsplit(".", 1)[1]
                if leaf == "qkv_proj":
                    sizes = [parent.q_proj.out_features, parent.k_proj.out_features,
                             parent.v_proj.out_features]
                    parts = list(zip(("q_proj", "k_proj", "v_proj"), Br.split(sizes, 0)))
                else:
                    parts = [(leaf, Br)]
                for sub, b in parts:
                    def hook(module, inp, out, A_=Ar, B_=b, c=mod.scale):
                        return out + c * (inp[0] @ A_.t()) @ B_.t()
                    hooks.append(getattr(parent, sub).register_forward_hook(hook))
            rl = ref(input_ids=ids).logits.reshape(-1, cfg.vocab_size)[pos]
            torch.nn.functional.cross_entropy(rl, tgt).backward()
        finally:
            qwen2.eager_attention_forward = saved
        worst = 0.0
        for name, mod in ad.modules.items():
            Ar, Br = leaves[name]
            for got, want in ((mod.B.grad, Br.grad), (mod.A.grad, Ar.grad)):
                if want.norm() > 0:
                    worst = max(worst, rel(got, want))
        self.assertLess(worst, 0.08)          # 0.028 measured

    def test_a_non_finite_value_is_refused_at_verify(self):
        from y_lang.exact import lora
        m = tiny_model()
        ad = lora.attach(m, r=8, alpha=8)
        sft = lora.SFT(m, ad, lr=3e-3)
        with torch.no_grad():
            m.model.embed_tokens.weight[5] = float("nan")
        ids, labels = batch()
        ids[0, 3] = 5
        sft.step(ids, labels)
        with self.assertRaises(OverflowError):
            sft.verify()

    def test_the_kv_cache_flag_is_refused(self):
        from y_lang.exact import kv, lora
        m = tiny_model()
        kv.USE_CACHE, old = True, kv.USE_CACHE
        try:
            with self.assertRaises(RuntimeError):
                lora.attach(m)
        finally:
            kv.USE_CACHE = old


class Flags(unittest.TestCase):
    def test_compile_flags_hold_in_every_thread(self):
        """model.py's Inductor flags and CompiledOps' `specialize_int`, read
        in a new thread and in autograd's backward (its own thread on a GPU):
        torch's config values are per thread, and an assignment alone reached
        neither (2026-11-22)."""
        import threading
        import torch._inductor.config as ic
        from y_lang.exact import lora
        lora.CompiledOps(exact=True, mm=lora.mm_f64, act=torch.float32)

        def read():
            return (ic.eager_numerics.division_rounding, ic.emulate_precision_casts,
                    torch._dynamo.config.specialize_int)
        out = {}
        t = threading.Thread(target=lambda: out.update(thread=read()))
        t.start()
        t.join()
        self.assertEqual(out["thread"], (True, True, True))
        if torch.cuda.is_available():
            class F(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x * 2

                @staticmethod
                def backward(ctx, g):
                    out["backward"] = (read(), threading.current_thread() is threading.main_thread())
                    return g * 2
            x = torch.ones(4, device="cuda", requires_grad=True)
            F.apply(x).sum().backward()
            self.assertEqual(out["backward"], ((True, True, True), False))


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA GPU")
class OnTheGpu(unittest.TestCase):
    def test_any_shape_on_ys_gemm(self):
        from y_lang.exact import lora
        g = torch.Generator().manual_seed(0)
        for M_, N_, K_ in ((5, 16, 896), (33, 13, 70), (4, 24, 131040 + 96)):
            a = torch.randint(-127, 128, (M_, K_), generator=g)
            b = torch.randint(-127, 128, (N_, K_), generator=g)
            got = lora.mm_y_any(a.cuda(), b.cuda()).cpu()
            self.assertTrue(torch.equal(got, a @ b.t()), (M_, N_, K_))

    def test_the_cpus_adapters_on_the_gpu(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        ref = train_tiny("cpu")[:2]
        self.assertEqual(train_tiny("cuda")[:2], ref)

    def test_a_llamas_adapters_on_the_cpu_and_compiled_on_the_gpu(self):
        """A Llama (Llama 3's rotary scaling): its attention has the exact
        backward (`attach` registers Llama's attention class), and the
        adapters and every step's p(target) are the CPU's."""
        from y_lang.exact import lora
        torch.backends.cuda.matmul.allow_tf32 = False
        ref = train_tiny("cpu", arch="llama")
        got = train_tiny("cuda", compile=True, arch="llama", lengths=(30, 27))
        import transformers.models.llama.modeling_llama as llama
        attn = [m for m in got[2].model.modules() if isinstance(m, llama.LlamaAttention)]
        self.assertEqual(len(attn), 2)
        self.assertTrue(all(isinstance(lora._OPS.get(id(m)), lora.CompiledOps) for m in attn))
        self.assertEqual(got[:2], train_tiny("cpu", arch="llama", lengths=(30, 27))[:2])
        self.assertNotEqual(ref[0], train_tiny("cpu")[0])          # not the Qwen2 run

    def test_compiled_steps_are_the_eager_ones(self):
        """Every compiled call of five steps (three batch lengths, so the
        shapes change under the compiled code) against the same call eager,
        bit for bit, every output and every check: the adapters' forward and
        backward, the norms', SiLU's and attention's (`lora._run`), and the
        products compiled alone (`CompiledOps.product`, the frozen head's);
        and the adapters and every step's p(target) the CPU's."""
        from unittest import mock
        from y_lang.exact import lora
        torch.backends.cuda.matmul.allow_tf32 = False
        lengths = (30, 27, 33)
        same = []
        compiled_product, compiled = lora.CompiledOps.product, lora._compiled

        def product(ops, a, b, bits_a, bits_b, batched, int8, qa, qb):
            want = lora.Ops.product(ops, a, b, bits_a, bits_b, batched, int8, qa, qb)
            got = compiled_product(ops, a, b, bits_a, bits_b, batched, int8, qa, qb)
            same.append(("product", bits_equal(want, got)))
            return got

        def both(fn):
            c = compiled(fn)

            def run(*args):
                want, got = fn(*args), c(*args)
                same.append((fn.__name__, bits_equal(want, got)))
                return got
            return run
        with mock.patch.object(lora.CompiledOps, "product", product), mock.patch.object(lora, "_compiled", both):
            got = train_tiny("cuda", steps=5, compile=True, lengths=lengths)
        self.assertIsInstance(got[2].ops, lora.CompiledOps)
        kinds = {k for k, _ in same}
        for k in ("_lora_forward", "_lora_backward", "_norm_backward", "_module_forward", "_silu_forward",
                  "_silu_backward", "_attn_forward", "_attn_backward", "product"):
            self.assertIn(k, kinds)
        self.assertGreater(len(same), 100)
        bad = [k for k, ok in same if not ok]
        self.assertEqual(bad, [], f"{len(bad)} of {len(same)} compiled calls differ")
        self.assertEqual(got[:2], train_tiny("cpu", steps=5, lengths=lengths)[:2])

    def test_compiled_products_keep_the_clamp(self):
        """A 12-bit gradient at the clamp, 2047 / 2048 times a weight of 1:
        exactly 2047 / 2048, eager and compiled. With symbolic widths (the
        hazard `CompiledOps` guards against) Inductor clamped it at 2046; the
        guard refuses to run without `specialize_int`, and the hazard itself
        is shown, so this test would see it."""
        from y_lang.exact import lora
        x = torch.full((4, 64), 0.25, device="cuda")
        x[:, 0] = 2047 / 2048                       # q8 at 12 bits: s = -11, q = 2047
        w = torch.zeros(8, 64, device="cuda")
        w[:, 0] = 1.0
        want = torch.full((4, 8), 2047 / 2048, device="cuda")
        eager = lora.make_ops("cuda")
        ops = lora.make_ops("cuda", compile=True)
        self.assertTrue(torch.equal(eager.matmul(x, w, grad="a"), want))
        self.assertTrue(torch.equal(ops.matmul(x, w, grad="a"), want))
        torch._dynamo.config.specialize_int = False
        try:
            with self.assertRaises(RuntimeError):
                ops.matmul(x, w, grad="a")
            f = torch.compile(lambda a, b, bits_a, bits_b: lora._checked_product(
                a, b, bits_a, bits_b, ops.mm, False, False, torch.float32, None, None)[0], dynamic=True)
            moved = f(x, w, 12, 8)
        finally:
            torch._dynamo.config.specialize_int = True
        if torch.equal(moved, want):
            self.skipTest("this torch keeps q8's clamp under symbolic widths (the guard is then "
                          "not needed, and harmless)")
        self.assertTrue(torch.equal(moved, torch.full_like(want, 2046 / 2048)))


if __name__ == "__main__":
    unittest.main()
