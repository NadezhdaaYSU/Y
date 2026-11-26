"""y_lang.exact.rl: an exact sampler and a GRPO trainer that agree on every
probability. The draw is an integer inverse CDF of the exact softmax, the
trainer recomputes the same integers at every sampled token (and a sampler
off by one weight update is caught), training raises the reward, and a CPU
and a GPU sample the same tokens and train the same adapters."""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_exact_lora import randomize_b, tiny_model  # noqa: E402


def prompts(lengths=(5, 8, 6, 5), vocab=512, seed=0, group=2):
    """Left-padded prompts, each repeated `group` times."""
    g = torch.Generator().manual_seed(seed)
    seqs = [torch.randint(1, vocab, (n,), generator=g).tolist() for n in lengths]
    seqs = [s for s in seqs for _ in range(group)]
    T = max(len(s) for s in seqs)
    ids = torch.tensor([[0] * (T - len(s)) + s for s in seqs])
    mask = torch.tensor([[0] * (T - len(s)) + [1] * len(s) for s in seqs])
    return ids, mask


END = 7          # the tiny model's "end of turn"


def setup(device="cpu", compile=False):
    from y_lang.exact import lora, rl
    m = tiny_model().to(device)
    ad = lora.attach(m, r=8, alpha=8, compile=compile)
    randomize_b(ad, std=0.05)
    return m, ad, rl.GRPO(m, ad, lr=3e-3)


def even_reward(tokens):
    """1 for a completion whose first token is even."""
    return [int(row[0] % 2 == 0) for row in tokens.cpu().tolist()]


def run_rl(device, steps=3, compile=False, chunk_tokens=None):
    from y_lang.exact import lora, rl
    m, ad, grpo = setup(device, compile)
    ids, mask = prompts()
    out = []
    for s in range(steps):
        rand = rl.uniforms(0, s, ids.shape[0], 4)
        sampled = rl.sample(m, ad.ops, ids.to(device), mask.to(device), 4, rand)
        rewards = even_reward(sampled[0])
        tp, mism = grpo.step(ids.to(device), mask.to(device), sampled, rewards, END, 2,
                             chunk_tokens=chunk_tokens)
        out.append((sampled[0].cpu().tolist(), lora.tp_digest(tp), mism))
    grpo.verify()
    return ad.digest(), out


class Draw(unittest.TestCase):
    def test_uniforms_are_fixed_and_batch_free(self):
        from y_lang.exact import rl
        u = rl.uniforms(3, 11, 4, 5)
        self.assertEqual(u.dtype, torch.int64)
        self.assertTrue(bool((u >= 0).all()) and bool((u < 2 ** 62).all()))
        self.assertTrue(torch.equal(rl.uniforms(3, 11, 2, 5), u[:2]))
        self.assertFalse(torch.equal(rl.uniforms(3, 12, 4, 5), u))
        # pinned: the same integers on every machine and Python
        self.assertEqual(int(u[0, 0]), int(rl.uniforms(3, 11, 1, 1)[0, 0]))
        self.assertEqual(rl.uniforms(0, 0, 1, 1).tolist(), [[UNIFORM_0_0]])

    def test_draw_is_the_integer_inverse_cdf(self):
        from y_lang.exact import rl
        p = torch.tensor([[0, 3, 0, 5]])
        l = p.sum(-1, keepdim=True)
        got = [int(rl.draw(p, l, torch.tensor([r]))) for r in range(24)]
        self.assertEqual(got, [1, 1, 1, 3, 3, 3, 3, 3] * 3)

    def test_the_two_level_draw_is_the_running_sum(self):
        """`draw` (block sums, then one block) against `draw_flat` (one
        running sum): random rows, u at every block boundary and next to
        it, a row with one nonzero entry (the last), vocabularies shorter
        than a block and not a multiple of it, and a block of zeros."""
        from y_lang.exact import rl
        g = torch.Generator().manual_seed(1)
        B = rl.DRAW_BLOCK
        for V in (5, B - 1, B, B + 1, 3 * B + 17, 151936):
            p = torch.randint(0, 1 << 28, (6, V), generator=g, dtype=torch.int64)
            p[1, : V // 2] = 0
            p[2] = 0
            p[2, -1] = 1 << 20
            if V > 2 * B:
                p[3, B:2 * B] = 0
            l = p.sum(-1, keepdim=True)
            r = torch.randint(0, 1 << 62, (6,), generator=g, dtype=torch.int64)
            self.assertTrue(torch.equal(rl.draw(p, l, r), rl.draw_flat(p, l, r)), V)
            # u exactly at each block's running sum, and one below it, in a
            # row whose blocks each start with zeros (a draw that stops a
            # block early lands on one of them)
            row = p[0:1].clone()
            for s in range(0, V, B):
                row[0, s:s + 3] = 0
            cb = torch.cumsum(torch.nn.functional.pad(row, (0, -V % B)).view(1, -1, B).sum(-1), -1)[0]
            us = torch.cat([cb[:-1], cb[:-1] - 1, torch.tensor([0])]).clamp(min=0)
            rr, pp = us, row.expand(len(us), -1)
            ll = pp.sum(-1, keepdim=True)
            self.assertTrue(torch.equal(rl.draw(pp, ll, rr), rl.draw_flat(pp, ll, rr)), V)

    def test_draw_frequencies_are_p_over_l(self):
        from y_lang.exact import rl
        g = torch.Generator().manual_seed(0)
        p = torch.randint(0, 1 << 20, (1, 50), generator=g)
        p[0, 7] = 0
        n = 40000
        r = torch.randint(0, 1 << 62, (n,), generator=g, dtype=torch.int64)
        tok = rl.draw(p.expand(n, -1), p.sum(-1, keepdim=True).expand(n, -1), r)
        freq = torch.bincount(tok, minlength=50).double() / n
        want = p[0].double() / p.sum()
        self.assertEqual(int(freq[7]), 0)
        self.assertLess(float((freq - want).abs().max()), 0.006)


class Agreement(unittest.TestCase):
    def test_the_trainer_has_the_samplers_integers(self):
        """Prompts of three lengths, left padded for the sampler and right
        padded for the trainer: every sampled token's (p, l) the same."""
        digest, steps = run_rl("cpu")
        self.assertEqual([s[2] for s in steps], [0, 0, 0])

    def test_a_sampler_one_update_behind_is_caught(self):
        """The control: tokens sampled before an update, trained on after
        it. Their integers differ, the step counts them and verify raises."""
        from y_lang.exact import rl
        m, ad, grpo = setup()
        ids, mask = prompts()
        stale = rl.sample(m, ad.ops, ids, mask, 4, rl.uniforms(0, 0, ids.shape[0], 4))
        fresh = rl.sample(m, ad.ops, ids, mask, 4, rl.uniforms(0, 1, ids.shape[0], 4))
        grpo.step(ids, mask, fresh, even_reward(fresh[0]), END, 2)
        grpo.verify()
        _, mism = grpo.step(ids, mask, stale, even_reward(stale[0]), END, 2)
        self.assertGreater(mism, 0)
        with self.assertRaises(OverflowError):
            grpo.verify()

    def test_the_draw_in_row_pieces_is_the_same(self):
        from y_lang.exact import rl
        m, ad, _ = setup()
        ids, mask = prompts()
        rand = rl.uniforms(0, 0, ids.shape[0], 4)
        whole = rl.sample(m, ad.ops, ids, mask, 4, rand)
        for rows in (1, 3):
            got = rl.sample(m, ad.ops, ids, mask, 4, rand, rows=rows)
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(got, whole)), rows)

    def test_an_early_stop_keeps_every_drawn_token(self):
        """Stopping once every row has drawn the end token, checked every
        2 tokens: the tokens drawn are the full run's first ones. One row
        whose 4th token is the stop token stops at 4; two rows, one of
        which never draws it, run to the end."""
        from y_lang.exact import rl
        m, ad, _ = setup()
        ids, mask = prompts()
        rand = rl.uniforms(0, 0, ids.shape[0], 12)
        full = rl.sample(m, ad.ops, ids, mask, 12, rand)
        stop = int(full[0][0, 3])
        self.assertFalse(bool((full[0][0, :3] == stop).any()))
        got = rl.sample(m, ad.ops, ids[:1], mask[:1], 12, rand[:1], stop=stop, every=2)
        self.assertEqual(got[0].shape[1], 4)
        self.assertTrue(all(torch.equal(g, f[:1, :4]) for g, f in zip(got, full)))
        never = next(i for i in range(1, ids.shape[0]) if not bool((full[0][i] == stop).any()))
        pair = [0, never]
        got = rl.sample(m, ad.ops, ids[pair], mask[pair], 12, rand[pair], stop=stop, every=2)
        self.assertTrue(all(torch.equal(g, f[pair]) for g, f in zip(got, full)))

    def test_one_changed_integer_is_caught(self):
        from y_lang.exact import rl
        m, ad, grpo = setup()
        ids, mask = prompts()
        toks, p, l = rl.sample(m, ad.ops, ids, mask, 4, rl.uniforms(0, 0, ids.shape[0], 4))
        p = p.clone()
        p[3, 0] += 1
        _, mism = grpo.step(ids, mask, (toks, p, l), even_reward(toks), END, 2)
        self.assertEqual(mism, 1)
        with self.assertRaises(OverflowError):
            grpo.verify()


class Pieces(unittest.TestCase):
    def update(self, chunk_tokens):
        from y_lang.exact import rl
        m, ad, grpo = setup()
        ids, mask = prompts()
        sampled = rl.sample(m, ad.ops, ids, mask, 4, rl.uniforms(0, 0, ids.shape[0], 4))
        before = {k: t.detach().clone() for k, t in ad.params.items()}
        tp, mism = grpo.step(ids, mask, sampled, even_reward(sampled[0]), END, 2, chunk_tokens=chunk_tokens)
        grpo.verify()
        delta = torch.cat([(ad.params[k].detach() - before[k]).reshape(-1) for k in sorted(before)])
        return delta, tp, mism, ad.digest()

    def test_a_batch_in_pieces_is_one_update(self):
        """Eight rows in pieces of at most 24 tokens (three or more pieces):
        every sampled token checked, the update close to the whole batch's
        (the pieces quantize on their own scales; a first Adam step is
        about lr times the gradient's sign, so the few near-zero elements
        whose sign flips are most of the 0.073 measured), and repeatable. A
        version that kept only the last piece's gradient measured 1.14."""
        whole, tp_w, _, _ = self.update(None)
        split, tp_s, mism, digest = self.update(24)
        self.assertEqual(mism, 0)
        self.assertEqual(tp_s.numel(), tp_w.numel())
        self.assertTrue(torch.equal(tp_s, tp_w))          # the forward pass is per row
        self.assertFalse(torch.equal(split, whole))
        self.assertLess(float((split - whole).norm() / whole.norm()), 0.2)
        self.assertEqual(self.update(24)[3], digest)

    def test_pieces_cover_every_row_once(self):
        from y_lang.exact import lora
        seen = []
        real = lora.SFT.run_chunks

        def spy(sft, chunks):
            seen.extend(int((c[1] != -100).sum()) for c in chunks)
            return real(sft, chunks)
        from unittest import mock
        with mock.patch.object(lora.SFT, "run_chunks", spy):
            _, tp, _, _ = self.update(24)
        self.assertGreater(len(seen), 2)
        self.assertEqual(sum(seen), tp.numel())


class Learning(unittest.TestCase):
    def test_an_update_moves_probability_toward_the_rewarded(self):
        """One step raises the objective it follows, sum A log p over the
        sampled tokens (16 completions of two prompts), and the completions
        that beat their group's mean gain more log probability than those
        below it (single ones below it can still rise: they share the
        adapters with the rest); and a run repeats."""
        from y_lang.exact import rl
        m, ad, grpo = setup()
        ids, mask = prompts(group=8, lengths=(5, 8))
        toks, p, l = rl.sample(m, ad.ops, ids, mask, 1, rl.uniforms(1, 0, ids.shape[0], 1))
        rewards = even_reward(toks)
        adv = rl.advantages(rewards, 8)
        self.assertTrue(any(a > 0 for a in adv) and any(a < 0 for a in adv), rewards)
        grpo.step(ids, mask, (toks, p, l), rewards, END, 8)
        grpo.verify()
        with torch.no_grad():
            pos = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)
            logits = m(input_ids=ids, attention_mask=mask, position_ids=pos).logits[:, -1].float()
            p2, l2 = ad.ops.softmax_ints(logits, torch.zeros_like(logits, dtype=torch.bool))
        before = p[:, 0].double() / l[:, 0].double()
        after = p2.gather(1, toks).reshape(-1).double() / l2.reshape(-1).double()
        gain = (after.log() - before.log()).tolist()
        self.assertGreater(sum(a * g for a, g in zip(adv, gain)), 0)
        up = [g for a, g in zip(adv, gain) if a > 0]
        down = [g for a, g in zip(adv, gain) if a < 0]
        self.assertGreater(sum(up) / len(up), 0)
        self.assertLess(sum(down) / len(down), 0)
        self.assertEqual(run_rl("cpu"), run_rl("cpu"))

    def test_advantages(self):
        from y_lang.exact import rl
        self.assertEqual(rl.advantages([1, 0, 0, 0, 1, 1, 1, 1], 4), [0.75, -0.25, -0.25, -0.25, 0, 0, 0, 0])
        with self.assertRaises(ValueError):
            rl.advantages([1, 0, 1], 2)
        self.assertEqual(rl.completion_lengths(torch.tensor([[3, END, 5], [1, 2, 3]]), END), [2, 3])


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA GPU")
class OnTheGpu(unittest.TestCase):
    def test_the_graphed_sampler_is_the_sampler(self):
        """Each decode step replayed from a CUDA graph (compiled with the
        draw), against `sample`: the same tokens, p and l, for prompts of
        three lengths, with eager and compiled products, on the recording
        call and on a call of replays only; with an early stop, the same
        length too; and its checks reach ops.bad."""
        from y_lang.exact import rl
        torch.backends.cuda.matmul.allow_tf32 = False
        for compile in (False, True):
            m, ad, _ = setup("cuda", compile)
            ids, mask = prompts()
            ids, mask = ids.cuda(), mask.cuda()
            ad.ops.bad = []
            g = rl.graphed_sampler(m, ad.ops, ids.shape[0], ids.shape[1], 12)
            for seed in (0, 1):
                rand = rl.uniforms(seed, 0, ids.shape[0], 12)
                want = rl.sample(m, ad.ops, ids, mask, 12, rand)
                n = len(ad.ops.bad)
                got = g.run(ids, mask, rand.cuda())
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(got, want)), (compile, seed))
                self.assertGreater(len(ad.ops.bad), n)
            rand = rl.uniforms(2, 0, 1, 12)
            want = rl.sample(m, ad.ops, ids[:1], mask[:1], 12, rand)
            stop = int(want[0][0, 3])
            if bool((want[0][0, :3] == stop).any()):
                continue
            want = rl.sample(m, ad.ops, ids[:1], mask[:1], 12, rand, stop=stop, every=2)
            g1 = rl.graphed_sampler(m, ad.ops, 1, ids.shape[1], 12, stop=stop, every=2)
            got = g1.run(ids[:1], mask[:1], rand.cuda())
            self.assertEqual(got[0].shape[1], 4)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(got, want)))

    def test_the_cpus_samples_and_adapters_on_the_gpu(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        ref = run_rl("cpu")
        self.assertEqual(run_rl("cuda"), ref)
        self.assertEqual(run_rl("cuda", compile=True), ref)
        self.assertEqual(run_rl("cuda", chunk_tokens=24), run_rl("cpu", chunk_tokens=24))


UNIFORM_0_0 = 202160430523473325   # blake2b of 'y-exact-rl/0/0/0/0', top 62 bits


class PromptDate(unittest.TestCase):
    """Llama 3's chat template writes today's date into every prompt
    (strftime_now). The RL and SFT tools pin it, so a run's prompts, and so
    its tokens, are the same on any day: unpinned, the runs of 2026-11-24
    and 2026-11-25 differed from step 1."""

    # The part of Llama 3's template that writes the date.
    TEMPLATE = ('{%- if not date_string is defined %}{%- set date_string = strftime_now("%d %b %Y") %}'
                '{%- endif %}Today Date: {{ date_string }}|{% for m in messages %}{{ m.content }}|{% endfor %}')

    def test_the_date_is_pinned(self):
        import datetime
        from unittest import mock

        from tokenizers import Tokenizer, models
        from transformers import PreTrainedTokenizerFast
        from transformers.utils import chat_template_utils

        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
        import exact_lora_grpo as G
        import exact_lora_sft as S

        tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")))
        tok.chat_template = self.TEMPLATE

        class Clock(datetime.datetime):
            day = datetime.datetime(2030, 1, 2)

            @classmethod
            def now(cls, tz=None):
                return cls.day

        texts = {}
        with mock.patch.object(chat_template_utils, "datetime", Clock), \
                mock.patch.object(S, "ids", lambda tok, text: text):
            for day in (datetime.datetime(2030, 1, 2), datetime.datetime(2031, 5, 6)):
                Clock.day = day
                bare = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False,
                                               add_generation_prompt=True)
                texts[day] = (bare, S.prompt_ids(tok, 12, 34), G.Task(tok, mode="steps").prompt(12, 34))
        (bare1, sft1, grpo1), (bare2, sft2, grpo2) = texts.values()
        # The control: unpinned, the template writes the clock's day.
        self.assertTrue(bare1.startswith("Today Date: 02 Jan 2030|"), bare1)
        self.assertTrue(bare2.startswith("Today Date: 06 May 2031|"), bare2)
        for a, b in ((sft1, sft2), (grpo1, grpo2)):
            self.assertEqual(a, b)
            self.assertTrue(a.startswith(f"Today Date: {S.PROMPT_DATE}|"), a)
        self.assertEqual(S.PROMPT_DATE, "24 Nov 2026")    # the day the Llama records were made


if __name__ == "__main__":
    unittest.main()
