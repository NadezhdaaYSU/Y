"""Exact reinforcement learning (T3b): a sampler and a trainer that agree on
every probability, bit for bit, on any device.

    from y_lang.exact import load, lora, rl
    model, tok = load("Qwen/Qwen2.5-0.5B-Instruct", device)
    ad = lora.attach(model, r=16, alpha=16)
    grpo = rl.GRPO(model, ad, lr=1e-4)
    out = rl.sample(model, ad.ops, ids, mask, 6, rl.uniforms(seed, step, rows, 6))
    ...rewards...
    grpo.step(ids, mask, out, rewards, group=8)

**The sampler** decodes over the KV cache, as `greedy` does, and draws each
token from the exact softmax of its logits (`Ops.softmax_ints`): integer
numerators p_i (Q0.28, the integer exp2) and their sum l. It draws by
integer inverse CDF: u = r mod l for a 62-bit integer r hashed on the host
from (seed, step, row, position) (`uniforms`), and the token is the first i
with p_0 + ... + p_i > u. Every operation is an integer one, so the draw is
the same on every device; P(i) = p_i / l to within the modulo's bias, below
l / 2^62 < 2^-16 (l < 2^28 times the vocabulary).

**The trainer** runs the batch the sampler produced through the training
forward (`lora.SFT.forward`, the served model's own code) and recomputes
p_i and l at every sampled token. Since the training forward's logits are a
cached decode's (`test_trainer_logits_are_a_cached_decodes`), they are the
same integers, and each step checks it: every sampled token's (p, l) from
the trainer against the sampler's, refused (deferred, `verify`) if any
differs. So the importance ratio of an on-policy step is exactly 1, where
float stacks need a correction for the sampler and trainer disagreeing.

**The update** is Dr. GRPO's: advantage = reward - the mean reward of the
prompt's group (no division by the standard deviation, no KL term), and the
loss -sum A log p(token) over the completion tokens divided by their count;
the gradient through the softmax is A (P - onehot) / n, P = p / l in
float32, the SFT step's with a weight per token.
"""

import hashlib

import torch

from . import lora as _lora
from .kv import new_cache

R_BITS = 62


def uniforms(seed: int, step: int, rows: int, n: int) -> torch.Tensor:
    """[rows][n] int64 random integers below 2^62, from blake2b of
    (seed, step, row, position): the same on every machine, and each one
    independent of the batch it is drawn in."""
    out = []
    for r in range(rows):
        row = []
        for t in range(n):
            h = hashlib.blake2b(f"y-exact-rl/{seed}/{step}/{r}/{t}".encode(), digest_size=8).digest()
            row.append(int.from_bytes(h, "little") >> (64 - R_BITS))
        out.append(row)
    return torch.tensor(out, dtype=torch.int64)


def draw(p: torch.Tensor, l: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """The integer inverse CDF: for each row, the first i with
    p_0 + ... + p_i > r mod l. p [rows][V] int64 >= 0, l [rows][1] its row
    sums (> 0), r [rows] int64 below 2^62."""
    c = torch.cumsum(p, dim=-1)
    u = torch.remainder(r[:, None], l)
    return torch.searchsorted(c, u, right=True).reshape(-1)


@torch.no_grad()
def sample(model, ops, input_ids, attention_mask, new_tokens, rand):
    """Sampled continuations: (tokens [B][new], p of each token [B][new]
    int64, l of each step [B][new] int64), all on the model's device.
    `input_ids`, `attention_mask` left padded as `greedy` takes them; `rand`
    [B][new] from `uniforms`. End of turn does not stop a row: every row
    gets `new_tokens` (the trainer reads each up to its first end)."""
    device = input_ids.device
    rand = rand.to(device)
    cache = new_cache(model.config.get_text_config(decoder=True))
    position = (attention_mask.long().cumsum(-1) - 1).masked_fill(attention_mask == 0, 0)
    out = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position,
                past_key_values=cache, use_cache=True, logits_to_keep=1, return_dict=True)
    toks, ps, ls = [], [], []
    position = position[:, -1:]
    ones = attention_mask.new_ones((attention_mask.shape[0], 1))
    for t in range(new_tokens):
        logits = out.logits[:, -1].to(torch.float32)
        p, l = ops.softmax_ints(logits, torch.zeros_like(logits, dtype=torch.bool))
        token = draw(p, l, rand[:, t])
        toks.append(token)
        ps.append(p.gather(1, token[:, None]).reshape(-1))
        ls.append(l.reshape(-1))
        if t + 1 == new_tokens:
            break
        del out
        position = position + 1
        attention_mask = torch.cat([attention_mask, ones], dim=-1)
        out = model(input_ids=token[:, None], attention_mask=attention_mask, position_ids=position,
                    past_key_values=cache, use_cache=True, logits_to_keep=1, return_dict=True)
    return torch.stack(toks, 1), torch.stack(ps, 1), torch.stack(ls, 1)


def completion_lengths(tokens: torch.Tensor, end: int) -> list:
    """Each row's completion length: up to and including its first `end`,
    or every token where it has none."""
    out = []
    for row in tokens.tolist():
        out.append(row.index(end) + 1 if end in row else len(row))
    return out


def advantages(rewards, group: int) -> list:
    """Dr. GRPO's: reward - the mean of its group of `group` consecutive
    rows, in float64 on the host (the mean of rewards in {0, 1} over a
    power-of-two group is exact)."""
    if len(rewards) % group:
        raise ValueError(f"rl: {len(rewards)} rewards are not groups of {group}")
    out = []
    for i in range(0, len(rewards), group):
        g = [float(x) for x in rewards[i:i + group]]
        m = sum(g) / group
        out.extend(x - m for x in g)
    return out


MISMATCH = "rl: the trainer's probabilities of the sampled tokens are not the sampler's"


def training_batch(prompt_ids, prompt_mask, tokens, lens, adv, pad=0):
    """(ids, labels, weights, rows, cols): each row's real prompt then its
    completion (the first `lens[i]` sampled tokens), right padded so its
    positions count from 0 as the sampler's did, labelled on the
    completion; one weight A_i / n per labelled token (n: all of them), in
    `lora.target_positions` order, which (rows, cols) index into the
    sampler's [B][new] outputs."""
    pm, pi, tk = prompt_mask.cpu().tolist(), prompt_ids.cpu().tolist(), tokens.cpu().tolist()
    n = sum(lens)
    seqs, labels, weights, rows, cols = [], [], [], [], []
    for i in range(len(pi)):
        prompt = [t for t, m in zip(pi[i], pm[i]) if m]
        comp = tk[i][:lens[i]]
        seqs.append(prompt + comp)
        labels.append([-100] * len(prompt) + comp)
        weights.extend([adv[i] / n] * lens[i])
        rows.extend([i] * lens[i])
        cols.extend(range(lens[i]))
    T = max(len(s) for s in seqs)
    device = prompt_ids.device
    return (torch.tensor([s + [pad] * (T - len(s)) for s in seqs], device=device),
            torch.tensor([l + [-100] * (T - len(l)) for l in labels], device=device),
            torch.tensor(weights, dtype=torch.float32, device=device),
            torch.tensor(rows, device=device), torch.tensor(cols, device=device))


class GRPO(_lora.SFT):
    """Dr. GRPO on the adapters, on-policy: one update per sampled batch."""

    def step(self, prompt_ids, prompt_mask, sampled, rewards, end, group, pad=0):
        """One update from `sample`'s output for `prompt_ids` (left padded,
        as the sampler had them) and one reward per row, the rows in groups
        of `group` consecutive rows sharing a prompt. Returns (p(target) of
        the completion tokens, as `SFT.step`; the number of them whose
        (p, l) differ from the sampler's, read on the host: 0 unless
        something is wrong, and then `verify` raises too)."""
        tokens, p_s, l_s = sampled
        lens = completion_lengths(tokens.cpu(), end)
        ids, labels, w, rows, cols = training_batch(prompt_ids, prompt_mask, tokens, lens,
                                                    advantages(rewards, group), pad)
        ps, ls = p_s[rows, cols], l_s[rows, cols]
        differs = {}

        def check(p, l):
            differs["x"] = (p != ps) | (l.reshape(-1) != ls)
            return differs["x"], MISMATCH
        tp = self.run(ids, labels, weights=w, check=check)[0]
        return tp, int(differs["x"].sum())
