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
from . import vocab_draw as _vocab_draw
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
    sums (> 0), r [rows] int64 below 2^62.

    In two levels, so no running sum over the whole vocabulary is written:
    the sums of blocks of `DRAW_BLOCK` entries, their running sum, the first
    block whose running sum passes u (every entry before it sums to at most
    u), then the running sum inside that block from the sum of the blocks
    before it. Integer sums, so the same token as one running sum over the
    row (`draw_flat`, tests)."""
    rows, V = p.shape
    u = torch.remainder(r[:, None], l)                                # [rows][1], below l
    nb = -(-V // DRAW_BLOCK)
    blocks = torch.nn.functional.pad(p, (0, nb * DRAW_BLOCK - V)).view(rows, nb, DRAW_BLOCK)
    cb = torch.cumsum(blocks.sum(-1), dim=-1)                         # [rows][nb]
    k = torch.searchsorted(cb, u, right=True)                         # [rows][1]
    before = torch.where(k > 0, cb.gather(1, (k - 1).clamp(min=0)), torch.zeros_like(k))
    inner = blocks.gather(1, k[:, :, None].expand(rows, 1, DRAW_BLOCK)).view(rows, DRAW_BLOCK)
    j = torch.searchsorted(torch.cumsum(inner, dim=-1) + before, u, right=True)
    return (k * DRAW_BLOCK + j).reshape(-1)


DRAW_BLOCK = 1024


def draw_flat(p: torch.Tensor, l: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """`draw` as one running sum over each row (its definition)."""
    c = torch.cumsum(p, dim=-1)
    u = torch.remainder(r[:, None], l)
    return torch.searchsorted(c, u, right=True).reshape(-1)


def draw_rows(ops, logits, r, rows=64):
    """(token, p of it, l) for each row of float32 `logits` [B][V], drawn
    with the integers `r` [B]: the exact softmax's integers and the inverse
    CDF, `rows` rows at a time. Each row's softmax and draw are its own, and
    the int64 numerators of a whole batch over the vocabulary do not fit
    (256 rows: about 8 GB at the peak), so any `rows` gives the same bits.

    On a GPU the softmax and the draw run as one kernel
    (`vocab_draw`, the same bits; `Y_EXACT_DRAW=torch` for this code),
    which never stores p for the whole vocabulary."""
    if logits.is_cuda and logits.dtype == torch.float32 and _vocab_draw.available(logits.device):
        return _vocab_draw.draw(logits.contiguous(), r, ops.table(logits.device), ops.bad)
    token, pt, lt = [], [], []
    for i in range(0, logits.shape[0], rows):
        z = logits[i:i + rows]
        p, l = ops.softmax_ints(z, torch.zeros_like(z, dtype=torch.bool))
        tk = draw(p, l, r[i:i + rows])
        token.append(tk)
        pt.append(p.gather(1, tk[:, None]).reshape(-1))
        lt.append(l.reshape(-1))
        del p, l
    return torch.cat(token), torch.cat(pt), torch.cat(lt)


@torch.no_grad()
def sample(model, ops, input_ids, attention_mask, new_tokens, rand, rows=64, stop=None, every=16):
    """Sampled continuations: (tokens [B][new], p of each token [B][new]
    int64, l of each step [B][new] int64), all on the model's device.
    `input_ids`, `attention_mask` left padded as `greedy` takes them; `rand`
    [B][new] from `uniforms`. End of turn does not stop a row: every row
    gets `new_tokens` (the trainer reads each up to its first end). The
    softmax and the draw run `rows` rows at a time, the same bits for any
    `rows`. `stop` (a token id): after every `every` tokens, end when every
    row has drawn it (a test of the tokens, read on the host, so every
    device stops at the same length); the outputs are then shorter."""
    device = input_ids.device
    rand = rand.to(device)
    cache = new_cache(model.config.get_text_config(decoder=True))
    position = (attention_mask.long().cumsum(-1) - 1).masked_fill(attention_mask == 0, 0)
    out = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position,
                past_key_values=cache, use_cache=True, logits_to_keep=1, return_dict=True)
    toks, ps, ls = [], [], []
    position = position[:, -1:]
    ones = attention_mask.new_ones((attention_mask.shape[0], 1))
    ended = None
    for t in range(new_tokens):
        token, pt, lt = draw_rows(ops, out.logits[:, -1].to(torch.float32), rand[:, t], rows)
        toks.append(token)
        ps.append(pt)
        ls.append(lt)
        if t + 1 == new_tokens:
            break
        if stop is not None:
            hit = token == stop
            ended = hit if ended is None else ended | hit
            if (t + 1) % every == 0 and bool(ended.all()):
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


def pieces(full, chunk_tokens=None) -> list:
    """`GRPO.step`'s split of a batch: lists of consecutive row indices,
    each piece at most `chunk_tokens` tokens counted as its rows times its
    longest row (`full`: each row's prompt plus completion length); None,
    one piece."""
    out, cur = [], []
    for i in range(len(full)):
        if cur and chunk_tokens is not None and (len(cur) + 1) * max(full[j] for j in cur + [i]) > chunk_tokens:
            out.append(cur)
            cur = []
        cur.append(i)
    out.append(cur)
    return out


MISMATCH = "rl: the trainer's probabilities of the sampled tokens are not the sampler's"


def training_batch(prompt_ids, prompt_mask, tokens, lens, adv, pad=0, which=None, n=None):
    """(ids, labels, weights, rows, cols): each row's real prompt then its
    completion (the first `lens[i]` sampled tokens), right padded so its
    positions count from 0 as the sampler's did, labelled on the
    completion; one weight A_i / n per labelled token (n: all of them, or
    the given `n` when this is one piece of a larger batch), in
    `lora.target_positions` order, which (rows, cols) index into the
    sampler's [B][new] outputs. `which`: the rows to include (all)."""
    pm, pi, tk = prompt_mask.cpu().tolist(), prompt_ids.cpu().tolist(), tokens.cpu().tolist()
    which = range(len(pi)) if which is None else which
    n = sum(lens) if n is None else n
    seqs, labels, weights, rows, cols = [], [], [], [], []
    for i in which:
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

    def step(self, prompt_ids, prompt_mask, sampled, rewards, end, group, pad=0, chunk_tokens=None):
        """One update from `sample`'s output for `prompt_ids` (left padded,
        as the sampler had them) and one reward per row, the rows in groups
        of `group` consecutive rows sharing a prompt. Returns (p(target) of
        the completion tokens, as `SFT.step`; the number of them whose
        (p, l) differ from the sampler's, read on the host: 0 unless
        something is wrong, and then `verify` raises too).

        `chunk_tokens`: run the batch in pieces of consecutive rows, each at
        most that many tokens (rows times the piece's longest row), the
        gradients added in order (`SFT.run_chunks`); None, one piece."""
        tokens, p_s, l_s = sampled
        lens = completion_lengths(tokens.cpu(), end)
        adv = advantages(rewards, group)
        pm = prompt_mask.cpu()
        full = [int(pm[i].sum()) + lens[i] for i in range(len(lens))]
        n = sum(lens)
        chunks, differs = [], []
        for which in pieces(full, chunk_tokens):
            ids, labels, w, rows, cols = training_batch(prompt_ids, prompt_mask, tokens, lens, adv, pad,
                                                        which=which, n=n)
            ps, ls = p_s[rows, cols], l_s[rows, cols]

            def check(p, l, ps=ps, ls=ls):
                differs.append((p != ps) | (l.reshape(-1) != ls))
                return differs[-1], MISMATCH
            chunks.append((ids, labels, w, check))
        tp = self.run_chunks(chunks)[0]
        return tp, int(sum(int(d.sum()) for d in differs))


def graphed_sampler(model, ops, batch, prompt_len, new_tokens, rows=64, stop=None, every=16):
    """A `GraphedSampler` (built here so that importing `rl` does not import
    the graphed decoder)."""
    from .graphed import GraphedGreedy

    class GraphedSampler(GraphedGreedy):
        """`sample`, each decode step replayed from one CUDA graph: the
        model's step and the exact softmax and draw compiled together
        (`torch.compile`, `dynamic=False`) and recorded once for inputs of
        one shape, as `GraphedGreedy` records greedy decoding. The prefill
        and its draw run as in `sample`. The random integers sit in a device
        buffer the step indexes; the checks of the step (the adapters'
        products, the softmax's) are or-ed on the device into one flag per
        message, inside the graph, and handed to `ops.bad` at the end, so a
        replayed step is checked as an eager one is. The same tokens and
        integers as `sample` (tests), and the trainer compares every sampled
        token's integers anyway."""

        def __init__(self):
            super().__init__(model, batch, prompt_len, new_tokens)
            dev = self.dev
            self.ops, self.rows, self.stop, self.every = ops, rows, stop, every
            self.rand = torch.zeros((batch, new_tokens), dtype=torch.int64, device=dev)
            self.p_out = torch.zeros((batch, new_tokens), dtype=torch.int64, device=dev)
            self.l_out = torch.zeros((batch, new_tokens), dtype=torch.int64, device=dev)
            self.ended = torch.zeros(batch, dtype=torch.bool, device=dev)
            self.flags = {}

        def _prefill(self, input_ids, attention_mask):
            """GraphedGreedy's prefill, with the first token drawn as
            `sample` draws it (its p and l kept for `_load_first`)."""
            cache = new_cache(self.cfg)
            position = (attention_mask.long().cumsum(-1) - 1).masked_fill(attention_mask == 0, 0)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position,
                             past_key_values=cache, use_cache=True, logits_to_keep=1, return_dict=True)
            first, self._p0, self._l0 = draw_rows(self.ops, out.logits[:, -1].to(torch.float32),
                                                  self.rand[:, 0], self.rows)
            rows_ = []
            for layer in cache.layers:
                k, v = layer._rows() if hasattr(layer, "_rows") else (layer.keys, layer.values)
                if k.dim() != 4 or k.shape[0] != self.batch or k.shape[2] != self.prompt_len:
                    raise RuntimeError(f"rl: a prefill cache layer holds {tuple(k.shape)}, not "
                                       f"[batch {self.batch}, heads, {self.prompt_len}, row]")
                rows_.append((k, v))
            return first, position[:, -1:] + 1, rows_

        def _step(self, token, pos, t, r):
            """The decode step and its draw, its checks collected here and
            or-ed inside the compiled graph into one flag per message (were
            they returned one by one, the step's ~950 conditions each cost
            an eager reduction and an or in the replayed graph)."""
            outer, self.ops.bad = self.ops.bad, []
            try:
                out = self._decode(token, pos, t, r)
                local = self.ops.bad
            finally:
                self.ops.bad = outer
            msgs = []
            for _, m in local:
                if m not in msgs:
                    msgs.append(m)
            flags = [torch.stack([c.any() for c, m2 in local if m2 == m]).any() for m in msgs]
            return (*out, flags, msgs)

        def _decode(self, token, pos, t, r):
            inner = self.model.model
            h = inner.embed_tokens(token[:, None])
            pe = inner.rotary_emb(h, pos)
            allowed = (self.ar <= t) & self.base
            if self.sdpa:
                mask = allowed.view(self.batch, 1, 1, self.cap)
            else:
                mask = torch.where(allowed, 0.0, float("-inf")).to(h.dtype)
                mask = mask.view(self.batch, 1, 1, self.cap)
            for layer in inner.layers:
                h = layer(h, attention_mask=mask, position_ids=pos, past_key_values=self.cache,
                          use_cache=True, position_embeddings=pe)
            h = inner.norm(h)
            logits = self.model.lm_head(h)[:, -1].to(dtype=torch.float32)
            return draw_rows(self.ops, logits, r, self.rows)

        def _advance(self):
            r = self.rand.index_select(1, (self.step_i + 1).view(1)).view(-1)
            nxt, pt, lt, flags, msgs = self._step_compiled(self.token, self.pos, self.t, r)
            for f, msg in zip(flags, msgs):
                if msg not in self.flags:
                    self.flags[msg] = torch.zeros((), dtype=torch.bool, device=self.dev)
                self.flags[msg] |= f
            self.token.copy_(nxt)
            self.pos.add_(1)
            self.t.add_(1)
            self.step_i.add_(1)
            col = self.step_i.view(1)
            self.out.index_copy_(1, col, nxt.view(self.batch, 1))
            self.p_out.index_copy_(1, col, pt.view(self.batch, 1))
            self.l_out.index_copy_(1, col, lt.view(self.batch, 1))
            if self.stop is not None:
                self.ended |= nxt == self.stop

        def run(self, input_ids, attention_mask, rand):
            """(tokens, p, l) as `sample` returns them."""
            if tuple(input_ids.shape) != (self.batch, self.prompt_len):
                raise ValueError(f"the graphed sampler was recorded for inputs "
                                 f"{(self.batch, self.prompt_len)}, not {tuple(input_ids.shape)}")
            if self.ops.bad is None:
                raise RuntimeError("rl: the graphed sampler needs ops.bad, a list for its checks")
            bad, self.ops.bad = self.ops.bad, []
            try:
                with torch.no_grad():
                    self.rand.copy_(rand)
                    with self._installed():
                        first, next_pos, rows_ = self._prefill(input_ids, attention_mask)
                    p0, l0 = self._p0, self._l0
                    self._load(first, next_pos, rows_, attention_mask)
                    self._load_first(first, p0, l0)
                    if self.new > 1 and self.graph is None:
                        self._record()
                        self._load(first, next_pos, rows_, attention_mask)
                        self._load_first(first, p0, l0)
                    for f in self.flags.values():
                        f.zero_()
                    n = 1
                    while n < self.new:
                        if self.stop is not None and n % self.every == 0 and bool(self.ended.all()):
                            break
                        self.graph.replay()
                        n += 1
            finally:
                for msg, f in self.flags.items():
                    bad.append((f.clone(), msg))
                self.ops.bad = bad
            return self.out[:, :n].clone(), self.p_out[:, :n].clone(), self.l_out[:, :n].clone()

        def _load_first(self, first, p0, l0):
            self.p_out.zero_()
            self.l_out.zero_()
            self.p_out[:, 0] = p0
            self.l_out[:, 0] = l0
            self.ended.copy_(first == self.stop if self.stop is not None else torch.zeros_like(self.ended))

    return GraphedSampler()
