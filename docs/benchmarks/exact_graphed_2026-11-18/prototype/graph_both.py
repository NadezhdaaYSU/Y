"""Prototype 2: graphed greedy decode for stock AND exact, one process.

Both arms are built by tools/exact_throughput.py's build() (compiled
forwards, used for the prefill). The decode step calls the model's own
layers with a fixed-size cache and a mask built on the device from a
device-held length, is compiled once (dynamic=False) and recorded once as
a CUDA graph per arm. Rows may be left padded: the mask also blocks each
row's padding, and positions advance per row.

Checks: each arm's graphed tokens against its own eager-cache greedy
(y_lang.exact.greedy, compiled forward). Exact must match bit for bit.
Stock is reported (masking over spare rows may change float bits).
Then timing, the arms in turns, best of REPS, prefill included.
"""
import os
import sys
import time

sys.path.insert(0, "/tmp/y_exact_nospace/tools")
import exact_throughput as T  # noqa: E402

import torch  # noqa: E402
import transformers.models.qwen2.modeling_qwen2 as qwen2  # noqa: E402
from transformers import AutoTokenizer, DynamicCache  # noqa: E402

from y_lang.exact import attention as A  # noqa: E402
from y_lang.exact import kv as KV  # noqa: E402
from y_lang.exact import model as EM  # noqa: E402
from y_lang.exact.greedy import greedy  # noqa: E402

B, NEW, REPS = int(os.environ.get("B", "32")), int(os.environ.get("NEW", "64")), int(os.environ.get("REPS", "5"))
RAGGED = os.environ.get("RAGGED") == "1"
torch.backends.cuda.matmul.allow_tf32 = False
tok = AutoTokenizer.from_pretrained(T.MODEL)
ids = tok(T.PROMPT, return_tensors="pt").to("cuda")
inp, att = ids.input_ids.repeat(B, 1), ids.attention_mask.repeat(B, 1)
if RAGGED:                     # left pad: row r drops its first r % 5 tokens
    for r in range(B):
        k = r % 5
        if k:
            att[r, :k] = 0
            inp[r, :k] = tok.eos_token_id
T0 = inp.shape[1]
CAP = ((T0 + NEW + 31) // 32) * 32


class StaticCache:
    def __init__(self, ks, vs, t):
        self.k, self.v, self.t = ks, vs, t

    def update(self, k, v, layer_idx, *a, **kw):
        self.k[layer_idx].index_copy_(2, self.t, k)
        self.v[layer_idx].index_copy_(2, self.t, v)
        return self.k[layer_idx], self.v[layer_idx]


class Graphed:
    def __init__(self, name, m, attention):
        self.name, self.m, self.attention = name, m, attention
        self.exact = name == "exact"
        self.dev = "cuda"
        self.ar = torch.arange(CAP, device=self.dev)
        self.base = torch.ones((B, CAP), dtype=torch.bool, device=self.dev)
        self.base[:, :T0] = att.bool()
        self.t = torch.tensor([T0], dtype=torch.int64, device=self.dev)
        self.step_i = torch.zeros((), dtype=torch.int64, device=self.dev)
        self.out = torch.zeros((B, NEW), dtype=torch.int64, device=self.dev)
        first, pos, rows = self._prefill()
        self.ks = [k.new_zeros((*k.shape[:2], CAP, k.shape[-1])) for k, _ in rows]
        self.vs = [v.new_zeros((*v.shape[:2], CAP, v.shape[-1])) for _, v in rows]
        self.cache = StaticCache(self.ks, self.vs, self.t)
        self.token = first.clone()
        self.pos = pos.clone()
        self._load(first, pos, rows)
        self.cstep = torch.compile(self._step, dynamic=False)
        self._capture()

    def _swap(self):
        prev = qwen2.eager_attention_forward
        if self.attention is not None:
            qwen2.eager_attention_forward = self.attention
        return prev

    def _prefill(self):
        cache = KV.new_cache(self.m.config.get_text_config(decoder=True)) if self.exact else DynamicCache(config=self.m.config)
        pos = att.long().cumsum(-1) - 1
        pos = pos.masked_fill(att == 0, 0)
        with torch.no_grad():
            out = self.m(input_ids=inp, attention_mask=att, position_ids=pos, past_key_values=cache,
                         use_cache=True, logits_to_keep=1, return_dict=True)
        first = out.logits[:, -1].to(torch.float32).argmax(-1)
        rows = [layer._rows() if hasattr(layer, "_rows") else (layer.keys, layer.values) for layer in cache.layers]
        return first, pos[:, -1:] + 1, rows

    def _load(self, first, pos, rows):
        for i, (k, v) in enumerate(rows):
            self.ks[i].zero_()
            self.vs[i].zero_()
            self.ks[i][:, :, :T0].copy_(k)
            self.vs[i][:, :, :T0].copy_(v)
        self.token.copy_(first)
        self.pos.copy_(pos)
        self.t.fill_(T0)
        self.step_i.zero_()
        self.out.zero_()
        self.out[:, 0] = first

    def _step(self, token, pos, t):
        m = self.m
        h = m.model.embed_tokens(token[:, None])
        pe = m.model.rotary_emb(h, pos)
        allowed = (self.ar <= t) & self.base                       # [B, CAP]
        if self.exact:
            mask = torch.where(allowed, 0.0, float("-inf")).to(torch.float32).view(B, 1, 1, CAP)
        else:
            mask = allowed.view(B, 1, 1, CAP)
        for layer in m.model.layers:
            h = layer(h, attention_mask=mask, position_ids=pos, past_key_values=self.cache,
                      use_cache=True, position_embeddings=pe)
        h = m.model.norm(h)
        return m.lm_head(h)[:, -1].to(torch.float32).argmax(-1)

    def _advance(self):
        nxt = self.cstep(self.token, self.pos, self.t)
        self.token.copy_(nxt)
        self.pos.add_(1)
        self.t.add_(1)
        self.step_i.add_(1)
        self.out.index_copy_(1, self.step_i.view(1), nxt.view(B, 1))

    def _capture(self):
        import contextlib
        import torch._inductor.config as ic
        prev = self._swap()
        # Inductor reads its config when it compiles (the first warm-up call):
        # stock compiles with PyTorch's defaults, as exact_throughput's stock arm.
        cfg = contextlib.nullcontext() if self.exact else ic.patch(EM.INDUCTOR_DEFAULTS)
        try:
            cfg.__enter__()
            snap = (self.token.clone(), self.pos.clone())
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side), torch.no_grad():
                for _ in range(3):
                    self._advance()
            torch.cuda.current_stream().wait_stream(side)
            self.g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g), torch.no_grad():
                self._advance()
        finally:
            cfg.__exit__(None, None, None)
            qwen2.eager_attention_forward = prev

    def run(self):
        first, pos, rows = self._prefill()
        self._load(first, pos, rows)
        for _ in range(NEW - 1):
            self.g.replay()
        torch.cuda.synchronize()
        return self.out.clone()


arms = {}
for name in ("stock", "exact"):
    m = T.build(name, "cuda", True)
    arms[name] = (m, A.exact_attention if name == "exact" else None)

for name, (m, attn) in arms.items():
    with torch.no_grad():
        ref = greedy(m, inp, att, NEW)        # m.forward installs its own attention
    gr = Graphed(name, m, attn)
    got = gr.run()
    same = torch.equal(got, ref)
    diff = (got != ref).any(dim=1).sum().item()
    print(f"  {name}: graphed == greedy on all {B} rows: {same} ({diff} rows differ); "
          f"ragged={RAGGED}; distinct {len(set(got[0].tolist()))}")
    arms[name] = (m, attn, gr, ref)


def timed(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t


res = {f"{n}_{k}": [] for n in arms for k in ("greedy", "graphed")}
for _ in range(REPS):
    for name, (m, attn, gr, ref) in arms.items():
        with torch.no_grad():
            res[f"{name}_greedy"].append(timed(lambda: greedy(m, inp, att, NEW)))
        res[f"{name}_graphed"].append(timed(gr.run))
best = {k: min(v) for k, v in res.items()}
for k, v in best.items():
    print(f"  {k:<15} {v:.4f} s  {NEW / v:7.1f} tok/s/seq")
print(f"  exact_graphed / stock_graphed = {best['exact_graphed'] / best['stock_graphed']:.3f}x")
print(f"  exact_graphed / stock_greedy  = {best['exact_graphed'] / best['stock_greedy']:.3f}x")
print(f"  stock_graphed / stock_greedy  = {best['stock_graphed'] / best['stock_greedy']:.3f}x")
