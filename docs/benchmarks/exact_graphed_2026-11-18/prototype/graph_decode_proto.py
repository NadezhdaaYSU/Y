"""Prototype: the exact path's decode step as one CUDA graph.

Run with every switch (Y attention reads packed rows; the cache is packed):
  Y_EXACT_V_INT8=1 Y_EXACT_NORM=triton Y_EXACT_ATTN=y Y_EXACT_KV_PACK=inplace

1. Reference: y_lang.exact.greedy, eager (the repo's loop).
2. Graphed: prefill as usual, copy the packed K/V rows into fixed buffers
   of CAP rows, then one compiled decode step whose shapes never change
   (the cache length lives in a device tensor and becomes a mask), recorded
   once with torch.cuda.graph and replayed. The step advances its own state.
The two must give the same token ids on all rows.
"""
import os
import sys
import time

sys.path.insert(0, "/tmp/y_exact_nospace/tools")
import _exact_pkg  # noqa: E402
_exact_pkg.add_path()

import torch  # noqa: E402
import transformers.models.qwen2.modeling_qwen2 as qwen2  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from y_lang.exact import attention as A  # noqa: E402
from y_lang.exact import kv as KV  # noqa: E402
from y_lang.exact import model as EM  # noqa: E402
from y_lang.exact.greedy import greedy  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = ("Explain, step by step and in detail, how a suspension bridge carries "
          "the load of the traffic crossing it down into the ground. ")
B, NEW = int(os.environ.get("B", "32")), int(os.environ.get("NEW", "64"))
assert KV.PACK and A.ATTN == "y", "run with every switch"
torch.backends.cuda.matmul.allow_tf32 = False

qwen2.eager_attention_forward = A.exact_attention
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager").cuda().eval()
EM.convert(m)
ids = tok(PROMPT, return_tensors="pt").to("cuda")
inp, att = ids.input_ids.repeat(B, 1), ids.attention_mask.repeat(B, 1)
T0 = inp.shape[1]
CAP = ((T0 + NEW + 31) // 32) * 32

with torch.no_grad():
    ref = greedy(m, inp, att, NEW)
print("reference", tuple(ref.shape), "distinct", len(set(ref[0].tolist())))


class StaticPackedCache:
    """Per layer, fixed [B, NKV, CAP, row] buffers of packed rows. `update`
    writes the new row at `t` (a device int64 [1]) and returns the whole
    buffers; rows past t are masked by the step."""

    def __init__(self, ks, vs, t):
        self.k, self.v, self.t = ks, vs, t

    def update(self, k, v, layer_idx, *a, **kw):
        self.k[layer_idx].index_copy_(2, self.t, k)
        self.v[layer_idx].index_copy_(2, self.t, v)
        return self.k[layer_idx], self.v[layer_idx]


def prefill():
    cache = KV.new_cache(m.config.get_text_config(decoder=True))
    pos = att.long().cumsum(-1) - 1
    pos = pos.masked_fill(att == 0, 0)
    out = m(input_ids=inp, attention_mask=att, position_ids=pos, past_key_values=cache,
            use_cache=True, logits_to_keep=1, return_dict=True)
    first = out.logits[:, -1].to(torch.float32).argmax(-1)
    ks, vs = [], []
    for layer in cache.layers:
        k, v = layer._rows() if hasattr(layer, "_rows") else (layer.keys, layer.values)
        kb = k.new_zeros((*k.shape[:2], CAP, k.shape[-1]))
        vb = v.new_zeros((*v.shape[:2], CAP, v.shape[-1]))
        kb[:, :, :T0].copy_(k)
        vb[:, :, :T0].copy_(v)
        ks.append(kb)
        vs.append(vb)
    return first, pos[:, -1:], ks, vs


with torch.no_grad():
    first, last_pos, ks, vs = prefill()
t_dev = torch.tensor([T0], dtype=torch.int64, device="cuda")
cache = StaticPackedCache(ks, vs, t_dev)
s_token = first.clone()
s_pos = (last_pos + 1).clone()
s_step = torch.zeros((), dtype=torch.int64, device="cuda")
out_buf = torch.zeros((B, NEW), dtype=torch.int64, device="cuda")
out_buf[:, 0] = first
ar = torch.arange(CAP, device="cuda")
layers = m.model.layers


def step(token, pos, t):
    h = m.model.embed_tokens(token[:, None])
    pe = m.model.rotary_emb(h, pos)
    mask = torch.where(ar <= t, 0.0, float("-inf")).to(torch.float32)
    mask = mask.view(1, 1, 1, CAP).expand(B, 1, 1, CAP)
    for layer in layers:
        h = layer(h, attention_mask=mask, position_ids=pos, past_key_values=cache,
                  use_cache=True, position_embeddings=pe)
    h = m.model.norm(h)
    logits = m.lm_head(h)
    return logits[:, -1].to(torch.float32).argmax(-1)


EM.fail_closed_recompiles()
cstep = torch.compile(step, dynamic=False)


def advance():
    nxt = cstep(s_token, s_pos, t_dev)
    s_token.copy_(nxt)
    s_pos.add_(1)
    t_dev.add_(1)
    s_step.add_(1)
    out_buf.index_copy_(1, s_step.view(1), nxt.view(B, 1))


# Warm up on a side stream (compiles), then put the state back.
saved = (s_token.clone(), s_pos.clone(), t_dev.clone(), s_step.clone())
side = torch.cuda.Stream()
side.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(side), torch.no_grad():
    for _ in range(3):
        advance()
torch.cuda.current_stream().wait_stream(side)
for d, s in zip((s_token, s_pos, t_dev, s_step), saved):
    d.copy_(s)
out_buf.zero_()
out_buf[:, 0] = first

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g), torch.no_grad():
    advance()
# Capture records without running; put the state back once more.
for d, s in zip((s_token, s_pos, t_dev, s_step), saved):
    d.copy_(s)
out_buf.zero_()
out_buf[:, 0] = first

for _ in range(NEW - 1):
    g.replay()
torch.cuda.synchronize()
same = torch.equal(out_buf, ref)
print("graphed == reference on all rows:", same,
      "" if same else f"first diff at {(out_buf != ref).nonzero()[:4].tolist()}")


# Timing: the whole 64-token decode (prefill included), graphed vs eager greedy.
def run_graphed():
    with torch.no_grad():
        f, lp, k2, v2 = prefill()
        for i in range(len(ks)):
            ks[i].copy_(k2[i])
            vs[i].copy_(v2[i])
        s_token.copy_(f)
        s_pos.copy_(lp + 1)
        t_dev.fill_(T0)
        s_step.zero_()
        out_buf[:, 0] = f
        for _ in range(NEW - 1):
            g.replay()
    torch.cuda.synchronize()


def timed(fn, n=5):
    best = 1e9
    for _ in range(n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t)
    return best


run_graphed()
print("graphed again == reference:", torch.equal(out_buf, ref))
tg = timed(run_graphed)
print(f"graphed: {tg:.3f} s for {NEW} tokens, {NEW / tg:.1f} tok/s/seq")


def only_prefill():
    with torch.no_grad():
        prefill()
    torch.cuda.synchronize()


def only_replays():
    best = 1e9
    for _ in range(5):
        s_token.copy_(saved[0]); s_pos.copy_(saved[1]); t_dev.fill_(T0); s_step.zero_()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(NEW - 1):
            g.replay()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t)
    return best


tp = timed(only_prefill)
tr = only_replays()
print(f"prefill (eager, uncompiled here): {1000 * tp:.1f} ms;  {NEW - 1} replays: {1000 * tr:.1f} ms "
      f"= {1000 * tr / (NEW - 1):.3f} ms per decode step")
