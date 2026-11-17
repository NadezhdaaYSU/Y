"""How much the hidden `logits_to_keep` cost each arm of tools/exact_throughput.py.

    cd tools && python3 ../docs/benchmarks/exact_speed_05b_2026-11-17/logits_to_keep_ab.py [--stock-dtype float16] [--model M]

`generate()` passes `logits_to_keep=1` only when it can see that argument in
the model's `forward`. From 9f0c685 (2026-11-04) the compiled stock arm was
wrapped in a bare `(*args, **kw)` function that hid it, and from 67b8fdc so
was every wrapped arm. Its prefill then computed the logits of every prompt
position, not one per sequence.

Four arms, compiled by the tool's own `build()` (which now keeps the
signature), batch 32, 64 new tokens, best of 5, taken in turns:
  stock, exact            as the tool builds them now;
  stock_all, exact_all    the same, with a bare wrapper put back around
                          `forward`, so the prefill computes every position.
The two forms of each arm must give the same tokens.
"""
import argparse
import os
import sys

TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tools")
sys.path.insert(0, os.path.abspath(TOOLS))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import exact_throughput as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stock-dtype", choices=("bfloat16", "float16"), default="bfloat16")
ap.add_argument("--model", default=T.MODEL)
ap.add_argument("--reps", type=int, default=5)
a = ap.parse_args()
T.STOCK_DTYPE = getattr(torch, a.stock_dtype)
T.MODEL = a.model
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def hide_signature(m):
    inner = m.forward

    def call(*args, _f=inner, **kw):           # the old wrappers' shape
        return _f(*args, **kw)

    m.forward = call
    assert not m._supports_logits_to_keep()
    return m


tok = AutoTokenizer.from_pretrained(T.MODEL)
models = {"stock": T.build("stock", "cuda", True), "exact": T.build("exact", "cuda", True),
          "stock_all": hide_signature(T.build("stock", "cuda", True)),
          "exact_all": hide_signature(T.build("exact", "cuda", True))}
for n, m in models.items():
    print(f"  {n:<10} generate passes logits_to_keep=1: {m._supports_logits_to_keep()}")
if not T.gate(models, tok, "cuda"):
    raise SystemExit(1)

ids = tok(T.PROMPT, return_tensors="pt").to("cuda")
print(f"  prompt: {ids.input_ids.shape[1]} tokens, batch 32")
inp, att = ids.input_ids.repeat(32, 1), ids.attention_mask.repeat(32, 1)
kw = dict(do_sample=False, num_beams=1, use_cache=True, pad_token_id=tok.eos_token_id,
          max_new_tokens=64, min_new_tokens=64)
with torch.no_grad():
    out = {n: m.generate(input_ids=inp, attention_mask=att, **kw) for n, m in models.items()}
for arm in ("stock", "exact"):
    same = torch.equal(out[arm], out[arm + "_all"])
    print(f"  {arm} and {arm}_all give the same token ids on all 32 rows: {same}")

res = T.run_interleaved(models, tok, 32, 64, "cuda", a.reps)
for n, (best, med) in res.items():
    print(f"  {n:<10} {best:7.3f} s  {64 / best:7.1f} tok/s/seq  median {100 * (med - best) / best:4.1f}% above")
for arm in ("stock", "exact"):
    print(f"  {arm}_all / {arm} = {res[arm + '_all'][0] / res[arm][0]:.3f}x")
print(f"  exact / stock (both keep one position, the fix)       = {res['exact'][0] / res['stock'][0]:.3f}x")
print(f"  exact / stock_all (9f0c685 up to 67b8fdc, compiled)   = {res['exact'][0] / res['stock_all'][0]:.3f}x")
print(f"  exact_all / stock_all (67b8fdc up to the fix)         = {res['exact_all'][0] / res['stock_all'][0]:.3f}x")
