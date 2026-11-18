"""Inductor's C++ wrapper against its Python one, for stock and exact.

    cd tools && python3 ../docs/benchmarks/exact_t4_profile_2026-11-17/cpp_wrapper_ab.py [--stock-dtype float16] [--model M]

A compiled decode step launches every kernel from Inductor's generated
wrapper. By default that wrapper is Python, and a profile put it at about
4 us of host time a kernel, which is most of the step's host time. With
`torch._inductor.config.cpp_wrapper = True` the wrapper is C++. Only host
code changes; the kernels are the same, so the bits must be too.

Four arms, built by tools/exact_throughput.py's own `build()`, batch 32,
64 new tokens, best of 5, taken in turns:
  stock, exact          as the tool builds them;
  stock_cpp, exact_cpp  the same, with `cpp_wrapper` set around every call
                        (Inductor reads it when it compiles, on the first).
Each arm must pass the tool's sanity gate, and each _cpp arm must give the
same token ids as its plain arm on all 32 rows. The exact arms' switches
come from the environment.
"""
import argparse
import os
import sys

TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tools")
sys.path.insert(0, os.path.abspath(TOOLS))

import torch  # noqa: E402
import torch._inductor.config as inductor_config  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import exact_throughput as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stock-dtype", choices=("bfloat16", "float16"), default="bfloat16")
ap.add_argument("--model", default=T.MODEL)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--arms", default="stock,stock_cpp,exact,exact_cpp")
a = ap.parse_args()
T.STOCK_DTYPE = getattr(torch, a.stock_dtype)
T.MODEL = a.model
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def with_cpp_wrapper(m):
    inner = m.forward

    @T.keeps_signature(inner)
    def call(*args, _f=inner, **kw):
        with inductor_config.patch(cpp_wrapper=True):
            return _f(*args, **kw)

    m.forward = call
    return m


tok = AutoTokenizer.from_pretrained(T.MODEL)
arms = a.arms.split(",")
models = {}
for n in arms:
    base = n.removesuffix("_cpp")
    m = T.build(base, "cuda", True)
    models[n] = with_cpp_wrapper(m) if n.endswith("_cpp") else m
env = {k: v for k, v in os.environ.items() if k.startswith("Y_EXACT")}
print(f"  arms {arms}, exact switches {env or 'none'}")
if not T.gate(models, tok, "cuda"):
    raise SystemExit(1)

ids = tok(T.PROMPT, return_tensors="pt").to("cuda")
inp, att = ids.input_ids.repeat(32, 1), ids.attention_mask.repeat(32, 1)
kw = dict(do_sample=False, num_beams=1, use_cache=True, pad_token_id=tok.eos_token_id,
          max_new_tokens=64, min_new_tokens=64)
with torch.no_grad():
    out = {n: m.generate(input_ids=inp, attention_mask=att, **kw) for n, m in models.items()}
for n in arms:
    if n.endswith("_cpp") and n.removesuffix("_cpp") in out:
        same = torch.equal(out[n], out[n.removesuffix("_cpp")])
        print(f"  {n} and {n.removesuffix('_cpp')} give the same token ids on all 32 rows: {same}")
        if not same:
            raise SystemExit(1)

res = T.run_interleaved(models, tok, 32, 64, "cuda", a.reps)
for n, (best, med) in res.items():
    print(f"  {n:<10} {best:7.3f} s  {64 / best:7.1f} tok/s/seq  median {100 * (med - best) / best:4.1f}% above")
for n in arms:
    if n.endswith("_cpp") and n.removesuffix("_cpp") in res:
        print(f"  {n} / {n.removesuffix('_cpp')} = {res[n][0] / res[n.removesuffix('_cpp')][0]:.3f}x")
for e in ("exact", "exact_cpp"):
    for s in ("stock", "stock_cpp"):
        if e in res and s in res:
            print(f"  {e} / {s} = {res[e][0] / res[s][0]:.3f}x")
