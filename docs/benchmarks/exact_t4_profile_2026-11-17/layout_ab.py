"""The int8 weight's storage order, timed: the exact path with the weight in
nn.Linear's order (what `convert` now picks when the linears take
`torch._int_mm`) against the old row-major [K][N] order, next to stock.

    cd tools && python3 ../docs/benchmarks/exact_t4_profile_2026-11-17/layout_ab.py [--stock-dtype float16] [--model M]

Three arms, built by tools/exact_throughput.py's own `build()`, compiled,
batch 32, 64 new tokens, best of 5, taken in turns (`run_interleaved`):
  stock     stock float16 or bf16 with SDPA;
  exact     the exact path as `convert` builds it;
  exact_nn  the same, with every `w8t` copied back to row-major [K][N]
            before its first call (so before anything is compiled).
Before timing, each arm passes the tool's sanity gate, and the two exact arms
must give the same token ids on all 32 rows. Then one profiled step per exact
arm lists the GEMM kernels it launched.
"""
import argparse
import os
import sys

TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tools")
sys.path.insert(0, os.path.abspath(TOOLS))

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import exact_throughput as T  # noqa: E402
from y_lang.exact.model import ExactLinear  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stock-dtype", choices=("bfloat16", "float16"), default="bfloat16")
ap.add_argument("--model", default=T.MODEL)
ap.add_argument("--reps", type=int, default=5)
a = ap.parse_args()
T.STOCK_DTYPE = getattr(torch, a.stock_dtype)
T.MODEL = a.model
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def layouts(m):
    ls = [m.w8t for m in m.modules() if isinstance(m, ExactLinear)]
    return {"count": len(ls),
            "nn.Linear order ([N][K] contiguous)": sum(w.t().is_contiguous() for w in ls),
            "row-major [K][N]": sum(w.is_contiguous() for w in ls)}


tok = AutoTokenizer.from_pretrained(T.MODEL)
models = {"stock": T.build("stock", "cuda", True), "exact": T.build("exact", "cuda", True)}
nn_arm = T.build("exact", "cuda", True)
for mod in nn_arm.modules():
    if isinstance(mod, ExactLinear):
        mod.w8t = mod.w8t.contiguous()
models["exact_nn"] = nn_arm
for n in ("exact", "exact_nn"):
    print(f"  layout {n:<9} {layouts(models[n])}")
if not T.gate(models, tok, "cuda"):
    raise SystemExit(1)

ids = tok(T.PROMPT, return_tensors="pt").to("cuda")
inp, att = ids.input_ids.repeat(32, 1), ids.attention_mask.repeat(32, 1)
kw = dict(do_sample=False, num_beams=1, use_cache=True, pad_token_id=tok.eos_token_id)
with torch.no_grad():
    out = {n: models[n].generate(input_ids=inp, attention_mask=att, max_new_tokens=64,
                                 min_new_tokens=64, **kw) for n in ("exact", "exact_nn")}
same = torch.equal(out["exact"], out["exact_nn"])
print(f"  exact and exact_nn give the same token ids on all 32 rows: {same}")
if not same:
    raise SystemExit(1)

GEMM = ("gemm", "igemm", "imma", "int_mm", "i16832", "i8816", "s8")
for n in ("exact", "exact_nn"):
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        with torch.no_grad():
            models[n].generate(input_ids=inp, attention_mask=att, max_new_tokens=2, min_new_tokens=2, **kw)
        torch.cuda.synchronize()
    names = sorted({e.name for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA
                    and any(g in e.name.lower() for g in GEMM)})
    print(f"  GEMM kernels, {n}:")
    for k in names:
        print(f"    {k[:140]}")

res = T.run_interleaved(models, tok, 32, 64, "cuda", a.reps)
for n, (best, med) in res.items():
    print(f"  {n:<9} {best:7.3f} s  {64 / best:7.1f} tok/s/seq  median {100 * (med - best) / best:4.1f}% above")
for n in ("exact", "exact_nn"):
    print(f"  {n} / stock = {res[n][0] / res['stock'][0]:.2f}x slower")
print(f"  exact_nn / exact = {res['exact_nn'][0] / res['exact'][0]:.2f}x")
