"""On a GPU where the exact path's linear layers run the Triton int8 kernel
(sm_80 and later), is `torch._int_mm` in the TN form faster?

    cd tools && python3 ../docs/benchmarks/exact_t4_profile_2026-11-17/gemm_path_ab.py [--model M]

Three arms, built by tools/exact_throughput.py's own `build()`, compiled,
batch 32, 64 new tokens, best of 5, taken in turns:
  stock         stock bf16 with SDPA;
  exact         the exact path as `convert` builds it (the Triton kernel);
  exact_int_mm  the same model with every layer's `triton_ok` cleared and
                its weight stored in nn.Linear's order before the first call,
                which is what `convert` does on a card without the Triton
                int8 dot: the layers take `torch._int_mm` (TN).
Each arm must pass the tool's sanity gate, and the two exact arms must give
the same token ids on all 32 rows. One profiled step per exact arm then
lists the GEMM kernels it launched.
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
from y_lang.exact import model as em  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=T.MODEL)
ap.add_argument("--reps", type=int, default=5)
a = ap.parse_args()
T.MODEL = a.model
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
if em.GEMM != "triton" or em.w8a8_matmul is None or not em.triton_int8_dot_ok("cuda"):
    raise SystemExit(f"this card's exact path does not run the Triton GEMM (GEMM={em.GEMM!r}); nothing to compare")

tok = AutoTokenizer.from_pretrained(T.MODEL)
models = {"stock": T.build("stock", "cuda", True), "exact": T.build("exact", "cuda", True)}
alt = T.build("exact", "cuda", True)
n = 0
for mod in alt.modules():
    if isinstance(mod, em.ExactLinear):
        mod.triton_ok = False
        mod.w8t = mod.w8t.t().contiguous().t()
        n += 1
models["exact_int_mm"] = alt
lay = [m.w8t.t().is_contiguous() for m in models["exact"].modules() if isinstance(m, em.ExactLinear)]
print(f"  exact: {len(lay)} layers, {sum(lay)} in nn.Linear order; exact_int_mm: {n} layers switched to _int_mm, TN")
if not T.gate(models, tok, "cuda"):
    raise SystemExit(1)

ids = tok(T.PROMPT, return_tensors="pt").to("cuda")
inp, att = ids.input_ids.repeat(32, 1), ids.attention_mask.repeat(32, 1)
kw = dict(do_sample=False, num_beams=1, use_cache=True, pad_token_id=tok.eos_token_id)
with torch.no_grad():
    out = {k: models[k].generate(input_ids=inp, attention_mask=att, max_new_tokens=64,
                                 min_new_tokens=64, **kw) for k in ("exact", "exact_int_mm")}
same = torch.equal(out["exact"], out["exact_int_mm"])
print(f"  exact and exact_int_mm give the same token ids on all 32 rows: {same}")
if not same:
    raise SystemExit(1)

for k in ("exact", "exact_int_mm"):
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        with torch.no_grad():
            models[k].generate(input_ids=inp, attention_mask=att, max_new_tokens=2, min_new_tokens=2, **kw)
        torch.cuda.synchronize()
    names = sorted({e.name for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA
                    and any(g in e.name.lower() for g in ("gemm", "w8a8", "int_mm", "i16832", "s8"))})
    print(f"  GEMM kernels, {k}:")
    for name in names:
        print(f"    {name[:140]}")

res = T.run_interleaved(models, tok, 32, 64, "cuda", a.reps)
for k, (best, med) in res.items():
    print(f"  {k:<13} {best:7.3f} s  {64 / best:7.1f} tok/s/seq  median {100 * (med - best) / best:4.1f}% above")
for k in ("exact", "exact_int_mm"):
    print(f"  {k} / stock = {res[k][0] / res['stock'][0]:.3f}x")
print(f"  exact_int_mm / exact = {res['exact_int_mm'][0] / res['exact'][0]:.3f}x")
