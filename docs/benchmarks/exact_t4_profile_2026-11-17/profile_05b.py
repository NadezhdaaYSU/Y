"""Where a 0.5B decode step's GPU time goes, stock against exact, kernel by kernel.

    cd tools && python3 ../docs/benchmarks/exact_t4_profile_2026-11-17/profile_05b.py ARM OUT.json [--stock-dtype float16]

ARM is `stock` or `exact`; the exact path's switches come from the
environment as usual. The model is built by tools/exact_throughput.py's own
`build()`, compiled, so this profiles what that tool times: batch 32, its
prompt, `generate()`. After warm-up, one generate of NEW tokens runs under
torch.profiler. Reported per new token: wall time, GPU busy time (kernels
and copies), the kernel count, time by kernel family, and the top kernels.

If GPU busy is far below the unprofiled wall time, the step waits on the
host (launches, Python), not on the GPU. Times include the prompt's prefill,
spread over the NEW tokens.

Decode only: the same is then measured at 3*NEW tokens, and the difference
divided by 2*NEW. The prefill is in both runs, so it cancels, and what is
left is the cost of one decode step ("decode_only" in the JSON).
"""
import argparse
import json
import os
import re
import sys
import time

TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tools")
sys.path.insert(0, os.path.abspath(TOOLS))

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import exact_throughput as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("arm", choices=("stock", "exact"))
ap.add_argument("out")
ap.add_argument("--stock-dtype", choices=("bfloat16", "float16"), default="bfloat16")
ap.add_argument("--model", default=T.MODEL)
ap.add_argument("--new", type=int, default=16)
a = ap.parse_args()
T.STOCK_DTYPE = getattr(torch, a.stock_dtype)
T.MODEL = a.model

# One family per kernel, first match wins. The patterns are names, so the
# "other" bucket is listed in full in the JSON to check nothing big hides there.
FAMILIES = [
    ("int8 GEMM", r"int_mm|imma|igemm|i8i8|s8s8|_s8_|int8|w8a8|cutlass.*(s8|int)"),
    ("Y kernel", r"exact_pv|exact_decode_attention|int8_gemm|y_mm|ysu"),
    ("attention", r"flash|fmha|attention|sdpa|efficient_attention"),
    ("float GEMM", r"gemm|gemv|cutlass|xmma|ampere_|turing_|sm75_|sm80_|sm89_|cublas|splitKreduce"),
    ("Inductor fused", r"^triton_"),
    ("other Triton", r"^_?[a-z0-9_]*kernel|^triton"),
    ("copy/fill", r"memcpy|memset|copy|fill|cat|index"),
]


def family(name):
    low = name.lower()
    for fam, pat in FAMILIES:
        if re.search(pat, low):
            return fam
    return "other"


tok = AutoTokenizer.from_pretrained(T.MODEL)
model = T.build(a.arm, "cuda", True)
ids = tok(T.PROMPT, return_tensors="pt").to("cuda")
inp, att = ids.input_ids.repeat(32, 1), ids.attention_mask.repeat(32, 1)
kw = dict(do_sample=False, num_beams=1, use_cache=True, pad_token_id=tok.eos_token_id)
with torch.no_grad():
    for n in (8, a.new, a.new):
        model.generate(input_ids=inp, attention_mask=att, max_new_tokens=n, min_new_tokens=n, **kw)
torch.cuda.synchronize()

# The profiler slows the host, so its wall time overstates a host-bound step.
# The same generate, unprofiled, best of 3, is the wall time to compare with.
def measure(new):
    """Best-of-3 unprofiled wall time, then one profiled generate: (plain ms,
    profiled ms, {kernel: [ms, calls]})."""
    plain = []
    for _ in range(3):
        t = time.perf_counter()
        with torch.no_grad():
            model.generate(input_ids=inp, attention_mask=att, max_new_tokens=new, min_new_tokens=new, **kw)
        torch.cuda.synchronize()
        plain.append((time.perf_counter() - t) * 1000)
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        t = time.perf_counter()
        with torch.no_grad():
            model.generate(input_ids=inp, attention_mask=att, max_new_tokens=new, min_new_tokens=new, **kw)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t) * 1000
    kernels = {}
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            k = kernels.setdefault(e.name, [0.0, 0])
            k[0] += e.device_time / 1000.0
            k[1] += 1
    return min(plain), wall, kernels


plain, wall, kernels = measure(a.new)
plain3, _, kernels3 = measure(3 * a.new)
steps = 2 * a.new
decode = {n: [(kernels3.get(n, [0.0, 0])[0] - kernels.get(n, [0.0, 0])[0]) / steps,
              (kernels3.get(n, [0.0, 0])[1] - kernels.get(n, [0.0, 0])[1]) / steps]
          for n in set(kernels) | set(kernels3)}
dfams = {}
for n, (ms, calls) in decode.items():
    f = dfams.setdefault(family(n), [0.0, 0.0])
    f[0] += ms
    f[1] += calls
dbusy = sum(v[0] for v in decode.values())
dwall = (plain3 - plain) / steps
busy = sum(v[0] for v in kernels.values())
launches = sum(v[1] for v in kernels.values())
fams = {}
for n, (ms, calls) in kernels.items():
    f = fams.setdefault(family(n), [0.0, 0])
    f[0] += ms
    f[1] += calls
top = sorted(kernels.items(), key=lambda kv: -kv[1][0])
env = {k: v for k, v in os.environ.items() if k.startswith("Y_EXACT")}
out = {"arm": a.arm, "env": env, "stock_dtype": a.stock_dtype, "new_tokens": a.new,
       "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
       "wall_ms_unprofiled": round(plain, 2), "wall_ms_profiled": round(wall, 2), "gpu_busy_ms": round(busy, 2), "kernels_launched": launches,
       "families": {f: {"ms": round(v[0], 3), "calls": v[1]} for f, v in sorted(fams.items(), key=lambda kv: -kv[1][0])},
       "kernels": [{"name": n, "family": family(n), "ms": round(v[0], 3), "calls": v[1]} for n, v in top],
       "decode_only": {"per_step_wall_ms_unprofiled": round(dwall, 3), "per_step_gpu_busy_ms": round(dbusy, 3),
                       "per_step_kernels": round(sum(v[1] for v in decode.values()), 1),
                       "families": {f: {"ms": round(v[0], 4), "calls": round(v[1], 2)}
                                    for f, v in sorted(dfams.items(), key=lambda kv: -kv[1][0])},
                       "kernels": [{"name": n, "family": family(n), "ms": round(v[0], 4), "calls": round(v[1], 2)}
                                   for n, v in sorted(decode.items(), key=lambda kv: -kv[1][0])]}}
json.dump(out, open(a.out, "w"), indent=1)

per = a.new
print(f"{a.arm} {env or ''} on {out['device']}, {per} new tokens, batch 32, compiled")
print(f"  per token: wall {plain/per:.2f} ms unprofiled ({wall/per:.2f} profiled), "
      f"GPU busy {busy/per:.2f} ms ({100*busy/plain:.0f}% of the unprofiled wall), "
      f"{launches/per:.0f} kernels")
for f, v in sorted(fams.items(), key=lambda kv: -kv[1][0]):
    print(f"  {f:<16} {v[0]/per:7.3f} ms/token {100*v[0]/busy:5.1f}%  {v[1]/per:6.1f} kernels/token")
print("  top kernels:")
for n, v in top[:20]:
    print(f"  {v[0]/per:7.3f} ms/token {v[1]/per:6.1f}x  [{family(n)}] {n[:100]}")
print(f"  decode only ({a.new} -> {3 * a.new} new tokens), per step: wall {dwall:.2f} ms unprofiled, "
      f"GPU busy {dbusy:.2f} ms ({100 * dbusy / dwall:.0f}%), {sum(v[1] for v in decode.values()):.0f} kernels")
for f, v in sorted(dfams.items(), key=lambda kv: -kv[1][0]):
    print(f"  {f:<16} {v[0]:7.3f} ms/step {100 * v[0] / dbusy:5.1f}%  {v[1]:6.1f} kernels/step")
print("  top kernels, decode only:")
for n, v in sorted(decode.items(), key=lambda kv: -kv[1][0])[:20]:
    print(f"  {v[0]:7.3f} ms/step {v[1]:6.1f}x  [{family(n)}] {n[:100]}")
