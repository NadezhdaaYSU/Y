"""Where a 7B exact decode step's GPU time goes, kernel by kernel.

    Y_EXACT_GEMM=y Y_EXACT_V_INT8=1 python3 profile_step.py <out.json>

The configuration exact_throughput times (batch 32, its prompt, compiled),
warmed up, then one generate of 16 new tokens under torch.profiler. Reports
the wall time, the GPU's busy time (kernels plus copies), and the busy time
per kernel name, so launch gaps and each kernel family's share are visible.
"""
import json
import os
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tools"))
from y_lang.exact import load  # noqa: E402
from y_lang.exact import model as em  # noqa: E402

PROMPT = ("Explain, step by step and in detail, how a suspension bridge carries "
          "the load of the traffic crossing it down into the ground. ")
NEW = 16
model, tok = load("Qwen/Qwen2.5-7B-Instruct", "cuda", low_memory=True)
em.compile_forward(model)
ids = tok(PROMPT, return_tensors="pt").to("cuda")
inp, att = ids.input_ids.repeat(32, 1), ids.attention_mask.repeat(32, 1)
kw = dict(max_new_tokens=NEW, min_new_tokens=NEW, do_sample=False, num_beams=1,
          use_cache=True, pad_token_id=tok.eos_token_id)
with torch.no_grad():
    for n in (8, NEW, NEW):
        model.generate(input_ids=inp, attention_mask=att, **dict(kw, max_new_tokens=n, min_new_tokens=n))
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    t = time.perf_counter()
    with torch.no_grad():
        model.generate(input_ids=inp, attention_mask=att, **kw)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t

kernels = {}
for e in prof.events():
    if e.device_type == torch.autograd.DeviceType.CUDA:
        k = kernels.setdefault(e.name, [0.0, 0])
        k[0] += e.device_time / 1000.0      # ms
        k[1] += 1
busy = sum(v[0] for v in kernels.values())
top = sorted(kernels.items(), key=lambda kv: -kv[1][0])
out = {"config": {k: os.environ.get(k, "") for k in ("Y_EXACT_GEMM", "Y_EXACT_V_INT8")},
       "new_tokens": NEW, "wall_ms": wall * 1000, "gpu_busy_ms": busy,
       "kernels": [{"name": n, "ms": round(v[0], 3), "calls": v[1]} for n, v in top]}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(f"wall {wall*1000:.1f} ms, GPU busy {busy:.1f} ms ({100*busy/(wall*1000):.0f}%), {NEW} tokens")
for n, v in top[:25]:
    print(f"{v[0]:9.2f} ms {100*v[0]/busy:5.1f}% {v[1]:6d}x  {n[:110]}")
