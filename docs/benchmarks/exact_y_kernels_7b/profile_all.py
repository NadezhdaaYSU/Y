"""Where a 7B exact decode step's time goes with Y's kernels on.

    Y_EXACT_GEMM=y Y_EXACT_V_INT8=1 Y_EXACT_ATTN=y Y_EXACT_NORM=triton \
    Y_EXACT_KV_PACK=1 python3 profile_all.py <out.json> [model]

A copy of `../exact_greedy_loop_7b_2026-11-08/profile_greedy.py` that also
records the three switches this folder adds.

The configuration exact_throughput times (batch 32, its prompt, compiled),
warmed up. Two measurements:

1. One `greedy` of 16 new tokens under torch.profiler: wall time, the GPU's
   busy time (the union of kernel and copy intervals), and busy time per
   kernel name.
2. The CPU's cost of issuing one decode step, apart from the GPU's: the step
   is issued onto an idle, synchronised GPU, so the time until the forward
   call returns is the CPU's, and the time until the GPU has finished is the
   step's whole latency. If issuing takes as long as the GPU's busy time per
   step, the CPU is what the GPU waits for.
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
from y_lang.exact.greedy import greedy  # noqa: E402
from transformers import DynamicCache  # noqa: E402

PROMPT = ("Explain, step by step and in detail, how a suspension bridge carries "
          "the load of the traffic crossing it down into the ground. ")
NEW = 16
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-7B-Instruct"
model, tok = load(MODEL, "cuda", low_memory=True)
em.compile_forward(model)
ids = tok(PROMPT, return_tensors="pt").to("cuda")
inp, att = ids.input_ids.repeat(32, 1), ids.attention_mask.repeat(32, 1)
for n in (8, NEW, NEW):
    greedy(model, inp, att, n)
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    t = time.perf_counter()
    greedy(model, inp, att, NEW)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t

kernels, spans = {}, []
for e in prof.events():
    if e.device_type == torch.autograd.DeviceType.CUDA:
        k = kernels.setdefault(e.name, [0.0, 0])
        k[0] += e.device_time / 1000.0      # ms
        k[1] += 1
        spans.append((e.time_range.start, e.time_range.end))
spans.sort()
busy_us, end = 0.0, None
for s, e in spans:                          # union of intervals, in us
    if end is None or s > end:
        busy_us += e - s
        end = e
    elif e > end:
        busy_us += e - end
        end = e
busy = busy_us / 1000.0

# 2. Issue cost of single decode steps on a synchronised GPU.
position_ids = (att.long().cumsum(-1) - 1).masked_fill(att == 0, 0)
issue, latency = [], []
with torch.no_grad():
    cache = DynamicCache(config=model.config.get_text_config(decoder=True))
    out = model(input_ids=inp, attention_mask=att, position_ids=position_ids,
                past_key_values=cache, use_cache=True, logits_to_keep=1, return_dict=True)
    token = out.logits[:, -1].float().argmax(-1)
    mask, pos = att, position_ids[:, -1:]
    for step in range(12):
        pos = pos + 1
        mask = torch.cat([mask, mask.new_ones((mask.shape[0], 1))], dim=-1)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(input_ids=token[:, None], attention_mask=mask, position_ids=pos,
                    past_key_values=cache, use_cache=True, logits_to_keep=1, return_dict=True)
        token = out.logits[:, -1].float().argmax(-1)
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if step >= 2:
            issue.append((t1 - t0) * 1000)
            latency.append((t2 - t0) * 1000)

top = sorted(kernels.items(), key=lambda kv: -kv[1][0])
out = {"model": MODEL, "config": {k: os.environ.get(k, "") for k in ("Y_EXACT_GEMM", "Y_EXACT_V_INT8", "Y_EXACT_MERGE",
                                                    "Y_EXACT_ATTN", "Y_EXACT_NORM", "Y_EXACT_KV_PACK")},
       "new_tokens": NEW, "wall_ms": wall * 1000, "gpu_busy_ms": busy,
       "decode_step_issue_ms": issue, "decode_step_latency_ms": latency,
       "kernels": [{"name": n, "ms": round(v[0], 3), "calls": v[1]} for n, v in top]}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(f"greedy {NEW} tokens: wall {wall*1000:.1f} ms, GPU busy {busy:.1f} ms "
      f"({100*busy/(wall*1000):.0f}%), idle {wall*1000-busy:.1f} ms")
print(f"one decode step on an idle GPU: CPU issue {min(issue):.2f} ms (median "
      f"{sorted(issue)[len(issue)//2]:.2f}), latency {min(latency):.2f} ms (median "
      f"{sorted(latency)[len(latency)//2]:.2f})")
print(f"launches per greedy run: {len(spans)}")
for n, v in top[:30]:
    print(f"{v[0]:9.2f} ms {100*v[0]/busy:5.1f}% {v[1]:6d}x  {n[:110]}")
