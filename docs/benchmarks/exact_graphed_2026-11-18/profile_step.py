"""Where the GPU time of a graphed decode step goes, kernel by kernel.

    L=/tmp/y_exact_nospace; cd $L/tools && $L/venv/bin/python3 \\
        $L/docs/benchmarks/exact_graphed_2026-11-18/profile_step.py [--stock-dtype float16]

(Absolute paths through the space-free link, as for capacity_sweep.py.)

For each arm `exact_throughput.build` makes (`Y_THR_ARMS`; the exact arm's
switches come from the environment) this records a `GraphedGreedy` (batch
32, 64 new tokens), then profiles 16 decode steps run through the compiled
step directly (`_advance`), which launches the same kernels the graph
replays. It prints each kernel's GPU time per step, grouped into: GEMM
(cuBLAS, CUTLASS, Y's int8 GEMM), attention, Inductor's fused kernels
(`triton_*`), and the rest. The step's wall time comes from graph replays,
as in capacity_sweep.py.
"""
import argparse
import os
import sys
import time
from collections import defaultdict

TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tools")
sys.path.insert(0, os.path.abspath(TOOLS))

import torch  # noqa: E402

if " " in torch.__file__ or " " in sys.executable:
    raise SystemExit(f"*** torch ({torch.__file__}) or Python ({sys.executable}) is reached through "
                     f"a path with a space; torch.compile will not link. Run through "
                     f"/tmp/y_exact_nospace (see the docstring).")
from torch.profiler import ProfilerActivity, profile  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import exact_throughput as T  # noqa: E402
from y_lang.exact.graphed import GraphedGreedy  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stock-dtype", choices=("bfloat16", "float16"), default="bfloat16")
ap.add_argument("--model", default=T.MODEL)
a = ap.parse_args()
T.MODEL, T.STOCK_DTYPE = a.model, getattr(torch, a.stock_dtype)
torch.backends.cuda.matmul.allow_tf32 = False

B, NEW, STEPS = 32, 64, 16
tok = AutoTokenizer.from_pretrained(T.MODEL)
ids = tok(T.PROMPT, return_tensors="pt").to("cuda")
inp, att = ids.input_ids.repeat(B, 1), ids.attention_mask.repeat(B, 1)
print(f"{torch.cuda.get_device_name()}, stock {a.stock_dtype}, batch {B}; switches: "
      f"{' '.join(k + '=' + v for k, v in sorted(os.environ.items()) if k.startswith('Y_EXACT_')) or 'none'}")


def group(name):
    # Attention first: PyTorch's mem-efficient SDPA kernel is a CUTLASS
    # kernel (`fmha_cutlassF_...`), and an Inductor kernel fused around SDPA
    # carries `scaled_dot_product` in its name. A strided-batched cuBLAS
    # kernel is a `bmm`, which in this step only attention runs (Q.K^T and
    # P.V on the exact path's float rows; its linears are `_w8a8_gemm`).
    n = name.lower()
    if any(s in n for s in ("attention", "fmha", "flash", "exact_pv", "decode_attn", "softmax",
                            "scaled_dot_product", "stridedbatched")):
        return "attention"
    if any(s in n for s in ("gemm", "cutlass", "sm80_xmma", "sm90_xmma", "ampere_", "volta_",
                            "turing_", "cublas", "gemv", "exact_linear", "int8_mm")):
        return "GEMM"
    if n.startswith("triton_"):
        return "Inductor fused"
    return "other"


def fresh(g):
    with torch.no_grad():
        with g._installed():
            first, nxt, rows = g._prefill(inp, att)
        g._load(first, nxt, rows, att)


for name in os.environ.get("Y_THR_ARMS", "stock,exact").split(","):
    m = T.build(name, "cuda", True)
    g = GraphedGreedy(m, B, inp.shape[1], NEW, attention=m._y_attention,
                      inductor_patch=m._y_inductor_patch)
    g(inp, att)                                         # compile and record
    walls = []
    for _ in range(5):
        fresh(g)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(NEW - 1):
            g.graph.replay()
        torch.cuda.synchronize()
        walls.append((time.perf_counter() - t) / (NEW - 1))
    fresh(g)
    with torch.no_grad(), g._installed():
        g._advance()
        g._advance()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(STEPS):
                g._advance()
            torch.cuda.synchronize()
    per = defaultdict(lambda: [0.0, 0])
    for e in prof.key_averages():
        us = getattr(e, "device_time_total", None)
        if us is None:
            us = e.cuda_time_total
        if us <= 0:
            continue
        per[e.key][0] += us / STEPS
        per[e.key][1] += e.count / STEPS
    total = sum(v[0] for v in per.values())
    groups = defaultdict(float)
    for k, (us, _) in per.items():
        groups[group(k)] += us
    print(f"\n{name}: graphed step {1e3 * min(walls):.3f} ms wall (best of 5); "
          f"GPU kernels {total / 1e3:.3f} ms a step, {sum(v[1] for v in per.values()):.0f} kernels")
    for gname, us in sorted(groups.items(), key=lambda x: -x[1]):
        print(f"  {gname:<15} {us / 1e3:7.3f} ms  {100 * us / total:5.1f}%")
    print("  top kernels (us a step, launches a step):")
    for k, (us, n) in sorted(per.items(), key=lambda x: -x[1][0])[:14]:
        print(f"    {us:8.1f}  {n:5.1f}  [{group(k)}] {k[:110]}")
    del g, m
    torch.cuda.empty_cache()
