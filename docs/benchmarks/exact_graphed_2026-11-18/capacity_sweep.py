"""What the graphed decode step pays for the spare rows of its cache.

    L=/tmp/y_exact_nospace; cd $L/tools && $L/venv/bin/python3 \
        $L/docs/benchmarks/exact_graphed_2026-11-18/capacity_sweep.py

(Absolute paths through the space-free link: torch.compile cannot link when
the interpreter or torch is reached through `.../NVME files/...`; see
tools/_nospace.py. A relative `../venv/bin/python3` resolves to the real
path and fails.)

`GraphedGreedy` attends over its whole cache buffer every step (the mask
blocks the rows not written yet). The buffer holds the prompt plus every
new token, rounded up to `capacity_multiple`. With the speed tool's prompt
(28 tokens) and 64 new tokens that is 96 rows, while a step sees 60 rows on
average.

This times the decode steps alone (graph replays, no prefill) for buffers of
92 rows (no rounding), 96, 128 and 256, for each arm `exact_throughput.build`
makes (`Y_THR_ARMS`, default stock and exact; the exact arm's switches come
from the environment). If time per step grows with the buffer, recording
one graph per length bucket would save that work. Each arm's tokens must be
the same at every size for the exact arm (a blocked row gets no weight);
for stock they are reported.

Batch 32, 64 new tokens, compiled, best of 5 rounds; sizes and arms in turns.
Options: `--stock-dtype float16` (a T4 has no bf16 tensor cores), `--model`.
"""
import argparse
import os
import sys
import time

TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tools")
sys.path.insert(0, os.path.abspath(TOOLS))

import torch  # noqa: E402

if " " in torch.__file__ or " " in sys.executable:
    raise SystemExit(f"*** torch ({torch.__file__}) or Python ({sys.executable}) is reached through "
                     f"a path with a space; torch.compile will not link. Run through "
                     f"/tmp/y_exact_nospace (see the docstring).")
import torch._dynamo  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import exact_throughput as T  # noqa: E402
from y_lang.exact.graphed import GraphedGreedy  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stock-dtype", choices=("bfloat16", "float16"), default="bfloat16")
ap.add_argument("--model", default=T.MODEL)
a = ap.parse_args()
T.MODEL, T.STOCK_DTYPE = a.model, getattr(torch, a.stock_dtype)

B, NEW, ROUNDS = 32, 64, 5
MULTIPLES = (1, 32, 128, 256)          # 92, 96, 128 and 256 rows here
torch._dynamo.config.recompile_limit = 64   # one compile per (arm, size)
torch.backends.cuda.matmul.allow_tf32 = False

tok = AutoTokenizer.from_pretrained(T.MODEL)
ids = tok(T.PROMPT, return_tensors="pt").to("cuda")
inp, att = ids.input_ids.repeat(B, 1), ids.attention_mask.repeat(B, 1)
print(f"{torch.cuda.get_device_name()}, stock {a.stock_dtype}; "
      f"prompt {inp.shape[1]} tokens, batch {B}, {NEW} new tokens; "
      f"switches: {' '.join(k + '=' + v for k, v in sorted(os.environ.items()) if k.startswith('Y_EXACT_')) or 'none'}")

arms = os.environ.get("Y_THR_ARMS", "stock,exact").split(",")
runs, tokens = {}, {}
for name in arms:
    m = T.build(name, "cuda", True)
    for mult in MULTIPLES:
        g = GraphedGreedy(m, B, inp.shape[1], NEW, attention=m._y_attention,
                          inductor_patch=m._y_inductor_patch, capacity_multiple=mult)
        tokens[name, g.cap] = g(inp, att)            # compiles and records
        runs[name, g.cap] = g


def steps_time(g):
    """The NEW - 1 replays alone, from a freshly loaded state."""
    with torch.no_grad():
        with g._installed():
            first, nxt, rows = g._prefill(inp, att)
        g._load(first, nxt, rows, att)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(NEW - 1):
        g.graph.replay()
    torch.cuda.synchronize()
    return time.perf_counter() - t


times = {k: [] for k in runs}
for _ in range(ROUNDS):
    for k, g in runs.items():
        times[k].append(steps_time(g))

for name in arms:
    caps = sorted(c for n, c in runs if n == name)
    ref = tokens[name, caps[0]]
    same = all(torch.equal(tokens[name, c], ref) for c in caps)
    print(f"\n{name}: tokens the same at every size: {same}")
    base = min(times[name, caps[0]]) / (NEW - 1)
    for c in caps:
        ms = 1e3 * min(times[name, c]) / (NEW - 1)
        print(f"  {c:>4} rows  {ms:7.3f} ms a step  {ms / (1e3 * base):.3f}x of {caps[0]} rows")
    if name == "exact" and not same:
        raise SystemExit("*** the exact arm's tokens depend on the buffer size")
