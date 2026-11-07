"""Which `exact_pv` route `e2e.sh`'s arm C took, and whether the graph held.

    Y_EXACT_V_INT8=1 /tmp/y_exact_nospace/venv/bin/python \
        /tmp/y_exact_nospace/docs/benchmarks/exact_pv_library_op_2026-11-07/route_check.py

Builds the exact arm as `tools/exact_throughput.py --compile` does, generates
8 tokens at batch 32, and reports: whether the kernel was loaded, how many
times the eager body ran (0 under compile: the graph calls the op) and how
many times the registered op did, the dtype of the V the kernel was handed,
and dynamo's graph and break counts. Run without `Y_EXACT_V_INT8` it shows
the digit split: no kernel, no op calls.
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
sys.path.insert(0, os.path.join(REPO, "tools"))
import torch  # noqa: E402
from torch._dynamo.utils import counters  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import batch_invariance_demo as D  # noqa: E402
import exact_kv  # noqa: E402
import exact_throughput as ET  # noqa: E402

calls = {"body": 0, "op": 0, "v": set()}
body = D._y_exact_pv


def counted_body(p, v):
    calls["body"] += 1
    calls["v"].add(str(v.dtype))
    return body(p, v)


def counted_op(p, v):
    calls["op"] += 1
    calls["v"].add(str(v.dtype))
    return body(p, v)


# The eager path calls the module's `_y_exact_pv`; the registered op's CUDA
# kernel is the function given at registration, so it is re-pointed here.
D._y_exact_pv = counted_body
D._PV_LIB.impl("exact_pv", counted_op, "CUDA", allow_override=True)

tok = AutoTokenizer.from_pretrained(ET.MODEL)
model = ET.build("exact", "cuda", True)
ids = tok(ET.PROMPT, return_tensors="pt").to("cuda")
inp = ids["input_ids"].repeat(32, 1)
att = ids["attention_mask"].repeat(32, 1)
with torch.no_grad():
    model.generate(input_ids=inp, attention_mask=att, max_new_tokens=8, do_sample=False,
                   num_beams=1, use_cache=True, pad_token_id=tok.eos_token_id)
torch.cuda.synchronize()
print(f"Y_EXACT_V_INT8={os.environ.get('Y_EXACT_V_INT8', '0')}  V cache dtype "
      f"{exact_kv.v_store_dtype()}")
print(f"kernel loaded on devices {sorted(D._Y_PV)}; eager body calls {calls['body']}, "
      f"op calls {calls['op']}, V dtypes handed to the kernel {sorted(calls['v'])}")
print("dynamo stats:", dict(counters["stats"]))
print("graph breaks:", sum(counters["graph_break"].values()), dict(counters["graph_break"]))
