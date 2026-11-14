"""Prefill head-to-head on one GPU: samebit v0.2 against Y's exact path, with
PyTorch float16 / bfloat16 as the shared yardstick. Qwen2.5-0.5B-Instruct,
WikiText-2 test windows of T tokens, batch 1, logits of every position.

Correctness first: each integer system's GPU logits must equal its own CPU
logits bit for bit (samebit: its NumPy forward; Y: the exact model built on
the CPU), on every window. Then speed: one process, every arm warm, R rounds
in a shuffled order each round, the minimum and the median per arm.

    python prefill_vs_samebit.py SAMEBIT_SRC QWEN_WORK Y_REPO [--tokens 256] [--rounds 11]
"""
import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("samebit_src")
ap.add_argument("qwen_work")
ap.add_argument("y_repo")
ap.add_argument("--tokens", type=int, default=256)
ap.add_argument("--windows", type=int, default=2)
ap.add_argument("--rounds", type=int, default=11)
ap.add_argument("--reps", type=int, default=5, help="forwards per timed sample")
ap.add_argument("--compiled", action="store_true", help="also time Y's exact path under torch.compile")
ap.add_argument("--out")
args = ap.parse_args()

os.environ["QWEN_WORK"] = args.qwen_work
sys.path.insert(0, args.samebit_src)
sys.path.insert(0, os.path.join(args.samebit_src, "..", "tools", "qwen2"))
sys.path.insert(0, os.path.join(args.y_repo, "tools"))

import numpy as np  # noqa: E402
import pickle  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import _exact_pkg  # noqa: E402
_exact_pkg.add_path()
import exact_ragged_batch as R  # noqa: E402
from y_lang.exact import model as exact_model  # noqa: E402
from samebit.v2 import ops_cuda, qwen2  # noqa: E402

T, W = args.tokens, args.qwen_work
MODEL = os.path.join(W, "model")


def digest(a, dtype):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=dtype).tobytes()).hexdigest()[:16]


tok = AutoTokenizer.from_pretrained(MODEL)
text = "".join(pq.read_table(os.path.join(W, "data/wt2-test.parquet")).column("text").to_pylist())
ids = tok(text)["input_ids"]
windows = [ids[j * T:(j + 1) * T] for j in range(args.windows)]
out = {"gpu": torch.cuda.get_device_name(0), "cpu": platform.processor() or platform.machine(),
       "torch": torch.__version__, "tokens": T, "windows": args.windows, "checks": {}, "times_ms": {}}

# --- samebit -------------------------------------------------------------
with open(os.path.join(W, "qwen2.5-0.5b.samebit.pkl"), "rb") as f:
    P = pickle.load(f)
G = ops_cuda.prepare(P)
sb_same = True
for j, x in enumerate(windows):
    d_cpu = digest(qwen2.forward(P, x), "<i8")
    d_gpu = digest(ops_cuda.asnumpy(qwen2.forward(G, x, ops=ops_cuda)), "<i8")
    out["checks"][f"samebit window {j}"] = {"gpu": d_gpu, "cpu": d_cpu}
    sb_same &= d_gpu == d_cpu
    print(f"samebit window {j}: GPU {d_gpu} CPU {d_cpu} {'IDENTICAL' if d_gpu == d_cpu else 'MISMATCH'}", flush=True)
del P

# --- Y's exact path ------------------------------------------------------
R.MODEL = MODEL
ycpu = R.build("exact", "cpu")
ygpu = R.build("exact", "cuda")
y_same = True
with torch.no_grad():
    for j, x in enumerate(windows):
        lc = ycpu(torch.tensor([x])).logits[0].float().numpy()
        lg = ygpu(torch.tensor([x], device="cuda")).logits[0].float().cpu().numpy()
        d_cpu, d_gpu = digest(lc, "<f4"), digest(lg, "<f4")
        out["checks"][f"Y exact window {j}"] = {"gpu": d_gpu, "cpu": d_cpu}
        y_same &= d_gpu == d_cpu
        print(f"Y exact window {j}: GPU {d_gpu} CPU {d_cpu} {'IDENTICAL' if d_gpu == d_cpu else 'MISMATCH'}", flush=True)
del ycpu

# --- the yardstick -------------------------------------------------------
f16 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
bf16 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()

x = windows[0]
xt = torch.tensor([x], device="cuda")
arms = {
    "samebit integers": lambda: qwen2.forward(G, x, ops=ops_cuda),
    "Y exact, eager": lambda: ygpu(xt).logits,
    "PyTorch float16": lambda: f16(xt).logits,
    "PyTorch bfloat16": lambda: bf16(xt).logits,
}
if args.compiled:
    yc = R.build("exact", "cuda")
    exact_model.compile_forward(yc)
    with torch.no_grad():
        lg = yc(xt).logits[0].float().cpu().numpy()
    d = digest(lg, "<f4")
    out["checks"]["Y exact compiled, window 0"] = {"gpu": d, "eager gpu": out["checks"]["Y exact window 0"]["gpu"]}
    print(f"Y exact compiled window 0: {d} (eager {out['checks']['Y exact window 0']['gpu']})", flush=True)
    y_same &= d == out["checks"]["Y exact window 0"]["gpu"]
    arms["Y exact, compiled"] = lambda: yc(xt).logits


def sync():
    torch.cuda.synchronize()
    ops_cuda.sync()


samples = {k: [] for k in arms}
with torch.no_grad():
    for fn in arms.values():          # warm
        for _ in range(3):
            fn()
    sync()
    rng = random.Random(0)
    for r in range(args.rounds):
        order = list(arms)
        rng.shuffle(order)
        for k in order:
            sync()
            t0 = time.perf_counter()
            for _ in range(args.reps):
                arms[k]()
            sync()
            samples[k].append((time.perf_counter() - t0) / args.reps * 1e3)

print(f"\n{out['gpu']}, {T}-token prompt, batch 1, {args.rounds} shuffled rounds x {args.reps} forwards")
print(f"{'':22s} {'min ms':>8s} {'median':>8s} {'tok/s (min)':>12s}")
for k, v in samples.items():
    out["times_ms"][k] = v
    print(f"{k:22s} {min(v):8.2f} {statistics.median(v):8.2f} {T / min(v) * 1e3:12.0f}")
sb, ye = min(samples["samebit integers"]), min(samples["Y exact, eager"])
print(f"\nY exact eager vs samebit: {sb / ye:.2f}x faster")
if "Y exact, compiled" in samples:
    print(f"Y exact compiled vs samebit: {sb / min(samples['Y exact, compiled']):.2f}x faster")
print(f"bits: samebit GPU == CPU: {sb_same};  Y GPU == CPU (and compiled == eager): {y_same}")
out["samebit_gpu_equals_cpu"], out["y_gpu_equals_cpu"] = bool(sb_same), bool(y_same)
if args.out:
    json.dump(out, open(args.out, "w"), indent=1)
