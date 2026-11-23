"""The speed table of exact RL against the stock arms, from the two exact
dumps, `stock.json` and `vllm.json` (`run.sh`). Each step's time is the
minimum over its repetitions (exact: the two runs; stock and vLLM: the
rounds); the table averages steps `--from` to the last (the first steps
include compiling and recording)."""
import argparse
import json
import statistics
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("exact", nargs=2)
ap.add_argument("stock")
ap.add_argument("vllm", nargs="?")
ap.add_argument("--from", dest="first", type=int, default=4)
args = ap.parse_args()

best = {}                                          # (arm, step) -> seconds


def keep(arm, step, s):
    best[(arm, step)] = min(s, best.get((arm, step), float("inf")))


dumps = [torch.load(p, weights_only=False) for p in args.exact]
for d in dumps:
    for s in d["steps"]:
        keep("exact_sample", s["step"], s["sample_s"])
        keep("exact_train", s["step"], s["train_s"])
a, b = dumps
for x, y in zip(a["steps"], b["steps"]):
    if not torch.equal(x["tokens"], y["tokens"]) or x["keep"] != y["keep"]:
        sys.exit(f"step {x['step']}: the two exact runs did different work")
stock = json.load(open(args.stock))
for t in stock["times"]:
    keep(t["arm"], t["step"], t["s"])
if args.vllm:
    for t in json.load(open(args.vllm))["times"]:
        keep(t["arm"], t["step"], t["s"])

steps = sorted({s for _, s in best if s >= args.first})
arms = sorted({arm for arm, _ in best})


def mean(arm):
    v = [best[(arm, s)] for s in steps if (arm, s) in best]
    return statistics.fmean(v) if len(v) == len(steps) else None


m = {arm: mean(arm) for arm in arms}
print(f"steps {steps[0]} to {steps[-1]}, each the minimum over its repetitions; seconds a step")
for kind in ("sample", "train"):
    print(f"\n{kind}:")
    ex = m[f"exact_{kind}"]
    for arm in arms:
        if kind in arm and m[arm] is not None:
            rel = "" if arm.startswith("exact") else f"   exact takes {ex / m[arm]:.2f}x its time"
            print(f"  {arm:16} {m[arm]:7.3f}{rel}")

samplers = [x for x in arms if "sample" in x and not x.startswith("exact") and m[x] is not None]
trainers = [x for x in arms if "train" in x and not x.startswith("exact") and m[x] is not None]
if samplers and trainers:
    fs, ft = min(samplers, key=m.get), min(trainers, key=m.get)
    ex = m["exact_sample"] + m["exact_train"]
    st = m[fs] + m[ft]
    print(f"\na whole step: exact {ex:.3f} s; stock's fastest sampler ({fs}) plus fastest trainer ({ft}) "
          f"{st:.3f} s; exact takes {ex / st:.2f}x its time")

print("\nchecks:")
agree = {}
for s in stock["steps"].values():
    for arm, v in s.get("agree", {}).items():
        agree.setdefault(arm, []).extend(v)
for arm, v in agree.items():
    print(f"  {arm}: its draw equals the exact token at {100 * statistics.fmean(v):.1f}% of positions")
med = [s["logp_vs_exact"]["median_abs"] for s in stock["steps"].values() if "logp_vs_exact" in s]
if med:
    print(f"  stock log p of the exact tokens against the exact p / l: median |difference| "
          f"{statistics.median(med):.2e} (per-step medians {min(med):.1e} to {max(med):.1e})")
worst = 0.0
for s in stock["steps"].values():
    ls = s.get("mean_logp", {})
    if "train_eager" in ls:
        for arm, v in ls.items():
            for x, y in zip(v, ls["train_eager"]):
                worst = max(worst, abs(x - y))
print(f"  training arms' mean log p of the labelled tokens against eager's, same step and round: "
      f"largest difference {worst:.4f}")
print(f"  compile and record times (first call): {stock.get('compile_s')}")
