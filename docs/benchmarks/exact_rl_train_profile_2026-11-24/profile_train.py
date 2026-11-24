"""Where an exact GRPO training step's time goes, on the GPU.

Replays the exact run's training steps from its dump (`exact_lora_grpo.py
--dump`) in order from the same adapters, so every step is the record's
(its p(target) digest is compared), then times one step bare and profiles
the next: each phase's wall time, the GPU's busy time, the number of kernels
and the kernels that take the most time.

    python3 profile_train.py DUMP RECORD [--steps 5] [--shapes] [--out profile.txt]
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path[:0] = [os.path.join(REPO, "tools"), os.path.join(REPO, "python")]

import json  # noqa: E402

from y_lang.exact import load, lora, rl  # noqa: E402
from torch.profiler import ProfilerActivity, profile, record_function  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dump")
ap.add_argument("record")
ap.add_argument("--steps", type=int, default=5, help="the last is profiled, the one before timed bare")
ap.add_argument("--out")
ap.add_argument("--eager", action="store_true", help="products eager too (no --compile)")
ap.add_argument("--shapes", action="store_true", help="also each product's GPU time by operand shapes "
                "(the profiler's record_shapes)")
ap.add_argument("--chunk-tokens", type=int, help="pieces of this many tokens, not the run's (a different "
                "exact function: p(target) is then not compared with the record)")
args = ap.parse_args()

d = torch.load(args.dump, weights_only=False)
rec = json.load(open(args.record))
cfg = d["config"]
G, end = d["group"], d["end"]
model, tok = load(cfg["model"], "cuda")
ad = lora.attach(model, r=cfg["r"], alpha=cfg["alpha"], seed=cfg["seed"], compile=not args.eager)
grpo = rl.GRPO(model, ad, lr=cfg["lr"])


def mark(obj, name, label):
    f = getattr(obj, name)

    def wrapped(*a, **k):
        with record_function(label):
            return f(*a, **k)
    setattr(obj, name, wrapped)


mark(lora.SFT, "forward", "phase:forward")
mark(lora.SFT, "logits_at", "phase:lm_head")
mark(lora, "softmax_rows", "phase:vocab_softmax")
mark(rl, "training_batch", "phase:host_batch")
mark(torch.Tensor, "backward", "phase:backward")
mark(grpo.opt, "update", "phase:adam")


def mark_fn(cls, label):
    for which in ("forward", "backward"):
        f = getattr(cls, which)

        def wrapped(*a, _f=f, _l=f"{label}.{which}", **k):
            with record_function("part:" + _l):
                return _f(*a, **k)
        setattr(cls, which, staticmethod(wrapped))


for cls, label in ((lora._FrozenFn, "frozen_linear"), (lora._LoRAFn, "lora_linear"), (lora._NormFn, "norm"),
                   (lora._SiLUFn, "silu"), (lora._AttnFn, "attention")):
    mark_fn(cls, label)


def train(s):
    r = torch.tensor([g * G + j for g in s["keep"] for j in range(G)])
    sub = tuple(t.cuda()[r] for t in (s["tokens"], s["p"], s["l"]))
    return grpo.step(s["ids"].cuda()[r], s["mask"].cuda()[r], sub, [s["rewards"][i] for i in r.tolist()],
                     end, G, chunk_tokens=args.chunk_tokens or cfg["chunk_tokens"])


lines = []


def say(x=""):
    print(x, flush=True)
    lines.append(x)


want = {e["step"]: e for e in rec["steps"]}
for s in d["steps"][:args.steps]:
    st = s["step"]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if st == args.steps:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=args.shapes) as prof:
            tp, mism = train(s)
            torch.cuda.synchronize()
    else:
        tp, mism = train(s)
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    same = args.chunk_tokens is not None or lora.tp_digest(tp) == want[st]["tp"]
    say(f"step {st}: {dt:.3f} s, {int(tp.numel())} tokens, mismatch {mism}, p(target) "
        f"{'the record' if same else 'NOT the record'}")
    if (not same or mism) and not args.chunk_tokens:
        sys.exit("the replay is not the record")
if not args.chunk_tokens:
    # (Other pieces train other adapters, so from step 2 the dump's sampled
    # integers are not this trainer's: a timing run, not a replay.)
    grpo.verify()

ev = prof.events()
kernels = [e for e in ev if e.device_type == torch.autograd.DeviceType.CUDA
           and not e.name.startswith(("phase:", "part:", "## "))]   # "## Call CompiledFxGraph": spans, not kernels
busy = sum(e.time_range.elapsed_us() for e in kernels) / 1e6
say(f"\nprofiled step: {len(kernels)} GPU kernels and copies, GPU busy {busy:.3f} s "
    f"(sum of their times; the step's wall time is above)")
phases = defaultdict(lambda: [0.0, 0])
for e in ev:
    if e.name.startswith("phase:") and e.device_type == torch.autograd.DeviceType.CPU:
        phases[e.name][0] += e.cpu_time_total / 1e6
        phases[e.name][1] += 1
say("\nphases (wall time on the host, calls):")
for k, (t, n) in sorted(phases.items(), key=lambda x: -x[1][0]):
    say(f"  {k[6:]:15} {t:7.3f} s  {n} calls")
# kernel launches (any thread) inside each phase's span on the host
spans = sorted((e.time_range.start, e.time_range.end, e.name[6:]) for e in ev if e.name.startswith("phase:"))
launch = defaultdict(int)
for e in ev:
    if e.device_type == torch.autograd.DeviceType.CPU and "Launch" in e.name:
        for a, b, name in spans:
            if a <= e.time_range.start <= b:
                launch[name] += 1
                break
say(f"\nkernel launches by phase: {dict(launch)}")
parts = sorted((e.time_range.start, e.time_range.end, e.name[5:]) for e in ev
               if e.name.startswith("part:") and e.device_type == torch.autograd.DeviceType.CPU)
plaunch, ptime, pcalls = defaultdict(int), defaultdict(float), defaultdict(int)
for a, b, name in parts:
    ptime[name] += (b - a) / 1e6
    pcalls[name] += 1
import bisect
starts = [a for a, _, _ in parts]
for e in ev:
    if e.device_type == torch.autograd.DeviceType.CPU and "Launch" in e.name:
        i = bisect.bisect_right(starts, e.time_range.start) - 1
        if i >= 0 and parts[i][0] <= e.time_range.start <= parts[i][1]:
            plaunch[parts[i][2]] += 1
# GPU time of the kernels each part launched (the profiler attaches a kernel
# to the host op that launched it; a part's total includes its children's).
pgpu = defaultdict(float)
for e in ev:
    if e.name.startswith("part:") and e.device_type == torch.autograd.DeviceType.CPU:
        pgpu[e.name[5:]] += e.device_time_total / 1e6
say("\nparts (host time inside, GPU time of their kernels, calls, kernel launches):")
for name in sorted(ptime, key=lambda n: -ptime[n]):
    say(f"  {name:26} {ptime[name]:7.3f} s {pgpu[name]:7.3f} s GPU {pcalls[name]:6d} calls {plaunch[name]:8d} launches "
        f"({plaunch[name] / max(pcalls[name], 1):.0f} a call)")
if args.shapes:
    prods = defaultdict(lambda: [0.0, 0])
    for e in ev:
        if e.device_type == torch.autograd.DeviceType.CPU and e.name.startswith("y_lang::"):
            k = (e.name, str(e.input_shapes))
            prods[k][0] += e.device_time_total / 1e6
            prods[k][1] += 1
    say("\nY's ops by operand shapes (GPU time of their kernels, calls):")
    for (name, shp), (t, n) in sorted(prods.items(), key=lambda x: -x[1][0])[:40]:
        say(f"  {t * 1e3:9.2f} ms {n:6d}x  {name} {shp[:120]}")
by = defaultdict(lambda: [0.0, 0])
for e in kernels:
    by[e.name][0] += e.time_range.elapsed_us() / 1e6
    by[e.name][1] += 1
say("\nkernels by GPU time:")
for name, (t, n) in sorted(by.items(), key=lambda x: -x[1][0])[:30]:
    say(f"  {t * 1e3:9.2f} ms {n:6d}x  {name[:110]}")
counts = sorted(by.items(), key=lambda x: -x[1][1])[:15]
say("\nkernels by count:")
for name, (t, n) in counts:
    say(f"  {n:6d}x {t * 1e3:9.2f} ms  {name[:110]}")
if args.out:
    open(args.out, "w").write("\n".join(lines) + "\n")
