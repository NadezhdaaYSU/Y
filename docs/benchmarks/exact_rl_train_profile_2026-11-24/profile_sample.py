"""Where the exact graphed sampler's time goes, on the GPU.

Builds the graphed sampler for one step of the exact run's dump (its
prompts, its random integers, adapters at their start), checks its tokens
against the dump's step 1, then times a run bare and profiles one: the
kernels inside the replayed graphs by GPU time, per decode step.

    python3 profile_sample.py DUMP [--out profile_sample.txt]
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path[:0] = [os.path.join(REPO, "tools"), os.path.join(REPO, "python")]

from torch.profiler import ProfilerActivity, profile  # noqa: E402

from y_lang.exact import load, lora, rl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dump")
ap.add_argument("--out")
ap.add_argument("--rows", type=int, default=64, help="the softmax and draw's rows at a time")
args = ap.parse_args()

d = torch.load(args.dump, weights_only=False)
cfg = d["config"]
s = d["steps"][0]
model, tok = load(cfg["model"], "cuda")
ad = lora.attach(model, r=cfg["r"], alpha=cfg["alpha"], seed=cfg["seed"], compile=True)
ids, mask = s["ids"].cuda(), s["mask"].cuda()
rand = rl.uniforms(cfg["seed"], s["step"], ids.shape[0], cfg["new"]).cuda()
sampler = rl.graphed_sampler(model, ad.ops, *ids.shape, cfg["new"], rows=args.rows, stop=d["end"])
lines = []


def say(x=""):
    print(x, flush=True)
    lines.append(x)


def run():
    ad.ops.bad = []
    try:
        out = sampler.run(ids, mask, rand)
        torch.cuda.synchronize()
    finally:
        ad.ops.bad = None
    return out


t0 = time.perf_counter()
out = run()
say(f"first run (compiles and records): {time.perf_counter() - t0:.1f} s")
same = torch.equal(out[0].cpu(), s["tokens"]) and torch.equal(out[1].cpu(), s["p"])
say(f"tokens and p: {'the dump step 1' if same else 'NOT the dump'}; {out[0].shape[1]} tokens")
if not same:
    sys.exit(1)
times = []
for _ in range(3):
    t0 = time.perf_counter()
    run()
    times.append(time.perf_counter() - t0)
n = out[0].shape[1]
say(f"bare runs: {', '.join(f'{t:.3f}' for t in times)} s; {1e3 * min(times) / n:.2f} ms a token")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    run()
ev = prof.events()
kernels = [e for e in ev if e.device_type == torch.autograd.DeviceType.CUDA]
busy = sum(e.time_range.elapsed_us() for e in kernels) / 1e3
say(f"profiled run: {len(kernels)} kernels, GPU busy {busy:.1f} ms, {busy / n:.2f} ms a token, "
    f"{len(kernels) / n:.0f} kernels a token")
launches = [e.time_range.start for e in ev if e.device_type == torch.autograd.DeviceType.CPU
            and e.name == "cudaGraphLaunch"]
if launches:
    first = min(launches)
    pre = [e for e in kernels if e.time_range.start < first]
    say(f"before the first graph replay (the prefill and the first draw): {len(pre)} kernels, "
        f"GPU busy {sum(e.time_range.elapsed_us() for e in pre) / 1e3:.1f} ms")
    pby = defaultdict(lambda: [0.0, 0])
    for e in pre:
        pby[e.name][0] += e.time_range.elapsed_us() / 1e3
        pby[e.name][1] += 1
    say("  its kernels by GPU time:")
    for name, (t, c) in sorted(pby.items(), key=lambda kv: -kv[1][0])[:25]:
        say(f"  {t:9.2f} ms {c:6d}x  {name[:120]}")
by = defaultdict(lambda: [0.0, 0])
for e in kernels:
    by[e.name][0] += e.time_range.elapsed_us() / 1e3
    by[e.name][1] += 1
say("\nkernels by GPU time (ms over the run, calls):")
for name, (t, c) in sorted(by.items(), key=lambda x: -x[1][0])[:40]:
    say(f"  {t:9.2f} ms {c:6d}x {100 * t / busy:5.1f}%  {name[:120]}")
if args.out:
    open(args.out, "w").write("\n".join(lines) + "\n")
