#!/usr/bin/env python3
"""Device-side timing (CUPTI) of Y paged decode attention variants for any
(head_dim, q heads, kv heads); no FlashInfer.

  cupti_gen.py --hd 128 --nqh 32 --nkvh 4 --cases b1c4096,b8c1024 \
      s:16:8:a.ptx p:0:32:b.ptx ...

kind s = split-K (both entries timed), p = single pass. Exactness against
the f32 reference is checked first (NaN-filled Out). Reports per kernel the
min over interleaved rounds of the mean device time, and the total."""
import argparse, os, subprocess, sys
sys.path.insert(0, os.environ.get("Y_TESTS", ""))
import torch
import benchmark_y_paged_decode_attention as B
from torch.profiler import profile, ProfilerActivity

ap = argparse.ArgumentParser()
ap.add_argument("variants", nargs="+")
ap.add_argument("--hd", type=int, default=128); ap.add_argument("--nqh", type=int, default=32)
ap.add_argument("--nkvh", type=int, default=8); ap.add_argument("--ps", type=int, default=16)
ap.add_argument("--cases", default="b1c1024,b1c4096,b8c1024,b32c1024")
ap.add_argument("--rounds", type=int, default=5); ap.add_argument("--reps", type=int, default=40)
a = ap.parse_args()
HD, NQH, NKVH, PS = a.hd, a.nqh, a.nkvh, a.ps


def warm():
    x = torch.randn(4096, 4096, dtype=torch.float16, device=B.DEV)
    for _ in range(30):
        x @ x
    torch.cuda.synchronize()


def clock():
    return subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()


def measure(run, reps):
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            run()
        torch.cuda.synchronize()
    per = {}
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            per.setdefault(e.name, []).append(e.time_range.elapsed_us())
    return {k: sum(v) / reps for k, v in per.items()}


specs = []
for v in a.variants:
    kind, splits, warps, ptx = v.split(":", 3)
    specs.append((kind, int(splits), int(warps), ptx))

warm()
for case in a.cases.split(","):
    batch, ctx = [int(x) for x in case[1:].split("c")]
    lens = [ctx] * batch
    num_pages = max(512, batch * ((ctx + PS - 1) // PS) * 2)
    q, kc, vc, pt, sl, mp = B.make_inputs(HD, NQH, NKVH, PS, lens, num_pages)
    ref = B.reference(q, kc, vc, pt, sl, HD, NQH, NKVH, PS)
    runners = []
    for kind, splits, warps, ptx in specs:
        out = torch.full((batch, NQH, HD), float("nan"), dtype=torch.float16, device=B.DEV)
        if kind == "s":
            name = "paged_decode_attention_split_%d_%d_%d_%d_%d_%d" % (HD, NQH, NKVH, PS, splits, warps)
            r = B.make_y_split_runner(ptx, name, warps * 32, splits, NKVH, q, kc, vc, pt, sl, mp, out)
        else:
            name = "paged_decode_attention_%d_%d_%d_%d_%d" % (HD, NQH, NKVH, PS, warps)
            r = B.make_y_runner(ptx, name, warps * 32, q, kc, vc, pt, sl, mp, out)
        r(); torch.cuda.synchronize()
        err = ((out.float() - ref).norm() / ref.norm()).item()
        if not (torch.isfinite(out).all().item() and err <= B.TOL_REL_L2):
            print("!! %s %s FAILED: rel L2 %.3e" % (case, ptx, err)); sys.exit(1)
        runners.append(("%s/%s%dx%dw" % (os.path.basename(ptx)[:-4], kind, splits, warps), r))
    for _, r in runners:
        for _ in range(10):
            r()
    best = {lab: None for lab, _ in runners}
    clocks = []
    for rnd in range(a.rounds):
        warm(); clocks.append(clock())
        for off in range(len(runners)):
            lab, r = runners[(off + rnd) % len(runners)]
            per = measure(r, a.reps)
            if best[lab] is None or sum(per.values()) < sum(best[lab].values()):
                best[lab] = per
    print("== hd%d %d/%d %s  (SM MHz per round: %s)" % (HD, NQH, NKVH, case, " ".join(clocks)))
    for lab, per in best.items():
        ks = "  ".join("%.2f" % v for _, v in sorted(per.items(), key=lambda kv: len(kv[0])))
        print("   %-34s total %8.2f us | %s" % (lab, sum(per.values()), ks))
