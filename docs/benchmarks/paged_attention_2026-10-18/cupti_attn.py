#!/usr/bin/env python3
"""Device-side kernel times (CUPTI, via torch.profiler) for Y's split-K paged
decode attention against FlashInfer, warm L2, ramped clocks.

  cupti_attn.py <splits>:<ptx> [<splits>:<ptx> ...] [--cases b1c1024,b1c4096,...]
               [--rounds 7] [--reps 50]

For every case and every variant: checks exactness against the f32 reference,
then runs `rounds` interleaved rounds of `reps` back-to-back launches under the
profiler and reports, per kernel name, the MINIMUM over rounds of the per-call
mean device duration, plus the end-to-end per-iteration span (first kernel
start to last kernel end, divided by reps) which includes the gap between Y's
two launches. Prints the SM clock per round.
"""
import argparse, os, subprocess, sys, time
sys.path.insert(0, os.environ.get("Y_TESTS", ""))
import torch
import benchmark_y_paged_decode_attention as B
from torch.profiler import profile, ProfilerActivity

ap = argparse.ArgumentParser()
ap.add_argument("variants", nargs="+")
ap.add_argument("--cases", default="b1c1024,b1c4096,b8c1024,b8c4096,b32c1024,b32c4096")
ap.add_argument("--rounds", type=int, default=7)
ap.add_argument("--reps", type=int, default=50)
ap.add_argument("--no-fi", action="store_true")
a = ap.parse_args()
HD, NQH, NKVH, PS = 128, 32, 8, 16


def clock():
    return subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,utilization.gpu", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()


def warm():
    x = torch.randn(4096, 4096, dtype=torch.float16, device=B.DEV)
    for _ in range(30):
        x @ x
    torch.cuda.synchronize()


def measure(run, reps):
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            run()
        torch.cuda.synchronize()
    ev = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    per = {}
    for e in ev:
        per.setdefault(e.name, []).append(e.time_range.elapsed_us())
    t0 = min(e.time_range.start for e in ev); t1 = max(e.time_range.end for e in ev)
    return {k: sum(v) / reps for k, v in per.items()}, (t1 - t0) / reps


fi_mod = None if a.no_fi else B.import_flashinfer()
variants = []
for v in a.variants:
    parts = v.split(":")
    if len(parts) == 2:
        parts = [parts[0], "8", parts[1]]
    variants.append((int(parts[0]), int(parts[1]), parts[2]))

warm()
print("[clock after warm] %s" % clock())
for case in a.cases.split(","):
    batch, ctx = [int(x) for x in case[1:].split("c")]
    lens = [ctx] * batch
    num_pages = max(512, batch * ((ctx + PS - 1) // PS) * 2)
    q, kc, vc, pt, sl, mp = B.make_inputs(HD, NQH, NKVH, PS, lens, num_pages)
    ref = B.reference(q, kc, vc, pt, sl, HD, NQH, NKVH, PS)
    runners = []
    for splits, warps, ptx in variants:
        name = "paged_decode_attention_split_%d_%d_%d_%d_%d_%d" % (HD, NQH, NKVH, PS, splits, warps)
        out = torch.full((batch, NQH, HD), float("nan"), dtype=torch.float16, device=B.DEV)
        r = B.make_y_split_runner(ptx, name, warps * 32, splits, NKVH, q, kc, vc, pt, sl, mp, out)
        r(); torch.cuda.synchronize()
        err = ((out.float() - ref).norm() / ref.norm()).item()
        ok = torch.isfinite(out).all().item() and err <= B.TOL_REL_L2
        runners.append(("%s/%dx%dw" % (os.path.basename(ptx)[:-4], splits, warps), r, err, ok))
    if fi_mod is not None:
        out_fi = None
        fr = B.make_flashinfer_runner(fi_mod, kc, vc, pt, sl, q, HD, NQH, NKVH, PS)
        o = fr(); torch.cuda.synchronize()
        runners.append(("flashinfer", fr, 0.0, True))
    for label, r, err, ok in runners:
        if not ok:
            print("!! %s %s FAILED exactness: rel L2 %.3e" % (case, label, err)); sys.exit(1)
        for _ in range(20):
            r()
    torch.cuda.synchronize()
    best = {lab: (None, 1e9) for lab, *_ in runners}
    clocks = []
    for rnd in range(a.rounds):
        warm()
        clocks.append(clock())
        for off in range(len(runners)):
            lab, r, _, _ = runners[(off + rnd) % len(runners)]
            per, span = measure(r, a.reps)
            if span < best[lab][1]:
                best[lab] = (per, span)
    print("== %s  (clocks,util per round: %s)" % (case, " ".join(clocks)))
    for lab, (per, span) in best.items():
        ks = "  ".join("%s=%.2f" % (k.split("(")[0][-26:], v) for k, v in sorted(per.items()))
        print("   %-28s span/iter %7.2f us | %s" % (lab, span, ks))
