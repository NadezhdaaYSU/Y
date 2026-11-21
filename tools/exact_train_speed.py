"""Step time of `y_lang.exact.train_lm` in each mode, with a bit check.

    python3 tools/exact_train_speed.py --size tiny --out tiny.json
    python3 tools/exact_train_speed.py --size medium --check 3 --steady 50

For each mode (arm:mode; arm exact, exact32 (float32 activations, Ops(act=
torch.float32)), f32 or bf16 (float32 with bf16 products); mode one of eager, graph, compile, compile+graph),
trains `--check` steps from the same start and reports the weight digest; an
exact mode's digest must equal the CPU's (eager, float64 BLAS), and the
script exits 1 if one does not. Then `--steady` more steps are timed (wall
clock, one device read at the end, so the GPU is never waited on inside).
The first step, which compiles and captures, is timed apart.

Sizes: `tiny` is T2 (width 128, 2 layers, 1,024 tokens a step); `medium` is
width 512, 4 layers, 4,096 tokens (12.8M parameters), the size at which
Inductor's store-then-reload race showed (y_lang.exact.model). float32 is
PyTorch's default (TF32 off).
"""

import argparse
import json
import os
import platform
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from y_lang.exact import train_lm as L  # noqa: E402

TEXT = os.path.expanduser("~/.cache/y-lang/data/tinyshakespeare.txt")
SIZES = {"tiny": (L.Config(), 16),
         "medium": (L.Config(d=512, heads=8, layers=4, ff=2048, ctx=256), 16)}
MODES = ("exact:compile+graph", "f32:eager", "f32:compile+graph")


def make(arm, mode, device, cfg, mm):
    if arm in ("exact", "exact32"):
        ops = L.Ops(exact=True, mm={"y": L.mm_y, "torch": L.mm_torch, "f64": L.mm_f64}[mm],
                    act=torch.float32 if arm == "exact32" else None)
        dtype = torch.float64
    elif arm == "bf16":                          # mixed precision: bf16 products
        ops, dtype = L.Ops(exact=False, float_mm=torch.bfloat16), torch.float32
    else:
        ops, dtype = L.Ops(exact=False), torch.float32
    return L.Trainer(ops, cfg, device, dtype, graph="graph" in mode, compile="compile" in mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=SIZES, default="tiny")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--mm", choices=("y", "torch", "f64"), default="y")
    ap.add_argument("--check", type=int, default=30, help="steps before the weights are compared")
    ap.add_argument("--steady", type=int, default=200, help="steps timed after them")
    ap.add_argument("--text", default=TEXT)
    ap.add_argument("--out", metavar="OUT.json")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg, B = SIZES[a.size]
    data = L.CharData(open(a.text, encoding="utf-8").read())
    gen = torch.Generator().manual_seed(1)
    xs, ys = zip(*(data.batch(gen, B, cfg.ctx) for _ in range(a.check + a.steady)))
    xs, ys = torch.stack(xs), torch.stack(ys)
    major, minor = torch.cuda.get_device_capability()
    rec = {"size": a.size, "config": vars(cfg), "batch": B, "mm": a.mm, "check": a.check,
           "steady": a.steady, "gpu": f"{torch.cuda.get_device_name()} (sm_{major}{minor})",
           "cpu": platform.processor() or platform.machine(), "torch": torch.__version__, "modes": {}}
    print(f"size {a.size}: {cfg}, batch {B}, on {rec['gpu']}, torch {torch.__version__}", flush=True)
    cpu = {}
    bad = False
    for m in a.modes.split(","):
        arm, mode = m.split(":")
        if arm.startswith("exact") and arm not in cpu:
            t = time.perf_counter()
            tr = make(arm, "eager", "cpu", cfg, "f64")
            for i in range(a.check):
                tr.step(xs[i], ys[i])
            tr.verify()
            cpu[arm] = L.digest(tr.params)
            rec[f"cpu_digest_{arm}"] = cpu[arm]
            print(f"cpu reference {arm}: weights@{a.check} {cpu[arm]}  ({time.perf_counter() - t:.0f} s)",
                  flush=True)
        tr = make(arm, mode, "cuda", cfg, a.mm)
        X, Y = xs.cuda(), ys.cuda()
        t = time.perf_counter()
        tr.step(X[0], Y[0])
        torch.cuda.synchronize()
        first = time.perf_counter() - t
        for i in range(1, a.check):
            tr.step(X[i], Y[i])
        tr.verify()
        d = L.digest(tr.params)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for i in range(a.check, a.check + a.steady):
            tr.step(X[i], Y[i])
        tr.verify()
        ms = (time.perf_counter() - t) / a.steady * 1e3
        same = None if not arm.startswith("exact") else d == cpu[arm]
        bad |= same is False
        rec["modes"][m] = {"ms_per_step": ms, "first_step_s": first, "digest": d, "same_as_cpu": same}
        note = "" if same is None else ("  same as CPU" if same else "  *** DIFFERS from CPU")
        print(f"{m:22s} {ms:8.2f} ms/step   first step {first:6.1f} s   weights@{a.check} {d}{note}",
              flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rec, f, indent=1)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
