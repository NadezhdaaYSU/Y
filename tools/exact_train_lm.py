"""Train the tiny character model of `y_lang.exact.train_lm` and record it,
so two machines' runs can be compared bit for bit.

    python3 tools/exact_train_lm.py --record cpu.json --device cpu
    python3 tools/exact_train_lm.py --record gpu.json --device cuda --mm y
    python3 tools/exact_train_lm.py --record fast.json --device cuda --mm y --graph --compile
    python3 tools/exact_train_lm.py --record f32.json --arm float32
    python3 tools/exact_train_lm.py --compare cpu.json gpu.json

A record holds every step's training loss (read on the host) and a digest of
the probabilities it gave each target token, the held-out
loss every `--eval-every` steps, a digest of all weights at each of those
points, and the steady-state time per step (`ms_per_step`; the first step,
which compiles and captures, is timed apart). `--compare` reports the first checkpoint whose weights differ and the
first step whose target probabilities differ. For the exact arm, two devices must agree
everywhere; the float32 arm is the control, which should not.

Data: Tiny Shakespeare (`--text`, default ~/.cache/y-lang/data/tinyshakespeare.txt,
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt;
its sha256 goes into the record).
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from y_lang.exact import train_lm as L  # noqa: E402

TEXT = os.path.expanduser("~/.cache/y-lang/data/tinyshakespeare.txt")


def device_name(dev):
    if dev == "cuda":
        major, minor = torch.cuda.get_device_capability()
        return f"{torch.cuda.get_device_name()} (sm_{major}{minor})"
    return f"cpu:{platform.machine()} {platform.processor() or ''}".strip()


def record(a):
    torch.backends.cuda.matmul.allow_tf32 = False
    raw = open(a.text, "rb").read()
    data = L.CharData(raw.decode("utf-8"))
    cfg = L.Config()
    if a.arm == "exact":
        mm = {"f64": L.mm_f64, "y": L.mm_y, "torch": L.mm_torch}[a.mm]
        ops, dtype = L.Ops(exact=True, mm=mm, grad_bits=a.grad_bits, bwd_bits=a.bwd_bits), torch.float64
    else:
        ops, dtype = L.Ops(exact=False), torch.float32
    tr = L.Trainer(ops, cfg, a.device, dtype, a.seed, a.lr, graph=a.graph, compile=a.compile)
    gen = torch.Generator().manual_seed(a.seed + 1)
    losses, evals, pts, digests = [], [], [], []
    sync = torch.cuda.synchronize if a.device == "cuda" else (lambda: None)
    t0 = time.time()
    step_s, first_s, done = 0.0, None, 0
    while done < a.steps:
        # One segment, to the next checkpoint: its batches drawn in order from
        # the one generator (the same batches as one draw per step) and moved
        # to the device in one copy.
        n = min(a.eval_every - done % a.eval_every, a.steps - done)
        xs, ys = zip(*(data.batch(gen, a.batch, cfg.ctx) for _ in range(n)))
        xs, ys = torch.stack(xs).to(a.device), torch.stack(ys).to(a.device)
        sync()
        t = time.perf_counter()
        tps = []
        for i in range(n):
            tps.append(tr.step(xs[i], ys[i]))
            if first_s is None:
                sync()
                first_s = time.perf_counter() - t
        tr.verify()                       # fail closed before reading the segment
        step_s += time.perf_counter() - t
        for tp in torch.stack(tps).cpu():
            losses.append(L.nll_on_host(tp))
            pts.append(L.tp_digest(tp))
        done += n
        ev = L.evaluate(ops, cfg, tr.params, data, a.device, a.eval_batches, a.batch)
        evals.append([done, ev])
        digests.append([done, L.digest(tr.params)])
        print(f"step {done:5d}  train {losses[-1]:.4f}  held-out {ev:.4f}  weights {digests[-1][1]}"
              f"  {time.time() - t0:.0f} s", flush=True)
    steady = (step_s - first_s) / max(1, a.steps - 1)
    print(f"steps: {step_s:.1f} s, the first {first_s:.1f} s (compile, capture), then "
          f"{steady * 1e3:.2f} ms/step", flush=True)
    rec = {"device": device_name(a.device), "arm": a.arm, "mm": a.mm if a.arm == "exact" else None,
           "torch": torch.__version__, "config": vars(cfg), "seed": a.seed, "lr": a.lr, "batch": a.batch,
           "steps": a.steps, "grad_bits": a.grad_bits, "bwd_bits": a.bwd_bits, "text_sha256": hashlib.sha256(raw).hexdigest(),
           "losses": losses, "target_probs": pts, "evals": evals, "digests": digests, "seconds": time.time() - t0,
           "graph": a.graph, "compile": a.compile, "step_seconds": step_s, "first_step_seconds": first_s,
           "ms_per_step": steady * 1e3}
    with open(a.record, "w") as f:
        json.dump(rec, f)
    print(f"written to {a.record}")


def compare(pa, pb):
    A, B = (json.load(open(p)) for p in (pa, pb))
    print(f"A: {A['device']}  arm {A['arm']}  mm {A['mm']}\nB: {B['device']}  arm {B['arm']}  mm {B['mm']}")
    same_work = (all(A[k] == B[k] for k in ("arm", "config", "seed", "lr", "batch", "steps", "text_sha256"))
                 and A.get("bwd_bits", 8) == B.get("bwd_bits", 8)
                 and A.get("grad_bits", A.get("bwd_bits", 8)) == B.get("grad_bits", B.get("bwd_bits", 8)))
    if not same_work:
        print("  *** not the same workload (arm, config, seed, lr, batch, steps, text or "
              "backward width differ)")
        return 2
    diff_ck = next((sa for (sa, da), (_, db) in zip(A["digests"], B["digests"]) if da != db), None)
    # p(target) bits per step: device-independent if training is. The losses
    # themselves pass through each host's log, so they are shown, not compared.
    diff_step = next((i for i, (x, y) in enumerate(zip(A["target_probs"], B["target_probs"])) if x != y), None)
    n = len(A["digests"])
    same = sum(da == db for (_, da), (_, db) in zip(A["digests"], B["digests"]))
    print(f"  weights identical at {same}/{n} checkpoints"
          + ("" if diff_ck is None else f"; first differ at step {diff_ck}"))
    print("  p(target) identical at every step" if diff_step is None
          else f"  p(target) first differs at step {diff_step + 1}")
    print(f"  final held-out loss: A {A['evals'][-1][1]:.6f}  B {B['evals'][-1][1]:.6f}")
    return 0 if diff_ck is None and diff_step is None else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", metavar="OUT.json")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--arm", choices=("exact", "float32"), default="exact")
    ap.add_argument("--mm", choices=("f64", "y", "torch"), default="f64",
                    help="the exact arm's linear-layer GEMM: float64 BLAS (exact for int8 "
                         "operands), Y's int8 kernel, or torch._int_mm")
    ap.add_argument("--grad-bits", type=int, default=12,
                    help="width of the backward pass's gradient operands (exact arm)")
    ap.add_argument("--bwd-bits", type=int, default=8,
                    help="width of the activations and weights the gradients meet in the backward pass")
    ap.add_argument("--graph", action="store_true", help="replay each step as one CUDA graph")
    ap.add_argument("--compile", action="store_true", help="fuse the step with torch.compile")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--text", default=TEXT)
    a = ap.parse_args()
    if a.compare:
        sys.exit(compare(*a.compare))
    if not a.record:
        ap.error("--record or --compare")
    if a.device == "cpu" and (a.mm != "f64" or a.graph or a.compile):
        ap.error("--mm y/torch, --graph and --compile need --device cuda")
    record(a)


if __name__ == "__main__":
    main()
