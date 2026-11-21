"""Train the tiny character model of `y_lang.exact.train_lm` and record it,
so two machines' runs can be compared bit for bit.

    python3 tools/exact_train_lm.py --record cpu.json --device cpu
    python3 tools/exact_train_lm.py --record gpu.json --device cuda --mm y
    python3 tools/exact_train_lm.py --record f32.json --arm float32
    python3 tools/exact_train_lm.py --compare cpu.json gpu.json

A record holds every step's training loss (read on the host) and a digest of
the probabilities it gave each target token, the held-out
loss every `--eval-every` steps, and a digest of all weights at each of those
points. `--compare` reports the first checkpoint whose weights differ and the
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
        ops, dtype = L.Ops(exact=True, mm=mm, bwd_bits=a.bwd_bits), torch.float64
    else:
        ops, dtype = L.Ops(exact=False), torch.float32
    digests = []

    def log(msg):
        print(msg, flush=True)

    t0 = time.time()
    params = L.init_params(cfg, a.seed, dtype, a.device)
    opt = L.Adam(params, lr=a.lr, exact=ops.exact)
    gen = torch.Generator().manual_seed(a.seed + 1)
    losses, evals, pts = [], [], []
    for step in range(a.steps):
        x, y = data.batch(gen, a.batch, cfg.ctx)
        x, y = x.to(a.device), y.to(a.device)
        _, probs, saved = L.forward(ops, cfg, params, x, y)
        losses.append(L.cross_entropy_on_host(probs, y))
        pts.append(L.target_prob_digest(probs, y))
        opt.step(params, L.backward(ops, cfg, params, probs, y, saved))
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            ev = L.evaluate(ops, cfg, params, data, a.device, a.eval_batches, a.batch)
            evals.append([step + 1, ev])
            digests.append([step + 1, L.digest(params)])
            log(f"step {step + 1:5d}  train {losses[-1]:.4f}  held-out {ev:.4f}  weights {digests[-1][1]}"
                f"  {time.time() - t0:.0f} s")
    rec = {"device": device_name(a.device), "arm": a.arm, "mm": a.mm if a.arm == "exact" else None,
           "torch": torch.__version__, "config": vars(cfg), "seed": a.seed, "lr": a.lr, "batch": a.batch,
           "steps": a.steps, "bwd_bits": a.bwd_bits, "text_sha256": hashlib.sha256(raw).hexdigest(),
           "losses": losses, "target_probs": pts, "evals": evals, "digests": digests, "seconds": time.time() - t0}
    with open(a.record, "w") as f:
        json.dump(rec, f)
    print(f"written to {a.record}")


def compare(pa, pb):
    A, B = (json.load(open(p)) for p in (pa, pb))
    print(f"A: {A['device']}  arm {A['arm']}  mm {A['mm']}\nB: {B['device']}  arm {B['arm']}  mm {B['mm']}")
    same_work = (all(A[k] == B[k] for k in ("arm", "config", "seed", "lr", "batch", "steps", "text_sha256"))
                 and A.get("bwd_bits", 8) == B.get("bwd_bits", 8))
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
    ap.add_argument("--bwd-bits", type=int, default=8,
                    help="width of the backward pass's product operands (exact arm; >8 needs --mm f64)")
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
    if a.device == "cpu" and a.mm != "f64":
        ap.error("--mm y/torch need --device cuda")
    record(a)


if __name__ == "__main__":
    main()
