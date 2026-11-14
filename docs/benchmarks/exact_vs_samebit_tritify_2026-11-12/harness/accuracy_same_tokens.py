"""Accuracy on identical evaluation tokens: samebit v0.2 and Y's exact path
against PyTorch float16 and float32 references, Qwen2.5-0.5B-Instruct.

Protocol (samebit's `tools/qwen2/evaluate.py`): WikiText-2-raw-v1 test, all
text joined, tokenized once; window j is ids[256j : 256(j+1)+1], the model
sees the first 256 and is scored on the next token at every position, so
N windows give 256N scored tokens. Every system gets the same ids. Logits are
widened to float64 (samebit: integer codes times `logit_unit`, as its own
evaluator does) and every metric is computed in float64.

Per system: mean per-token NLL and perplexity; against each reference the
paired per-token NLL difference (mean, 95% bootstrap CI over windows), mean
KL(ref || system), top-1 agreement and top-5 overlap; and Y against samebit
directly. Per-token NLLs are saved for re-analysis.

    python accuracy_same_tokens.py SAMEBIT_SRC QWEN_WORK Y_REPO [--windows 40] --out PREFIX
"""
import argparse
import hashlib
import json
import math
import os
import pickle
import sys

ap = argparse.ArgumentParser()
ap.add_argument("samebit_src")
ap.add_argument("qwen_work")
ap.add_argument("y_repo")
ap.add_argument("--windows", type=int, default=40)
ap.add_argument("--boot", type=int, default=10000)
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.environ["QWEN_WORK"] = args.qwen_work
sys.path.insert(0, args.samebit_src)
sys.path.insert(0, os.path.join(args.y_repo, "tools"))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import _exact_pkg  # noqa: E402
_exact_pkg.add_path()
import exact_ragged_batch as R  # noqa: E402
from samebit.v2 import ops_cuda, qwen2  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
W, T, N = args.qwen_work, 256, args.windows
MODEL = os.path.join(W, "model")
tok = AutoTokenizer.from_pretrained(MODEL)
text = "".join(pq.read_table(os.path.join(W, "data/wt2-test.parquet")).column("text").to_pylist())
ids = tok(text)["input_ids"]
assert len(ids) >= N * T + 1, (len(ids), N * T + 1)
eval_ids = ids[:N * T + 1]
ids_sha = hashlib.sha256(np.asarray(eval_ids, dtype="<i8").tobytes()).hexdigest()

with open(os.path.join(W, "qwen2.5-0.5b.samebit.pkl"), "rb") as f:
    P = pickle.load(f)
unit = float(P["logit_unit"])
G = ops_cuda.prepare(P)
del P

# References are sdpa so Y's patch of the eager attention cannot reach them.
f32 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="sdpa").cuda().eval()
f16 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
R.MODEL = MODEL
y = R.build("exact", "cuda")

SYS = ["fp32", "fp16", "Y exact", "samebit"]
nll = {s: np.zeros((N, T)) for s in SYS}
kl = {(r, s): np.zeros((N, T)) for r in ("fp32", "fp16") for s in SYS if s != r}
top1 = {(r, s): np.zeros((N, T), bool) for r in ("fp32", "fp16") for s in SYS if s != r}
top5 = {(r, s): np.zeros((N, T)) for r in ("fp32", "fp16") for s in SYS if s != r}
top1_ys = np.zeros((N, T), bool)

with torch.no_grad():
    for j in range(N):
        win = eval_ids[j * T:(j + 1) * T + 1]
        x, tgt = win[:-1], torch.tensor(win[1:], device="cuda")
        xt = torch.tensor([x], device="cuda")
        logits = {
            "fp32": f32(xt).logits[0].double(),
            "fp16": f16(xt).logits[0].double(),
            "Y exact": y(xt).logits[0].double(),
            "samebit": torch.from_numpy(ops_cuda.asnumpy(qwen2.forward(G, x, ops=ops_cuda)).astype(np.float64) * unit).cuda(),
        }
        lp = {s: torch.log_softmax(v, -1) for s, v in logits.items()}
        am = {s: v.argmax(-1) for s, v in logits.items()}
        t5 = {s: v.topk(5, -1).indices for s, v in logits.items()}
        for s in SYS:
            nll[s][j] = (-lp[s].gather(-1, tgt[:, None])[:, 0]).cpu().numpy()
        for r in ("fp32", "fp16"):
            pr = lp[r].exp()
            for s in SYS:
                if s == r:
                    continue
                kl[(r, s)][j] = (pr * (lp[r] - lp[s])).sum(-1).cpu().numpy()
                top1[(r, s)][j] = (am[r] == am[s]).cpu().numpy()
                top5[(r, s)][j] = torch.tensor([len(set(a.tolist()) & set(b.tolist())) / 5
                                                for a, b in zip(t5[r], t5[s])]).numpy()
        top1_ys[j] = (am["Y exact"] == am["samebit"]).cpu().numpy()
        del logits, lp
        print(f"window {j:2d}: " + "  ".join(f"{s} {math.exp(nll[s][:j + 1].mean()):.3f}" for s in SYS), flush=True)

rng = np.random.default_rng(0)


def boot_ci(per_window):                 # per-window means -> CI of the mean
    idx = rng.integers(0, len(per_window), (args.boot, len(per_window)))
    b = per_window[idx].mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


res = {"model": "Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775",
       "data": "Salesforce/wikitext wikitext-2-raw-v1 test parquet, joined, tokenized once",
       "windows": N, "tokens_per_window": T, "scored_tokens": N * T, "eval_ids_sha256": ids_sha,
       "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "systems": {}, "pairs": {}}
print(f"\n{N} windows x {T} = {N * T} scored tokens, ids sha256 {ids_sha[:16]}")
print(f"{'':10s} {'NLL/token':>10s} {'PPL':>8s}")
for s in SYS:
    m = float(nll[s].mean())
    res["systems"][s] = {"nll_per_token": m, "ppl": math.exp(m), "nll_sd_per_token": float(nll[s].std())}
    print(f"{s:10s} {m:10.5f} {math.exp(m):8.3f}")
for r in ("fp16", "fp32"):
    print(f"\nagainst {r}:  dNLL/token [95% CI over windows]   dPPL     KL/token   top-1   top-5")
    for s in SYS:
        if s == r:
            continue
        d = nll[s] - nll[r]
        lo, hi = boot_ci(d.mean(1))
        e = {"dnll_per_token": float(d.mean()), "dnll_ci95": [lo, hi],
             "dppl": res["systems"][s]["ppl"] - res["systems"][r]["ppl"],
             "kl_per_token": float(kl[(r, s)].mean()), "top1_agree": float(top1[(r, s)].mean()),
             "top5_overlap": float(top5[(r, s)].mean())}
        res["pairs"][f"{s} vs {r}"] = e
        print(f"  {s:9s} {e['dnll_per_token']:+.5f} [{lo:+.5f}, {hi:+.5f}]  {e['dppl']:+7.3f}  "
              f"{e['kl_per_token']:.5f}  {e['top1_agree']:.4f}  {e['top5_overlap']:.4f}")
d = nll["Y exact"] - nll["samebit"]
lo, hi = boot_ci(d.mean(1))
wins = int((d.mean(1) < 0).sum())
res["pairs"]["Y exact vs samebit"] = {"dnll_per_token": float(d.mean()), "dnll_ci95": [lo, hi],
                                      "windows_Y_lower_nll": wins, "top1_agree": float(top1_ys.mean())}
print(f"\nY exact - samebit: dNLL/token {d.mean():+.5f} [95% CI {lo:+.5f}, {hi:+.5f}], "
      f"Y lower on {wins}/{N} windows, same top-1 on {top1_ys.mean():.4f}")
json.dump(res, open(args.out + ".json", "w"), indent=1)
np.savez_compressed(args.out + "_nll.npz", **{s.replace(" ", "_"): v for s, v in nll.items()})
