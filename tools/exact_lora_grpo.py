"""Exact reinforcement learning (T3b) of Qwen2.5-0.5B-Instruct on two-digit
multiplication: LoRA adapters trained by Dr. GRPO on samples drawn by the
exact sampler (`y_lang.exact.rl`), with records that compare across devices.

    python3 tools/exact_lora_grpo.py --device cuda --steps 200 --record gpu.json --save gpu.pt
    python3 tools/exact_lora_grpo.py --device cpu --steps 10 --record cpu.json
    python3 tools/exact_lora_grpo.py --compare gpu.json cpu.json

Each step draws `--prompts` training problems (a seeded host RNG), samples
`--group` completions of each, rewards a completion 1 when it is the right
number followed by the end of turn and 0 otherwise, and makes one update.
The task, the 600 held-out problems and their greedy evaluation are
`exact_lora_sft.py`'s. A record holds, at every step, digests of the
sampled tokens and of the sampler's integers (p and l of every sampled
token), the number of sampled tokens whose trainer integers differ from the
sampler's (0 unless something is wrong; the step's checks then fail too),
the rewards and p(target); the adapters' digest every `--checkpoint` steps;
and each evaluation's accuracy and token digest.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
sys.path.insert(0, HERE)

import exact_lora_sft as S  # noqa: E402


def prompts_batch(tok, train, step, n, group, seed):
    rng = random.Random(seed * 1_000_003 + step + 500_000_000)
    probs = [rng.choice(train) for _ in range(n)]
    rows = [p for p in probs for _ in range(group)]
    seqs = [S.prompt_ids(tok, a, b) for a, b in rows]
    T = max(len(s) for s in seqs)
    ids = torch.tensor([[0] * (T - len(s)) + s for s in seqs])
    mask = torch.tensor([[0] * (T - len(s)) + [1] * len(s) for s in seqs])
    return rows, ids, mask


def reward(tok, a, b, row, end):
    """1 for the right number followed at once by the end of turn."""
    digits = tok(str(a * b)).input_ids
    return int(row[:len(digits) + 1] == digits + [end])


def sha(*ts):
    h = hashlib.sha256()
    for t in ts:
        h.update(t.detach().cpu().to(torch.int64).contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def run(args):
    from y_lang.exact import load, lora, rl
    torch.backends.cuda.matmul.allow_tf32 = False
    device = args.device
    model, tok = load(S.MODEL, device)
    ad = lora.attach(model, r=args.r, alpha=args.alpha, seed=args.seed, compile=args.compile)
    train, test = S.pairs()
    test = test[:args.eval]
    end = tok.convert_tokens_to_ids("<|im_end|>")
    rec = {"config": vars(args), "machine": S.machine(device), "steps": [], "checkpoints": {},
           "evals": {}, "init": ad.digest()}
    if args.load:
        ad.load(torch.load(args.load))
        rec["loaded"] = ad.digest()
    grpo = rl.GRPO(model, ad, lr=args.lr)

    def do_eval(step):
        t0 = time.time()
        e = S.evaluate(model, tok, ad, test, device, batch=args.eval_batch)
        e["seconds"] = round(time.time() - t0, 2)
        rec["evals"][str(step)] = e
        print(f"eval @{step}: {e['right']}/{e['n']} right, {e['clean']} clean, tokens "
              f"{e['tokens']} ({e['seconds']} s)", flush=True)

    if args.eval and args.eval_every:
        do_eval(0)
    times = []
    for step in range(1, args.steps + 1):
        rows, ids, mask = prompts_batch(tok, train, step, args.prompts, args.group, args.seed)
        rand = rl.uniforms(args.seed, step, ids.shape[0], args.new)
        if device != "cpu":
            torch.cuda.synchronize()
        t0 = time.time()
        bad = []
        ad.ops.bad = bad
        try:
            sampled = rl.sample(model, ad.ops, ids.to(device), mask.to(device), args.new, rand)
        finally:
            ad.ops.bad = None
        if bad:
            from y_lang.exact.train_lm import reduce_checks
            flags, messages = reduce_checks(bad)
            if bool(flags.any()):
                raise OverflowError("; ".join(m for m, f in zip(messages, flags.tolist()) if f))
        t1 = time.time()
        toks = sampled[0].cpu().tolist()
        rewards = [reward(tok, a, b, row, end) for (a, b), row in zip(rows, toks)]
        tp, mism = grpo.step(ids.to(device), mask.to(device), sampled, rewards, end, args.group)
        times.append((t1 - t0, time.time() - t1))
        rec["steps"].append({"step": step, "tokens": sha(sampled[0]), "ints": sha(sampled[1], sampled[2]),
                             "mismatch": mism, "reward": sum(rewards) / len(rewards),
                             "tp": lora.tp_digest(tp), "n": int(tp.numel())})
        if step % args.checkpoint == 0 or step == args.steps:
            grpo.verify()
            rec["checkpoints"][str(step)] = ad.digest()
        if step % args.log == 0 or step == args.steps:
            s = rec["steps"][-1]
            print(f"step {step}: reward {s['reward']:.3f} mismatch {mism} tokens {s['tokens']} "
                  f"sample {times[-1][0]:.2f} s train {times[-1][1]:.2f} s", flush=True)
        if args.eval and args.eval_every and (step % args.eval_every == 0 or step == args.steps):
            do_eval(step)
    grpo.verify()
    skip = min(3, len(times) - 1)
    if times:
        rec["seconds_per_step"] = {"sample": sum(t[0] for t in times[skip:]) / max(len(times) - skip, 1),
                                   "train": sum(t[1] for t in times[skip:]) / max(len(times) - skip, 1)}
    rec["final"] = ad.digest()
    rec["mismatches"] = sum(s["mismatch"] for s in rec["steps"])
    if args.save:
        torch.save(ad.state(), args.save)
    if args.record:
        with open(args.record, "w") as f:
            json.dump(rec, f, indent=1)
    return rec


def compare(a_path, b_path):
    a, b = (json.load(open(p)) for p in (a_path, b_path))
    out, ok = [], True
    for key in ("tokens", "ints", "tp"):
        sa = {s["step"]: s[key] for s in a["steps"]}
        sb = {s["step"]: s[key] for s in b["steps"]}
        common = sorted(set(sa) & set(sb))
        same = [s for s in common if sa[s] == sb[s]]
        out.append(f"{key} digests: {len(same)}/{len(common)} steps the same")
        ok &= len(same) == len(common)
    for r in (a, b):
        out.append(f"{r['machine'].get('gpu', r['machine']['cpu'])}: {r['mismatches']} sampled tokens "
                   f"whose trainer integers differ from the sampler's, over {len(r['steps'])} steps")
    ca, cb = a["checkpoints"], b["checkpoints"]
    for s in sorted(set(ca) & set(cb), key=int):
        out.append(f"adapters @{s}: {ca[s]} {cb[s]} {'same' if ca[s] == cb[s] else 'DIFFERENT'}")
        ok &= ca[s] == cb[s]
    ea, eb = a["evals"], b["evals"]
    for s in sorted(set(ea) & set(eb), key=int):
        x, y = ea[s], eb[s]
        same = x["tokens"] == y["tokens"] and x["n"] == y["n"]
        out.append(f"eval @{s}: {x['right']}/{x['n']} vs {y['right']}/{y['n']}, tokens "
                   f"{x['tokens']} {y['tokens']} {'same' if same else 'DIFFERENT'}")
        ok &= same
    out.append("SAME" if ok else "DIFFERENT")
    print("\n".join(out))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--group", type=int, default=8, help="completions per prompt (a power of two)")
    ap.add_argument("--new", type=int, default=6, help="tokens sampled per completion")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", type=int, default=S.HELD_OUT)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--eval-batch", type=int, default=100)
    ap.add_argument("--checkpoint", type=int, default=10)
    ap.add_argument("--log", type=int, default=10)
    ap.add_argument("--record")
    ap.add_argument("--save")
    ap.add_argument("--load")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        sys.exit(0 if compare(*args.compare) else 1)
    if args.group & (args.group - 1):
        ap.error("--group must be a power of two (the group mean is then exact)")
    run(args)


if __name__ == "__main__":
    main()
