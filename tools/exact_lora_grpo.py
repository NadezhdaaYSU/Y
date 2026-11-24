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

Larger runs:

    # worked answers: the model writes its steps, the reward reads the last
    # "Answer: N"; 32 problems sampled a step, the first 8 whose 8 tries are
    # neither all right nor all wrong trained on (dynamic sampling), in
    # pieces of 2048 tokens
    python3 tools/exact_lora_grpo.py --mode steps --new 128 --sample-prompts 32 \
        --chunk-tokens 2048 --record steps.json
    # another model, three-digit problems
    python3 tools/exact_lora_grpo.py --model Qwen/Qwen2.5-7B-Instruct --digits 3 ...

With `--sample-prompts`, a step whose sampled groups are all flat makes no
update; the record says how many groups were all right, all wrong and kept.
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
sys.path.insert(0, HERE)

import exact_lora_sft as S  # noqa: E402


STEPS_SYSTEM = "Solve the problem step by step, then end with 'Answer: <number>'."


class Task:
    """The problems, prompts and rewards of one setting. `mode` "number"
    (answer with the number only: `exact_lora_sft.py`'s task) or "steps"
    (work it out, end with "Answer: N"); `digits` per factor. Two digits
    and "number" are exactly the earlier runs' problems, prompts and
    rewards."""

    def __init__(self, tok, mode="number", digits=2):
        if mode not in ("number", "steps"):
            raise ValueError(f"--mode must be number or steps, not {mode!r}")
        self.tok, self.mode, self.digits = tok, mode, digits
        lo, hi = 10 ** (digits - 1), 10 ** digits
        every = [(a, b) for a in range(lo, hi) for b in range(lo, hi)]
        self.test = random.Random(1234).sample(every, S.HELD_OUT)
        held = set(self.test)
        self.train = [p for p in every if p not in held]
        self.end = S.end_of_turn(tok)
        self.system = S.SYSTEM if mode == "number" else STEPS_SYSTEM

    def prompt(self, a, b):
        text = self.tok.apply_chat_template([{"role": "system", "content": self.system},
                                             {"role": "user", "content": f"{a}*{b}"}],
                                            tokenize=False, add_generation_prompt=True)
        return S.ids(self.tok, text)

    def reward(self, a, b, row):
        if self.mode == "number":
            return reward(self.tok, a, b, row, self.end)
        if self.end not in row:
            return 0
        text = self.tok.decode(row[:row.index(self.end)], skip_special_tokens=True)
        found = re.findall(r"Answer:\s*\**\s*([\d,]+)", text)
        return int(bool(found) and found[-1].replace(",", "") == str(a * b))


def evaluate(model, task, ad, device, n, new, batch=100):
    """Greedy over the first `n` held-out problems, `new` tokens each:
    (number rewarded 1, digest of every generated token)."""
    from y_lang.exact.greedy import greedy
    from y_lang.exact.train_lm import reduce_checks
    right = 0
    h = hashlib.sha256()
    bad = []
    ad.ops.bad = bad
    try:
        for i in range(0, n, batch):
            part = task.test[i:min(i + batch, n)]
            seqs = [task.prompt(a, b) for a, b in part]
            T = max(len(s) for s in seqs)
            ids = torch.tensor([[0] * (T - len(s)) + s for s in seqs], device=device)
            am = torch.tensor([[0] * (T - len(s)) + [1] * len(s) for s in seqs], device=device)
            out = greedy(model, ids, am, new).cpu()
            h.update(out.to(torch.int64).numpy().tobytes())
            right += sum(task.reward(a, b, row) for (a, b), row in zip(part, out.tolist()))
    finally:
        ad.ops.bad = None
    if bad:
        flags, messages = reduce_checks(bad)
        if bool(flags.any()):
            raise OverflowError("; ".join(m for m, f in zip(messages, flags.tolist()) if f))
    return {"right": right, "clean": right, "n": n, "acc": right / n, "tokens": h.hexdigest()[:16]}


def prompts_batch(tok, train, step, n, group, seed, prompt=None):
    prompt = prompt or (lambda a, b: S.prompt_ids(tok, a, b))
    rng = random.Random(seed * 1_000_003 + step + 500_000_000)
    probs = [rng.choice(train) for _ in range(n)]
    rows = [p for p in probs for _ in range(group)]
    seqs = [prompt(a, b) for a, b in rows]
    T = max(len(s) for s in seqs)
    ids = torch.tensor([[0] * (T - len(s)) + s for s in seqs])
    mask = torch.tensor([[0] * (T - len(s)) + [1] * len(s) for s in seqs])
    return rows, ids, mask


def reward(tok, a, b, row, end):
    """1 for the right number followed at once by the end of turn."""
    digits = S.ids(tok, str(a * b))
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
    model, tok = load(args.model, device)
    ad = lora.attach(model, r=args.r, alpha=args.alpha, seed=args.seed, compile=args.compile)
    task = Task(tok, args.mode, args.digits)
    train, test = task.train, task.test[:args.eval]
    end = task.end
    old = args.mode == "number" and args.digits == 2 and args.model == S.MODEL
    rec = {"config": vars(args), "machine": S.machine(device), "steps": [], "checkpoints": {},
           "evals": {}, "init": ad.digest()}
    if args.load:
        ad.load(torch.load(args.load))
        rec["loaded"] = ad.digest()
    grpo = rl.GRPO(model, ad, lr=args.lr)

    def do_eval(step):
        t0 = time.time()
        if old and args.eval_new == 6:
            e = S.evaluate(model, tok, ad, test, device, batch=args.eval_batch)
        else:
            e = evaluate(model, task, ad, device, len(test), args.eval_new, batch=args.eval_batch)
        e["seconds"] = round(time.time() - t0, 2)
        rec["evals"][str(step)] = e
        print(f"eval @{step}: {e['right']}/{e['n']} right, {e['clean']} clean, tokens "
              f"{e['tokens']} ({e['seconds']} s)", flush=True)

    def save_partial(step):
        """The record so far, written at each checkpoint and evaluation, so a
        run stopped early (a time limit) keeps its steps; the last write is
        the whole record, without `partial_through`."""
        if args.record:
            part = dict(rec, partial_through=step, mismatches=sum(s["mismatch"] for s in rec["steps"]))
            with open(args.record, "w") as f:
                json.dump(part, f, indent=1)

    if args.eval and args.eval_every:
        do_eval(0)
        save_partial(0)
    times = []
    samplers = {}
    dump = [] if args.dump else None
    for step in range(1, args.steps + 1):
        rows, ids, mask = prompts_batch(tok, train, step, args.sample_prompts or args.prompts, args.group,
                                        args.seed, task.prompt)
        rand = rl.uniforms(args.seed, step, ids.shape[0], args.new)
        if device != "cpu":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        bad = []
        ad.ops.bad = bad
        try:
            if args.graphed:
                key = tuple(ids.shape)
                if key not in samplers:
                    samplers[key] = rl.graphed_sampler(model, ad.ops, *key, args.new, rows=args.sample_rows,
                                                       stop=end if args.early_stop else None)
                sampled = samplers[key].run(ids.to(device), mask.to(device), rand.to(device))
            else:
                sampled = rl.sample(model, ad.ops, ids.to(device), mask.to(device), args.new, rand,
                                    rows=args.sample_rows, stop=end if args.early_stop else None)
        finally:
            ad.ops.bad = None
        if bad:
            from y_lang.exact.train_lm import reduce_checks
            flags, messages = reduce_checks(bad)
            if bool(flags.any()):
                raise OverflowError("; ".join(m for m, f in zip(messages, flags.tolist()) if f))
        t1 = time.time()
        toks = sampled[0].cpu().tolist()
        rewards = [task.reward(a, b, row) for (a, b), row in zip(rows, toks)]
        G = args.group
        sums = [sum(rewards[i:i + G]) for i in range(0, len(rewards), G)]
        groups = {"sampled": len(sums), "all_right": sums.count(G), "all_wrong": sums.count(0)}
        keep = list(range(len(sums)))
        if args.sample_prompts:
            keep = [g for g, k in enumerate(sums) if 0 < k < G][:args.prompts]
        groups["kept"] = len(keep)
        entry = {"step": step, "tokens": sha(sampled[0]), "ints": sha(sampled[1], sampled[2]),
                 "reward": sum(rewards) / len(rewards), "groups": groups}
        if keep:
            r = torch.tensor([g * G + j for g in keep for j in range(G)], device=device)
            sub = tuple(t[r] for t in sampled)
            tp, mism = grpo.step(ids.to(device)[r], mask.to(device)[r], sub,
                                 [rewards[i] for i in r.tolist()], end, G, chunk_tokens=args.chunk_tokens)
            entry.update(mismatch=mism, tp=lora.tp_digest(tp), n=int(tp.numel()))
        else:
            mism = 0
            entry.update(mismatch=0, tp=None, n=0)
        if not args.sample_prompts:
            del entry["groups"]["kept"]
        times.append((t1 - t0, time.time() - t1))
        entry.update(sample_s=round(times[-1][0], 3), train_s=round(times[-1][1], 3))
        if device != "cpu":
            entry["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        if dump is not None:
            dump.append({"step": step, "ids": ids, "mask": mask, "tokens": sampled[0].cpu(),
                         "p": sampled[1].cpu(), "l": sampled[2].cpu(), "rewards": rewards,
                         "keep": keep, "n": entry["n"], "sample_s": times[-1][0], "train_s": times[-1][1]})
        rec["steps"].append(entry)
        if step % args.checkpoint == 0 or step == args.steps:
            grpo.verify()
            rec["checkpoints"][str(step)] = ad.digest()
            save_partial(step)
        if step % args.log == 0 or step == args.steps:
            s = rec["steps"][-1]
            print(f"step {step}: reward {s['reward']:.3f} mismatch {mism} tokens {s['tokens']} "
                  f"groups {s['groups']} sample {times[-1][0]:.2f} s train {times[-1][1]:.2f} s"
                  + (f" peak {s['peak_gb']} GB" if "peak_gb" in s else ""), flush=True)
        if args.eval and args.eval_every and (step % args.eval_every == 0 or step == args.steps):
            do_eval(step)
            save_partial(step)
    grpo.verify()
    skip = min(3, len(times) - 1)
    if times:
        rec["seconds_per_step"] = {"sample": sum(t[0] for t in times[skip:]) / max(len(times) - skip, 1),
                                   "train": sum(t[1] for t in times[skip:]) / max(len(times) - skip, 1)}
    rec["final"] = ad.digest()
    rec["mismatches"] = sum(s["mismatch"] for s in rec["steps"])
    if args.save:
        torch.save(ad.state(), args.save)
    if args.dump:
        torch.save({"config": vars(args), "end": end, "group": args.group, "steps": dump,
                    "machine": rec["machine"]}, args.dump)
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
    ap.add_argument("--model", default=S.MODEL)
    ap.add_argument("--mode", default="number", choices=("number", "steps"))
    ap.add_argument("--digits", type=int, default=2)
    ap.add_argument("--eval-new", type=int, default=6, help="tokens generated per held-out problem")
    ap.add_argument("--sample-prompts", type=int, default=0,
                    help="problems sampled a step; the first --prompts with mixed rewards are trained on")
    ap.add_argument("--early-stop", action="store_true",
                    help="stop sampling once every row has ended (checked every 16 tokens)")
    ap.add_argument("--graphed", action="store_true",
                    help="sample with each decode step replayed from a CUDA graph (the same bits)")
    ap.add_argument("--sample-rows", type=int, default=64,
                    help="rows of the sampler's vocabulary softmax and draw at a time (the same bits for any)")
    ap.add_argument("--chunk-tokens", type=int, default=None, help="train in pieces of at most this many tokens")
    ap.add_argument("--dump", help="save each step's prompts, sampled tokens, integers, rewards, kept groups "
                    "and times (torch.save), for a baseline to repeat the same work")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        sys.exit(0 if compare(*args.compare) else 1)
    if args.group & (args.group - 1):
        ap.error("--group must be a power of two (the group mean is then exact)")
    run(args)


if __name__ == "__main__":
    main()
