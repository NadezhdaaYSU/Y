"""Exact LoRA fine-tuning of Qwen2.5-0.5B-Instruct on two-digit
multiplication (T3), with records that compare across devices.

    # train on a GPU, evaluate, keep the adapters
    python3 tools/exact_lora_sft.py --device cuda --steps 300 --record gpu.json --save gpu.pt
    # the same steps on a CPU, then compare
    python3 tools/exact_lora_sft.py --device cpu --steps 20 --record cpu.json
    python3 tools/exact_lora_sft.py --compare gpu.json cpu.json
    # serve the GPU's adapters on the CPU: the same tokens as the GPU's eval
    python3 tools/exact_lora_sft.py --device cpu --load gpu.pt --eval-only --record serve_cpu.json

The task: "a*b" for a, b in 10..99, answered with the number and the end of
turn. 600 of the 8,100 pairs are held out (never trained on) and are the
evaluation; training batches are drawn from the rest by a seeded host RNG,
so every device sees the same batches. A record holds each step's digest of
p(target) (`lora.tp_digest`), the adapters' digest every `--checkpoint`
steps, and each evaluation's accuracy and a digest of every generated
token. The losses printed pass through the host's log and are for reading.
"""

import argparse
import hashlib
import json
import os
import platform
import random
import re
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM = "Answer with the number only."
HELD_OUT = 600


def pairs():
    every = [(a, b) for a in range(10, 100) for b in range(10, 100)]
    test = random.Random(1234).sample(every, HELD_OUT)
    held = set(test)
    return [p for p in every if p not in held], test


def end_of_turn(tok):
    """The chat's end-of-turn token: Qwen's <|im_end|>, Llama 3's <|eot_id|>,
    else the tokenizer's end of sequence."""
    vocab = tok.get_vocab()
    for t in ("<|im_end|>", "<|eot_id|>"):
        if t in vocab:
            return vocab[t]
    return tok.eos_token_id


def ids(tok, text):
    """`text`'s token ids with no special tokens added: the chat template
    writes its own (Llama's tokenizer would add a second <|begin_of_text|>,
    and one before an answer's digits); Qwen's adds none, so its ids, and the
    earlier records, are unchanged."""
    return tok(text, add_special_tokens=False).input_ids


# Llama 3's chat template writes today's date into the system turn ("Today
# Date: 25 Nov 2026", from strftime_now), so the same run's prompts, and so
# its tokens, changed at midnight: the records of 2026-11-24 and those of
# 2026-11-25 differed from step 1 (found 2026-11-26). The date is pinned to
# the one the Llama records here were made under, so they replay on any day;
# templates without a date (Qwen2, SmolLM2) ignore it.
PROMPT_DATE = os.environ.get("Y_EXACT_PROMPT_DATE", "24 Nov 2026")


def chat_text(tok, messages):
    """The chat template's text for `messages`, the turn to answer opened,
    with the date pinned (`PROMPT_DATE`)."""
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   date_string=PROMPT_DATE)


def prompt_ids(tok, a, b):
    text = chat_text(tok, [{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": f"{a}*{b}"}])
    return ids(tok, text)


def example(tok, a, b):
    p = prompt_ids(tok, a, b)
    ans = ids(tok, str(a * b)) + [end_of_turn(tok)]
    return p + ans, [-100] * len(p) + ans


def make_batch(tok, train, step, B, seed):
    rng = random.Random(seed * 1_000_003 + step)
    rows = [example(tok, *rng.choice(train)) for _ in range(B)]
    T = max(len(i) for i, _ in rows)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.tensor([i + [pad] * (T - len(i)) for i, _ in rows])
    labels = torch.tensor([l + [-100] * (T - len(l)) for _, l in rows])
    return ids, labels


def evaluate(model, tok, ad, test, device, batch=100, new=6):
    """Greedy decoding of the held-out problems: (number right, number right
    and then the end of turn, digest of every generated token)."""
    from y_lang.exact.greedy import greedy
    from y_lang.exact.train_lm import reduce_checks
    tok.padding_side = "left"
    end = end_of_turn(tok)
    right = clean = 0
    h = hashlib.sha256()
    bad = []
    ad.ops.bad = bad
    try:
        for i in range(0, len(test), batch):
            part = test[i:i + batch]
            seqs = [prompt_ids(tok, a, b) for a, b in part]
            T = max(len(s) for s in seqs)
            ids = torch.tensor([[0] * (T - len(s)) + s for s in seqs], device=device)
            am = torch.tensor([[0] * (T - len(s)) + [1] * len(s) for s in seqs], device=device)
            out = greedy(model, ids, am, new).cpu()
            h.update(out.to(torch.int64).numpy().tobytes())
            for (a, b), row in zip(part, out.tolist()):
                text = tok.decode(row, skip_special_tokens=True).strip()
                m = re.match(r"\d+", text)
                ok = bool(m) and m.group() == str(a * b)
                right += ok
                digits = ids(tok, str(a * b))
                clean += ok and row[:len(digits) + 1] == digits + [end]
    finally:
        ad.ops.bad = None
    if bad:
        flags, messages = reduce_checks(bad)
        hit = flags.cpu().tolist()
        if any(hit):
            raise OverflowError("; ".join(m for m, f in zip(messages, hit) if f))
    n = len(test)
    return {"right": right, "clean": clean, "n": n, "acc": right / n, "clean_acc": clean / n,
            "tokens": h.hexdigest()[:16]}


def machine(device):
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "cpu": platform.processor() or platform.machine()}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    if torch.device(device).type == "cuda":
        info["gpu"] = torch.cuda.get_device_name()
        info["cuda"] = torch.version.cuda
    import transformers
    info["transformers"] = transformers.__version__
    from y_lang.exact import model as M
    info["gemm"] = M.GEMM
    return info


def run(args):
    from y_lang.exact import load, lora
    torch.backends.cuda.matmul.allow_tf32 = False
    device = args.device
    model, tok = load(MODEL, device)
    ad = lora.attach(model, r=args.r, alpha=args.alpha, seed=args.seed, compile=args.compile)
    train, test = pairs()
    test = test[:args.eval]
    rec = {"config": vars(args), "machine": machine(device), "steps": [], "checkpoints": {},
           "evals": {}, "init": ad.digest()}
    if args.load:
        ad.load(torch.load(args.load))
        rec["loaded"] = ad.digest()

    def do_eval(step):
        t0 = time.time()
        e = evaluate(model, tok, ad, test, device, batch=args.eval_batch)
        e["seconds"] = round(time.time() - t0, 2)
        rec["evals"][str(step)] = e
        print(f"eval @{step}: {e['right']}/{e['n']} right, {e['clean']} clean, tokens "
              f"{e['tokens']} ({e['seconds']} s)", flush=True)

    if args.eval_only:
        do_eval(0)
    else:
        sft = lora.SFT(model, ad, lr=args.lr)
        if args.eval_every and args.eval:
            do_eval(0)
        times = []
        for step in range(1, args.steps + 1):
            ids, labels = make_batch(tok, train, step, args.batch, args.seed)
            if device != "cpu":
                torch.cuda.synchronize()
            t0 = time.time()
            tp = sft.step(ids.to(device), labels.to(device))
            d = lora.tp_digest(tp)
            times.append(time.time() - t0)
            rec["steps"].append({"step": step, "tp": d, "nll": lora.nll(tp), "n": int(tp.numel())})
            if step % args.checkpoint == 0 or step == args.steps:
                sft.verify()
                rec["checkpoints"][str(step)] = ad.digest()
            if step % args.log == 0 or step == args.steps:
                print(f"step {step}: nll {rec['steps'][-1]['nll']:.4f} tp {d} "
                      f"{times[-1]:.3f} s", flush=True)
            if args.eval_every and args.eval and (step % args.eval_every == 0 or step == args.steps):
                do_eval(step)
        sft.verify()
        skip = min(3, len(times) - 1)
        rec["seconds_per_step"] = sum(times[skip:]) / max(len(times) - skip, 1)
        rec["final"] = ad.digest()
        if args.eval and not args.eval_every:
            do_eval(args.steps)
    if args.save:
        torch.save(ad.state(), args.save)
    if args.record:
        with open(args.record, "w") as f:
            json.dump(rec, f, indent=1)
    return rec


def compare(a_path, b_path):
    a, b = (json.load(open(p)) for p in (a_path, b_path))
    out, ok = [], True
    sa = {s["step"]: s["tp"] for s in a["steps"]}
    sb = {s["step"]: s["tp"] for s in b["steps"]}
    common = sorted(set(sa) & set(sb))
    same = [s for s in common if sa[s] == sb[s]]
    out.append(f"p(target) digests: {len(same)}/{len(common)} steps the same")
    ok &= len(same) == len(common)
    ca, cb = a["checkpoints"], b["checkpoints"]
    for s in sorted(set(ca) & set(cb), key=int):
        out.append(f"adapters @{s}: {ca[s]} {cb[s]} {'same' if ca[s] == cb[s] else 'DIFFERENT'}")
        ok &= ca[s] == cb[s]
    ea, eb = a["evals"], b["evals"]
    for s in sorted(set(ea) & set(eb), key=int):
        x, y = ea[s], eb[s]
        same_tokens = x["tokens"] == y["tokens"] and x["n"] == y["n"]
        out.append(f"eval @{s}: {x['right']}/{x['n']} vs {y['right']}/{y['n']}, tokens "
                   f"{x['tokens']} {y['tokens']} {'same' if same_tokens else 'DIFFERENT'}")
        ok &= same_tokens
    if "final" in a and "final" in b and a["config"]["steps"] == b["config"]["steps"]:
        ok &= a["final"] == b["final"]
        out.append(f"final adapters: {a['final']} {b['final']}")
    out.append("SAME" if ok else "DIFFERENT")
    print("\n".join(out))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", type=int, default=HELD_OUT, help="held-out problems evaluated (0: none)")
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--eval-batch", type=int, default=100)
    ap.add_argument("--checkpoint", type=int, default=10)
    ap.add_argument("--log", type=int, default=10)
    ap.add_argument("--record")
    ap.add_argument("--save")
    ap.add_argument("--load")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--compile", action="store_true", help="the exact products compiled (a GPU; the same bits)")
    args = ap.parse_args()
    if args.compare:
        sys.exit(0 if compare(*args.compare) else 1)
    run(args)


if __name__ == "__main__":
    main()
