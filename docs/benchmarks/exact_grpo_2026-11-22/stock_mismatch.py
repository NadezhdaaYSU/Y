"""The control for the exact sampler: how often does a stock model's sampler
disagree with its own trainer about the probability of a sampled token?

The stock Qwen2.5-0.5B-Instruct (not converted), in bfloat16 and in float32,
decodes the GRPO run's first-step prompts over the KV cache (greedy, 6 new
tokens), keeping log p of each chosen token from the float32 softmax of its
logits; then a training forward over prompt + completion (one batch, right
padded, no cache) recomputes log p of the same tokens. Counted: tokens whose
two values are not the same bits, and the largest difference. The exact model
(`y_lang.exact.rl`) is held to 0 at every step of its records.

    python3 stock_mismatch.py --device cuda --out stock_mismatch.json
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "..", "..", "tools")
sys.path.insert(0, TOOLS)

import exact_lora_grpo as G  # noqa: E402
import exact_lora_sft as S  # noqa: E402


@torch.no_grad()
def measure(model, tok, device, dtype_name, step=1):
    train, _ = S.pairs()
    rows, ids, mask = G.prompts_batch(tok, train, step, 8, 8, 0)
    ids, mask = ids.to(device), mask.to(device)
    from transformers import DynamicCache
    cache = DynamicCache()
    pos = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)
    out = model(input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True)
    am = mask
    toks, lps = [], []
    for t in range(6):
        lp = torch.log_softmax(out.logits[:, -1].float(), -1)
        nxt = lp.argmax(-1)
        toks.append(nxt)
        lps.append(lp.gather(1, nxt[:, None]).reshape(-1))
        am = torch.cat([am, torch.ones_like(am[:, :1])], -1)
        pos = pos[:, -1:] + 1
        out = model(input_ids=nxt[:, None], attention_mask=am, position_ids=pos, past_key_values=cache,
                    use_cache=True)
    toks = torch.stack(toks, 1)
    lps = torch.stack(lps, 1)
    # the trainer: each row's real prompt then its six tokens, right padded
    seqs = []
    for i in range(ids.shape[0]):
        prompt = ids[i][mask[i] == 1].tolist()
        seqs.append(prompt + toks[i].tolist())
    T = max(len(s) for s in seqs)
    batch = torch.tensor([s + [0] * (T - len(s)) for s in seqs], device=device)
    logits = model(input_ids=batch).logits.float()
    train_lp = torch.empty_like(lps)
    for i, s in enumerate(seqs):
        P = len(s) - 6
        lp = torch.log_softmax(logits[i, P - 1:P + 5], -1)
        train_lp[i] = lp.gather(1, toks[i][:, None]).reshape(-1)
    differ = (train_lp != lps)
    return {"dtype": dtype_name, "tokens": int(differ.numel()), "differ": int(differ.sum()),
            "max_abs_logp_diff": float((train_lp - lps).abs().max()),
            "max_ratio_minus_1": float((train_lp - lps).exp().sub(1).abs().max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(S.MODEL)
    res = {"machine": S.machine(args.device), "results": []}
    for name, dt in (("bfloat16", torch.bfloat16), ("float32", torch.float32)):
        model = AutoModelForCausalLM.from_pretrained(S.MODEL, dtype=dt, attn_implementation="sdpa").to(args.device).eval()
        r = measure(model, tok, args.device, name)
        print(r, flush=True)
        res["results"].append(r)
        del model
        torch.cuda.empty_cache()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
