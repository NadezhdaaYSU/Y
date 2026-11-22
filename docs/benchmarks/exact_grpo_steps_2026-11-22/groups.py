"""How many training problems give GRPO a gradient? For the base model and
the 200-step adapters: 8 sampled tries at each of 512 training problems,
the number right per problem, and what the wrong answers look like."""
import collections, json, random, sys
import os
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, REPO + "/python"); sys.path.insert(0, REPO + "/tools")
import torch
import exact_lora_sft as S, exact_lora_grpo as G
from y_lang.exact import load, lora, rl
torch.backends.cuda.matmul.allow_tf32 = False
model, tok = load(S.MODEL, "cuda")
ad = lora.attach(model, compile=True)
train, _ = S.pairs()
end = tok.convert_tokens_to_ids("<|im_end|>")
probs = random.Random(77).sample(train, 512)
out = {}
for name, path in (("base", None), ("grpo200", REPO + "/docs/benchmark_data/exact_lora/rtx4070tis_grpo200_adapters.pt")):
    if path:
        ad.load(torch.load(path))
    right = []
    wrong = collections.Counter()
    for i in range(0, 512, 8):
        rows = [p for p in probs[i:i + 8] for _ in range(8)]
        seqs = [S.prompt_ids(tok, a, b) for a, b in rows]
        ids = torch.tensor(seqs).cuda(); mask = torch.ones_like(ids)
        toks = rl.sample(model, ad.ops, ids, mask, 6, rl.uniforms(999, i, 64, 6))[0].cpu().tolist()
        for j in range(8):
            (a, b) = probs[i + j]
            k = 0
            for row in toks[j * 8:(j + 1) * 8]:
                ok = G.reward(tok, a, b, row, end)
                k += ok
                if not ok:
                    text = tok.decode(row, skip_special_tokens=False)
                    num = "".join(c for c in text.split("<|im_end|>")[0] if c.isdigit())
                    kind = ("no number" if not num else "ends late" if num == str(a * b)
                            else "off by a digit count" if len(num) != len(str(a * b)) else "wrong digits")
                    wrong[kind] += 1
            right.append(k)
    h = collections.Counter(right)
    big = [k for (a, b), k in zip(probs, right) if a * b >= 1000]
    small = [k for (a, b), k in zip(probs, right) if a * b < 1000]
    res = {"histogram (right of 8 -> problems)": dict(sorted(h.items())),
           "no gradient (0/8 or 8/8)": h[0] + h[8], "of": 512,
           "mean right of 8, products >= 1000": round(sum(big) / len(big), 2), "n big": len(big),
           "mean right of 8, products < 1000": round(sum(small) / max(len(small), 1), 2), "n small": len(small),
           "wrong answers": dict(wrong)}
    out[name] = res
    print(name, json.dumps(res), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
