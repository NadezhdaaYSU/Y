"""Base accuracy when the model may show its work: greedy, held-out problems."""
import re, sys, time
import os
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, REPO + "/python"); sys.path.insert(0, REPO + "/tools")
import torch
import exact_lora_sft as S
from y_lang.exact import load
from y_lang.exact.greedy import greedy
torch.backends.cuda.matmul.allow_tf32 = False
model, tok = load(S.MODEL, "cuda")
_, test = S.pairs()
test = test[:200]
SYSTEMS = {
    "steps": "Solve the problem step by step, then end with 'Answer: <number>'.",
    "short steps": "Multiply step by step in a few short lines, then end with 'Answer: <number>'.",
}
NEW = int(sys.argv[1]) if len(sys.argv) > 1 else 160
tok.padding_side = "left"
for name, system in SYSTEMS.items():
    right = done = 0
    lens = []
    t0 = time.time()
    for i in range(0, len(test), 100):
        part = test[i:i + 100]
        seqs = [tok(tok.apply_chat_template([{"role": "system", "content": system},
                                             {"role": "user", "content": f"{a}*{b}"}],
                                            tokenize=False, add_generation_prompt=True)).input_ids for a, b in part]
        T = max(map(len, seqs))
        ids = torch.tensor([[0] * (T - len(s)) + s for s in seqs]).cuda()
        am = torch.tensor([[0] * (T - len(s)) + [1] * len(s) for s in seqs]).cuda()
        out = greedy(model, ids, am, NEW).cpu().tolist()
        end = tok.convert_tokens_to_ids("<|im_end|>")
        for (a, b), row in zip(part, out):
            n = row.index(end) + 1 if end in row else len(row)
            lens.append(n)
            text = tok.decode(row[:n], skip_special_tokens=True)
            m = re.findall(r"Answer:\s*\**\s*([\d,]+)", text)
            done += bool(m)
            right += bool(m) and m[-1].replace(",", "") == str(a * b)
    lens.sort()
    print(f"{name}: {right}/{len(test)} right, {done} gave an answer, tokens median {lens[len(lens)//2]} "
          f"max {lens[-1]}, hit the limit {sum(l == NEW for l in lens)} ({time.time() - t0:.0f} s)", flush=True)
print("example:", tok.decode(out[0], skip_special_tokens=True)[:400])
