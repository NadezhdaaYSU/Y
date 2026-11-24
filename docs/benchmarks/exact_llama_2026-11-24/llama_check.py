"""An exact Llama-architecture model on the CPU and the GPU: greedy tokens
of four chat prompts in one batch on each (the same?), each prompt alone
(the batch's tokens?), and stock bf16's text beside them.

    python3 llama_check.py MODEL [NEW_TOKENS [OUT.json]]

OUT.json keeps both devices' tokens and the machine, to compare with another
machine's.
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python"))
from y_lang.exact import load
from y_lang.exact.greedy import greedy
torch.backends.cuda.matmul.allow_tf32 = False
MODEL = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 24
prompts = ["What is 17*23? Answer with the number.", "Name three colors.", "Write one sentence about the sea.",
           "Explain what a CPU is in one sentence."]
def batch(tok, ps, device):
    seqs = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True) for p in ps]
    seqs = [s if isinstance(s, list) else s["input_ids"] for s in seqs]
    T = max(map(len, seqs))
    ids = torch.tensor([[tok.pad_token_id or 0] * (T - len(s)) + s for s in seqs], device=device)
    am = torch.tensor([[0] * (T - len(s)) + [1] * len(s) for s in seqs], device=device)
    return ids, am
outs = {}
for dev in ("cpu", "cuda"):
    m, tok = load(MODEL, dev)
    ids, am = batch(tok, prompts, dev)
    outs[dev] = greedy(m, ids, am, N).cpu()
    alone = torch.cat([greedy(m, *batch(tok, [p], dev), N).cpu() for p in prompts])
    print(dev, "alone == in batch:", torch.equal(alone, outs[dev]))
    del m
print("CPU == GPU tokens:", torch.equal(outs["cpu"], outs["cuda"]))
if len(sys.argv) > 3:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tools"))
    from exact_lora_sft import machine
    json.dump({"model": MODEL, "new": N, "prompts": prompts, "machine": machine("cuda"),
               "tokens": {d: t.tolist() for d, t in outs.items()}}, open(sys.argv[3], "w"), indent=1)
for p, row in zip(prompts, outs["cuda"].tolist()):
    print(repr(p[:30]), "->", repr(tok.decode(row, skip_special_tokens=True)[:80]))
from transformers import AutoModelForCausalLM
st = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
ids, am = batch(tok, prompts, "cuda")
with torch.no_grad():
    so = st.generate(input_ids=ids, attention_mask=am, max_new_tokens=N, do_sample=False)[:, ids.shape[1]:]
for p, row in zip(prompts, so.tolist()):
    print("stock", repr(p[:30]), "->", repr(tok.decode(row, skip_special_tokens=True)[:80]))
