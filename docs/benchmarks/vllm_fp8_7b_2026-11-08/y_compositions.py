"""Y's exact path at 7B on the same batch compositions as vllm_7b.py: the
target prompt alone, then with other prompts around it (left padded), 160
greedy tokens each; which compositions changed the target's tokens.

    PYTHONPATH=python python3 y_compositions.py <out.json>

The prompts are read out of vllm_7b.py, so the two runs cannot drift apart.
Eager (exactness does not depend on compiling); Y_EXACT_GEMM / Y_EXACT_V_INT8
from the environment.
"""
import ast
import json
import os
import sys
from pathlib import Path

import torch

tree = ast.parse((Path(__file__).with_name("vllm_7b.py")).read_text())
lit = {}
for node in tree.body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        try:
            lit.setdefault(node.targets[0].id, ast.literal_eval(node.value))
        except ValueError:
            pass
PROMPT, OTHERS = lit["PROMPT"], (lit["OTHERS"] * 3)[:31]   # as vllm_7b.py does
COMPOSITIONS = {
    "alone": [PROMPT],
    "+2 after": [PROMPT] + OTHERS[:2],
    "+7 after": [PROMPT] + OTHERS[:7],
    "+31 after": [PROMPT] + OTHERS[:31],
    "middle of 11": OTHERS[:5] + [PROMPT] + OTHERS[5:10],
    "alone again": [PROMPT],
}

from y_lang.exact import load  # noqa: E402

model, tok = load("Qwen/Qwen2.5-7B-Instruct", "cuda", low_memory=True)
tok.padding_side = "left"
tokens = {}
for name, prompts in COMPOSITIONS.items():
    enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=160, min_new_tokens=160, do_sample=False,
                             num_beams=1, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    row = prompts.index(PROMPT)
    tokens[name] = out[row, enc.input_ids.shape[1]:].tolist()
    print(name, len(tokens[name]), flush=True)
ref = tokens["alone"]
changed = {n: next((i for i, (a, b) in enumerate(zip(t, ref)) if a != b), None)
           for n, t in tokens.items() if n != "alone"}
result = {"engine": "Y exact (y_lang.exact, eager)",
          "Y_EXACT_GEMM": os.environ.get("Y_EXACT_GEMM", ""), "Y_EXACT_V_INT8": os.environ.get("Y_EXACT_V_INT8", ""),
          "first_differing_token_vs_alone": changed,
          "compositions_changed": sum(v is not None for k, v in changed.items() if k != "alone again"),
          "alone_reproduces": changed["alone again"] is None,
          "alone_text_start": tok.decode(ref[:40]), "tokens": tokens}
json.dump(result, open(sys.argv[1], "w"), indent=1)
print("RESULT", json.dumps({k: result[k] for k in ("compositions_changed", "alone_reproduces", "first_differing_token_vs_alone")}))
