"""The Kaggle records (t4_kaggle/) against the RTX 4070 Ti SUPER's and the
Ryzen 9 9950X's (rtx4070tis_and_cpu/): greedy tokens of both Llamas on four
machines, the 3 GRPO steps on the T4 (two ways) and on Kaggle's Intel CPU,
and the overnight run's first 10 steps. Prints, and writes
t4_kaggle/compare.txt.

    python3 compare_t4.py
"""
import contextlib
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tools"))
from exact_lora_grpo import compare  # noqa: E402

L, K = os.path.join(HERE, "rtx4070tis_and_cpu"), os.path.join(HERE, "t4_kaggle")
out, ok = [], True
for name, local, kaggle in (("SmolLM2-135M-Instruct", "tokens_smollm2.json", "t4_smollm2.json"),
                            ("Llama-3.2-1B-Instruct", "tokens_llama1b.json", "t4_llama1b.json")):
    a, b = (json.load(open(os.path.join(d, f))) for d, f in ((L, local), (K, kaggle)))
    if a["prompts"] != b["prompts"] or a["new"] != b["new"]:
        raise SystemExit(f"{name}: the records ran different prompts")
    devs = [(a["machine"]["cpu"], a["tokens"]["cpu"]), (a["machine"]["gpu"], a["tokens"]["cuda"]),
            (b["machine"]["cpu"], b["tokens"]["cpu"]), (b["machine"]["gpu"], b["tokens"]["cuda"])]
    same = all(t == devs[0][1] for _, t in devs)
    ok &= same
    out.append(f"{name}, {a['new']} greedy tokens of {len(a['prompts'])} prompts: "
               + ", ".join(d for d, _ in devs) + (": the same tokens" if same else ": DIFFERENT"))
for title, x, y in (("3 GRPO steps, the T4 compiled", "llama_gpu.json", "t4_grpo3.json"),
                    ("3 GRPO steps, the T4 graphed with every switch", "llama_gpu.json", "t4_grpo3_fast.json"),
                    ("3 GRPO steps, Kaggle's Intel CPU", "llama_gpu.json", "kaggle_cpu_grpo3.json"),
                    ("worked answers, the 4070 Ti's 10 steps against the T4's 150", "rtx4070tis_steps10.json",
                     "t4_llama_steps150.json")):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok &= compare(os.path.join(L, x), os.path.join(K, y))
    out.append(f"\n== {title}\n" + buf.getvalue().rstrip())
out.append("\nALL SAME" if ok else "\nNOT ALL THE SAME")
text = "\n".join(out) + "\n"
print(text, end="")
open(os.path.join(K, "compare.txt"), "w").write(text)
sys.exit(0 if ok else 1)
