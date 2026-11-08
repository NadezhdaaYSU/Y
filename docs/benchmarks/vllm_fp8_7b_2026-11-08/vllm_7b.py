"""vLLM at 7B with FP8 W8A8 (weights quantised on load, activations per token):
the same weight bytes as Y's int8 exact path, from a production engine.

    ~/vllm_env/bin/python vllm_7b.py <graphs|eager> <out.json>

1. Speed, as tools/exact_throughput.py measures Y: 32 copies of its PROMPT,
   64 new tokens, greedy, wall time of one generate call including prefill,
   best of 5 after a warm-up. Prefix caching is off (it would compute the
   shared prompt once, which the HF harness does 32 times) and EOS is
   ignored, so every row does the same work as Y's.
2. Batch invariance: the target prompt decoded alone and in 5 other batch
   compositions, 160 tokens; which compositions changed its tokens.
VLLM_BATCH_INVARIANT=1 in the environment is vLLM's deterministic mode.
"""
import json
import os
import sys
import time

from vllm import LLM, SamplingParams

MODE, OUT = sys.argv[1], sys.argv[2]
MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROMPT = ("Explain, step by step and in detail, how a suspension bridge carries "
          "the load of the traffic crossing it down into the ground. ")
OTHERS = [
    "Write a short poem about the sea at night.",
    "List the planets of the solar system in order from the sun, with one fact about each.",
    "What is the derivative of x^3 sin(x)? Show the steps.",
    "Summarise the causes of the French Revolution in a paragraph.",
    "Translate into German: The library opens at nine and closes at six.",
    "Explain what a hash table is to a ten-year-old.",
    "Give three tips for keeping houseplants alive in winter.",
    "Describe how a refrigerator moves heat out of its inside.",
    "Why is the sky blue? Answer briefly.",
    "Write a Python function that reverses a linked list.",
    "What were the main achievements of the Ottoman architect Mimar Sinan?",
    "Compare TCP and UDP in a short table.",
]
OTHERS = (OTHERS * 3)[:31]

llm = LLM(model=MODEL, quantization="fp8", dtype="bfloat16", max_model_len=1024,
          gpu_memory_utilization=0.88, enable_prefix_caching=False,
          enforce_eager=(MODE == "eager"), seed=0)
speed = SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True)


def timed(prompts, params):
    t = time.perf_counter()
    outs = llm.generate(prompts, params, use_tqdm=False)
    return time.perf_counter() - t, outs


timed([PROMPT] * 32, SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True))
timed([PROMPT] * 32, speed)
times = []
for _ in range(5):
    dt, outs = timed([PROMPT] * 32, speed)
    assert all(len(o.outputs[0].token_ids) == 64 for o in outs)
    times.append(dt)
best = min(times)

inv = SamplingParams(temperature=0.0, max_tokens=160, ignore_eos=True)
compositions = {
    "alone": [PROMPT],
    "+2 after": [PROMPT] + OTHERS[:2],
    "+7 after": [PROMPT] + OTHERS[:7],
    "+31 after": [PROMPT] + OTHERS[:31],
    "middle of 11": OTHERS[:5] + [PROMPT] + OTHERS[5:10],
    "alone again": [PROMPT],
}
tokens = {}
for name, prompts in compositions.items():
    _, outs = timed(prompts, inv)
    tokens[name] = list(next(o for o in outs if o.prompt == PROMPT).outputs[0].token_ids)
ref = tokens["alone"]
changed = {n: next((i for i, (a, b) in enumerate(zip(t, ref)) if a != b), None)
           for n, t in tokens.items() if n != "alone"}
text = llm.get_tokenizer().decode(ref[:40])

result = {
    "mode": MODE, "batch_invariant_env": os.environ.get("VLLM_BATCH_INVARIANT", ""),
    "model": MODEL, "quantization": "fp8", "batch": 32, "tokens": 64,
    "times_s": times, "best_s": best, "tok_s_per_seq": 64 / best, "tok_s_total": 64 * 32 / best,
    "first_differing_token_vs_alone": changed,
    "compositions_changed": sum(v is not None for k, v in changed.items() if k != "alone again"),
    "alone_reproduces": changed["alone again"] is None,
    "alone_text_start": text,
}
json.dump(result, open(OUT, "w"), indent=1)
print("RESULT", json.dumps({k: result[k] for k in ("mode", "batch_invariant_env", "best_s", "tok_s_per_seq",
                                                    "compositions_changed", "alone_reproduces")}))
