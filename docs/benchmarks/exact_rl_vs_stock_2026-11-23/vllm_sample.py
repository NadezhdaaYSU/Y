"""vLLM sampling the exact RL run's prompts, for its speed: at every step the
step's prompts, `group` completions of each (`SamplingParams(n=group)`), each
exactly as many tokens as the exact sampler drew that step (`min_tokens` =
`max_tokens`, the end of turn ignored): the same rows and lengths, vLLM's own
tokens. Temperature 1, no top-k or top-p, as the exact sampler. The base
model's bfloat16 weights (adapters merged into them would cost the same);
vLLM's defaults otherwise (CUDA graphs, prefix caching, which shares each
prompt among its group). Moving trained weights into vLLM each step, which
an RL loop pays, is not timed.

    ~/vllm_env/bin/python vllm_sample.py vllm_in.json --record vllm.json

`vllm_in.json` is written by `tools/exact_rl_stock_speed.py --vllm-input`.
"""
import argparse
import json
import platform
import time

import torch
import vllm
from vllm import LLM, SamplingParams

ap = argparse.ArgumentParser()
ap.add_argument("input")
ap.add_argument("--record")
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--eager", action="store_true", help="enforce_eager: no CUDA graphs")
args = ap.parse_args()

data = json.load(open(args.input))
G = data["group"]
llm = LLM(model=data["model"], dtype="bfloat16", max_model_len=1024, gpu_memory_utilization=0.6,
          enforce_eager=args.eager, seed=0)


def run(step, seed):
    params = SamplingParams(n=G, temperature=1.0, top_p=1.0, top_k=0, max_tokens=step["new"],
                            min_tokens=step["new"], ignore_eos=True, seed=seed)
    prompts = [{"prompt_token_ids": p} for p in step["prompts"]]
    torch.cuda.synchronize()
    t = time.perf_counter()
    outs = llm.generate(prompts, params, use_tqdm=False)
    dt = time.perf_counter() - t
    rows = [o for out in outs for o in out.outputs]
    if len(rows) != G * len(prompts) or any(len(o.token_ids) != step["new"] for o in rows):
        raise SystemExit(f"step {step['step']}: vLLM returned {len(rows)} rows of lengths "
                         f"{sorted({len(o.token_ids) for o in rows})}, not {G * len(prompts)} of {step['new']}")
    return dt


run(data["steps"][0], 12345)                      # warm-up: not timed
times = []
for rnd in range(args.rounds):
    for s in data["steps"]:
        dt = run(s, 1000 * rnd + s["step"])
        times.append({"round": rnd, "step": s["step"], "arm": "sample_vllm" + ("_eager" if args.eager else ""),
                      "s": dt, "rows": G * len(s["prompts"]), "new": s["new"]})
        print(f"round {rnd} step {s['step']} vllm: {dt:.3f} s ({G * len(s['prompts'])} rows x {s['new']})",
              flush=True)
rec = {"vllm": vllm.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
       "python": platform.python_version(), "eager": args.eager, "times": times}
if args.record:
    json.dump(rec, open(args.record, "w"), indent=1)
