"""tritify's `generate_backends` request through Y's exact path, aligned with
it and with every step's logits hashed.

- Input: Qwen's chat template over the same two messages; the token ids are
  printed and, with --tritify-ids, compared with tritify's own (its
  PROMPT_IDS line) id by id.
- Stopping: EOS ids from the generation config (151645, 151643, as tritify),
  or --no-eos for exactly N tokens. Counts follow tritify: the EOS a step
  chooses is not counted as a token, but that step's logits are hashed.
- Bits: one untimed run per device with every step's full logits tensor
  (vocabulary-wide, float32) chained into one SHA-256; it must be the same on
  every device, and so must the tokens.
- Time: warm, then R runs. Each reports the total, the first token (the
  prompt as one batched forward plus the first choice) and decode ms/token
  after it, from timestamps taken (after a device sync) as each token is
  produced. Min, median, mean and standard deviation.

    python chat_request_y2.py Y_REPO MODEL_DIR [N=32] [--rounds 7] [--no-eos]
        [--compiled] [--devices cuda,cpu] [--tritify-ids FILE] [--out JSON]
"""
import argparse
import ast
import hashlib
import json
import os
import statistics
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("y_repo")
ap.add_argument("model_dir")
ap.add_argument("n", type=int, nargs="?", default=32)
ap.add_argument("--rounds", type=int, default=7)
ap.add_argument("--no-eos", action="store_true")
ap.add_argument("--compiled", action="store_true")
ap.add_argument("--devices", default="cuda,cpu")
ap.add_argument("--tritify-ids", help="a tritify log with a PROMPT_IDS line")
ap.add_argument("--out")
args = ap.parse_args()
sys.path.insert(0, os.path.join(args.y_repo, "tools"))

import torch  # noqa: E402
from transformers import AutoTokenizer, GenerationConfig  # noqa: E402
from transformers.generation.streamers import BaseStreamer  # noqa: E402

import _exact_pkg  # noqa: E402
_exact_pkg.add_path()
import exact_ragged_batch as R  # noqa: E402
from y_lang.exact import model as exact_model  # noqa: E402

tok = AutoTokenizer.from_pretrained(args.model_dir)
gen_cfg = GenerationConfig.from_pretrained(args.model_dir)
eos = gen_cfg.eos_token_id if isinstance(gen_cfg.eos_token_id, list) else [gen_cfg.eos_token_id]
chat = [{"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "What is the capital of France? Answer in one sentence."}]
prompt = tok.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]
ids = prompt[0].tolist()
res = {"prompt_ids": ids, "eos": None if args.no_eos else eos, "max_new": args.n, "devices": {}}
print(f"prompt {len(ids)} tokens, max {args.n}, eos {'off' if args.no_eos else eos}")
print("PROMPT_IDS", ids)
if args.tritify_ids:
    line = next(l for l in open(args.tritify_ids) if l.startswith("PROMPT_IDS"))
    theirs = ast.literal_eval(line.split(None, 1)[1].strip())
    same = theirs == ids
    res["tritify_prompt_ids_equal"] = same
    print(f"tritify's prompt ids: {'IDENTICAL' if same else 'DIFFERENT'} ({len(theirs)} vs {len(ids)})")
    if not same:
        first = next((i for i, (a, b) in enumerate(zip(theirs, ids)) if a != b), min(len(theirs), len(ids)))
        print(f"  first difference at {first}: tritify {theirs[first:first + 5]} Y {ids[first:first + 5]}")


class Stamps(BaseStreamer):
    """`generate` calls put() with the prompt, then once per new token."""
    def __init__(self, dev):
        self.dev, self.t0, self.times, self.calls = dev, None, [], 0

    def put(self, value):
        self.calls += 1
        if self.calls == 1:
            return
        if self.dev == "cuda":
            torch.cuda.synchronize()
        self.times.append((time.perf_counter() - self.t0) * 1e3)

    def end(self):
        pass


def stats(v):
    if not v:
        return None
    return {"min": min(v), "median": statistics.median(v), "mean": statistics.fmean(v),
            "sd": statistics.stdev(v) if len(v) > 1 else 0.0, "n": len(v)}


def fmt(s):
    return "-" if s is None else f"min {s['min']:.2f} median {s['median']:.2f} mean {s['mean']:.2f} sd {s['sd']:.2f} (n={s['n']})"


R.MODEL = args.model_dir
for dev in args.devices.split(","):
    m = R.build("exact", dev)
    compiled = args.compiled and dev == "cuda"
    if compiled:
        exact_model.compile_forward(m)
    x = prompt.to(dev)
    kw = dict(max_new_tokens=args.n, do_sample=False, num_beams=1, top_p=None, top_k=None, temperature=None,
              eos_token_id=None if args.no_eos else eos, attention_mask=torch.ones_like(x))
    if args.no_eos:
        kw["min_new_tokens"] = args.n
    with torch.no_grad():
        out = m.generate(x, output_logits=True, return_dict_in_generate=True, **kw)
    new = out.sequences[0, x.shape[1]:].tolist()
    h = hashlib.sha256()
    for step in out.logits:                       # every step, the whole vocabulary
        h.update(step[0].float().contiguous().cpu().numpy().tobytes())
    toks = [t for t in new if args.no_eos or t not in eos]
    logits_hash = h.hexdigest()
    del out
    totals, firsts, per_tok = [], [], []
    with torch.no_grad():
        m.generate(x, **kw)                       # warm
        for _ in range(args.rounds):
            st = Stamps(dev)
            if dev == "cuda":
                torch.cuda.synchronize()
            st.t0 = time.perf_counter()
            again = m.generate(x, streamer=st, **kw)
            if dev == "cuda":
                torch.cuda.synchronize()
            totals.append((time.perf_counter() - st.t0) * 1e3)
            assert again[0, x.shape[1]:].tolist() == new, "not reproducible on one device"
            firsts.append(st.times[0])
            if len(st.times) > 1:
                per_tok.append((st.times[-1] - st.times[0]) / (len(st.times) - 1))
    label = f"Y exact {dev}{' compiled' if compiled else ''}"
    res["devices"][label] = {"tokens": toks, "steps": len(new), "logits_hash": logits_hash,
                             "total_ms": stats(totals), "first_token_ms": stats(firsts),
                             "decode_ms_per_token": stats(per_tok)}
    print(f"  {label:22s} tokens {len(toks):3d} (steps {len(new)})  logits_hash {logits_hash[:16]}")
    print(f"  {label:22s} text: {tok.decode(toks)!r}")
    print(f"  {label:22s} GEN_IDS {toks}")
    print(f"  {label:22s} BENCH total ms: {fmt(stats(totals))}")
    print(f"  {label:22s} BENCH first token ms (batched prompt + first choice): {fmt(stats(firsts))}")
    print(f"  {label:22s} BENCH decode ms/token (after the first): {fmt(stats(per_tok))}")
    del m
    if dev == "cuda":
        torch.cuda.empty_cache()
labels = list(res["devices"])
if len(labels) > 1:
    same = len({(str(v["tokens"]), v["logits_hash"]) for v in res["devices"].values()}) == 1
    res["identical"] = same
    print("VERDICT:", "every step's logits and the tokens identical on every device" if same else "MISMATCH")
if args.out:
    json.dump(res, open(args.out, "w"), indent=1)
