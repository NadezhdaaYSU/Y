"""Stock bfloat16 PyTorch doing the work of an exact RL run, for its speed.

    python3 tools/exact_lora_grpo.py ... --dump exact.pt          # the exact run
    python3 tools/exact_rl_stock_speed.py exact.pt --record stock.json --vllm-input vllm_in.json

The exact run's dump holds, at every step, the prompts, the tokens its
sampler drew, the rewards and the groups it trained on. Here the stock model
(transformers, bfloat16 weights, SDPA attention; LoRA r 16 on q, k, v, o,
gate, up and down of every layer, float32 master weights used in bfloat16)
does the same work, step by step:

  sample_eager    the same prompts decoded over a KV cache for as many tokens
                  as the exact sampler drew, each step's float32 softmax and
                  draw computed (the exponential race vLLM samples with), and
                  the exact run's token fed back: the same rows, lengths and
                  tokens. Each draw is kept (it is not dead code) and how often
                  it equals the exact token is reported.
  sample_graphed  the same, each decode step compiled and replayed from one
                  CUDA graph (`GraphedGreedy`'s step, as the exact graphed
                  sampler's).
  train_eager     Dr. GRPO on the kept groups, in the exact run's pieces:
  train_compile   forward, float32 log-softmax at the labelled positions,
  train_autotune  backward, one fused Adam step; eager, `torch.compile`
                  (dynamic shapes, PyTorch's defaults: cuBLAS by its own
                  heuristics) and `torch.compile(mode="max-autotune-no-cudagraphs")`
                  (each matrix product's kernel timed among cuBLAS and Triton's).

Each training arm has its own model and adapters, fed the same batches in
the same order, so their mean log p of the labelled tokens should agree to
bfloat16 rounding at every step (reported; the loss itself nearly cancels
within a group, so it is no measure). A sanity gate: the eager trainer's log p of the exact
tokens against the exact sampler's p / l (the same model, two arithmetics).

`--rounds` repeats every step, the arms in a seeded shuffled order each time;
the record keeps every time, so a summary can take each step's minimum. The
first call of a compiled arm compiles: it is timed apart (`compile_s`).

`--vllm-input` writes the prompts and lengths for
`docs/benchmarks/exact_rl_vs_stock_2026-11-23/vllm_sample.py`.
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
sys.path.insert(0, HERE)

import exact_lora_sft as S  # noqa: E402

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
TRAIN_ARMS = ("train_eager", "train_compile", "train_autotune")
SAMPLE_ARMS = ("sample_eager", "sample_graphed")


class LoRALinear(nn.Module):
    """base(x) + x A^T B^T * alpha / r, A and B float32 used in x's type."""

    def __init__(self, base, r, alpha, gen):
        super().__init__()
        self.base = base
        K, N = base.in_features, base.out_features
        dev = base.weight.device
        bound = 1.0 / math.sqrt(K)
        self.A = nn.Parameter(((torch.rand(r, K, generator=gen) * 2 - 1) * bound).to(dev))
        self.B = nn.Parameter(torch.zeros(N, r, device=dev))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.to(x.dtype).t()) @ self.B.to(x.dtype).t() * self.scale


def load_stock(model_name, device, r, alpha, seed):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    gen = torch.Generator().manual_seed(seed)
    n = 0
    for path, mod in list(model.named_modules()):
        for name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and name in TARGETS and ".layers." in f".{path}.":
                setattr(mod, name, LoRALinear(child, r, alpha, gen))
                n += 1
    if n == 0:
        raise ValueError("no projection found for the adapters")
    params = [p for p in model.parameters() if p.requires_grad]
    return model, params


def draw(logits):
    """vLLM's random sampling at temperature 1: argmax of p / q, q ~ Exp(1)."""
    probs = torch.softmax(logits.to(torch.float32), -1)
    q = torch.empty_like(probs).exponential_()
    return probs.div_(q).argmax(-1)


@torch.no_grad()
def sample_eager(model, ids, mask, force):
    """Decode `force.shape[1]` tokens, drawing each and feeding back `force`'s.
    Returns the draws [B][L]."""
    from transformers import DynamicCache
    cache = DynamicCache(config=model.config)
    position = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 0)
    out = model(input_ids=ids, attention_mask=mask, position_ids=position, past_key_values=cache,
                use_cache=True, logits_to_keep=1, return_dict=True)
    position = position[:, -1:]
    ones = mask.new_ones((mask.shape[0], 1))
    drawn = []
    L = force.shape[1]
    for t in range(L):
        drawn.append(draw(out.logits[:, -1]))
        if t + 1 == L:
            break
        del out
        position = position + 1
        mask = torch.cat([mask, ones], dim=-1)
        out = model(input_ids=force[:, t:t + 1], attention_mask=mask, position_ids=position,
                    past_key_values=cache, use_cache=True, logits_to_keep=1, return_dict=True)
    return torch.stack(drawn, 1)


def graphed_stock(model, batch, prompt_len, new_tokens):
    """`GraphedGreedy`'s step with `draw` in place of argmax and the token
    fed back read from a device buffer of forced tokens."""
    from y_lang.exact.graphed import GraphedGreedy
    from y_lang.exact import kv as _kv

    class GraphedStock(GraphedGreedy):
        def __init__(self):
            super().__init__(model, batch, prompt_len, new_tokens, attention=None)
            self.force = torch.zeros((batch, new_tokens), dtype=torch.int64, device=self.dev)
            self.drawn = torch.zeros((batch, new_tokens), dtype=torch.int64, device=self.dev)

        def _prefill(self, input_ids, attention_mask):
            cache = _kv.new_cache(self.cfg)
            position = (attention_mask.long().cumsum(-1) - 1).masked_fill(attention_mask == 0, 0)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position,
                             past_key_values=cache, use_cache=True, logits_to_keep=1, return_dict=True)
            self._d0 = draw(out.logits[:, -1])
            rows = [(layer.keys, layer.values) for layer in cache.layers]
            return self.force[:, 0].clone(), position[:, -1:] + 1, rows

        def _step(self, token, pos, t):
            inner = self.model.model
            h = inner.embed_tokens(token[:, None])
            pe = inner.rotary_emb(h, pos)
            allowed = (self.ar <= t) & self.base
            mask = allowed.view(self.batch, 1, 1, self.cap)
            for layer in inner.layers:
                h = layer(h, attention_mask=mask, position_ids=pos, past_key_values=self.cache,
                          use_cache=True, position_embeddings=pe)
            h = inner.norm(h)
            return draw(self.model.lm_head(h)[:, -1])

        def _advance(self):
            d = self._step_compiled(self.token, self.pos, self.t)
            self.step_i.add_(1)
            col = self.step_i.view(1)
            self.token.copy_(self.force.index_select(1, col).view(-1))
            self.pos.add_(1)
            self.t.add_(1)
            self.drawn.index_copy_(1, col, d.view(self.batch, 1))

        def run(self, ids, mask, force):
            if not self.sdpa:
                raise RuntimeError("the stock graphed sampler builds SDPA masks only")
            L = force.shape[1]
            with torch.no_grad():
                self.force.zero_()
                self.force[:, :L].copy_(force)
                first, next_pos, rows = self._prefill(ids, mask)
                self._load(first, next_pos, rows, mask)
                if self.graph is None:
                    self._record()
                    self._load(first, next_pos, rows, mask)
                self.drawn.zero_()
                self.drawn[:, 0] = self._d0
                for _ in range(L - 1):
                    self.graph.replay()
            return self.drawn[:, :L].clone()

    return GraphedStock()


def piece_loss(model, ids, pos, tgt, w):
    """-sum(w * log p(target)) at the labelled positions, and log p."""
    h = model.model(input_ids=ids, use_cache=False).last_hidden_state
    hs = h.reshape(-1, h.shape[-1]).index_select(0, pos)
    logits = model.lm_head(hs).to(torch.float32)
    lp = torch.log_softmax(logits, -1).gather(1, tgt[:, None]).squeeze(1)
    return -(w * lp).sum(), lp.detach()


def batches(step, end, group, chunk_tokens, device):
    """The exact `GRPO.step`'s pieces of the kept groups: (ids, positions,
    targets, weights) each, and the row/column of every labelled token."""
    from y_lang.exact import lora, rl
    r = [g * group + j for g in step["keep"] for j in range(group)]
    if not r:
        return [], 0
    tokens, pid, pm = step["tokens"][r], step["ids"][r], step["mask"][r]
    rewards = [step["rewards"][i] for i in r]
    lens = rl.completion_lengths(tokens, end)
    adv = rl.advantages(rewards, group)
    full = [int(pm[i].sum()) + lens[i] for i in range(len(lens))]
    n = sum(lens)
    out = []
    for which in rl.pieces(full, chunk_tokens):
        ids, labels, w, rows, cols = rl.training_batch(pid.to(device), pm.to(device), tokens, lens, adv,
                                                       0, which=which, n=n)
        pos, tgt = lora.target_positions(labels)
        rr = torch.tensor(r, device=device)[rows]
        out.append((ids, pos, tgt, w, rr, cols))
    return out, n


class Trainer:
    def __init__(self, arm, args, device):
        self.arm = arm
        self.model, params = load_stock(args.model, device, args.r, args.alpha, args.seed)
        self.opt = torch.optim.Adam(params, lr=args.lr, fused=True)
        if arm == "train_eager":
            self.fn = piece_loss
        elif arm == "train_compile":
            self.fn = torch.compile(piece_loss, dynamic=True)
        elif arm == "train_autotune":
            self.fn = torch.compile(piece_loss, dynamic=True, mode="max-autotune-no-cudagraphs")
        else:
            raise ValueError(arm)
        self.calls = 0

    def step(self, pieces):
        """One update; (loss, log p of every labelled token)."""
        self.opt.zero_grad(set_to_none=True)
        total, lps = 0.0, []
        for ids, pos, tgt, w, _, _ in pieces:
            loss, lp = self.fn(self.model, ids, pos, tgt, w)
            loss.backward()
            total = total + loss.detach()
            lps.append(lp)
        self.opt.step()
        self.calls += 1
        return float(total), torch.cat(lps)


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def run(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    device = args.device
    data = torch.load(args.dump, weights_only=False)
    cfg = data["config"]
    steps = data["steps"][:args.steps] if args.steps else data["steps"]
    end, group, chunk = data["end"], data["group"], cfg.get("chunk_tokens")
    if cfg.get("model") != args.model:
        raise SystemExit(f"the dump's model is {cfg.get('model')}, not {args.model}")
    if args.vllm_input:
        with open(args.vllm_input, "w") as f:
            json.dump({"model": args.model, "group": group, "steps": [
                {"step": s["step"],
                 "prompts": [[t for t, m in zip(row, mrow) if m]
                             for row, mrow in zip(s["ids"][::group].tolist(), s["mask"][::group].tolist())],
                 "new": int(s["tokens"].shape[1])} for s in steps]}, f)
        # The groups' rows share their prompt (n = group per prompt in vLLM).
        for s in steps:
            g = s["ids"].view(-1, group, s["ids"].shape[1])
            if not bool((g == g[:, :1]).all()):
                raise SystemExit("a group's rows do not share their prompt")
    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        if a not in TRAIN_ARMS + SAMPLE_ARMS:
            raise SystemExit(f"unknown arm {a}")
    trainers = {a: Trainer(a, args, device) for a in arms if a in TRAIN_ARMS}
    sampler_model = None
    if any(a in SAMPLE_ARMS for a in arms):
        sampler_model = (trainers["train_eager"].model if "train_eager" in trainers
                         else load_stock(args.model, device, args.r, args.alpha, args.seed)[0])
    graphed = {}
    rec = {"config": vars(args), "dump_config": cfg, "exact_machine": data.get("machine"),
           "machine": S.machine(device), "times": [], "compile_s": {}, "steps": {}}
    rng = random.Random(args.seed)
    for rnd in range(args.rounds):
        for s in steps:
            st = s["step"]
            ids, mask = s["ids"].to(device), s["mask"].to(device)
            force = s["tokens"].to(device)
            info = rec["steps"].setdefault(str(st), {"rows": int(ids.shape[0]), "new": int(force.shape[1]),
                                                     "n_exact": s.get("n"), "losses": {}, "agree": {}})
            order = arms[:]
            rng.shuffle(order)
            for a in order:
                sync(device)
                t0 = time.perf_counter()
                if a in TRAIN_ARMS:
                    pieces, n = batches(s, end, group, chunk, device)
                    if not pieces:
                        continue
                    loss, lp = trainers[a].step(pieces)
                    sync(device)
                    dt = time.perf_counter() - t0
                    info["n"] = n
                    if s.get("n") is not None and n != s["n"]:
                        raise SystemExit(f"step {st}: {n} labelled tokens, the exact run trained on {s['n']}")
                    info["losses"].setdefault(a, []).append(loss)
                    info.setdefault("mean_logp", {}).setdefault(a, []).append(float(lp.double().mean()))
                    if a == "train_eager" and rnd == 0:
                        rr = torch.cat([p[4] for p in pieces]).cpu()
                        cc = torch.cat([p[5] for p in pieces]).cpu()
                        pe = s["p"][rr, cc].to(torch.float64) / s["l"][rr, cc].to(torch.float64)
                        d = (lp.cpu().to(torch.float64) - torch.log(pe)).abs()
                        info["logp_vs_exact"] = {"median_abs": float(d.median()), "max_abs": float(d.max())}
                else:
                    if a == "sample_eager":
                        drawn = sample_eager(sampler_model, ids, mask, force)
                    else:
                        key = tuple(ids.shape)
                        if key not in graphed:
                            graphed[key] = graphed_stock(sampler_model, *key, cfg["new"])
                        drawn = graphed[key].run(ids, mask, force)
                    sync(device)
                    dt = time.perf_counter() - t0
                    if drawn.shape != force.shape:
                        raise SystemExit(f"{a} drew {tuple(drawn.shape)}, not {tuple(force.shape)}")
                    info["agree"].setdefault(a, []).append(float((drawn == force).double().mean()))
                first = a not in rec["compile_s"] and a in ("train_compile", "train_autotune", "sample_graphed")
                if first:
                    rec["compile_s"][a] = dt
                else:
                    rec["times"].append({"round": rnd, "step": st, "arm": a, "s": dt})
                print(f"round {rnd} step {st} {a}: {dt:.3f} s{' (first call, compiles)' if first else ''}",
                      flush=True)
    if args.record:
        with open(args.record, "w") as f:
            json.dump(rec, f, indent=1)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model", default=S.MODEL)
    ap.add_argument("--arms", default=",".join(SAMPLE_ARMS + TRAIN_ARMS))
    ap.add_argument("--steps", type=int, default=0, help="the first this many steps of the dump (0: all)")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record")
    ap.add_argument("--vllm-input")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
