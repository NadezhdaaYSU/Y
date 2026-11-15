"""Run proofs/ExactDecodeStep.v's model against the real exact path, bit for bit.

    PYTHONPATH=python python3 tools/exact_decode_check.py                  # CPU
    PYTHONPATH=python python3 tools/exact_decode_check.py --device cuda    # GPU vs CPU model

The Coq file models one decode step as a function of a MACHINE (how every sum,
max and argmax is computed) and of the trusted float operations. This script
evaluates that model with the reference machine: every sum is an exact
integer sum (Python / numpy int64, far inside its range), every max is
Python's `max` over the integer keys or float values, the argmax is the first
maximal index. The integer functions are transcribed from the Coq
definitions, not imported from the code: `act_levels_for`, `k_levels_for`,
the integer `exp2`, the norm's grid limbs, the digit split. The float
operations that are not reductions are the Section variables of the proof,
and here they are the same torch elementwise operations the code uses,
evaluated on the CPU.

It runs the real path (`y_lang.exact`, on `--device`) through a prefill and
`--steps` decode steps of a pinned Qwen2.5-0.5B-Instruct, captures each
chosen layer's input, output and new cache row, recomputes the layer from
its input with the model, and compares the bits; then the final norm, the
logits and the token. On a GPU the real path runs there and the model on the
CPU, so equality also tests the trust assumption that the elementwise float
operations are the same function on both devices.

Exit status 0 only if every compared tensor is bit-identical and the run
compared something. Results go to `--out` as JSON.
"""
import argparse
import json
import math
import os
import platform
import sys
import time

import numpy as np
import torch

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
PROMPT = "The capital of France is"

# --------------------------------------------------------------------------
# Transcribed from proofs/ExactDecodeWidths.v and proofs/ExactDecodeStep.v.
# Constants from proofs/ExactDecodeConstants.v (read from the file, so the
# transcription uses the proof's values, not the code's).
# --------------------------------------------------------------------------

def coq_constants(path):
    """The `Definition NAME : Z := INT.` lines and the exp table."""
    import re
    text = open(path).read()
    consts = {m.group(1): int(m.group(2))
              for m in re.finditer(r"Definition (\w+) : Z := (-?\d+)\.", text)}
    table = re.search(r"Definition EXP_TABLE : list Z :=\s*\[(.*?)\]\.", text, re.S).group(1)
    consts["EXP_TABLE"] = [int(x) for x in table.replace("\n", " ").split(";")]
    return consts


C = None   # filled in main()


def bit_length(z):
    return 0 if z == 0 else (abs(z)).bit_length()


def floor_pow2m1(cap):
    n = max(1, bit_length(cap))
    if cap < (1 << n) - 1:
        n -= 1
    return max(127, (1 << n) - 1)


def act_levels_for(k, w_levels, requested):
    return floor_pow2m1(min(min(C["ACC_BUDGET"] // max(k * w_levels, 1) - 128, C["DIGIT_MAX"]),
                            requested))


def k_levels_for(d, q_levels):
    return floor_pow2m1(min(C["SCORE_BUDGET"] // max(d * q_levels, 1), C["K_LEVELS"]))


def split_hi(q):
    return q >> 7


def split_lo(q):
    return q & 127


def exp2_neg_q16_16(t0):
    t = max(t0, 0)
    n = t >> 16
    f = t & 65535
    idx = min(max(f >> 10, 0), 63)
    base = C["EXP_TABLE"][idx]
    d = f & 1023
    y = (d * C["LN2_Q30"]) >> 16
    y2 = (y * y) >> 30
    y3 = (y2 * y) >> 30
    y3o6 = (y3 * C["RECIP6_Q32"]) >> 32
    corr = (1 << 30) - y + (y2 >> 1) - y3o6
    g = (base * corr) >> 30
    sh = min(2 + n, 62)
    out = (g + (1 << (sh - 1))) >> sh
    return 0 if n >= 30 else out


def ng_hi(m2, s):
    return m2 >> min(s + 8, 63)


def ng_lo(m2, s):
    sa = min(s, 32)
    if s <= 32:
        return (m2 & (((1 << 40) - 1) >> (32 - sa))) << (32 - sa)
    return (m2 >> min(max(s - 32, 0), 63)) & ((1 << 40) - 1)


def f32_square(xb):
    ex = (xb >> 23) & 255
    frac = xb & 8388607
    m = frac if ex == 0 else frac | 8388608
    return m * m, 2 * max(ex, 1)


def clampz(lv, x):
    return max(-lv, min(lv, x))


def clamp0(hi, x):
    return max(0, min(hi, x))


# --------------------------------------------------------------------------
# The trusted float operations: the code's own torch elementwise operations.
# --------------------------------------------------------------------------

def f32_bits(x):
    """float32 tensor -> list of int32 bit patterns (signed)."""
    return x.contiguous().view(torch.int32).reshape(-1).tolist()


def from_bits_f32(b):
    return torch.tensor([b], dtype=torch.int32).view(torch.float32)[0]


def divisor(value, like):
    return torch.tensor(float(value), dtype=like.dtype)


def quant_ref(x, lv):
    """`_q_per_row` / `quantize_rows` / `q8_per_tensor` on one row x (float32 [n]).
    Returns (python int list, float32 0-d scale)."""
    absbits = f32_bits(x.abs())
    amax = from_bits_f32(max(absbits))                   # the max, by its integer key
    s = amax / divisor(lv, x)
    s = torch.where(s == 0, torch.ones_like(s), s)
    r = torch.round(x / s).tolist()
    return [clampz(lv, int(v)) for v in r], s


def linear_ref(lin, x, a_levels):
    """`ExactLinear` on one row x (float32 [K]); `lin` the module (CPU)."""
    K = lin.in_features
    assert x.numel() == K
    q, sx = quant_ref(x, a_levels)
    w = lin.w8t.t().contiguous().to(torch.int64).numpy()            # [N][K]
    qa = np.array(q, dtype=np.int64)
    if a_levels <= 127:
        acc = w @ qa
    else:
        hi = np.array([split_hi(v) for v in q], dtype=np.int64)
        lo = np.array([split_lo(v) for v in q], dtype=np.int64)
        acc = (w @ hi) * 128 + (w @ lo)
        assert np.array_equal(acc, w @ qa)
    assert np.abs(acc).max() < 2 ** 31
    y = torch.from_numpy(acc).to(torch.float32) * sx.to(torch.float32) * lin.sw
    if lin.bias is not None:
        y = y + lin.bias
    return y


def norm_ref(normmod, x):
    """`ExactRMSNorm` on one row x (float32 [H])."""
    H = x.numel()
    sq = [f32_square(b) for b in f32_bits(x)]
    assert all(((b >> 23) & 255) != 255 for b in f32_bits(x)), "inf/NaN row: outside the model"
    e = max(p for _, p in sq)
    big0 = sum(ng_hi(m2, max(e - p, 0)) for m2, p in sq)
    small0 = sum(ng_lo(m2, max(e - p, 0)) for m2, p in sq)
    big, small = big0 + (small0 >> 40), small0 & ((1 << 40) - 1)
    assert big < 2 ** 53 and small < 2 ** 40
    total = (torch.tensor(big, dtype=torch.int64).to(torch.float64) * float(1 << 40)
             + torch.tensor(small, dtype=torch.int64).to(torch.float64))
    scale = torch.tensor((e - (300 + 32) + 1023) << 52, dtype=torch.int64).view(torch.float64)
    total = total * scale
    xd = x.to(torch.float64)
    var = total / divisor(H, xd)
    y = xd / torch.sqrt(var + normmod.eps)
    return (y.to(torch.float32) * normmod.weight).to(x.dtype)


def attention_ref(q, kq, vq, sc, d):
    """`exact_attention_core` for one query position. q [nh, d] float32 after
    RoPE; kq / vq the quantised K and V rows of each KV group, as lists of
    (integer list, float32 scale). Returns [nh * d] float32."""
    nh, nkv, T = q.shape[0], len(kq), len(kq[0])
    rep = nh // nkv
    outs = []
    for h in range(nh):
        g = h // rep
        qi, sq = quant_ref(q[h], C["Q_LEVELS"])
        s_int = [sum(a * b for a, b in zip(qi, kq[g][t][0])) for t in range(T)]
        assert max(abs(v) for v in s_int) < 2 ** 24
        sqd = sq.to(torch.float64)
        a = sqd * sc * math.log2(math.e) * 65536.0
        sk = torch.stack([kq[g][t][1] for t in range(T)]).to(torch.float64)
        logit = torch.tensor(s_int, dtype=torch.float64) * a * sk
        m = torch.tensor(max(logit.tolist()), dtype=torch.float64)
        tq = torch.round(m - logit).tolist()
        p = [exp2_neg_q16_16(clamp0(1 << 30, int(v))) for v in tq]
        l = sum(p)
        sv = torch.stack([vq[g][t][1] for t in range(T)]).to(torch.float64)
        w = torch.tensor(p, dtype=torch.int64).to(torch.float64) * sv
        wmax = torch.tensor(max(w.tolist()), dtype=torch.float64)
        safe = torch.where(wmax > 0, wmax, torch.ones_like(wmax))
        W = torch.round(w * (float(1 << C["P_BITS"]) / safe))
        Wi = [clamp0(1 << C["P_BITS"], int(v)) for v in W.tolist()]
        wscale = safe / float(1 << C["P_BITS"])
        acc = [sum(Wi[t] * vq[g][t][0][dd] for t in range(T)) for dd in range(d)]
        acc_t = torch.tensor(acc, dtype=torch.int64).to(torch.float64)
        out = (acc_t * wscale) / torch.tensor(float(max(l, 1)), dtype=torch.float64)
        outs.append(out.to(torch.float32))
    return torch.cat(outs)


def prev_rows(cache_rows, g, d, lv, side):
    """The cached rows of KV group g, quantised: float32 rows are quantised
    here; packed int8 rows (`Y_EXACT_KV_PACK`) were quantised by the code
    when they arrived and are read back with its unpacker (a byte layout, no
    arithmetic). For a decode step's own new row the check compares the
    packed bytes against this file's quantisation."""
    rows = cache_rows[g]
    if rows.dtype == torch.float32:
        return [quant_ref(rows[t], lv) for t in range(rows.shape[0])]
    from y_lang.exact import kv as exact_kv
    ints, scales = (exact_kv.unpack_k if side == "k" else exact_kv.unpack_v)(rows.contiguous(), d)
    return [([int(v) for v in ints[t].tolist()], scales[t, 0]) for t in range(rows.shape[0])]


def layer_ref(layer, x, k_prev, v_prev, cos, sin, cfg, a_of):
    """One decoder layer for one position: x [H] float32, k_prev/v_prev
    [nkv, T-1, d]. Returns (out [H], k_new [nkv, d], v_new [nkv, d])."""
    import transformers.models.qwen2.modeling_qwen2 as qwen2
    att = layer.self_attn
    d = cfg.hidden_size // cfg.num_attention_heads
    nh, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
    h1 = norm_ref(layer.input_layernorm, x)
    qkv = linear_ref(att.qkv_proj, h1, a_of(att.qkv_proj.in_features))
    q = qkv[:nh * d].reshape(1, nh, 1, d)
    k = qkv[nh * d:(nh + nkv) * d].reshape(1, nkv, 1, d)
    v = qkv[(nh + nkv) * d:].reshape(nkv, d)
    q, k = qwen2.apply_rotary_pos_emb(q, k, cos, sin)
    q, k = q.reshape(nh, d), k.reshape(nkv, d)
    kl = k_levels_for(d, C["Q_LEVELS"])
    kq_new = [quant_ref(k[g], kl) for g in range(nkv)]
    vq_new = [quant_ref(v[g], C["V_LEVELS"]) for g in range(nkv)]
    kq = [prev_rows(k_prev, g, d, kl, "k") + [kq_new[g]] for g in range(nkv)]
    vq = [prev_rows(v_prev, g, d, C["V_LEVELS"], "v") + [vq_new[g]] for g in range(nkv)]
    attn = attention_ref(q, kq, vq, att.scaling, d)
    o = linear_ref(att.o_proj, attn, a_of(att.o_proj.in_features))
    x2 = x + o
    h2 = norm_ref(layer.post_attention_layernorm, x2)
    mlp = layer.mlp
    g = linear_ref(mlp.gate_proj, h2, a_of(mlp.gate_proj.in_features))
    u = linear_ref(mlp.up_proj, h2, a_of(mlp.up_proj.in_features))
    dn = linear_ref(mlp.down_proj, mlp.act_fn(g) * u, a_of(mlp.down_proj.in_features))
    return x2 + dn, k, v


# --------------------------------------------------------------------------
# The real path, captured
# --------------------------------------------------------------------------

def bits_equal(a, b):
    a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    if a.shape != b.shape or a.dtype != b.dtype:
        return False, f"shape/dtype {tuple(a.shape)} {a.dtype} vs {tuple(b.shape)} {b.dtype}"
    if a.dtype.is_floating_point:
        ia = a.view(torch.int32 if a.element_size() == 4 else torch.int64)
        ib = b.view(torch.int32 if b.element_size() == 4 else torch.int64)
    else:
        ia, ib = a, b
    diff = int((ia != ib).sum())
    return diff == 0, f"{diff} of {a.numel()} values differ"


def resolve(model, revision):
    """The local snapshot at `revision`: the cache's directory if it holds the
    files a load needs, else a download of only those files."""
    from huggingface_hub import snapshot_download
    need = ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
            "tokenizer_config.json", "vocab.json", "merges.txt"]
    return snapshot_download(model, revision=revision, allow_patterns=need)


def load(path, device):
    from y_lang.exact.attention import build_exact_model
    return build_exact_model(path, device)


def new_row_checks(ck, cvv, k_new, v_new, cfg):
    """(what, code's value, model's value) for a step's new K and V rows: the
    float rows, or with a packed cache the packed integers and scales read
    back with the code's unpacker against this file's quantisation."""
    if ck.dtype == torch.float32:
        return [("new K row", ck, k_new), ("new V row", cvv, v_new)]
    from y_lang.exact import kv as exact_kv
    d = cfg.hidden_size // cfg.num_attention_heads
    kl = k_levels_for(d, C["Q_LEVELS"])
    kint, ksc = exact_kv.unpack_k(ck.detach().cpu()[:, None].contiguous(), d)
    vint, vsc = exact_kv.unpack_v(cvv.detach().cpu()[:, None].contiguous(), d)
    kr = [quant_ref(k_new[g], kl) for g in range(k_new.shape[0])]
    vr = [quant_ref(v_new[g], C["V_LEVELS"]) for g in range(v_new.shape[0])]
    return [("new K row, packed integers", kint[:, 0].to(torch.int64),
             torch.tensor([r[0] for r in kr], dtype=torch.int64)),
            ("new K row, packed scales", ksc[:, 0, 0], torch.stack([r[1] for r in kr])),
            ("new V row, packed integers", vint[:, 0].to(torch.int64),
             torch.tensor([r[0] for r in vr], dtype=torch.int64)),
            ("new V row, packed scales", vsc[:, 0, 0], torch.stack([r[1] for r in vr]))]


def prefill_check(ref, cfg, cache, pcap, layers, a_of, report):
    """The prefill computes every prompt token in one pass. Row t of each
    chosen layer must be the model's decode step for token t over the cache
    rows before it, with the new K and V rows the cache holds at t."""
    for i in layers:
        x_in, x_out = pcap[i]
        x_out = x_out[0] if isinstance(x_out, tuple) else x_out
        x_in, x_out = x_in.detach().cpu()[0], x_out.detach().cpu()[0]
        K, V = cache.layers[i].keys.detach().cpu()[0], cache.layers[i].values.detach().cpu()[0]
        for t in range(x_in.shape[0]):
            cos, sin = ref.model.rotary_emb(torch.zeros(1, 1, cfg.hidden_size), torch.tensor([[t]]))
            y, k_new, v_new = layer_ref(ref.model.layers[i], x_in[t], K[:, :t], V[:, :t], cos, sin, cfg, a_of)
            rows = [("prefill row output", x_out[t], y)] + new_row_checks(K[:, t], V[:, t], k_new, v_new, cfg)
            for what, a, b in rows:
                ok, why = bits_equal(a, b)
                report["checks"].append({"prefill": True, "token": t, "layer": i, "what": what,
                                         "ok": ok, "detail": why})


LONG_PROMPT = "Write a short story about a lighthouse keeper who discovers something unusual."


def decode_run(model, ids, mask, steps, layers):
    """Prefill and `steps` greedy decode steps of a batch, as `y_lang.exact.greedy`
    drives the model (positions from the attention mask). Returns the per-step
    float32 last-position logits, the tokens, each chosen layer's decode-step
    outputs, and the cache."""
    from transformers import DynamicCache
    cache = DynamicCache(config=model.config)
    pos = mask.long().cumsum(-1) - 1
    pos = pos.masked_fill(mask == 0, 0)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, position_ids=pos,
                    past_key_values=cache, use_cache=True)
    logits = [out.logits[:, -1].float()]
    tokens = [logits[-1].argmax(-1)]
    outs = {i: [] for i in layers}
    hooks = [model.model.layers[i].register_forward_hook(
        lambda m, a, o, i=i: outs[i].append((o[0] if isinstance(o, tuple) else o).detach().clone()))
        for i in layers]
    p = pos[:, -1:]
    ones = mask.new_ones((mask.shape[0], 1))
    try:
        for _ in range(steps):
            p = p + 1
            mask = torch.cat([mask, ones], dim=-1)
            with torch.no_grad():
                out = model(input_ids=tokens[-1][:, None], attention_mask=mask, position_ids=p,
                            past_key_values=cache, use_cache=True)
            logits.append(out.logits[:, -1].float())
            tokens.append(logits[-1].argmax(-1))
    finally:
        for h in hooks:
            h.remove()
    # The prefill's hook calls come first; keep the decode steps' only.
    outs = {i: v[-steps:] for i, v in outs.items()}
    return logits, tokens, outs, cache


def ragged_check(real, tok, args, report):
    """A left-padded batch [prompt, longer prompt]: the padded row's logits,
    tokens, decode-step layer outputs and real cache rows must be the bits of
    the prompt decoded alone. Every compared value must also be finite, the
    condition under which `attend_pad_ok` holds."""
    side = tok.padding_side
    tok.padding_side = "left"
    try:
        enc = tok([args.prompt, LONG_PROMPT], return_tensors="pt", padding=True)
    finally:
        tok.padding_side = side
    ids, mask = enc.input_ids.to(args.device), enc.attention_mask.to(args.device)
    pad = int((mask[0] == 0).sum())
    alone = tok(args.prompt, return_tensors="pt").input_ids.to(args.device)
    report["ragged"] = {"padding_rows": pad, "batch_len": int(ids.shape[1])}
    lb, tb, ob, cb = decode_run(real, ids, mask, args.steps, args.layers)
    la, ta, oa, ca = decode_run(real, alone, torch.ones_like(alone), args.steps, args.layers)
    for s in range(args.steps + 1):
        ok, why = bits_equal(lb[s][0], la[s][0])
        report["checks"].append({"ragged": True, "step": s, "what": "logits, padded row vs alone",
                                 "ok": ok and bool(torch.isfinite(lb[s][0]).all()), "detail": why})
        report["checks"].append({"ragged": True, "step": s, "what": "token, padded row vs alone",
                                 "ok": int(tb[s][0]) == int(ta[s][0]),
                                 "detail": f"{int(tb[s][0])} vs {int(ta[s][0])}"})
    for i in args.layers:
        for s in range(args.steps):
            ok, why = bits_equal(ob[i][s][0], oa[i][s][0])
            report["checks"].append({"ragged": True, "step": s, "layer": i,
                                     "what": "layer output, padded row vs alone",
                                     "ok": ok and bool(torch.isfinite(ob[i][s][0]).all()), "detail": why})
        for what, a, b in (("K cache rows", cb.layers[i].keys[0, :, pad:], ca.layers[i].keys[0]),
                           ("V cache rows", cb.layers[i].values[0, :, pad:], ca.layers[i].values[0])):
            ok, why = bits_equal(a, b)
            report["checks"].append({"ragged": True, "layer": i, "what": f"{what}, padded row vs alone",
                                     "ok": ok, "detail": why})


def main():
    global C
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 11, 23])
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--constants", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "proofs", "ExactDecodeConstants.v"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--prefill", action="store_true",
                    help="also check each prefill row against the model's decode step for that token")
    ap.add_argument("--ragged", action="store_true",
                    help="also decode a left-padded batch and compare the padded row with the "
                         "prompt alone (left_padding_changes_nothing)")
    args = ap.parse_args()
    C = coq_constants(args.constants)

    torch.manual_seed(0)
    from transformers import AutoTokenizer, DynamicCache
    from y_lang.exact import model as exact_model
    path = resolve(args.model, args.revision)
    tok = AutoTokenizer.from_pretrained(path)
    t0 = time.time()
    real = load(path, args.device)
    ref = real if args.device == "cpu" else load(path, "cpu")
    cfg = ref.config
    req = exact_model.ACT_LEVELS
    a_of = lambda k: act_levels_for(k, C["W_LEVELS"], req)

    report = {"device": args.device, "model": args.model, "revision": args.revision,
              "act_levels_requested": req, "layers": args.layers, "steps": args.steps,
              "torch": torch.__version__, "python": platform.python_version(),
              "machine": platform.machine(), "cpu": platform.processor(),
              "gpu": torch.cuda.get_device_name(0) if args.device != "cpu" else None,
              "checks": [], "env": {k: v for k, v in os.environ.items() if k.startswith("Y_EXACT")}}

    # The widths the code derived are the ones the proof derives.
    for name, mod in ref.named_modules():
        if isinstance(mod, exact_model.ExactLinear):
            assert mod.a_levels == a_of(mod.in_features), (name, mod.a_levels)
    # The int8 weights built on the device are the ones built on the CPU.
    if ref is not real:
        rb = dict(ref.named_buffers())
        for name, b in real.named_buffers():
            if name.endswith(("w8t", "sw")):
                ok, why = bits_equal(b, rb[name]) if b.dtype != torch.int8 else (
                    torch.equal(b.cpu(), rb[name]), "int8 differ")
                report["checks"].append({"what": f"weights {name}", "ok": ok, "detail": why})

    ids = tok(args.prompt, return_tensors="pt").input_ids.to(args.device)
    cache = DynamicCache(config=real.config)
    pcap, phooks = {}, []
    if args.prefill:
        for i in args.layers:
            phooks.append(real.model.layers[i].register_forward_hook(
                lambda m, a, kw, o, i=i: pcap.__setitem__(i, (kw.get("hidden_states", a[0] if a else None), o)),
                with_kwargs=True))
    try:
        with torch.no_grad():
            out = real(input_ids=ids, past_key_values=cache, use_cache=True)
    finally:
        for h in phooks:
            h.remove()
    if args.prefill:
        with torch.no_grad():
            prefill_check(ref, cfg, cache, pcap, args.layers, a_of, report)
    token = out.logits[:, -1].float().argmax(-1)
    pos = ids.shape[1]
    tokens = [int(token)]
    for step in range(args.steps):
        prev = {i: (cache.layers[i].keys.detach().cpu().clone(), cache.layers[i].values.detach().cpu().clone())
                for i in args.layers}
        cap = {}
        hooks = []
        for i in args.layers:
            lay = real.model.layers[i]
            hooks.append(lay.register_forward_hook(
                lambda m, a, kw, o, i=i: cap.__setitem__(i, (kw.get("hidden_states", a[0] if a else None), o)),
                with_kwargs=True))
        hooks.append(real.model.norm.register_forward_hook(
            lambda m, a, o: cap.__setitem__("norm", (a[0], o))))
        with torch.no_grad():
            out = real(input_ids=token[:, None], past_key_values=cache, use_cache=True,
                       position_ids=torch.tensor([[pos]], device=args.device))
        for h in hooks:
            h.remove()
        logits = out.logits[0, -1].float()
        with torch.no_grad():
            pid = torch.tensor([[pos]])
            dummy = torch.zeros(1, 1, cfg.hidden_size)
            cos, sin = ref.model.rotary_emb(dummy, pid)
            for i in args.layers:
                x_in, x_out = cap[i]
                x_out = x_out[0] if isinstance(x_out, tuple) else x_out
                k_prev, v_prev = prev[i]
                y, k_new, v_new = layer_ref(ref.model.layers[i], x_in.detach().cpu().reshape(-1),
                                            k_prev[0], v_prev[0], cos, sin, cfg, a_of)
                rows = [("layer output", x_out.reshape(-1), y)]
                rows += new_row_checks(cache.layers[i].keys[0, :, -1], cache.layers[i].values[0, :, -1],
                                       k_new, v_new, cfg)
                for what, a, b in rows:
                    ok, why = bits_equal(a, b)
                    report["checks"].append({"step": step, "layer": i, "what": what, "ok": ok, "detail": why})
            n_in, n_out = cap["norm"]
            hn = norm_ref(ref.model.norm, n_in.detach().cpu().reshape(-1))
            ok, why = bits_equal(n_out.reshape(-1), hn)
            report["checks"].append({"step": step, "what": "final norm", "ok": ok, "detail": why})
            lg = linear_ref(ref.lm_head, hn, a_of(ref.lm_head.in_features))
            ok, why = bits_equal(logits, lg)
            report["checks"].append({"step": step, "what": "logits", "ok": ok, "detail": why})
            vals = lg.tolist()
            ref_tok = vals.index(max(vals))
            token = logits.argmax(-1, keepdim=True)
            report["checks"].append({"step": step, "what": "token", "ok": int(token) == ref_tok,
                                     "detail": f"real {int(token)} model {ref_tok}"})
        tokens.append(int(token))
        pos += 1

    # Bound stress: real activations keep most sums far below their bound, so
    # a float32 accumulation planted in a linear layer can pass the steps
    # above. Here each linear layer of the chosen layers and the LM head gets
    # the input that saturates one column: x = sign of the row with the
    # largest sum of |w|, so every quantised activation is +-a and that
    # column's sum is a * sum |w|, its largest possible value at that width.
    with torch.no_grad():
        mods = [(f"layers.{i}.{n}", m, dict(ref.named_modules())[f"model.layers.{i}.{n}"])
                for i in args.layers
                for n, m in real.model.layers[i].named_modules()
                if isinstance(m, exact_model.ExactLinear)]
        mods.append(("lm_head", real.lm_head, ref.lm_head))
        for name, mreal, mref in mods:
            w = mref.w8t.t().to(torch.int64)                       # [N][K]
            row = int(w.abs().sum(dim=1).argmax())
            x = torch.where(w[row] < 0, -1.0, 1.0).to(torch.float32)
            a = a_of(mref.in_features)
            y_real = mreal(x[None].to(args.device))[0]
            y_ref = linear_ref(mref, x, a)
            ok, why = bits_equal(y_real, y_ref)
            peak = a * int(w[row].abs().sum())
            report["checks"].append({"what": f"stress {name}", "ok": ok,
                                     "detail": f"{why}; peak |sum| {peak} ({peak / 2 ** 24:.2f} x 2^24)"})

    if args.ragged:
        ragged_check(real, tok, args, report)

    report["tokens"] = tokens
    report["text"] = tok.decode(tokens)
    report["seconds"] = round(time.time() - t0, 1)
    n_ok = sum(c["ok"] for c in report["checks"])
    report["summary"] = f"{n_ok} of {len(report['checks'])} compared tensors bit-identical"
    for c in report["checks"]:
        if not c["ok"]:
            print("MISMATCH", c)
    print(report["summary"], "| tokens", tokens, repr(report["text"]))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
    return 0 if report["checks"] and n_ok == len(report["checks"]) else 1


if __name__ == "__main__":
    sys.exit(main())
