"""Build the exact and stock arms of a model too large to hold in float32.

    python3 tools/exact_cross_device.py --record cpu.json --device cpu \\
        --model Qwen/Qwen2.5-7B-Instruct --low-memory
    python3 tools/exact_cross_device.py --record gpu.json --compile \\
        --model Qwen/Qwen2.5-7B-Instruct --low-memory

The harnesses' own builders load `dtype=float32` and then move the model: at
7B that is 30 GB on the host before anything reaches the device. Here the
checkpoint loads in bf16 on the host, and each `nn.Linear` is moved to the
target device, upcast to float32 there and quantised there by `ExactLinear`, one
layer at a time, so the scales are still computed on-device the way the
float32 builders compute them. The rest of the model is then cast to float32
and `exact_model.convert` does norms, SiLU, RoPE and divisors.

**That is the same model only for a bf16 checkpoint.** bf16 -> float32 widens
exactly, so loading bf16 and upcasting gives the bits a float32 load gives. A
float32 or float16 checkpoint loaded in bf16 is rounded first, so
`build_exact` reads the safetensors headers and refuses anything but BF16.
`python3 tools/exact_large_model.py --check MODEL` compares the two builders on
a model small enough for both: on Qwen2.5-0.5B all 460 tensors and the logits
are bit-identical. Compare the tensors, not only the logits: rounding the
weights through fp16 changed the embedding and left one prompt's logits
bit-identical; quantising in bf16 changed 338 tensors.

Stock 7B is 14.2 GiB of bf16, which does not fit beside a desktop on a 16 GB
card. `build_stock` keeps the 1.06 GiB token embedding on the host: a gather,
no arithmetic, so the rows that reach the device are the same bits wherever the
table lives. `host_lm_head=True` moves `lm_head` too, which DOES change that
layer's arithmetic (CPU bf16); acceptable for the control, whose only job is to
show that float stock diverges across devices, and recorded in the artifact.
"""
import glob
import json
import os
import struct

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


def checkpoint_dtypes(model_id):
    """The set of tensor dtypes in the cached safetensors files of `model_id`.

    Read from each file's JSON header, not from `config.json`, which states the
    dtype the model was exported for and not the one the tensors are stored in.
    """
    from huggingface_hub import snapshot_download
    root = snapshot_download(model_id, allow_patterns=["*.safetensors"],
                             local_files_only=True)
    files = sorted(glob.glob(os.path.join(root, "*.safetensors")))
    if not files:
        raise RuntimeError(f"no .safetensors files cached for {model_id} "
                           f"under {root}")
    return _safetensors_dtypes(files)


def _safetensors_dtypes(files):
    dtypes = set()
    for path in files:
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        dtypes.update(v["dtype"] for k, v in header.items()
                      if k != "__metadata__")
    return dtypes


def build_exact(model_id, device):
    import exact_model
    import batch_invariance_demo as D
    import transformers.models.qwen2.modeling_qwen2 as qwen2
    qwen2.eager_attention_forward = D.exact_attention
    m = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="eager").eval()
    stored = checkpoint_dtypes(model_id)
    if stored != {"BF16"}:
        raise RuntimeError(
            f"{model_id} stores {sorted(stored)}, not only BF16: loading it in "
            f"bf16 rounds the weights, so this is not the model the float32 "
            f"builder quantises. Use the harness without --low-memory.")
    rot = m.model.rotary_emb
    assert rot.inv_freq.dtype == torch.float32, rot.inv_freq.dtype
    shapes = []
    # Hold PARENTS only: a list of every module would keep each original Linear
    # alive after its float32 copy moved to the device (measured: 13.3 GiB, OOM).
    parents = [mod for mod in m.modules()
               if any(isinstance(c, nn.Linear) for c in mod.children())]
    for mod in parents:
        for cn in [n for n, c in mod.named_children() if isinstance(c, nn.Linear)]:
            lin = getattr(mod, cn).to(device=device, dtype=torch.float32)
            new = exact_model.ExactLinear(lin)
            setattr(mod, cn, new)
            shapes.append((new.out_features, new.in_features))
            del lin
    del parents
    m.to(device=device, dtype=torch.float32)
    n_lin, n_norm = exact_model.convert(m)
    assert n_lin == 0, "every Linear should already be ExactLinear"
    if (device != "cpu" and exact_model.GEMM == "triton"
            and exact_model.w8a8_matmul is not None):
        import w8a8_gemv
        w8a8_gemv.prime(sorted(set(shapes)), torch.device(device))
    return m


class HostEmbedding(nn.Embedding):
    """The token embedding kept on the host. A subclass of nn.Embedding so the
    weight keeps its name: the cross-device weight fingerprint compares names."""

    def forward(self, ids):
        return super().forward(ids.to(self.weight.device)).to(self.target)


class HostLinear(nn.Linear):
    """A Linear computed on the host; only stock's `lm_head`, see the module
    docstring for what that changes."""

    def forward(self, x):
        return super().forward(x.to(self.weight.device)).to(self.target)


def _pin_device(module, device):
    # HF's `.device` is the first parameter's device - the host embedding's -
    # and generate() builds position ids there. Pin it to the compute device.
    cls = module.__class__
    module.__class__ = type(cls.__name__, (cls,),
                            {"device": property(lambda self, _d=torch.device(device): _d)})


def build_stock(model_id, device, host_lm_head=False):
    import batch_invariance_demo as D
    import transformers.models.qwen2.modeling_qwen2 as qwen2
    qwen2.eager_attention_forward = D._orig
    m = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    if device == "cpu":
        return m
    emb = m.model.embed_tokens
    m.model.embed_tokens = None
    head = m.lm_head if host_lm_head else None
    if head is not None:
        m.lm_head = None
    m.to(device)
    if head is not None:
        hl = HostLinear(head.in_features, head.out_features,
                        bias=head.bias is not None, device="meta")
        hl.weight = head.weight
        hl.bias = head.bias
        hl.target = torch.device(device)
        m.lm_head = hl
    he = HostEmbedding(emb.num_embeddings, emb.embedding_dim,
                       padding_idx=emb.padding_idx, _weight=emb.weight,
                       _freeze=True, device="cpu", dtype=emb.weight.dtype)
    he.target = torch.device(device)
    m.model.embed_tokens = he
    _pin_device(m, device)
    _pin_device(m.model, device)
    return m


def build(arm, model_id, device, host_lm_head=False):
    if arm == "exact":
        return build_exact(model_id, device)
    if arm == "stock":
        return build_stock(model_id, device, host_lm_head)
    raise ValueError(f"--low-memory builds only the stock and exact arms, "
                     f"not {arm!r}")


def differing(a, b):
    """Names whose tensors are not the same dtype, shape and BITS in two state
    dicts (floats compared as bytes, so -0.0 != 0.0 and NaN == NaN)."""
    if a.keys() != b.keys():
        return sorted(set(a) ^ set(b))
    bad = []
    for k in a:
        x, y = a[k], b[k]
        if x.dtype != y.dtype or x.shape != y.shape:
            bad.append(k)
            continue
        if x.dtype.is_floating_point:
            x = x.contiguous().reshape(-1).view(torch.uint8)
            y = y.contiguous().reshape(-1).view(torch.uint8)
        if not torch.equal(x, y):
            bad.append(k)
    return bad


def check(model_id, device):
    """The low-memory exact arm against the float32 builder, on a model small
    enough for both: every parameter and buffer, and the prompt's logits, bit
    for bit. The comparison must also see a one-bit change in one int8 weight,
    or an all-equal result would mean nothing."""
    import exact_ragged_batch as R
    from transformers import AutoTokenizer
    R.MODEL = model_id
    tok = AutoTokenizer.from_pretrained(model_id)
    ids = tok([R.TARGET], return_tensors="pt").input_ids.to(device)
    got = []
    for fn in (lambda: R.build("exact", device),
               lambda: build_exact(model_id, device)):
        m = fn()
        with torch.no_grad():
            logits = m(ids).logits.detach().cpu()
        got.append(({k: v.detach().cpu().clone()
                     for k, v in m.state_dict().items()}, logits))
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (sa, la), (sb, lb) = got
    bad = differing(sa, sb)
    bad_logits = differing({"logits": la}, {"logits": lb})
    k = next(k for k, v in sb.items() if v.dtype == torch.int8)
    mutated = dict(sb)
    mutated[k] = sb[k].clone()
    mutated[k].view(-1)[0] ^= 1
    caught = differing(sb, mutated) == [k]
    print(f"  {model_id} on {device}: {len(sa)} tensors, "
          f"{len(bad)} differ{' ' + str(bad[:5]) if bad else ''}; logits "
          f"{'differ' if bad_logits else 'bit-identical'}; control (one bit of "
          f"{k}) {'caught' if caught else '*** NOT CAUGHT ***'}")
    return 0 if sa and not bad and not bad_logits and caught else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="check the low-memory exact builder against the float32 one")
    ap.add_argument("--check", metavar="MODEL", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    raise SystemExit(check(a.check, a.device))
