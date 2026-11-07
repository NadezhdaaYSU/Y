"""`check`: the low-memory exact builder against the float32 one.

    python3 tools/exact_large_model.py --check Qwen/Qwen2.5-0.5B-Instruct --device cpu

The builders moved to `python/y_lang/exact/large_model.py` (see its docstring);
this file keeps the check, which uses the ragged-batch harness, and makes this
name refer to the builders there.
"""
import _exact_pkg

_exact_pkg.add_path()

import torch  # noqa: E402

from y_lang.exact.large_model import (  # noqa: E402,F401
    HostEmbedding, HostLinear, build, build_exact, build_stock, checkpoint_dtypes, differing)


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
