"""Greedy decoding that never waits for the GPU until it has finished.

Hugging Face's `generate` asks the GPU every step whether every row has
finished (`unfinished_sequences.max() == 0`, then a Python `if`). That is a
device-to-host copy, so the CPU stops issuing work until the GPU has drained
its queue, and then the GPU idles while the CPU prepares the next step. A 7B
exact decode step launches about a thousand kernels, so that idle time is
large: about 2 ms of a 19 ms step (`docs/benchmarks/exact_7b_profile_2026-11-08`).

`greedy` decodes a fixed number of tokens and takes the argmax on the device,
so nothing is copied to the host until the end. It drives the model with the
inputs `generate` builds (positions from the attention mask, a dynamic cache,
`logits_to_keep=1`), and it picks tokens the way `generate` does when no
logits processor is active: argmax of the float32 last-position logits. It
applies no logits processor; in particular not the repetition penalty that
some checkpoints' generation configs set (Qwen2.5-Instruct sets 1.05), so
compare it with `generate(..., repetition_penalty=1.0)`. End-of-sequence
tokens do not stop it: every row gets exactly `max_new_tokens`.
"""

import torch

__all__ = ["greedy"]


@torch.no_grad()
def greedy(model, input_ids, attention_mask, max_new_tokens):
    """The new token ids, a `(batch, max_new_tokens)` tensor on the model's device.

    `input_ids` and `attention_mask` are `generate`'s: `(batch, length)`, left
    padded where rows differ in length.
    """
    from .kv import new_cache

    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be at least 1, not {max_new_tokens}")
    cache = new_cache(model.config.get_text_config(decoder=True))
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=cache, use_cache=True,
                logits_to_keep=1, return_dict=True)
    token = out.logits[:, -1].to(dtype=torch.float32).argmax(-1)
    new = [token]
    position = position_ids[:, -1:]
    ones = attention_mask.new_ones((attention_mask.shape[0], 1))
    for _ in range(max_new_tokens - 1):
        del out
        position = position + 1
        attention_mask = torch.cat([attention_mask, ones], dim=-1)
        out = model(input_ids=token[:, None], attention_mask=attention_mask,
                    position_ids=position, past_key_values=cache, use_cache=True,
                    logits_to_keep=1, return_dict=True)
        token = out.logits[:, -1].to(dtype=torch.float32).argmax(-1)
        new.append(token)
    return torch.stack(new, dim=1)
