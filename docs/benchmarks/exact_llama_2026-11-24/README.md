# Exact inference and RL on Llama, 2026-11-24

## What was asked

Does the exact path (`y_lang.exact`) carry over from Qwen2 to Llama, the same
bits on a CPU and a GPU, for inference and for RL?

## What changed

Llama's attention and MLP forwards in transformers 5.15 contain every line
the merged q/k/v and gate/up forwards were copied from, and its
`apply_rotary_pos_emb`, `rotate_half` and `eager_attention_forward` are
Qwen2's line for line. So `model.merge_projections` gives Llama's attention
and MLP the same merged forwards, which reach the exact attention; `convert`
refuses a Llama whose attention was not merged (its own forward would call
its own float attention). The rotary table (`ExactRotary`) is built from the
module's own `inv_freq`, which transformers scales for Llama 3 when it builds
the module, so the scaling is kept (a test compares the table with the scaled
and the unscaled embeddings). `lora.attach` registers Llama's attention.
The RL and SFT tools take each model's end of turn (`<|eot_id|>` for Llama 3)
and tokenize with no added special tokens (Llama's tokenizer adds
`<|begin_of_text|>`; Qwen's ids are unchanged, so its records are too).
Gemma stays refused.

## Results (RTX 4070 Ti SUPER and the Ryzen 9 9950X)

`llama_check.py`, 4 chat prompts, 24 greedy tokens:

| model | CPU == GPU tokens | alone == in a batch (GPU) |
|---|---|---|
| HuggingFaceTB/SmolLM2-135M-Instruct (9 query heads over 3 K/V) | yes | yes |
| unsloth/Llama-3.2-1B-Instruct (Llama 3 rotary scaling, 32 over 8) | yes | yes |

Llama 3.2 1B answers 17*23 = 391 as stock bf16 does; three of the four
answers begin word for word as stock's.

`rtx4070tis_and_cpu/`: 3 GRPO steps of Llama 3.2 1B (`exact_lora_grpo.py
--model unsloth/Llama-3.2-1B-Instruct --prompts 4 --group 4 --new 6 --steps 3`),
compiled on the GPU and eager on the CPU: the same sampled tokens, sampler
integers and p(target) at every step, the same adapters after step 3
(`compare.txt`), and no sampled token whose trainer integers differ from the
sampler's on either.

## Not covered

- Meta's own checkpoint is gated; `unsloth/Llama-3.2-1B-Instruct` holds the
  same weights ungated. Larger Llamas (8B) and the low-memory loader
  (`low_memory=True`, which builds Qwen2 layers by hand, refuses Llama).
- No training run past 3 steps, no T4/A100/H100 run of Llama yet.
