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

## On a Tesla T4 and an Intel CPU (Kaggle), 2026-11-24

`make_t4_llama_job.py` wrote the two Kaggle jobs (Y at ab01008 for the checks,
44dc9df for the 150 steps; the models
pinned to the revisions the local records used: Llama 5a8abab4, SmolLM2
12fd25f7; liby.so built on the machine, with Y's Turing int8 GEMM).
`compare_t4.py` compares their records (`t4_kaggle/`) with the local ones and
writes `t4_kaggle/compare.txt`: all the same.

- **Inference**: both Llamas give the same 24 greedy tokens of the four
  prompts on four machines: the Ryzen 9 9950X, the RTX 4070 Ti SUPER (sm_89),
  Kaggle's Intel Xeon @ 2.00GHz and the Tesla T4 (sm_75); alone and in a
  batch the same on each.
- **3 GRPO steps** (the settings above) on the T4 compiled, on the T4 with
  the graphed sampler and every inference switch on Y's GEMM, and eagerly on
  the Intel CPU (about 75 minutes a step there): at every step the 4070 Ti's
  sampled tokens, sampler integers and p(target), and its adapters after
  step 3 (707719f314858a53).
- **Worked answers, 150 steps on the T4** (`t4_kaggle/t4_llama_steps150.json`;
  the 0.5B run's settings: `--mode steps --new 128 --sample-prompts 32
  --early-stop --chunk-tokens 1024`, held-out 600 every 25 steps, greedy 160
  tokens; graphed sampler, every switch, `--sample-rows 64`):

  | step | 0 | 25 | 50 | 75 | 100 | 125 | 150 |
  |---|---|---|---|---|---|---|---|
  | right of 600 | 421 | 471 | 508 | 526 | 537 | 548 | 553 |

  70% to 92%, still rising at the end (Qwen2.5-0.5B, the same settings: 49%
  to 66% at best). The trainer checked all 479,441 sampled tokens: every one
  had the sampler's integers. All-wrong groups fell from about 3 of 32 a
  step to about 1.
- **The 4070 Ti ran the same run's first 10 steps**
  (`rtx4070tis_and_cpu/rtx4070tis_steps10.json`): the T4's sampled tokens,
  sampler integers and p(target) at all 10, the same adapters at step 10
  (04d7c90285a11156) and the same held-out tokens at step 0.
- **Speed** (steps 6 to 10, both graphed with every switch): the T4 9.5 s
  sampling and 6.8 s training a step, the 4070 Ti 2.8 s and 1.5 s. On the
  T4 a step took 8.8 s sampling and 6.6 s training after step 10; the 150
  steps and seven evaluations took 66 minutes. Peak GPU memory 10.4 GB of
  the T4's 15 (`--sample-rows 256` ran a 16 GB card out of memory: 256 rows
  of int64 numerators over Llama's 128,256 tokens).

Two things the T4 found, fixed before or after the runs:

- The fused adapter product (`qmm`, 7accac9) is Triton's int8 dot, which
  does not build for sm_75; `qmm.available` now asks for sm_80 or later, so
  on a T4 the adapters keep `exact_product` on Y's GEMM (the same bits; a
  test reports sm_75 and fails without the check).
- A model converted on the CPU and moved to the T4 still called the Triton
  GEMM, which does not compile there: five GPU training tests failed on the
  T4 (`t4_kaggle/exact_tests_t4_before_fix.log`; `load` on the GPU was not
  affected, nor were the runs above). `ExactLinear._apply` now decides again
  on the move (726cc1b; a test, which fails without it). The other failures
  in that log are a proof file the job's source lacked and the GEMM tile
  cache placed as for an installed package (the job points `Y_LIB_PATH` at
  the wheel's library).

## Not covered

- Meta's own checkpoint is gated; `unsloth/Llama-3.2-1B-Instruct` holds the
  same weights ungated. Larger Llamas (8B) and the low-memory loader
  (`low_memory=True`, which builds Qwen2 layers by hand, refuses Llama).
- No A100/H100 run of Llama; the T4's later steps (11 to 150) were not
  replayed elsewhere.
- No stock control on Llama here: whether stock bf16 changes its tokens
  between these machines was measured for Qwen2.5 (`../exact_cross_gpu_t4_2026-11-11/`),
  not for Llama.
