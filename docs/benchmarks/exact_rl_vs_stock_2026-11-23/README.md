# Exact RL against stock bf16 and vLLM, same work, 2026-11-23

## What was asked

How fast is an exact RL step (`python/y_lang/exact/rl.py`) against what
people run without the exact guarantee: bfloat16 PyTorch with cuBLAS, and
vLLM for sampling? Stock was given its best setup.

## What was run

`run.sh`, on the RTX 4070 Ti SUPER (AMD Ryzen 9 9950X host, nothing else on
the GPU: `gpu_before.txt`), in this order:

1. **Exact** (`exact_a`): the first 10 steps of the worked-answer run
   (`../exact_grpo_steps_2026-11-22/`): Qwen2.5-0.5B-Instruct, LoRA r 16, 32
   problems times 8 tries sampled a step, up to 128 tokens with the early
   stop, up to 8 mixed groups trained on, in pieces of 1,024 tokens; graphed
   sampler, compiled products. `--dump` keeps each step's prompts, tokens
   and kept groups.
2. **Stock bf16 PyTorch** (`tools/exact_rl_stock_speed.py`, torch 2.13.0,
   transformers 5.15.0, SDPA attention), on that dump:
   - sampling the same prompts for the same number of tokens, with the
     float32 softmax and draw every step, the exact run's token fed back:
     eager over a KV cache (`sample_eager`) and with each step compiled and
     replayed from a CUDA graph (`sample_graphed`);
   - training on the same pieces (Dr. GRPO, float32 log-softmax at the
     labelled positions, fused Adam), LoRA on q, k, v, o, gate, up and down:
     eager (`train_eager`), `torch.compile` with PyTorch's defaults, so cuBLAS
     by its own heuristics (`train_compile`), and `torch.compile` in
     `max-autotune-no-cudagraphs` mode, which times cuBLAS against Triton's
     kernels for each product and keeps the faster (`train_autotune`).
   Two rounds, the arms in a shuffled order at every step.
3. **vLLM 0.21.0** (`vllm_sample.py`, torch 2.11.0): the same prompts, 8
   completions each (`n = 8`), each exactly as long as the exact step's,
   temperature 1; vLLM's defaults: CUDA graphs and prefix caching (each
   prompt computed once for its 8 tries). Base weights (merged adapters cost
   the same). Two rounds.
4. **Exact again** (`exact_b`).

## Same work (`summary.txt`, `exact_ab_compare.txt`)

- The two exact runs: the same tokens, sampler integers, p(target) and
  adapters at every step, and the same as the 150-step record's steps 5 and
  10. No sampled token's trainer integers differed from the sampler's.
- Stock trained on the same number of labelled tokens as exact at every step
  (the tool refuses otherwise); every vLLM completion had the step's length
  (the script refuses otherwise).
- The same model: stock's log p of the exact tokens against the exact p / l,
  median difference 2.9e-4 (per step 1.4e-4 to 1.3e-2, larger at the first
  steps, where the model is less sure). On a few uncertain digits the two
  differ by up to 7 in log p: an int8 model and a bf16 model choosing
  differently on close calls.
- The stock draw equals the exact token at 64% of positions (eager and
  graphed alike), so the stock sampler draws from the same distribution and
  is not dead code.
- The three stock trainers, each with its own adapters, agree on the mean log
  p of the labelled tokens to 0.04 at every step and round.

## Speed

Seconds a step, steps 4 to 10, each step the minimum over its two
repetitions (the repetitions differ by about 2%):

| | exact | stock | exact takes |
|---|---|---|---|
| sampling, vLLM | 4.18 | 1.14 | **3.7x** the time |
| sampling, bf16 PyTorch eager | | 2.00 | 2.1x |
| sampling, bf16 PyTorch graphed | | 3.11 | 1.3x |
| training, compiled, max-autotune | 4.42 | 0.296 | **14.9x** |
| training, compiled, default | | 0.299 | 14.8x |
| training, eager | | 0.382 | 11.6x |
| a whole step (vLLM + max-autotune) | 8.60 | 1.43 | **6.0x** |

First calls, not in the table: stock's compiled trainer 139 s, max-autotune
220 s, graphed sampler 20 s; exact's first step 52 s (its graphed sampler
compiles and records).

## What it says

- On this GPU an exact RL step costs 6 times stock's best. The exact
  guarantee is not free here, and the price is mostly training.
- **Training is the gap: 15 times.** Exact training runs its products
  compiled and everything else eagerly, about 100,000 small kernels a step
  (`../exact_lora_2026-11-22/`). Stock compiled is 1.3 times its own eager;
  `max-autotune` gains nothing over cuBLAS's own choice at these shapes.
- **Sampling is 3.7 times vLLM** and 2.1 times plain PyTorch. vLLM shares each
  prompt among its 8 tries and decodes with paged attention; the exact
  sampler computes every row's prompt and its own integer softmax and draw.
- The stock graphed sampler was slower than stock eager (3.11 against
  2.00 s): its step attends over the whole buffer with a mask, and its
  compiled softmax and draw over 256 x 151,936 logits were not profiled.
  vLLM is the stock sampler that counts.

## Not covered

- One GPU, one model size, 10 steps. Exact's training time includes the
  rewards (decoding 256 completions on the host), stock's does not; it is a
  small part of 4.4 s.
- An RL loop with vLLM also moves the trained weights into vLLM every step;
  not timed.
- Stock's numbers are bf16 throughout; exact's arithmetic is integer sums
  with float32 elementwise work, and its checks of every sampled token are
  part of its time.
