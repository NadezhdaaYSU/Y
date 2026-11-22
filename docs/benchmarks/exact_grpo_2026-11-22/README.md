# Exact reinforcement learning (T3b) on an RTX 4070 Ti SUPER, an H100 and a CPU, 2026-11-22

## What was asked

Can a model be trained by reinforcement learning so that the sampler that
draws the completions and the trainer that learns from them agree exactly on
the probability of every sampled token, and so that a CPU and a GPU draw the
same completions and train the same adapters?

## What was run

Qwen2.5-0.5B-Instruct (revision `7ae55760`), converted by `y_lang.exact` (Y at
411b619; `machine.txt`),
with the LoRA adapters of the SFT run (`../exact_lora_2026-11-22/`): r = 16,
alpha = 16 on the five projections of all 24 layers, starting from B = 0
(the base model). The task and its 600 held-out problems are the SFT run's
(two-digit multiplication).

- The sampler (`python/y_lang/exact/rl.py`, `rl.sample`) decodes over the
  KV cache, as greedy serving does, and draws each token from the exact
  softmax's integers: numerators p_i (Q0.28, the integer exp2) and their
  sum l, by integer inverse CDF with u = r mod l, r a 62-bit integer
  hashed on the host from (seed, step, row, position). The draw is the same
  on every device; it follows p / l to within the modulo's bias, below
  2^-16.
- The update (`rl.GRPO`) is Dr. GRPO's, on policy: 8 problems a step, 8
  completions of each (6 tokens drawn, each read up to its first end of
  turn), reward 1 for the right number followed at once by the end of turn
  and 0 otherwise, advantage = reward minus the group's mean, loss
  -sum A log p over the completion tokens divided by their count. Adam at
  lr 1e-4. No KL term, no standard deviation.
- Every step the trainer recomputes p and l at every sampled token (the
  training forward over prompt and completion, right padded, no cache) and
  compares them with the sampler's. Any difference fails the step's checks.

Records:

- `rtx4070tis/grpo200.json`: 200 steps on the RTX 4070 Ti SUPER, the exact
  products compiled (`--compile`, the same bits as eager:
  `../exact_lora_2026-11-22/README.md`), evaluated every 20 steps.
- `cpu_grpo20.json`: steps 1 to 20 on the development machine's CPU (AMD
  Ryzen 9 9950X), eager, evaluated at steps 0 and 20.
- `h100/`: an H100 80GB HBM3 on Modal (`modal_grpo.py`): the RL and LoRA
  GPU tests (26 passed), and steps 1 to 20 eager (`h100_grpo20.json`) and
  with compiled products (`h100_grpo20_compiled.json`; on an H100 the
  quantizers' rounding runs in the float64 formula, not the integer one,
  `train_lm.FAST_FLOAT64`).
- `rtx4070tis/grpo200_lr3e-5.json` and `grpo200_lr3e-4.log`: the same run
  at two other learning rates.
- `rtx4070tis/stock_mismatch.json`: the control (`stock_mismatch.py`).
- `compare.txt`: the CPU's record against the 4070 Ti's and both H100 runs'.

## The sampler is the trainer

Over the 200 GPU steps the trainer checked 61,690 sampled tokens: for every
one, its p and l were the sampler's integers. So the importance ratio of
each update was exactly 1.

The control is the unconverted model, decoding the first step's 64 prompts
greedily over the KV cache and then running the same tokens through a
training forward, in the same process on the same GPU:

| model | tokens whose log p differs | largest difference |
|---|---|---|
| stock, bfloat16 | 368 of 384 | 0.97 in log p (a ratio off by 62%) |
| stock, float32 | 120 of 384 | 2.6e-5 in log p |
| exact | 0 of 61,690 | 0 |

The stock numbers are for one batch of one step, with the sampler and the
trainer in one framework and one process. Systems that sample with one
engine and train with another were not measured here.

## Same bits on a CPU and two GPUs

The CPU's first 20 steps are the 4070 Ti's and the H100's, eager and
compiled (`compare.txt`): at all 20 steps the
same sampled tokens, the same sampler integers and the same p(target) of
the trainer; the same adapters at steps 10 and 20 (`51bc97149251eb34`,
`c16a4d98f89fdd43`); every token of the held-out evaluation the same at
steps 0 and 20. The CPU, eager, also had its trainer's integers equal to
its sampler's at all 6,157 sampled tokens, and so did both H100 runs. Four
ways of computing (the CPU's eager products, the 4070 Ti's compiled ones,
the H100's eager and compiled ones, with its float64 rounding formula) give
one RL run.

## Learning

Held-out problems answered right (greedy, the SFT run's evaluation):

| step | 0 | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160 | 180 | 200 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| right of 600 | 276 | 306 | 337 | 323 | 319 | 361 | 358 | 367 | 364 | 359 | 355 |

46% to 59% at step 200 (61% at its best, step 140). The mean training
reward over each 40 steps: 0.49, 0.52, 0.58, 0.59, 0.55. One seed.

Other learning rates, the same run otherwise:

| lr | step 20 | 40 | 80 | 100 | 140 | 200 |
|---|---|---|---|---|---|---|
| 3e-4 | 1 | 1 | stopped | | | |
| 1e-4 | 306 | 337 | 319 | 361 | 367 | 355 |
| 3e-5 | 287 | 317 | 348 | 349 | 366 | 357 |

At 3e-4 the policy collapsed within 20 steps (every reward 0 by step 20, so
every advantage 0 and nothing left to learn from); the run was stopped at
step 40 and has its log, not a record. 1e-4 and 3e-5 level off at the same
place, about 60%, so the plateau is not the learning rate. Its cause was not
studied (one candidate: problems none of 8 completions get right give no
gradient). The 3e-5 run also checked 61,633 sampled tokens, every one the
sampler's integers. SFT on the answers reached 382 by step 20 and 463
by step 400 (`../exact_lora_2026-11-22/`); RL here learns from a 0 or 1 per
completion, and is slower.

## Speed

0.61 s to sample a step's 64 completions and 1.22 s to train on them, on the
4070 Ti (1.8 s a step, 200 steps in about 6 minutes plus evaluations); 7.7
and 28 s on the CPU; on the H100 2.44 and 1.46 s eager, 1.25 and 1.20 s
compiled. The H100 samples slower than the 4070 Ti; sampling is mostly
kernel launches, and the cloud machine's host was not profiled. The H100
job cost about $0.40.

## Tests

`python/tests/test_exact_rl.py`: the draw is the integer inverse CDF, its
frequencies are p / l, the hashed integers are pinned and independent of
the batch; the trainer has the sampler's integers for prompts of three
lengths (left padded for the sampler, right padded for the trainer); a
sampler one update behind is caught, and so is one changed integer; an
update raises the objective it follows; a run repeats; and a CPU, a GPU and
a GPU with compiled products sample the same tokens and train the same
adapters. Two mutants of the sampler (positions counted through the
padding, the training batch left padded) are each caught by three tests.

## Not covered

- One seed; accuracy flattens near 60% at two learning rates, for a reason
  not studied.
- The CPU and the H100 ran 20 of the 200 steps. Steps 21 to 200 are the
  4070 Ti's alone.
- The draw's modulo bias (below 2^-16 relative) is bounded, not removed.
- The stock control measures one batch, greedy, in one framework, as a
  measure of the gap the exact path closes; it is not a measurement of any
  particular RL system.
