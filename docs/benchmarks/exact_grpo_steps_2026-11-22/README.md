# Exact RL with worked answers, 2026-11-22

## What was asked

The first exact GRPO run (`../exact_grpo_2026-11-22/`) stopped improving near
60% of the 600 held-out problems, at two learning rates. Why, and does a
larger setting get past it, with the sampler still the trainer?

## Why it stopped

`groups.py` drew 8 tries at each of 512 training problems, before and after
the 200 steps (`plateau_groups.json`):

| | always wrong (0 of 8) | always right (8 of 8) | no gradient in all |
|---|---|---|---|
| base model | 214 | 122 | 336 of 512 (66%) |
| after 200 steps | 130 | 193 | 323 of 512 (63%) |

GRPO learns only from groups with both right and wrong answers. About two
thirds of every batch taught nothing; training turned "sometimes right" into
"always right", and the problems the model never got right stayed out of
reach. Almost every wrong answer had wrong digits (1,820 of 1,871 after
training), not a wrong format.

## What was changed

- **Worked answers** (`--mode steps`): the model may write out its work and
  ends with "Answer: N"; the reward reads the last such number and requires
  the end of turn within 128 tokens. Allowed this, the base model gets 101 of
  200 held-out problems right greedy (`cot.py`), against 46% with the number
  alone.
- **Dynamic sampling** (`--sample-prompts 32`): 32 problems are sampled a
  step, and the first 8 whose 8 tries are neither all right nor all wrong
  are trained on (fewer if fewer qualify; the record counts them).
- **Pieces** (`--chunk-tokens 1024`, `lora.SFT.run_chunks`): a step's batch
  runs forward and backward in pieces, the gradients added in a fixed order,
  then one Adam step. Each piece quantizes on its own scales, so this is a
  different function from one whole batch, and as exact.
- **An early stop** (`--early-stop`): sampling ends once every row has
  ended, tested every 16 tokens on the tokens themselves, so every device
  stops at the same length.
- **A graphed sampler** (`--graphed`, `rl.graphed_sampler`): each decode
  step, with the exact softmax and the draw, compiled together and replayed
  from one CUDA graph; the step's checks are or-ed into flags inside the
  graph.

## Results

`rtx4070tis/steps150.json`: 150 steps on the RTX 4070 Ti SUPER, Qwen2.5-0.5B-
Instruct with r = 16 adapters from zero, lr 1e-4, compiled products, eager
sampler, held-out evaluation every 25 steps (greedy, 160 tokens):

| step | 0 | 25 | 50 | 75 | 100 | 125 | 150 |
|---|---|---|---|---|---|---|---|
| right of 600 | 294 | 367 | 393 | 386 | 395 | 378 | 380 |

49% to 66% at its best (step 100), 63% at the end; the number-only run's best
was 61%. The trainer checked 398,624 sampled tokens over the 150 steps; every
one had the sampler's integers. About 6 of each step's 32 groups were still
all wrong at the end.

## Same bits (`compare.txt`)

- **The CPU**: `cpu_steps3.json`, steps 1 to 3 on the AMD Ryzen 9 9950X,
  eager everything, against the GPU record: the same sampled tokens, sampler
  integers and p(target) at all 3 steps, so the updates after steps 1 and 2
  were the same too (the tokens of steps 2 and 3 are drawn from them).
- **The graphed sampler**: `rtx4070tis/timing10_graphed.json`, the first 10
  steps again with `--graphed`: the record's sampled tokens, sampler
  integers and p(target) at all 10 steps, and the same adapters at step 10.
  `timing10_eager.json` repeats them eagerly, also the same.

## Speed

The two 10-step runs, one after the other, nothing else on the machine but
a browser (steps 4 to 10):

| sampler | sample | train | a step |
|---|---|---|---|
| eager | 15.84 s | 4.63 s | 20.5 s |
| graphed | 4.12 s | 4.31 s | 8.4 s |

The graphed sampler's first call compiles and records the step, about 4.5
minutes, once per batch shape. A profile of eager sampling at 256 rows had
put 17% of its time in the vocabulary softmax and the rest in the model's
own elementwise work, which the compiled step fuses.

## Not covered

- One seed, one learning rate; accuracy flattens after step 100 here too,
  higher than before. The problems never solved in 8 tries are still not
  learned from.
- The CPU ran 3 of the 150 steps.
- Qwen2.5-1.5B-Instruct on three-digit problems (a check of the larger-model
  path, 4 steps, not kept): 11 of 200 right at the start, 31 of 32 groups all
  wrong; the path ran, with no mismatch, but the task is too hard for it.
