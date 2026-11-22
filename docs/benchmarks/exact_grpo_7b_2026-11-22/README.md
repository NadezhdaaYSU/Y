# Exact RL of a 7B model on an H100 and an A100, 2026-11-22

## What was asked

Does the exact RL path (`python/y_lang/exact/rl.py`) hold at 7B parameters:
the trainer's integers equal to the sampler's at every sampled token, and
two different datacenter GPUs running the same steps to the same bits?

## What was run

Qwen2.5-7B-Instruct (revision `a09a3545`), converted by `y_lang.exact`, LoRA
r = 16 on the five projections of all 28 layers, from B = 0. Three-digit
multiplication answered with the number alone (`--digits 3`, 600 problems
held out), 8 tokens sampled per try, 32 problems sampled a step and up to 8
of the groups with both right and wrong tries trained on (dynamic sampling),
pieces of up to 4,096 tokens, Adam at lr 1e-4, eager everything. 20 steps,
held-out evaluation at steps 0 and 20 (greedy). `modal_grpo7b.py` ran the
same command on both GPUs at once, each built from Y at fb783a6.

- `cloud/h100_grpo20.json`: an H100 80GB HBM3.
- `cloud/a100_grpo20.json`: an A100-SXM4-80GB. On both, the quantizers'
  rounding runs in the float64 formula (`train_lm.FAST_FLOAT64`).
- `cloud/a100_rl_tests.log`: `python/tests/test_exact_rl.py` on the A100,
  14 tests passed.

## Same bits (`compare.txt`)

At all 20 steps the same sampled tokens, the same sampler integers (p and l
of every sampled token) and the same p(target) of the trainer; the same
adapters at steps 5, 10, 15 and 20; every token of the held-out evaluation
the same at steps 0 and 20. On each GPU the trainer checked 5,688 sampled
tokens, every one the sampler's integers.

## Learning, briefly

83 of 600 held-out problems right at step 0 and 95 at step 20. The task is
hard for the model without working: on average 24 of each step's 32 groups
were all wrong and 5.4 were trained on. This run is about scale and
agreement; 20 steps are not a training result.

## Speed and cost

A step on the H100: 4.86 s sampling, 2.16 s training; on the A100 6.84 and
3.08 s (steps 4 to 20). Both eager. The job cost $0.97 on Modal's bill
(H100 $0.41, A100 $0.32, CPU, memory and network $0.24). A first attempt
cost $0.29 and ran no step: the image had downloaded the model by revision,
which writes no `main` ref, and the tool loads `main` offline; the script
now downloads `main` and fails its build unless that is the pinned
revision.

## Not covered

- 20 steps, one seed; no CPU run of the 7B (it is not run locally).
- Number-only answers, 8 tokens a try: the worked-answer setting of
  `../exact_grpo_steps_2026-11-22/` was not run at 7B.
