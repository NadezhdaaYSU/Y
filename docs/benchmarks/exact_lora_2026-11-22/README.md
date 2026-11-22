# Exact LoRA fine-tuning (T3) on a CPU and an H100, 2026-11-22

## What was asked

Can a real model be fine-tuned with exact training, so that the adapters
after any number of steps are the same bits on a CPU and on a GPU, and the
fine-tuned model is then served on either with the same tokens?

## What was run

Qwen2.5-0.5B-Instruct (revision `7ae55760`, `model.safetensors` sha256
`fdf756fa...` on both machines), converted by `y_lang.exact`, with LoRA
adapters (`python/y_lang/exact/lora.py`, Y at b51a00d) on the five
projections of all 24 decoder layers: r = 16, alpha = 16, 8,110,080 adapter
weights, Adam at lr 3e-4, batches of 64 problems.

The task (`tools/exact_lora_sft.py`): "a*b" for a, b in 10..99, answered
with the number and the end of turn. 600 of the 8,100 pairs are held out
and never trained on; they are the evaluation (greedy decoding over the KV
cache, 6 new tokens). Training batches come from a seeded host RNG, so every
device sees the same batches.

- `cpu_sft20.json`: the development machine's CPU (AMD Ryzen 9 9950X),
  steps 1 to 20, evaluated at steps 0 and 20.
- `h100/h100_sft400.json`: an H100 80GB HBM3 on Modal (`modal_lora.py`),
  steps 1 to 400, evaluated every 20 steps.
- `h100/h100_serve_cpu20.json`: the CPU's step-20 adapters served on the H100.
- `cpu_serve_h100_400.json`: the H100's step-400 adapters served on the CPU.
- `h100/gpu_tests.log`: `python/tests/test_exact_lora.py` on the H100, 14
  tests passed, including a tiny model's adapters on the GPU against the CPU's.

The adapter files (65 MB each) are not in the repository; their digests are
in the records.

## Same bits (`compare.txt`)

- **Training**: the H100's first 20 steps are the CPU's. p(target) of every
  labelled token has the same digest at all 20 steps, and the adapters are the
  same at steps 10 and 20 (`8b994789036285f5`, `ba36215735e33487`).
- **Serving**: every token generated for the 600 held-out problems is the
  same on both at step 0 (the base model) and step 20.
- **Trained on one, served on the other**: the CPU's step-20 adapters served
  on the H100 give the CPU's tokens; the H100's step-400 adapters
  (`1aabfadac1e65edb`) served on the CPU give the H100's tokens.

What makes the last two hold for a model being trained, not only a fixed
one: the forward pass of training is the served model's own forward code,
and `test_trainer_logits_are_a_cached_decodes` checks that its logits are a
left-padded cached decode's, bit for bit.

## Accuracy

Held-out problems answered right (all right answers also ended the turn
cleanly):

| step | 0 | 20 | 100 | 200 | 300 | 400 |
|---|---|---|---|---|---|---|
| right of 600 | 276 (46%) | 382 (64%) | 405 (68%) | 429 (72%) | 437 (73%) | 463 (77%) |

Training loss on the labelled tokens fell from 0.45 to 0.08 (one seed, one run).

## Speed and cost

The backward is eager PyTorch, not compiled or graphed: 1.12 s a step on the
H100, 29 s on the CPU; an evaluation of 600 problems 11 to 13 s on the H100,
55 to 78 s on the CPU. No speed work has been done on this path. The H100 job
cost $0.97.

## Not covered

- One seed, one run; no float twin of the fine-tune for an accuracy
  comparison. The gain shown is the exact fine-tune's against the base model.
- The CPU ran 20 of the 400 steps; steps 21 to 400 are the H100's alone. The
  step-400 adapters were checked by serving them on the CPU, not by training
  them there.
- The gradient is the straight-through gradient of the quantized model
  (`lora.py`'s docstring), checked against float64 autograd piece by piece,
  not the float gradient.
