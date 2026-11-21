# Exact training, T2: a tiny GPT trained from scratch, 2026-11-21

## What was asked

Can a small transformer be trained from scratch so that its weights, after
thousands of steps, are the same bits on a CPU and on a GPU, and how much
accuracy does that cost against ordinary float32 training?

## The model and the run

`python/y_lang/exact/train_lm.py`, harness `tools/exact_train_lm.py`. A
character-level GPT: 2 layers, width 128, 4 heads of 32, a ReLU MLP of 512,
pre-RMSNorm with gains, learned token and position embeddings, an untied
output layer; vocabulary padded to 96 (65 characters used); context 64;
426,624 parameters. Tiny Shakespeare (1,115,394 characters, sha256
`86c4e6aa9db7c042`), the last tenth held out. Adam (lr 3e-3, betas 0.9 and
0.99), batch 16 x 64 tokens, 5000 steps, seed 0. Batches are drawn on the
CPU from a seeded generator, so every device trains on the same tokens.
Held-out loss: 8 fixed batches every 500 steps.

## The rule that makes it device-independent

The inference path's rule (`attention.exact_attention_core`), carried to the
backward pass:

- **Every sum is a sum of integers.** Linear layers: int8 x int8 products on
  an integer GEMM (Y's, cuBLAS's or the CPU's float64 BLAS, which is exact for
  them). Attention products: int8 values multiplied in float32, every partial
  sum an integer below 2^24. Other sums (a norm's mean square, a softmax row,
  an embedding's gradient, a norm gain's gradient): int64, after rounding to
  one power-of-two fixed-point scale per sum. Every int8 operand has one
  power-of-two scale per row along the summed dimension, so scales factor out
  of the sum exactly.
- **Everything else is elementwise float64** +, -, * and tensor-by-tensor /.
  Square roots are integer (`det_sqrt`); constant divisors are multiplies by
  a reciprocal computed once in Python (two library findings below).
- **The softmax's exp** is the integer exp2 of the inference path.
- **Widths.** The forward pass is int8, the format the model would be served
  in. The backward pass's gradient operands are `grad_bits` wide and the
  activations and weights they meet `bwd_bits`; past 8 bits an operand is
  split into int8 digits (q = sum d_i * 128^i), so every product still runs
  on an int8 GEMM and is still exact.

## Results

Held-out loss after 5000 steps (nats per character, 8 fixed batches), the
mean training loss over the last 500 steps, and the last weight digest.
CPU: AMD Ryzen 9 9950X (Zen 5), torch 2.13.0; GPU: RTX 4070 Ti SUPER (sm_89).

| Run | Gradient / other backward bits | Device, GEMM | Held-out | Train | Weights |
|---|---|---|---|---|---|
| `cpu_exact` | 8 / 8 | CPU, float64 BLAS | 2.0083 | 1.6836 | `6febd608d3a93ae7` |
| `gpu_exact_y` | 8 / 8 | GPU, Y's int8 GEMM | 2.0083 | 1.6836 | `6febd608d3a93ae7` |
| `cpu_exact_bwd12` | 12 / 12 | CPU, float64 BLAS | 1.7058 | 1.5179 | `7bd7590109a971db` |
| `cpu_exact_g12b12_digits` | 12 / 12 | CPU, int8 digits | 1.7058 | 1.5179 | `7bd7590109a971db` |
| `gpu_exact_y_g12b12` | 12 / 12 | GPU, Y's int8 GEMM, digits | 1.7058 | 1.5179 | `7bd7590109a971db` |
| `cpu_exact_g12b8` | 12 / 8 | CPU, int8 digits | 1.6883 | 1.5108 | `8a7018b19d9b7a00` |
| `gpu_exact_y_g12b8` | 12 / 8 | GPU, Y's int8 GEMM, digits | 1.6883 | 1.5108 | `8a7018b19d9b7a00` |
| `cpu_exact_bwd16` | 16 / 16 | CPU, float64 BLAS | 1.6832 | 1.5114 | `654d26ea889387b8` |
| `cpu_float32` | float32 | CPU | 1.7060 | 1.5091 | `33a1a9763f65c217` |
| `gpu_float32` | float32 | GPU | 1.7055 | 1.5089 | `e90abb0f5ff03bb7` |

**The same bits** (`compare.txt`): every pair of exact runs with the same
widths, CPU against GPU and with digits against without, has identical
weights at all 10 checkpoints and identical target probabilities at all
5000 steps. The float32 pair, the control, differs from step 1 (the target
probabilities) and at every checkpoint.

**Accuracy**: with 12-bit gradients the exact model trains as well as
float32 here: held-out 1.6883 (12 / 8) and 1.7058 (12 / 12) against 1.7060
and 1.7055, training loss 1.511 to 1.518 against 1.509. One seed and 8
held-out batches: the two float32 runs differ by 0.0005 between
themselves, and the exact runs' spread (1.683 to 1.706) is not a ranking.
The 12 / 8 setting is the default (`Ops(grad_bits=12, bwd_bits=8)`): two int8
GEMMs per backward product where int8 takes one.

## Findings

1. **PyTorch's CPU float64 `torch.sqrt` is not correctly rounded** on this
   AVX-512 build (torch 2.13.0): 226 of 20,000 inputs differ from
   `math.sqrt`, and in all 200 disagreements checked exactly (with
   fractions) the CPU was wrong; CUDA matched all 20,000. The first CPU ==
   GPU test failed on it. Square roots are now integer: a float estimate,
   then integer corrections to floor(sqrt(n)), the one integer it must be.
   The inference norm (`ExactRMSNorm`) uses the same `torch.sqrt` with a
   comment saying it is correctly rounded; logged in `docs/known_issues.md`.
2. **PyTorch's CUDA divide by a Python scalar multiplies by its reciprocal**:
   64,925 of 100,000 float64 quotients differ from the CPU's. Adam's bias
   corrections divided that way.
3. **int8 gradients stall training; 12-bit ones do not.** With int8 gradients the training
   loss stopped falling near step 3000 (1.68 over the last 500 steps against
   float32's 1.51) and the held-out loss ended at 2.01 against 1.71. 12 or
   16 bits for the backward pass's operands removed the gap; 12 bits for the
   gradients alone, with int8 activations and weights, did too. The forward
   pass is int8 in every run: the format the model would be served in costs
   nothing measurable here, the gradients' rounding did.
4. **The loss shown must be order-free too.** The first row-order test failed
   on the printed loss, a float mean over tokens, while the weights matched.
   It is now `math.fsum`, and records compare the bits of each target's
   probability rather than the loss, which passes through each host's `log`.

## Checks

`python/tests/test_exact_train_lm.py`, 19 tests: the hand-written backward
against PyTorch autograd in float64 (all 16 parameter gradients within
1e-15 relative); `isqrt` against `math.isqrt`; `det_sqrt` within 2^-29 of
`math.sqrt` and the same bits on CPU and CUDA; the digit split rebuilds
every value and a wide product equals the direct exact one; the exact sum
ignores order; training lowers the loss; reversing each batch's rows (which
reorders every sum over tokens) leaves the exact weights unchanged and moves
the float32 twin's; on a GPU, Y's GEMM and cuBLAS give the CPU's weights
(12-bit gradients, int8 gradients, and 16 / 12 bits through digits on
cuBLAS) and the float32 twin does not.

Mutations, each caught: the exact sum made a float sum (5 tests fail); the
norm's integer square root made `torch.sqrt` (3); Adam's reciprocal multiply
made a scalar divide (3). The last two are caught only by the GPU tests, so
a run of the suite on a machine without a GPU cannot see them.

## Not covered

- Speed. The exact step is unoptimised Python over many small kernels:
  5000 steps took 164 s on the GPU with 12-bit gradients
  (117 s with int8 ones) against float32's 16 s, and 823 s on the CPU
  (sharing it with another run) against 41 s. No kernel work has been done for training.
- One small model, one dataset, one seed. Other GPUs (T4, A100, H100) not yet
  run. The order freedom across data-parallel GPUs is shown by reversing a
  batch's rows, not by a multi-GPU run.
- Adam's state and the master weights are float64, updated elementwise.

## Files

- `*.json`: one record per run (`--record`): every step's training loss,
  a digest of the target probabilities per step, held-out losses and weight
  digests every 500 steps. `*.log`: the runs' output.
- `compare.txt`: `tools/exact_train_lm.py --compare` on each pair.
