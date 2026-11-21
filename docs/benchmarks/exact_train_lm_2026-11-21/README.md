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

## Speed (later the same day)

The first runs above took 164 s for 5000 GPU steps against float32's 16 s.
`Trainer(graph=True, compile=True)` (harness: `--graph --compile`) now runs
the same training at 0.99 ms a step, and gives the same bits:
`fast/gpu_exact_y_g12b8_fast.json` has the CPU record's weights at all 10
checkpoints and its target probabilities at all 5000 steps (`fast/compare.txt`),
final weights `8a7018b19d9b7a00`.

Steady-state ms per step, RTX 4070 Ti SUPER, Y's int8 GEMM, 12 / 8 bits.
Both arms get the same treatment; float32 is PyTorch's default (no TF32).

| Model | exact, eager (before) | exact, compiled + graphed | float32, eager | float32, compiled + graphed | ratio, same treatment |
|---|---|---|---|---|---|
| T2 (width 128, 2 layers, 1,024 tokens) | 22.4 | 0.99 | 1.08 | 0.40 | 2.5x |
| width 512, 4 layers, 4,096 tokens, 12.8M parameters | (not run) | 43.8 | 24.6 | 18.5 | 2.4x |

What changed, each measured on its own:

1. **Checks read once per step.** Every fail-closed check used to read the
   device (about 1,000 host waits a step). They now record a flag on the
   device and `Trainer.verify` reads them all in one transfer and raises
   before any value of the steps is read. Eager: 22.4 to 11.8 ms.
2. **Adam over one flat buffer**, updated in place: the same arithmetic per
   element in a few dozen kernels, and replayable by a CUDA graph.
3. **One CUDA graph per step** (eager kernels): 6.8 ms.
4. **`torch.compile`** with the flags under which Inductor rounds as eager
   PyTorch does: 3.2 ms.
5. **Quantizers on the integers inside the floats.** frexp, round, clamp
   and the digit split were float64 arithmetic, which this GPU runs at 1/64
   of float32's rate. They are now integer operations on the float64's bits
   (the max of |x| is an integer max of the bit patterns; rounding is an
   integer shift with round-half-even): the same function, checked on ties,
   zeros, subnormals and the 5000-step record above. 16.6 against 133 us for
   one [1024][512] 12-bit quantizer: 1.47 ms a step.
6. **Fewer kernels**: the checks reduced once per message instead of once
   per check, a 12-bit product as one GEMM over stacked digits, quantizer
   scales bounded so a product's scale needs no check: 1.02 ms.

At width 512 the GEMMs are not the gap: the exact step's GEMMs (Y's int8
GEMM 2.5 ms, attention's fp32 products 3.3 ms) take 5.8 ms against
float32's 14.3 ms. The other kernels take 38.4 ms against 4.4 ms, 14 ms of
it in four kernels over the attention tensors (32768 rows of 256), int64
work on 53-bit mantissas and float64 arithmetic. Activations in float32
(still integer sums, IEEE float32 elementwise, 24-bit mantissas in int32)
would be the next step; it would change the model's numbers, so it needs
new reference runs.

## Found while making it fast

1. **Inductor generated a data race** (torch 2.13.0, Triton 3.7.1). At width
   512 the compiled step's weights differed from the CPU's after one step,
   with every check passing; eager and graphed eager matched. Bisected to
   one gradient (the output layer's input gradient), wrong in 786,357 of
   2,097,152 values, then to one kernel. It quantized the output weight for
   the forward pass and wrote its transposed int8 copy for the backward
   pass. It stored the copy's int64 values in one loop and loaded them back
   in a later loop with no barrier between. Inductor assumes the thread
   that stored an element loads it, since the index is the same, but
   Triton gave the store the layout [1, 2] per thread and the load [1, 4]
   (its GPU IR), so threads loaded values other threads had not stored yet.
   The fix (`y_lang.exact.model`, applied to every Triton kernel Inductor
   writes in the process, and keyed into Inductor's cache) puts
   `tl.debug_barrier()` before any load of a buffer the same kernel stored.
   The width-512 model has 361 such loads, T2 one. Tests: the generated
   source of T2's compiled step has no unguarded reload, and does with the
   fix removed; one width-512 compiled step has the CPU's weights, and with
   the fix removed it does not. The compiled inference path imports the same
   module, so it has the fix too.
2. **A NaN or inf was not refused.** The quantizers and `det_sqrt` took a
   non-finite value without error, and CUDA converts a NaN to the integer
   0, so `det_sqrt(NaN)` gave 0 on the GPU. The integer forms read the
   exponent field and refuse them (test).
3. **The digit count was one short for 14, 15, 21, 22, ... bits.**
   ceil((bits - 1) / 7) gave two digits for 14 and 15 bits, but two digits
   in [-64, 63] reach 8,127, not 8,191. It was refused at run time, not
   wrong; now `ndigits` counts what the digits reach (test for 8 to 22
   bits).
4. **float32 is not reproducible between code paths either.** The float32
   twin gave held-out 1.6933 eager, 1.6974 compiled and 1.7060 with the
   first code, on one GPU; the exact arm gave 1.6883 every time.

## Not covered

- Speed past the two models above, and on other GPUs. The eager exact step
  is now slower than before (24 ms, many small integer kernels): eager is
  for the CPU and for checking, the fast path is compiled.
- One small model, one dataset, one seed. Other GPUs (T4, A100, H100) not yet
  run. The order freedom across data-parallel GPUs is shown by reversing a
  batch's rows, not by a multi-GPU run.
- Adam's state and the master weights are float64, updated elementwise.

## Files

- `*.json`: one record per run (`--record`): every step's training loss,
  a digest of the target probabilities per step, held-out losses and weight
  digests every 500 steps. `*.log`: the runs' output.
- `compare.txt`: `tools/exact_train_lm.py --compare` on each pair.
- `fast/`: the compiled, graphed runs (`--graph --compile`), with
  `ms_per_step` in each record, and their `compare.txt`.
