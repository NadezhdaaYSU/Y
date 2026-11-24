# Why exact RL training was slow, and the fixes, 2026-11-24

## What was asked

`../exact_rl_vs_stock_2026-11-23/` measured exact GRPO training at 14.9
times the time of stock bf16 training on the same work. Where did the time
go, and how much of it can go?

## How it was looked at

`profile_train.py` replays the exact run's training steps from its dump
(`exact_lora_grpo.py --dump`), in order from the same adapters, and checks
every step's p(target) against the run's record: the same bits, or it
stops. It times steps bare, then profiles one with `torch.profiler`:
the wall time of each phase and of each kind of layer, the kernels launched
in each, and the kernels by GPU time and by count. Qwen2.5-0.5B, RTX 4070 Ti
SUPER, the worked-answer run's first steps.

## What it found

`profile_step3_parts.txt`: one training step launched **605,472 GPU
kernels**, 510,681 of them in the backward. The adapters' backward made
347,666 of them (362 a call), though their products were meant to be
compiled (`lora.CompiledOps`). The kernels were PyTorch's eager ones
(`at::native::...`: int64 shifts, adds, clamps, copies), not Triton's.

The cause (`TORCH_LOGS=recompiles`): `torch._dynamo hit
config.recompile_limit (8)` on the compiled product. Past eight compiled
versions, PyTorch runs the function eagerly, and says so in one log line.
New versions kept coming because `mm_y_any` split K in a Python loop
(`for k0 in range(0, K, ...)`), which fixed K to one value, so every token
count (the contracted length of the adapters' gradients) compiled again;
and the products' `requires_grad`, types and strides doubled them. The bits
were right all along, so no test saw it.

## The fixes, in order

| | kernels a step | GPU busy |
|---|---|---|
| before | 605,472 | 3.95 s |
| 1. one GEMM call when K fits (K stays symbolic), operands detached, recompile limit 64 | 221,945 | 2.09 s |
| 2. the norm's backward compiled | 160,793 | 1.93 s |
| 3. each adapter's whole forward and backward one compiled call | 143,356 | 1.87 s |
| 4. attention's forward and backward compiled | 101,668 | 1.68 s |
| 5. the norm's and SiLU's forward, SiLU's backward compiled | 74,116 | 1.26 s |
| 6. the vocabulary softmax and its gradient, 64 rows a compiled call | 72,050 | 0.88 s |
| 7. the GEMM's row padding and the adapters' float64 casts fused | 64,511 | 0.90 s |

(`profile_after_fix*.txt`, one profiled step each; the bare step times in
these short replays move with compiles of new shapes, so the speed is
measured in `../exact_rl_vs_stock_2026-11-23/rtx4070tis_train_fixed/`.)

Two more things that sent compiled code eager or recompiled it per length,
found on the way to fix 4:

- `train_lm._ceil_log2` used `(n - 1).bit_length()`, which fixes a
  symbolic length to one value (and under Dynamo a symbolic int passes
  `isinstance(n, int)`). Compiling, it now counts powers of two: the same
  integer for every n (checked 1 to 4,999), a guard only on the
  power-of-two range.
- `model.divisor` refuses to create a divisor while compiling (a constant
  there becomes a reciprocal multiply); attention's quantizers met that on
  first use, so the rest of that frame ran eagerly and recompiled per
  length. `lora._warm_attention` now runs the attention once, uncompiled,
  on a tiny input of the same head layout first.

## Same bits

- Every replay above matched the record's p(target) at every step.
- `test_exact_lora.OnTheGpu.test_compiled_steps_are_the_eager_ones` runs
  every compiled call of five steps (three batch lengths) eagerly too and
  compares every output and check bit for bit (every kind of call), then
  the adapters and p(target) against the CPU. With Inductor's two rounding
  flags turned off (`division_rounding`, `emulate_precision_casts`) it
  fails: 88 of the 185 compiled calls differ.
- The 10-step run after the fixes has the same sampled tokens, sampler
  integers, p(target) and adapters as the run before them.

## The sampler (`profile_sample.py`)

The graphed sampler at 256 rows, the run's first step (its tokens and p
checked against the dump each time), ms a token:

| | ms a token | kernels a token |
|---|---|---|
| before (`profile_sample.txt`) | 33.8 | 3,401 |
| the inference switches: packed K/V, Y's decode attention, V int8 (`_switches`) | 25.6 | 3,313 |
| checks or-ed in the graph, the prefill's norms, SiLU, attention compiled (`_fix1`) | 18.5 | 2,332 |
| cuBLAS's int8 GEMM (`_gemm_torch`; Y's: `_gemm_y`, 17.3) | 17.2 | 2,302 |
| the two-level draw, 64 / 256 rows at a time (`_draw_rows64`, `_draw_rows256`) | 15.7 / 15.2 | 2,287 |

Before, 950 conditions a token were each reduced and or-ed by two eager
kernels inside the replayed graph (9% of the time); the attention
re-quantized every cached K/V row every token; the draw wrote a running sum
over the whole vocabulary. What is left: Y's decode attention 17%, the
GEMMs 15%, the vocabulary softmax 12%, the adapters' products 6%.

## The fused product, and what did not help (later the same day)

- **The adapters' products in one kernel** (`y_lang.exact.qmm`, 7accac9):
  40,773 kernels a step (was 54,567), GPU busy 0.67 s (0.75); the median
  of steps 4 to 9 1.05 s against 1.13 s (`profile_train_qmm1.txt`,
  `_qmm0.txt`), and 0.99 s in a later run (`_qmm_custom_B.txt`). Alone the
  kernel is slower than Y's GEMM on the same digits (64 against 44 us on
  the large products, 19 against 5 to 7 on the rank-16 ones); it wins by
  replacing four or five kernels with one.
- **The same kernel as a `torch.library.triton_op`** (launched from the
  compiled code, tiles by size class): the adapters' backward used half the
  host time, but under compilation the kernel ran 2.7 times slower; steps
  1.32 and 1.37 s against 0.99 s, run new, old, new
  (`_qmm_op_A.txt`, `_qmm_custom_B.txt`, `_qmm_op_C.txt`). Not kept.
- **cuBLAS's int8 GEMM for the adapters' products**: cuBLASLt refused the
  layout Inductor gave the transposed operand (`CUBLAS_STATUS_NOT_SUPPORTED`).
- **CUDA graphs**: a piece's shape changes from step to step, so a graph
  would be recorded again almost every call; fixed shapes need padding,
  which moves the quantizers' scales and so changes the bits.
- **Larger pieces** (`--chunk-tokens`, a different exact function): 4,096
  tokens ran out of the 16 GB; 2,048 halved the kernels (20,333) but the
  steps stayed within the noise (`_chunk2048.txt`).
- Measuring needs the GPU to itself: while a browser tab used it, Y's GEMM
  alone took 4 to 8 times as long.

## Not done

- The adapters' backward still launches 44 kernels a call: each of its five
  products calls Y's int8 GEMM as its own op, which Inductor cannot fuse
  into. The whole step is not one CUDA graph (its shapes change every step).
- The SFT path's vocabulary softmax (no per-token weights) stays eager; its
  gradient multiplies by a Python constant, a rounding case not checked here.
