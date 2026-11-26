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

## Where the GPU time goes now, 2026-11-25

After the whole-layer path (4f0f63f), training waits on the GPU on both
cards measured, so the next gains are GPU time. `profile_train.py` now also
prints each part's GPU time (the kernels its host code launched) and, with
`--shapes`, Y's ops by operand shape. Kernels are attributed to parts only on
the per-module path (`Y_EXACT_LAYER=0`), whose GPU time is within 10% of the
whole-layer path's (0.630 against 0.691 s on the 4070 Ti,
`rtx4070tis/profile_train_layer.txt`).

- **RTX 4070 Ti SUPER, Qwen2.5-0.5B** (`rtx4070tis/profile_train_parts_gpu.txt`,
  the fast2 record's step 10; 0.624 s of GPU time):

  | part | GPU s |
  |---|---|
  | adapters' layers (`lora_linear`), forward and backward | 0.249 |
  | attention, forward and backward | 0.176 |
  | LM head (`frozen_linear`) | 0.061 |
  | norms and SiLU | 0.035 |
  | between the parts (residuals, rotary, loss) | 0.103 |

  The products themselves (`_qmm_kernel` 122 ms, cuBLAS int8 18 ms, Y's GEMM
  10 ms, `exact_pv` 14 ms) are about a quarter; the rest is the quantizers'
  passes (each tensor's maximum, then its rounding and digits) and the
  softmax's.
- **Tesla T4, Llama 3.2 1B** (`t4_kaggle/t4_profile_train.log`, step 8 of the
  T4 runs' settings, `make_t4_llama_job.py pf`; 5.83 s of GPU time). There
  the fused product does not build (sm_75), so every product is Y's int8
  GEMM with its digits made in separate kernels. The 30 largest kernels hold
  4.46 s: Y's GEMM 1.22 s (5,120 calls: 8 for each of the 640 adapter
  layer calls, 3 forward and 5 backward), quantizer
  reductions 1.22 s, digits and int8 packing for the GEMM 0.78 s, other
  fused pointwise 0.55 s, PyTorch's elementwise kernels and copies 0.30 s,
  attention's P·V 0.28 s, float32 GEMMs 0.12 s.
- **Attention's softmax runs in float64.** The largest attention kernel (the
  forward's softmax, fused by Inductor: the logits, the row maxima, the exp,
  the row sums and the V scales' products; recomputed in the backward)
  forms the logits `s_int * (sq * sc * log2e * 2^16) * sk` in float64, rounds
  `m - logit` in float64, and stores two float64 tensors. GeForce cards run
  float64 at 1/64 of float32's rate, the T4 at 1/32. On the 4070 Ti, one
  layer's call (6 x 14 heads x 163 x 163 scores) took 305 us; the same kernel
  with float32 in place of float64 (a timing copy, wrong values) 95 us; the
  softmax chain alone in float32, 26 us. The exp itself (five int64
  products a score) is about 6 of those 26 us. This kernel family is about 9%
  of a training step on the 4070 Ti and 10% on the T4 (0.59 of 5.83 s).
  Keeping the bits while dropping float64 means each float64 product built
  from 32-bit integer pieces; power-of-two scales for Q and K would make the
  logits cheap, but are a different function (every record and the accuracy
  measured again).

## Why training compiled again at steps 2, 3 and 5 on the T4, and the fixes, 2026-11-25

`make_t4_llama_job.py rc` ran each training path for 6 steps of the T4 runs'
settings with `TORCH_LOGS=recompiles`, each path with its own cold caches,
and printed PyTorch's compile time breakdown at exit
(`t4_kaggle/recompiles/rc_*`). Most of the whole-layer path's compile time
was one function: `_layer_backward` compiled 12 times, 3 to 4 minutes each
on the T4's host. Its guards:

- `need_dh`: whether the layer passes a gradient to its input (the first
  layer does not). Two versions, as expected.
- `(B * T) % 32 == 0`: `_pad_to`'s `if not extra:` (the K padding for Y's
  GEMM). A piece whose length was a multiple of 32 got its own version.
- `B * T * 8192 >= 5,242,880`: Inductor fuses a "mix-order" reduction only
  past 5 * 2^20 elements, and with dynamic shapes that test is a guard.
- The length's power-of-two range (`_ceil_log2`). This one is by design:
  the exact sums' widths depend on it.
- A piece of one row (step 5, below).

The per-module path had the same causes in `_lora_backward` (13
recompiles). There were two smaller ones. Duck sizing tied the exp2 table's
64 entries to the 64 sampled rows (`_vocab_rows`). And the device's float64
and int8 decisions were first made inside a traced call, so their cache
dicts failed a guard on the next call.

The fixes:

- `_pad_to`, and `ExactLinear`'s `_int_mm` path, pad every length when
  compiled. A pad of 0 gives the same values.
- Inductor's `triton.mix_order_reduction_non_strict_mode` is on, so there
  is no size test. The reductions it fuses here are integer sums and maxima,
  the same in any order.
- The exp2 table is marked static.
- `_warm_attention` makes both device decisions before any compiled call.

The tests (`test_exact_lora.Recompiles`) check that one compiled graph
serves lengths in and out of each class. Each test fails with its fix undone.

`make_t4_llama_job.py rc3` ran the same 6 steps with the fixes
(`t4_kaggle/recompiles/rc3_*`). Its inputs were `rc`'s, and so were its
tokens, sampler integers and p(target) at every step, and the adapters at
step 6 (fcdf1cbf742fd812):

| arm | training s, steps 1, 2, 3, 5 | Dynamo's compile total | recompiles | the arm's wall time |
|---|---|---|---|---|
| whole-layer, before | 901, 703, 445, 431 | 2,701 s | 21 | 2,930 s |
| whole-layer, after | 955, 9, 8, 432 | 1,659 s | 13 | 1,846 s |
| per-module, before | 266, 69, 7, 169 | 787 s | 34 | 951 s |
| per-module, after | 231, 8, 7, 150 | 672 s | 26 | 825 s |

`_layer_backward` now compiles 6 times. At step 1 it compiles 4 versions:
`need_dh` times two length ranges. At step 5 it compiles 2 more: `need_dh`
again, for a piece of one row.

Why step 5: `GRPO.step` cuts the 64 rows into pieces of at most 1,024
tokens (rows times the longest row). At step 5 that gave pieces of 9 rows
and a last piece of 1. With one row, PyTorch specializes on the size 1, and
the causal mask has a real stride where it had 0. So every function
compiles again: 432 s on the T4. Cutting the rows evenly would avoid it.
But each piece quantizes on its own scales, so a different cut is a
different function, and the bits of every such step would change. This is
not done.

`make_t4_llama_job.py rc2` ran the fixed code a day later
(`t4_kaggle/recompiles/rc2_*`). It had other prompts: Llama 3's chat
template writes the day's date into each prompt (see
`../exact_llama_2026-11-24/README.md`). So its tokens differed from step 1,
and its counts (16 and 28 recompiles) are not a paired comparison.

## Not done

- A piece of one row compiles every training function again (above). An
  even cut of the rows would change the bits of the steps it changes.
- The adapters' backward still launches 44 kernels a call: each of its five
  products calls Y's int8 GEMM as its own op, which Inductor cannot fuse
  into. The whole step is not one CUDA graph (its shapes change every step).
- The SFT path's vocabulary softmax (no per-token weights) stays eager; its
  gradient multiplies by a Python constant, a rounding case not checked here.
