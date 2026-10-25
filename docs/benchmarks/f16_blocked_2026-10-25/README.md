# F16 GEMM blocked summation, 2026-10-25

RTX 4070 Ti SUPER, driver 615.71.09, ptxas 13.4, PyTorch 2.13.0+cu130 (CUDA
13.0), NumPy 2.5.1; Y's compiler at b67116d plus this change, under the
re-tuned profile. The accuracy runs had a game (Sober) on the GPU, which
changes timing, not results; the timing ran with no game, on the desktop GPU,
whose compositor held 9-26% utilisation at idle just before the runs: contended.
`env.txt` beside each run holds the samples taken between runs.

## What changed

Above 32 rows Y's F16 GEMM chained its whole K loop in the tensor core's
accumulator (`mma d, a, b, d`), which truncates as it adds. cuBLAS does the
same at 256^3-4096^3 and at most of the LLM layer shapes, and there the two
give the same bits. At 8192^3, 512 x 512 x K >= 4096 and 64 x 4096 x 4096
cuBLAS splits the sum and is 2-70x more accurate
(`../f16_accuracy_2026-10-20/`, and `accuracy/` here for 64 x 4096 x 4096).

**Blocked summation** (`F16Accumulation::Blocked`, `Y_F16_ACCUMULATE=blocked`)
chains the mmas into a block sum of their own, adds it into the accumulator
with `add.rn.f32` after every block of K (`Y_F16_BLOCK_K`, default 256: eight
32-deep K-tiles) and zeroes it. The truncation then acts on one block's sum,
not on all of K. The block sums take one register per accumulator.

**The defaults** (`PtxEmitter::f16_accumulation`), with the user's approval
after the speed below:

- above 32 rows, unsplit, K >= 4096 and longer than a block: blocked, where
  the block sums fit (`2 x accumulators + 100` at most the registers a thread
  can have: 65,536 / threads, at most 255). The chain kept 54-103 registers
  besides its accumulators in the kernels measured; a 64 x 64 warp tile's 128
  accumulators do not fit (4096^3 blocked: 844 bytes of spill stores), so
  4096^3 and the LLM shapes on 128x128 tiles keep the chain, where cuBLAS
  chains too (same bits, `accuracy/`);
- a split of a split-K GEMM with more than 32 rows and a K longer than a
  block: blocked, where its block sums fit; other splits stay float32 (at
  M <= 32 cuBLAS runs a float32 GEMV that per-mma float32 adds already lose
  to on same-sign inputs, so blocks there were not tried);
- the fused SwiGLU kernel, blocked, asks for the CTAs per SM its shared
  memory allows (`.maxnreg` 128 for its 64x64 tile on 8 warps) where that
  leaves `2 x accumulators + 64` registers: 138 -> 122 registers, no spills,
  one CTA an SM -> two.

Below K = 4096 nothing changes: the chain gives cuBLAS's bits at every shape
measured with K <= 2048 (256^3, 1024^3, 2048^3, 512 x 512 x 1024: 36 cases).

## Where the fold sits

The first version put the fold behind a branch in the K loop's body, taken
at the end of every 8th K-tile. `ptxas` if-converted it (with `bra.uni`
too): the 64 FADDs and 32 CS2Rs were issued, predicated off, on every K-tile,
and 8192^3's K loop grew from 94 SASS instructions to 193 (`check/sass.txt`).

The folds now sit on the exit of an inner loop over one block's K-tiles.
A second version tested "a multiple of 8, and not all of K" after each tile;
`ptxas` fused the two tests into a two-predicate `BRA P0, ...` back edge,
which `tools/ptxas_tval/loopcfg.py` does not parse, so `unroll.py` refused
the four regenerated kernels. This version computes the block's end once,
`min(k + 8, K-tiles)`, below the K loop's exit test, and the inner loop runs
while the count is below it. Its K loop is 92 SASS instructions at 8192^3
(the chain's 94), the same as the chain's at the other tile kinds, and the
folds run once a block; `unroll.py` matches it. Every path out of the K loop
passes a fold, so nothing is added after it.

All three versions compute the same blocks in the same order: rerun on the
final kernels, every column of all 144 accuracy cases gave the same output
bits as on the first version's (`accuracy/versions.txt`, 1,260 columns).

## Accuracy

`accuracy/accuracy_blocked.py` runs `../f16_accuracy_2026-10-20/accuracy.py`'s
nine distributions, seeds, reference and metrics unchanged on every kernel in
`<ptx dir>/<shape>/<column>.ptx`. For E = C - R, R the float64 product and
S = |A| @ |B|:

- rel_F = ||E||_F / ||R||_F;
- max_norm = max_ij |E_ij| / S_ij over S_ij > 0;
- per output against cuBLAS: the share of outputs whose |E_ij| is larger.

Shapes outside that record's eleven (2048^3 and five LLM layer shapes) are
seeded as its case (d, s) would be with s = 100 + their index in `EXTRA`.
The 90 cases shared with `../f16_accuracy_2026-10-20/acc_1.json` have its
inputs and its output bits for the chain, float32, cuBLAS, SGEMM and RN32
(`accuracy/repro.txt`).

Below or level with cuBLAS in both rel_F and max_norm (`accuracy/totals.txt`):

| column | 8 shapes (`acc.json`, 72 cases) | 6 more (`acc2.json`, 54) | 4 splits (`acc_split.json`, 18) |
| :--- | :--- | :--- | :--- |
| chain (`Y_mma`) | 0 below, 36 same bits | 0 below, 45 same bits | 0 |
| float32 per mma | 70 | 54 | 18 (`Y_f32_s4`) |
| blocked 64, 128 | 72, 72 | -, 54 | 18 (128) |
| blocked 256 | 63, 9 same bits (256^3: one block) | 54 | 18 |
| blocked 512, 1024 | 63, 45 (the rest same bits) | - | - |

cuBLAS's error over blocked 256's, rel_F / max_norm, ranges over each
group's distributions (`accuracy/tables.md` has every column):

| shape | mixed sign | same sign | cancellation | outputs where blocked is worse |
| :--- | :--- | :--- | :--- | :--- |
| 8192^3 | 8.7-9.1 / 8.0-8.7 | 12-14 / 11-12 | 7.9-16 / 6.8-12 | 0-8.2% |
| 512 x 512 x 4096 | 6.6-7.0 / 6.1-7.9 | 9.3-9.4 / 8.8-9.0 | 2.9-5.7 / 5.3-5.9 | 0-27% |
| 512 x 512 x 16384 | 6.4-6.7 / 4.9-6.1 | 9.3-9.4 / 7.9-8.1 | 2.5-5.5 / 2.6-4.9 | 0-26% |
| 512 x 512 x 65536 | 22-24 / 13-17 | 38 / 27-28 | 3.5-20 / 3.4-12 | 0-18% |
| 64 x 4096 x 4096 | 6.6-7.0 / 6.8-9.9 | 9.3-9.4 / 8.9 | 2.8-6.0 / 5.2-6.2 | 0-27% |
| 1024 x 4096 x 4096 | 13-14 / 13-17 | 19 / 18 | 12-26 / 10-20 | 0-5.2% |
| 128 x 14336 x 4096 | 13-14 / 12-16 | 19 / 18 | 11-26 / 11-20 | 0-5.6% |
| 512 x 4096 x 14336 | 44-46 / 37-51 | 65-70 / 57-60 | 39-79 / 33-61 | 0-1.6% |
| 4 splits of 512 x 512 x 16384, 65536 | 6.6-26 / 4.8-26 | 9.3-39 / 8.7-35 | 2.6-21 / 3.7-16 | 0-24% |

- **Shorter blocks are more accurate**, roughly in proportion: on N(0, 1) at
  512 x 512 x 4096 rel_F is 2.1e-7 (64), 2.9e-7 (128), 4.9e-7 (256), 9.0e-7
  (512) and 1.7e-6 (1024), against cuBLAS's 3.4e-6 and float32's 3.0e-7.
- **Under heavy cancellation** blocked 256's rel_F is at most 3.8e-4 (the
  paired inputs), against 1.2e-3 for float32 accumulation and 8.7e-2 for the
  chain at 512 x 512 x 65536.
- **On same-sign inputs float32 per mma stays ahead** (cuBLAS's rel_F
  26-202x its own, against 9.3-70x blocked): each block still truncates. At
  8192^3 the mean signed error on |N(0, 1)| is -2.6e-6 blocked, -3.2e-5
  cuBLAS, -1.0e-4 chained, -7.7e-8 float32.
- **Not on every output:** under cancellation 0.7-27% of blocked 256's
  outputs are farther from R than cuBLAS's; on the other inputs at most 12%.

## Speed

`run_speed.sh` times each shape's kernels and cuBLAS with
`../f16_splitk_2026-10-20/sweep/sbench.py` (CUDA graphs, 21 interleaved
rounds, minimum), two runs each on small-integer and N(0, 1) data, every
column checked exact first; the fused kernels with
`../swiglu_tile_2026-10-20/bench_swiglu.py` and
`../quiet_2026-10-20/bias_relu/bench_bias.py`. `all_timing.sh` ran them in
turn; nothing ran while a game did. Ratios are minimum over minimum, as
ranges over the four (fused: two) runs; `speed/tables.md` has every column.

| shape | blocked 256 / chain speed | vs cuBLAS: chain -> blocked |
| :--- | :--- | :--- |
| 8192^3 | 0.977-0.985 (`one_block` run: 0.988-0.999) | 1.021-1.036 -> 1.005-1.020 |
| 16384^3 | 0.978-0.994 | 0.993-1.027 -> 0.980-1.005 |
| bias + ReLU 8192^3 | 0.990-1.007 | (vs addmm + ReLU) 1.12 -> 1.11-1.13 |
| SwiGLU 4096^3, register limit 128 | 0.971-0.975 (0.770-0.782 without it) | (vs PyTorch) 1.06-1.09 -> 1.04-1.06 |
| 512 x 512 x 4096 | 0.969-0.976 | 1.084-1.111 -> 1.052-1.077 |
| 512 x 512 x 16384 | 0.957-0.973 | 1.013-1.016 -> 0.970-0.985 |
| 512 x 512 x 65536 | 0.964-0.981 | 0.951-0.963 -> 0.920-0.935 |
| 22 LLM layer shapes | 0.963-1.021, median 0.989; 16 within 2% | `speed/tables.md` |
| 4 splits of 512 x 512 x 16384 | 1.032-1.033 over float32 splits | 1.006-1.008 -> 1.039-1.040 |
| 4 splits of 512 x 512 x 65536 | 1.023-1.059 over float32 splits | 0.976-0.991 -> 1.014-1.033 |

**Where the unsplit cost comes from.** Not the instruction count: blocked
256's K loop is the chain's length or shorter at every tile kind
(`check/sass.txt`). With one block of all K (`speed/one_block/`: the folds
once, after the loop) it still costs 2.5-2.7% at 512 x 4096 x 4096 and
1.1-1.9% at 512 x 512 x 65536, against 3.3% and 2.7-3.4% with blocks of 256.
In that kernel's SASS the K loop holds the chain's instructions, but `ptxas`
places the wait and barrier for the next K-tile after seven of the second
16-deep step's eight HMMAs, where the chain's has four after it, so the next
tile's `ldmatrix` loads are covered by fewer HMMAs. The folds every 8 K-tiles
add the rest, 0.6-1.6 points at these shapes.

Blocks of 512 and 1024 cost less at 8192^3 (0.982-0.995, 0.985-1.008) and
about the same at 512 x 512 (0.957-0.991); blocks of 64 and 128 cost more
(0.925-0.972, 0.934-0.984). 256 is the default: below cuBLAS at every shape
measured, where 1024 is only 1.4-2.1x below at 512 x 512 x 4096.

**Occupancy** (`check/occupancy.txt`): the block sums leave every GEMM tile
measured at its CTAs per SM, which registers or shared memory already set;
the fused SwiGLU kernel went from two to one until the register limit.

## Checks

- `tests/ptx_f16_accumulate.rs`: the default at K = 2048/4096/8192 and with
  128 accumulators a thread, the knob and `Y_F16_BLOCK_K`, the loop structure
  (`assert_blocked`), SwiGLU's two arrays and its register limit; on the GPU,
  exact integer products at K = 32 (the single-stage loop), 96, 2080 and
  4096 and the same-sign bias (-2.6e-6, 1/19 of the chain's).
- `tests/ptx_f16_split_k.rs`: the split default by rows, depth and fit; on
  the GPU each blocked slot bit for bit equal to the unsplit blocked kernel
  on its K range, and its bias (-2.6e-6 against the chained splits'
  -1.2e-5).
- `mutants/`: 30 mutants of the selection, the loop structure and the
  SwiGLU limit, each caught (`mutants.txt`; `mutants_cpu.txt` was the first
  24 without the GPU tests).
- `gates.txt`: the validator's gates against HEAD's tree.
- `check/regen.txt`: of the 60 committed `tests/*.ptx`, four change
  (`gemm_f16_8192`, `gemm_f16_16384`, `gemm_f16_bias_relu_8192`,
  `gemm_f16_swiglu_4096`) and `test_drift`'s latency comments, as under every
  compiler.

## Files

- `gen.sh`, `gen_split.sh`, `gen_fused.sh`, `regen_v3.sh`: the kernels;
  `gen_ptx*.log` their tiles, registers and spills.
- `accuracy/`: the harness, raw results (`acc*.json`), logs, environment,
  `tables.md`, `totals.txt`, `repro.txt`, `versions.txt` and its scripts.
- `speed/`: the raw timing (`core`, `split`, `llm`, `fusedrun`, `one_block`,
  `readme_table`), `tables.md`, and `run_speed.sh`, `run_fused.sh`,
  `all_timing.sh`, `speed_table.py`, `llm_table.py`, `fused_table.py` beside
  this file.
- `check/`: `sass.txt`, `sass_loops.py`, `occupancy.txt`, `occupancy.py`,
  `regen.txt`.
