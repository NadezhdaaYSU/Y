# Y's int8 GEMM on Turing (sm_75), 2026-11-18

## Why

On a Tesla T4 the exact path's graphed decode step spends 3.75 of its 5.71
ms of GPU time in int8 linear layers, all on one cuBLAS kernel
(`cutlass_75_tensorop_i8816gemm_s8_64x64_tn_align16`) at 41% of the T4's
320 GB/s (`../exact_graphed_2026-11-18/`, job i). Neither int8 GEMM the exact
path has on Ada can take that work: Triton's int8 dot and Y's own GEMM
(`mma.sync.m16n8k32`) both need sm_80. Turing has `mma.m8n8k16` for int8
and `ldmatrix`, but no `cp.async`.

## What changed

`src/ptx_emitter/int8_staged_gemm.rs` emits Y's int8 GEMM for a target
below sm_80 with two substitutions; the tiles' indexing, the shared-memory
swizzle, the `ldmatrix` fragments, the K striping over `gridDim.z` and the
epilogue are the sm_80 kernel's.

1. **Each `m16n8k32` is four `m8n8k16` on the same registers.** From the PTX
   ISA's fragment tables (g = laneid >> 2, t = laneid & 3), m16n8k32's A
   registers a0..a3 hold (rows 0-7, k 0-15), (rows 8-15, k 0-15),
   (rows 0-7, k 16-31), (rows 8-15, k 16-31); b0 and b1 hold k 0-15 and
   16-31; d0,d1 and d2,d3 hold rows 0-7 and 8-15. Each is an m8n8k16
   fragment, so `{d0,d1} += a0.b0 + a2.b1` and `{d2,d3} += a1.b0 + a3.b1`.
   An integer sum is exact in any order.
2. **No `cp.async`.** A K tile is loaded into registers (`ld.global.v4`)
   where its cp.async was issued, with the registers zeroed first where the
   cp.async would have been zero-sized (past the last tile, a row past the
   matrix, a chunk past K). The registers are stored into the tile's stage
   (`st.shared.v4`) just before the barrier the cp.async's `wait_group` came
   before. That stage is the one tile it-1 used, whose last fragments were
   loaded before the previous iteration's barrier.
3. **Its own tiles** (below). Without cp.async a CTA has one K tile of loads
   in flight, so what pays is bytes in flight per thread and CTAs per SM,
   not stages.
4. A Turing CTA's 64 KB of shared memory is checked at compile time (the
   sm_89 limit, 99 KB, was the only one).
5. `ARCH_FEATURES` has a row for `mma.sync ... m8n8k16` at sm_75 and PTX ISA
   6.5, measured with ptxas 13.4 and Triton's 12.8 ("requires .target sm_75"
   at sm_72, "requires PTX ISA .version 6.5" at 6.4), with its probe in
   `tests/ptx_target_features.rs`. Before, every mma shape without a row of
   its own (all but m16n8k8, m8n8k4 and FP8) fell under the sm_80 row, so
   this kernel would have been refused.

**For sm_80 and later nothing changed**: the PTX of seven shapes for sm_80
and sm_89 is byte for byte what it was before this work (built from the
sources with and without it; the PTX cache keys on the library's hash).

## Checks

- **The layout, on the T4, before the emitter** (`probe_m8n8k16.py`,
  `t4_probe/`). One hand-written warp each, against a CPU int64 matmul,
  full-range int8 with -128 and 127 planted, 3 seeds: one m8n8k16 64/64
  outputs exact; four m8n8k16 as one m16n8k32 (fragments loaded in Y's
  m16n8k32 register order) 128/128 exact; the control, a1 and a2 swapped,
  128/128 wrong.
- **The CPU PTX interpreter** (`python/y_lang/interpreter.py`) now runs
  `ldmatrix` (not `.trans`) and the int8 mma (m8n8k16, m16n8k32; a sum past
  int32 is refused). It reproduces the sm_80 kernel, whose bits are
  established on GPUs, at the same 13 shapes (0 wrong), so its new instructions do
  not rest only on my reading of the tables. On it the sm_75 kernel gives
  0 wrong at 13 shapes: rows, columns and K past the tile, K split over
  gridDim.z, two and four stages, a 32-wide K step, every Turing default
  tile (`python/y_lang/tests/test_int8_mm.py`, `TheKernelOnTheInterpreter`).
- **Mutations.** Swapping a1 and a2 in the emitted mmas: caught (8960
  wrong). Not zeroing a skipped load: it passed every shape at first,
  because the interpreter's registers start at zero and every ragged K tile
  was loaded into fresh registers. Shapes where a full K tile goes through
  the loop's registers before a ragged one (K 224, 288, 416) catch it (5504
  wrong); they are in the test. The interpreter runs a block's warps in
  lockstep, so it cannot see a race across a barrier; that part rests on
  the argument in point 2 and on the GPU runs.
- **On the T4** (`t4_b/`, `t4_t/`): `int8_mm` gives `torch._int_mm`'s bits
  and a CPU int64 product's at 14 shapes (decode, prefill at 2048 rows,
  ragged, M = 16 and 17), and so does every tile variant at every decode
  shape. `test_int8_mm.py` passes on the T4, all 20 tests, its 8 GPU tests
  among them (row buckets, the largest licensed K with full-range products,
  a split launch, cuBLASLt's bits, zero rows, the caller's stream,
  `torch.compile` with no graph break).

The proofs (`proofs/Int8GemmSchedule.v`, `Int8GemmExact.v`) are about the
m16n8k32 form's schedule. That four m8n8k16 compute one m16n8k32's product
is checked (probe, interpreter, GPU runs), not proved.

## The tiles

`gemm_t4.py`, Tesla T4 (40 SMs, 4-core Xeon at 2.00 GHz), torch 2.13.0+cu126,
liby.so built on the Kaggle machine from these sources (Rust 1.97.1).
Weights cold: copies rotated past the T4's 4 MB L2. Min of 9 rounds of 50
calls, the variants in a shuffled order each round. Round 1 (`t4_b/`) swept
six tiles at M = 32; round 2 (`t4_b2/`) the neighbours of round 1's best at
M = 32, 128 and 2048. Time of `torch._int_mm` over Y's, same machine:

| M | Tile | q/k/v 1152x896 | o 896x896 | gate/up 4864x896 | down 896x4864 | lm_head 151936x896 |
|---|---|---|---|---|---|---|
| 32 | sm_80 rule's (64x64x64, 4 stages) | 0.96 | 0.96 | 0.88 | 1.33 | 0.85 |
| 32 | **32x64x128, 2 stages** | 1.38 | 1.34 | 1.10 | 1.53 | 1.03 |
| 128 | sm_80 rule's | 1.14 | 1.11 | 0.85 | 0.73 | |
| 128 | **64x64x128, 2 stages** | 1.28 | 1.28 | 1.09 | 1.26 | |
| 2048 | sm_80 rule's (128x128x64, 3 stages) | 0.71 | 0.79 | 0.78 | 0.82 | |
| 2048 | **128x128x64, 2 stages** | 0.92 | 0.88 | 0.91 | 1.02 | |

The rule for sm_75 (`int8_staged_tile`): M <= 32 the 32x64x128 tile, M <=
128 the 64x64x128 one, above that 128x128x64 where it fits in N and leaves
64 tiles, else 64x64x128. All two-stage (24 to 32 KB of shared memory, so
two to four CTAs an SM), all without register spills (100, 124 and 234
registers). Measured at those three row counts only. With the rule in place
(`t4_t/`, another T4): 1.39, 1.39, 1.31, 1.62 and 1.02 at M = 32.

At 2048 rows, where the GEMM is bound by arithmetic, cuBLAS keeps a small
lead (0.94x over a layer's four shapes); four m8n8k16 issue four times the
instructions of one m16n8k32, which Turing does not have.

## End to end

`tools/exact_throughput.py --loop graphed --stock-loop greedy` on the T4:
Qwen2.5-0.5B-Instruct, batch 32, 64 new tokens, compiled, best of 5, the
exact path's decode graphed against stock float16 on its faster loop there
(greedy), in turns inside one process. One Kaggle machine a job, three
processes in turn. Exact's speed over stock's (tokens per second per
sequence in brackets):

| Setup | Job e: the sm_80 rule's tile (`t4_e/`) | Job e2: the Turing tiles (`t4_e2/`) |
|---|---|---|
| Every switch, `torch._int_mm` | 1.42x (137.3 / 96.4) | 1.43x (130.0 / 90.9) |
| Every switch, Y's GEMM | 1.41x (134.8 / 95.5) | **1.61x** (146.8 / 91.2) |
| Defaults, Y's GEMM | 1.11x (104.2 / 93.6) | 1.25x (109.0 / 87.3) |

Every process passed the token checks: exact graphed == exact greedy ==
`generate()`, and stock greedy == `generate()`, on all 32 rows.

- With the Turing tiles, Y's GEMM makes the every-switch exact path 13%
  faster than on cuBLAS (146.8 against 130.0, one machine, two processes),
  and 1.61x faster than stock's faster loop.
- With the sm_80 rule's tile it gained nothing (134.8 against 137.3): that
  tile was slower than cuBLAS at four of the five decode shapes.
- At its defaults (float cache rows, PyTorch attention) the exact path on
  Y's GEMM is 1.25x faster than stock; on `torch._int_mm` it was 1.13x in
  `../exact_graphed_2026-11-18/` (job g, another machine).
- The first T4 runs of this work (2026-11-17) had the exact path 1.48x
  slower than stock there.

## Files

- `probe_m8n8k16.py`, `t4_probe/`: the layout probe and its T4 run.
- `gemm_t4.py`: same bits and timing against `torch._int_mm`; `t4_b/`
  (round 1), `t4_b2/` (round 2), `t4_t/` (the rule, and the T4 test run).
- `t4_e/`, `t4_e2/`: end to end with the sm_80 rule's tile and with the
  Turing tiles.
- `make_t4_turing_job.py`: writes the Kaggle jobs. Each builds liby.so and
  ysu_gpu_probe on the Kaggle machine and copies them over the installed
  wheel's.
- Each `t4_*/` holds the job's logs and its notebook console
  (`job_console.log`); `cargo_build.log` is the build on the Kaggle machine.
