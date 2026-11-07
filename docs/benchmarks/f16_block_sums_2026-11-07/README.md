# Block sums from K = 8192, whatever the tile (2026-11-07)

## The problem

From K = 4096 an F16 or BF16 GEMM of more than 32 rows sums K in blocks of
256: the mmas chain into a block sum that is added into the accumulator with
`add.rn.f32`. That needs one more register per accumulator, so a tile that
cannot hold them still chained K in the tensor core. The tile came from the
profile's measurement, a measurement at a nearby shape, or the analytic
model. So the accumulation, and with it the error, depended on which tile
was recorded:

- **The analytic model** gives 8192³ and 16384³ the 128x128 tile on 2x2
  warps: 128 accumulators a thread, no room. Every compile without a
  measured tile chained there. The committed `tests/gemm_f16_8192.ptx` and
  `_16384.ptx` were blocked only because the profile that emitted them
  held the 2026-10-20 re-tune's 4x2 warps.
- **This machine's `.ysu_hw_profile`** is dated 2026-10-17, before that
  re-tune. 9 of its 13 F16 tile lines are the older ones, 8192³'s among
  them (2x2 warps, 2 stages). `bf16_gemm_2026-11-07` compiled against it,
  and reported 8192³ at 3x cuBLAS's largest error.

cuBLAS splits K at 8192³: its trace there is a memset, then
`ampere_s16816gemm_fp16_256x128_ldg8_stages_32x3_nn`
(`f16_accuracy_2026-10-20/acc_2.json`). Each of its chains is half as long
as Y's.

## The change

`PtxEmitter::f16_block_sums_required(m, k)`: above 32 rows, from K = 8192,
unless `Y_F16_ACCUMULATE` chooses the mode. For such a GEMM,
`Autotuner::fit_f16_block_sums` runs a tile with no room on twice the warps
(on M where that tiles and leaves room, else on N) and at least 3 stages.
`resolve_split` applies it to every tile it returns: measured, borrowed,
analytic, and each candidate a measurement times. It does not apply it to
`Y_CTA_OVERRIDE`, which is taken as given. The autotuner prints a line when
it fits a tile.

At K = 4096 nothing changed. There a tile with no room still chains, which
is cuBLAS's own sum at every large shape measured, and 1-2% faster (below).

## Measurements

RTX 4070 Ti SUPER, idle (the desktop compositor only). `ab_tiles.py`: the
same CTA tile under `Y_CTA_OVERRIDE`, each arm checked exact on small
integers first, then on N(0, 1) inputs:
- max |C - R| / sum|a||b| ("maxn") and rel_F against float64, on a
  512 x 1024 corner;
- the share of outputs bit-identical to cuBLAS (`torch.mm(...,
  out_dtype=torch.float32)`).

Timing is the per-call minimum of 15 rounds, arms shuffled each round.

**2x2 warps (chain) against 4x2 (blocked), both on 3 stages**
(`ab_run1.log` BF16; `ab_run2.log` BF16 and F16):

| shape | blocked / chain speed | maxn: chain, blocked, cuBLAS (BF16) |
|---|---|---|
| 8192³ | 1.002, 1.007; F16 1.004 | 1.2e-6, 3.4e-8, 4.1e-7 |
| 16384³ | 0.994; F16 0.999 | 1.5e-6, 2.6e-8, 5.6e-7 |
| 2048 x 4096 x 14336 | 1.008, 1.004; F16 1.015 | 1.3e-6, 2.8e-8, 1.3e-6 |
| 2048 x 4096 x 11008 | 1.004; F16 1.006 | 1.3e-6, 2.8e-8, 1.3e-6 |
| 4096³ (K = 4096) | 0.998, 0.989; F16 0.988 | 7.0e-7, 3.9e-8, 7.0e-7 |
| 2048 x 11008 x 4096 | 0.979, 0.983; F16 0.985 | 7.4e-7, 4.2e-8, 7.4e-7 |
| 2048 x 4096 x 4096 | 0.990, 0.989; F16 0.990 | 7.8e-7, 4.2e-8, 7.8e-7 |

- **From K = 8192 the blocked tile is as fast** (0.994-1.015x). The chain
  is 3x cuBLAS's largest error at 8192³ and 16384³, and equal to cuBLAS's
  at 2048 rows, where cuBLAS chains too (100% the same bits). The blocked
  sum is 12-46x below cuBLAS's at every one of these shapes.
- **At K = 4096 it costs 1-2%,** and the chain gives cuBLAS's bits. So the
  rule starts at 8192.

**Stages** (`stages.log`, BF16, Y / cuBLAS):

| tile | 8192³ | 2048 x 4096 x 14336 | 2048 x 2048 x 8192 |
|---|---|---|---|
| 2x2 warps, s2 (chain) | 1.006 | 0.973 | 0.978 |
| 2x2 warps, s3 (chain) | 1.010 | 0.992 | 0.990 |
| 4x2 warps, s2 | **0.836** | **0.821** | **0.827** |
| 4x2 warps, s3 | 1.010 | 0.988 | 0.993 |
| 4x2 warps, s4 | 1.006 | 0.998 | 0.995 |

The 4x2 tile's registers hold one CTA an SM, so its own pipeline is all
that hides the copies, and 2 stages are not enough. The first version of the
change kept the profile's stages and gave 8192³ 0.842-0.854x cuBLAS on this
machine's profile (two runs of `bf16_bench.py`). So a fitted tile takes at
least 3 stages.

**Afterwards**, `bf16_gemm_2026-11-07/bf16_bench.py 8192x8192x8192` against
this machine's profile, three runs (the third is
`../bf16_gemm_2026-11-07/run3_8192_block_sums.log`). The profile's 2x2/s2
line is fitted to 4x2/s3:
- Y / cuBLAS 1.018, 1.015 and 1.013;
- largest error 4.5e-8 against cuBLAS's 4.2e-7 (it was 1.3e-6, 3x cuBLAS's);
- integers exact.

## Tests and mutants

`tests/ptx_f16_accumulate.rs`:
- `from_k_8192_a_tile_with_no_room_for_the_block_sums_runs_on_twice_the_warps`
  (no GPU) covers:
  - the analytic tile, F16 and BF16;
  - the K = 4096 control (2x2, chained);
  - 32 rows (compensated, untouched);
  - `Y_F16_ACCUMULATE=mma` (2x2, chained);
  - a measured 2x2/s2 line (4x2/s3) and a measured 2x2/s4 line (4x2/s4);
  - `Y_CTA_OVERRIDE` (2x2, chained).
- `the_fitted_tile_counts_every_product_once_and_beats_the_chain_on_the_gpu`:
  1024 x 2048 x 8192 on the device, on a lattice of outputs crossing every
  CTA:
  - integers exact and every output written;
  - largest relative error on same-sign inputs 2.9e-6 blocked, against
    1.05e-4 chained (the control).

`fit_mutate.py`: 9 mutants of the rule, each killed (`mutants.log`).
`tiles.sh PROFILE ELEM M N K ...` prints the tile and the count of
block-sum adds the compiler emits.
