# The F16 GEMM's K-loop barrier between a tile's last two steps, 2026-10-18

At 512^3 Y's F16 GEMM takes 1.13-1.17x cuBLAS's time with the same 64x64
tile on the same 64-CTA grid (`../f16_square_2026-10-18/512/`). This record
looks for where, tries one change to the K loop, keeps it where it measured
faster, and finds a precision gap on the way. RTX 4070 Ti SUPER (48 MB L2),
driver 615.71.09, PyTorch 2.13.0+cu130, Triton 3.7.1. Every timing run was
started by `gpu_quiet_then.sh` (`../paged_attention_2026-10-18/`) at a
10-sample mean of at most 5% with no game running; each directory's
`queue.txt` holds the reading it started at. Every column is checked first;
interleaved rounds, minimum of 15 (`../f16_square_2026-10-18/bench_ab.py`).

## Where the 512^3 kernel's time goes

`profile/ncu_src.sh` runs `profile/prof512.py` (the committed
`tests/gemm_f16_512.ptx` of `a525d7e`, and `torch.mm` with a float32 output,
both checked exact) under Nsight Compute with per-instruction stall sampling
(`profile/src512.ncu-repz`). `profile/regions.py` and the cuts in
`profile/phases.txt` split the samples by region; `profile/hot_y.txt` and
`profile/hot_cublas.txt` list every SASS line with its samples. The source
page is exported with `ncu -i src512.ncu-repz --page source --csv
--print-source sass -k regex:<kernel> --launch-count 1`.

| region | Y samples | cuBLAS samples |
|---|---|---|
| setup and pipeline fill | 224 (125 + 99 on the first wait) | 222 |
| K loop | 666: math 323, barrier 107, long scoreboard 106, wait 68 | 576: math 383, wait 99, barrier 26, long scoreboard 29 |
| epilogue | 83 | 108 |

`ptxas` unrolls Y's 16-iteration K loop whole (128 HMMA in straight-line
code). The extra samples are in the loop, at its barrier and on long
scoreboards: each iteration began with `cp.async.wait_group; bar.sync`, then
the `ldmatrix` loads and the MMAs that need them, so every warp of the CTA
stopped on the barrier and then on the loads together.

## The change

CUTLASS's multistage order: the wait for the next K-tile and the barrier sit
between this tile's last two 16-deep K steps, and the next tile's first
fragments load while the last step's MMAs run. Two register sets alternate
between the steps. The prefetch slot is still freed by the barrier one
iteration earlier, and the wait still lets `stages - 2` groups stay pending.

`hand/mk_xbar.py` wrote it into the committed 512 kernel by hand first
(`hand/g512_xbar.ptx`); `hand/check.py` checked it bit for bit against the
committed kernel on random inputs (the MMAs are in the same order) and exact
on small integers, and a copy with one B fragment swapped failed both. Then
the emitter, behind a switch while it was measured: off, all 19 committed
F16 GEMM files came out byte for byte; on, 10 changed, those with three
stages or more. Tiles of two stages (2048 and up) are left alone: there the
wait would be for the copies the same iteration had just issued.

## Measured

Microseconds, run 1 / run 2 (`tables.py`). "before / after" is the committed
kernel's time over the new one's; above 1, the new one is faster.

`hand/`, 512^3:

| committed | hand-pipelined | cuBLAS | committed / hand |
|---|---|---|---|
| 6.661 / 6.676 | 6.687 / 6.672 | 5.873 / 5.924 | 0.996 / 1.001 |

`sizes/` (`ptx_before/`, `ptx_after/`):

| shape | before | after | cuBLAS | before / after | cuBLAS / after |
|---|---|---|---|---|---|
| 256 | 3.040 / 3.067 | 3.019 / 3.046 | 3.917 / 4.014 | 1.007 / 1.007 | 1.297 / 1.318 |
| 512 | 6.675 / 6.703 | 6.881 / 6.689 | 5.898 / 5.893 | 0.970 / 1.002 | 0.857 / 0.881 |
| 1024 | 29.784 / 29.871 | 29.437 / 29.416 | 28.926 / 28.920 | 1.012 / 1.015 | 0.983 / 0.983 |

`decode/`, `tests/gemm_f16_decode_m<M>.ptx` with the kernel renamed:

| shape | before | after | cuBLAS | before / after | cuBLAS / after |
|---|---|---|---|---|---|
| 1x4096x4096 | 21.197 / 20.739 | 20.249 / 19.891 | 27.104 / 26.639 | 1.047 / 1.043 | 1.339 / 1.339 |
| 4x4096x4096 | 21.943 / 21.830 | 21.724 / 21.642 | 53.924 / 54.425 | 1.010 / 1.009 | 2.482 / 2.515 |
| 8x4096x4096 | 23.004 / 22.824 | 22.976 / 22.794 | 50.811 / 50.463 | 1.001 / 1.001 | 2.211 / 2.214 |
| 32x4096x4096 | 31.375 / 31.093 | 31.268 / 30.985 | 24.037 / 23.772 | 1.003 / 1.003 | 0.769 / 0.767 |

B (33.5 MB) fits in the 48 MB L2 and the loop reads it again on every call;
with B read from DRAM these shapes were not measured.

`tiles512/`: nine tiles at 512^3 (`Y_CTA_OVERRIDE`, `ptx/`), each before and
after, all 18 bit for bit equal to the committed kernel on random inputs
(`check_all.py`):

| tile (cta_m, cta_n, cta_k, warps_m, warps_n, stages) | before | after | cuBLAS / before |
|---|---|---|---|
| 64, 64, 32, 4, 2, 4 | 6.825 / 6.639 | 6.748 / 6.650 | 0.884 / 0.890 |
| 32, 64, 32, 2, 2, 3 | 8.412 / 8.374 | 8.381 / 8.379 | 0.718 / 0.706 |
| 32, 64, 32, 2, 2, 4 | 8.499 / 8.576 | 8.504 / 8.569 | 0.710 / 0.689 |
| 32, 64, 32, 2, 2, 6 | 8.520 / 8.606 | 8.412 / 8.493 | 0.709 / 0.687 |
| 32, 64, 64, 2, 2, 3 | 8.535 / 8.576 | 8.463 / 8.494 | 0.707 / 0.689 |
| 64, 32, 32, 2, 2, 4 | 9.477 / 9.385 | 9.483 / 9.384 | 0.637 / 0.630 |
| 32, 64, 64, 2, 2, 4 | 9.795 / 9.774 | 9.807 / 9.723 | 0.616 / 0.605 |
| 32, 32, 64, 2, 1, 3 | 10.834 / 10.844 | 11.085 / 11.054 | 0.557 / 0.545 |
| 32, 32, 32, 2, 1, 4 | 11.950 / 11.806 | 11.978 / 11.816 | 0.505 / 0.500 |
| cuBLAS | 6.036 / 5.908 | | |
| Triton | 8.629 / 8.936 | | |

- The change is 1.2-1.5% faster at 1024^3, 0.7% at 256^3, 4.3-4.7% at
  M = 1 and 0.9-1.0% at M = 4, in both runs.
- At 512^3 it is level: six comparisons of the same tile give 0.970, 1.002,
  1.011, 0.998, 0.996 and 1.001. The 0.970 is run 1 of `sizes/`, whose
  samples sat at 6.83-6.85 us before and 6.88-6.92 after; the same tile
  varied by 3% between the sweep's runs (6.825 and 6.639 before).
- A reading, not measured: the earlier wait also leaves the copies half a
  K-tile less time to land, which would offset what the overlap gains.
- Tiles of half the size (two CTAs to an SM) take 1.23-1.47x the tuned
  tile's time and of a quarter 1.59-1.78x. The two-warp 32x32x64 tile is the
  one the change slows, by 1.9-2.3%; no committed kernel uses it.
- So neither the barrier's place nor the tile closes 512^3.

`table/` re-measures the README's square table with the new kernels
(`table.py`); 2048 and up did not change.

## Precision

`precision/` (all on the GPU, with the compiler of this change):

- `prec.py`: every pinned tile of `python/y_lang/tests/test_f16_gemm.py` at
  256 x 256 x 768 on same-sign inputs (|N(0, 1)| as f16) gives the same
  bits, 8.1e-6 below float64 on average and 9.8e-6 at most; a float32 sum of
  the exact 16-deep partial products is 4.3e-7 off at most, a half-precision
  result 4.9e-4 and a half-precision accumulation 3.2e-3.
- `vs_cublas_sizes.py`: the committed `tests/gemm_f16_<N>.ptx` give
  cuBLAS's bits exactly from 256^3 to 4096^3, on random normal and on
  same-sign inputs; at 8192^3 every element differs.
- `err_8192.py`: on same-sign inputs (rows 0-255 against float64) Y and
  cuBLAS are both 5.0e-5 below on average at 4096^3. At 8192^3 Y is 1.0e-4
  below and cuBLAS 3.2e-5, every element of both below. The float32 sum of
  exact partials is at most 1.2e-6 and 1.7e-6 off. The tensor core's
  accumulation, chained over the whole K loop, loses more than float32
  rounding, and cuBLAS at 8192^3 avoids part of it; recorded in
  `docs/known_issues.md`.
- `cublas.py`: cuBLAS alone at 256 x 256 x 768 and 256 x 256 x 4096.

## Tests and mutants

`python/y_lang/tests/test_f16_gemm.py` gains:
- `test_the_next_tile_is_waited_for_between_the_last_two_k_steps`. It pins
  the loop's structure: one wait for `stages - 2` groups after the only
  commit, then one barrier, then only the next tile's first fragments
  (addressed from `k_iter + 1`) and the last step's MMAs. Two-stage tiles
  keep the barrier at one end of the loop.
- `test_every_tile_rounds_alike_and_close`. Every pinned tile gives the same
  bits on same-sign inputs, within 2e-5 of float64; both half-precision
  controls exceed the bound.

On the compiler without the change, the structural test fails on the three
pinned tiles of three or four stages. `mutants/run.sh` applies seven mutants
of the new emitter code (`mutants/mut.py`), rebuilding each and running the
whole file; `mutants/summary.txt` has every one failing (the control row
passes). The ordering test that was there before catches two of them:

| mutant | failing tests |
|---|---|
| wait for one group fewer | 12 |
| no barrier | 16 |
| next fragments loaded above the wait | 13 |
| next fragments addressed from `k_iter` | 17 |
| no commit before the wait | 5 |
| two-stage tiles too | 2 (structure only; their output is right) |
| no first fragments before the loop | 14 |
