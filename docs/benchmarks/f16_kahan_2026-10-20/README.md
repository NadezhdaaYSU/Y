# Compensated F16 GEMM accumulation (`Y_F16_ACCUMULATE=kahan`), 2026-10-20 to 2026-10-24

RTX 4070 Ti SUPER (66 SMs, 48 MB L2), driver 615.71.09, `ptxas` from CUDA
13.4, PyTorch 2.13.0+cu130. Accuracy runs compare output bits, which do not
depend on the clock; some ran while a game held the GPU (`*.env`). The
final speed runs (`speed/`, 2026-10-24) had no game running; the desktop
read 1-20% utilisation between runs (`speed/env.txt`), so by the 5% rule
they are contended. The first per-mma runs (`variants/1_kahan_per_mma/speed/`)
had the Steam client open.

## What changed

Since this record, this mode is the default for GEMMs of at most 32 rows
outside split K (`../f16_kahan_default_2026-10-24/`). As recorded here it
was opt-in.

`Y_F16_ACCUMULATE=kahan` (`F16Accumulation::Kahan`) is opt-in; without it
nothing changes (59 of the 60 committed `tests/*.ptx` regenerate byte for
byte, `check/regen.txt`; `test_drift` differs under every compiler, as
before). Under it, as under `=f32`, each mma starts from a zero register, and
then:

- each K-tile's mma results are summed in a register of their own, the
  tile sum `x` (one `add.rn.f32` per mma, as `=f32` adds into the
  accumulator);
- after the tile's last 16-deep step, `x` is folded into the accumulator `s`
  and the add's rounding error into a compensation `c`, and `x` is zeroed:

  ```
  add.rn.f32 s', s, x      // the tile sum into the accumulator
  sub.rn.f32 b, s, s'
  add.rn.f32 e, b, x       // e = (s - s') + x: what the add rounded off, exactly while |s| >= |x|
  add.rn.f32 c, c, e       // summed apart, never fed back into s
  ```

- after the K loop each tile sum is folded once more (a no-op where the last
  step already did, and what keeps the sum whole on a path that leaves a tile
  sum pending), and `c` is added into `s` where `s` is finite: `s - s` is +0
  exactly when it is, checked with `mov.b32` and `setp.eq.b32`, no float
  compare.

That is Kahan and Babuska's compensated summation with the error kept apart
from the sum, applied once per K-tile: four float32 operations per tile on
top of `=f32`'s one per mma, and three registers per accumulator. On the
decode tile (32x64x64, four 16-deep steps) that is 8 operations per tile and
accumulator where `=f32` has 4. Registers: 118 -> 139 on the decode tile,
124 -> 158 on 512 x 512 x 4096's 64x64 tile, 218 -> 255 on the 128x128 split
tile, none spilling (`speed/gen.txt`, `accuracy/gen_big.txt`). The plain
GEMM, the split-K GEMM and the fused SwiGLU kernel (both of its accumulator
arrays) take it.

## How it got there (`variants/`)

Each variant ran on the same shapes, distributions and harnesses
(`run_speed.sh` and `run_cold.sh` for variant 1, `run_speed2.sh` after; in
each variant's `speed/`, `kahan1` is variant 1 and `kahan2` that variant).
Decode is M = 1, N = K = 4096 from L2; minimum us.

| variant | ops per decode tile and accumulator | decode | 512 x 512 x 4096 | rel_F | an infinite input |
| :--- | ---: | ---: | ---: | :--- | :--- |
| `=f32`, no compensation | 4 | 17.6 | 28.4 | the baseline | +-Inf |
| 1. Kahan per mma | 16 | 23.9 | 39.2 | the lowest | **NaN** |
| 2. Kahan per K-tile | 8 | 20.4 | 39.0 | 1.0-1.43x variant 1's | **NaN** |
| 3. 2, `c` reset when not finite | 10 | 24.2 | 47.3 | 2's, bit for bit | +-Inf |
| 4. TwoSum per K-tile, error apart | 11 | 23.0 | 45.0 | not run | +-Inf |
| 5. Fast2Sum per K-tile, error apart (shipped) | 8 | 21.0 | 39.8 | 0.91-1.00x variant 2's | +-Inf |

1. **Kahan per mma** (`variants/1_kahan_per_mma/`): every mma's result added
   compensated, `y = x - c; s' = s + y; c = (s' - s) - y; s = s'`. Free from
   DRAM; 1.36x `=f32`'s time from L2 and 1.27-1.38x on the 512 x 512 shapes.
   Subtracting the last compensation after the loop moved no rel_F by 1e-4
   of itself in 36 cases, 11 of them bit for bit (`ablation/`), so that
   version had dropped it.
2. **Kahan per K-tile** (`variants/2_kahan_per_tile/`): the tile sum, added
   with Kahan once per tile, cost a third of variant 1's extra time. Same-sign
   accuracy is unchanged; on mixed signs and under cancellation the tile sums
   are as large as the result and their own rounding shows: rel_F 1.15x
   variant 1's on mixed signs, up to 1.43x under cancellation.
3. **Both Kahan forms turn +Inf into NaN** (`variants/*/inf_check.txt`): once
   `s` is infinite, `s' - s` is NaN, and `c` feeds the next add. Resetting a
   non-finite `c` after each fold (`testp.finite.f32` and a predicated
   `mov`, `variants/3_kahan_per_tile_guarded/`) fixed it, gave the same bits
   on finite inputs, and cost as much as variant 1: these kernels run one
   warp per scheduler, and two more dependent instructions per fold showed.
4. **TwoSum, error kept apart** (`variants/4_twosum_per_tile/`): `s' = s + x`,
   and the exact error of that add (Knuth's six operations) summed into `c`,
   which no longer feeds `s`, so an infinity needs one check at the end
   only. 23.0 us: the six operations cost more than the dependency they
   removed.
5. **Fast2Sum, error kept apart** (shipped): the same structure with
   Kahan's three-operation error, exact while `|s| >= |x|`, the condition
   Kahan's step also relies on. 21.0 us, and rel_F 0.91-1.00x variant 2's.

## Accuracy (`accuracy/`)

`accuracy_kahan.py` (`gen.sh` builds the kernels), at M = 1, 8 and 32 with
N = K = 4096, 512 x 512 x 4096 (unsplit and split 2), and split 4 and 16 at
512 x 512 x 65536, on the nine input distributions, seeds and float64
reference of `../f16_accuracy_2026-10-20/accuracy.py`, imported unchanged.
With R = A @ B in float64 (sampled outputs checked against `math.fsum`) and
S = |A| @ |B|:

- rel_F = ‖C − R‖_F / ‖R‖_F;
- max_norm = max_ij |C − R|_ij / S_ij;
- mean_norm = mean_ij (C − R)_ij / S_ij, the signed bias;
- per output, against cuBLAS, SGEMM and the `=f32` kernel of the same split:
  the fraction of (i, j) where |C − R|_ij is larger.

cuBLAS is `torch.mm(a, b, out_dtype=torch.float32)` (a GEMV at M = 1,
tensor-core kernels elsewhere; `accuracy/tables.md` names each); SGEMM is
float32 FMAs on the same f16 values, TF32 off. `accuracy/summary.md`
(`summary.py`) and `accuracy/tables.md` (`tables.py`) hold every figure below.

**Lower rel_F and lower max_norm than both cuBLAS and SGEMM in all 63
cases** (seven columns, nine distributions). `=f32` is not: at M = 1 it
loses to cuBLAS's GEMV in 2 of 9 (rel_F) and 3 of 9 (max_norm), and to SGEMM
at M = 8 and 32 in 2 of 9 (rel_F) and 7-8 of 9 (max_norm).

cuBLAS's rel_F over Y's, by input group (above 1, Y's error is the smaller):

| shape | Y | mixed sign (5) | same sign (2) | cancellation (2) |
| :--- | :--- | :--- | :--- | :--- |
| 1 x 4096 x 4096 | `=f32` | 1.7-1.9 | 0.93-0.95 | 1.8-2.9 |
| 1 x 4096 x 4096 | `=kahan` | 5.6-7.6 | 2.8-3.0 | 7.3-8.9 |
| 8 x 4096 x 4096 | `=kahan` | 73-89 | 584-616 | 55-71 |
| 32 x 4096 x 4096 | `=kahan` | 73-89 | 586-617 | 55-71 |
| 512 x 512 x 4096 | `=kahan` | 39-49 | 289-304 | 9.6-33 |
| 512 x 512 x 65536 | `=kahan`, split 4 | 138-166 | 1102-1178 | 16-115 |

Against SGEMM the ratios are 3.8-10 (mixed), 1.4-3.1 (same sign) and 6.4-14
(cancellation).

**Same-sign rel_F is 7.6e-8 to 8.6e-8 at every shape and K** (split 16:
9.3e-8 to 1.0e-7), against 1.1e-7 to 2.6e-7 for `=f32` and 1.1e-7 to 2.5e-7
for SGEMM. Most of it is the bias each mma's truncation of its own 16
products leaves, which no float32 summation can reach: mean_norm is -7.2e-8
to -7.8e-8 in every compensated column, as in every `=f32` one. Mixed-sign
rel_F is 6.8e-8 to 1.1e-7 (`=f32`: 2.1e-7 to 5.8e-7). Under cancellation:
paired inputs 6.2e-5 to 1.0e-4 (`=f32` 2.9e-4 to 8.2e-4, cuBLAS 5.5e-4 to
5.4e-3), offset inputs 4.7e-6 to 6.9e-6 (`=f32` 1.7e-5 to 5.3e-5).

**Not on every output.** Per output it is farther from R than cuBLAS's on
6.2-20.6% at M = 1 (the GEMV) and on 0-9.9% elsewhere; than SGEMM's on
4.2-34%; and than the `=f32` kernel's on 5.9-27%, where `=f32`'s rounding
happens to offset the truncation.

**With compensation, splitting K no longer helps accuracy.** At K = 65536
split 16 is a little worse than split 4 (same-sign rel_F 9.3e-8 to 1.0e-7,
against 8.3e-8 to 8.6e-8): the reduction adds the S partials in plain
float32, and with the splits' own error gone those adds count.

The fused SwiGLU kernel, `gemm_compute_schedule_gpu`'s two `--ignored`
checks against a float64 CPU reference, relative L2 (K = 32 and 128 only,
where accumulation matters little; `check/swiglu_modes.txt`):

| mode | 128 x 256 x 32 | 256 x 128 x 128 |
| :--- | ---: | ---: |
| mma | 1.61e-7 | 4.67e-7 |
| f32 | 1.16e-7 | 1.33e-7 |
| kahan | 1.16e-7 | 1.20e-7 |

## Infinities and NaN (`check/`)

`check/inf_check.py`: one +Inf, then one NaN, in row 3 of A (8 x 4096 x 4096,
B positive). Every mode and cuBLAS give +Inf across that row for the +Inf
and NaN for the NaN, and leave the other rows finite (`check/inf_check.txt`).
Both Kahan variants gave NaN for the +Inf.

## Speed (`speed/`)

`run_speed2.sh`: decode with `../f16_square_2026-10-18/bench_ab.py` (B
L2-resident, 32 MB) and `../decode_dram_2026-10-18/bench_cold.py` (four
33.5 MB copies of B, read from DRAM), and the 512 x 512 kernels with
`../f16_splitk_2026-10-20/sweep/sbench.py` (CUDA graphs, small-integer and
N(0, 1) timing data); every column checked exact first. Two runs, minimum us
(`speed/table.md` has all of it; `kahan1` there is variant 1, `kahan2` the
shipped form):

| M x 4096 x 4096 | `=f32` | `=mma` | Kahan per mma | shipped | cuBLAS |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1, from L2 | 17.58 / 17.64 | 18.04 / 18.04 | 23.90 / 23.97 | 20.95 / 21.00 | 26.63 / 26.70 |
| 4, from L2 | 17.65 / 17.69 | 18.10 / 18.18 | 23.99 / 24.11 | 21.00 / 21.17 | 53.66 / 53.67 |
| 8, from L2 | 17.82 / 17.75 | 18.19 / 18.09 | 24.13 / 24.01 | 21.13 / 21.09 | 50.47 / 50.55 |
| 32, from L2 | 20.07 / 20.24 | 20.01 / 20.16 | 23.79 / 23.84 | 21.27 / 21.42 | 23.15 / 23.28 |
| 1, from DRAM | 55.05 / 54.87 | 55.08 / 54.79 | 54.86 / 54.97 | 55.02 / 55.12 | 55.47 / 55.81 |
| 4, from DRAM | 55.07 / 54.91 | 54.91 / 54.91 | 54.83 / 55.00 | 54.76 / 55.08 | 56.61 / 56.41 |
| 8, from DRAM | 55.00 / 55.12 | 55.00 / 55.08 | 54.91 / 55.00 | 55.04 / 55.26 | 57.63 / 57.82 |
| 32, from DRAM | 55.21 / 55.19 | 55.07 / 56.42 | 55.31 / 55.22 | 55.32 / 55.35 | 55.95 / 56.19 |

**From DRAM, compensation costs nothing measurable:** the shipped form's
best of two is within 0.3% of `=f32`'s at every M, and the same kernel's two
runs differ by up to 2.4%. **From L2 it costs 1.19-1.20x at M = 1-8 and
1.06x at M = 32**, and is still faster than cuBLAS everywhere: 1.27x at
M = 1, 2.4-2.6x at 4 and 8, 1.09x at 32.

It is no candidate above the decode tiles: 512 x 512 x 4096 runs at
0.74-0.75x cuBLAS (`=f32` 1.04-1.05x), and split 4 at 512 x 512 x 65536 at
0.72x (`=f32` 0.98-0.99x). Those tiles take two 16-deep steps per K-tile, so
a fold is shared by two mmas, not four, and they run one 4-warp CTA per SM,
with little to hide the extra latency behind.

## Tests and mutants

`tests/ptx_f16_accumulate.rs`:

- the knob selects the compensated form at M = 8 and 256, and no form
  without it is compensated: every add of an mma's result goes into a tile
  sum that a fold adds into an accumulator;
- each tile sum takes one add per 16-deep step of the K-tile and is folded
  twice (end of tile, after the loop), in the GEMM and in both of the fused
  SwiGLU kernel's arrays;
- in a module of two GEMMs whose tiles hold different numbers of
  accumulators (asserted), each kernel's compensations and tile sums are its
  own registers (register numbers start again in each kernel), zeroed there,
  never an accumulator, and the compensations written only by the folds;
- on the GPU at 8 x 256 x 4096 and 128 x 128 x 4096, in every mode: exact
  integer products; a same-sign mean error within 1e-6; a spread (standard
  deviation of the relative error) below 0.3x `=f32`'s, measured 2.7e-8
  against 2.5e-7; and one +Inf in A giving +Inf across its row with every
  other output unchanged.

`mutants/mutants.py`: 17 mutants of the shipped form, all killed
(`mutants.json`): the knob and the default, each array's compensations, each
operand of the fold, the error never summed or never added in, the finite
check inverted or dropped, the fold cadence (only after the loop, after every
mma), the tile sum kept after its fold, no fold after the loop, and the
per-kernel reset. The earlier rounds (`mutants_first.json`,
`mutants_second.json` and `mutants_permma.json` for variant 1,
`mutants_twolevel_kahan.json` for variant 3) twice found the per-kernel reset
untested: the first two-kernel test's tiles held 16 accumulators each, so a
stale pairing named the same registers, and the test now asserts the counts
differ.
