# Automatic split K for the F16 GEMM, 2026-10-25

RTX 4070 Ti SUPER (66 SMs, 48 MB L2), driver 615.71.09, `ptxas` from CUDA 13.4,
PyTorch 2.13.0+cu130. Every timed run was contended by the 5% rule: the desktop
was in use (0-12% utilisation before the first batch, 26-28% before the second;
`speed/*/env.txt`), and no game ran. Status: **opt-in (`_splitk`, `Y_F16_ACCUMULATE=blocked_kahan`); no default changed.**

## What changed

A kernel named `<name>_splitk` (`PtxEmitter::gemm_split_k`, `SplitK::Auto`)
takes `_splitk<S>`'s four parameters (A, B, C, Partial) and its two entries.
`Autotuner::split_k_plan` chooses S and the tile:

- split only where the unsplit kernel's tile gives fewer CTAs than SMs, and the
  GEMM does at least `SPLIT_K_MIN_INTENSITY` = 96 flops per byte of A, B and C
  (2MNK / (2MK + 2KN + 4MN)), so not where it is bound to DRAM; the intensity is
  below min(M, N), so at most 32 rows never split;
- then S of 2, 4, 8 (each split at least `SPLIT_K_MIN_DEPTH` = 4096 deep) and a
  tile of `SPLIT_K_TILES` (128x128 4x2 s3, 64x128 4x2 s4, 64x64 4x2 s4) no
  larger than M x N, with the most CTAs from the unsplit count up to one an SM;
  a tie keeps the smaller S.

The rule's tile yields to `Y_CTA_OVERRIDE` and to a measurement of the same
shape and split, not to a borrowed or analytic one
(`Autotuner::autotune_split_planned`); the analytic model gives every split of
512 x 512 the unsplit 64x64 tile. Where the rule does not split, the main entry
is the plain kernel (the same SASS) writing C, Partial is not touched and
`_reduce` returns at once. A plain `@tile` GEMM keeps its launch; the CLI prints
the split it would take and the workspace.

## Design (step 2)

`census/`: the tile each shape gets from the profile, unsplit and in 2, 4 and 8,
no timing. The choice could not live in a plain kernel: every host passes three
pointers and a z = 1 grid, and none checks the parameter count, so a split plain
kernel would read a fourth pointer from past the argument array and compute only
split 0. A module-level workspace would race between streams; a serial split
waits on other CTAs. So the split is opt-in by name, with the existing launch
contract, and the reduction stays a second launch (the in-kernel last-CTA
reduction needs zeroed counters, a host obligation that fails silently).

## Speed (step 4)

`gen.sh` builds every column, `run_speed.sh` times them with `sbench.py` (CUDA
graphs, 21 interleaved rounds, minimum; two runs on integer and N(0, 1) data;
every column exact first; the reduction included), `analyze.py` and
`columns.py` summarise. `speed/ad`: the 7 shapes above 32 rows with fewer than
66 unsplit CTAs and the 12 decode shapes; `speed/b`: the other 32. The `auto`
column there is the first version of the rule (S = 4 on 128x128 at
512 x 512 x 16384 and 65536, S = 2 on 64x128 2x4 at 64 x 4096 x 11008 and
14336); the revised rule's three splits are the pinned columns
`s2_64x128w42s4` and `s4_64x64w42s4` byte for byte (`check/twins.txt`).

What the first version taught:

- **64 x 4096 x 11008 and 14336** are bound to DRAM unsplit (about 600 GB/s);
  every split was 0.88-0.99x. Hence the intensity bound.
- **At 512 x 512 the fewer splits were faster**: S = 2 on 64x128 (64 CTAs)
  1.085-1.089x unsplit at K = 16384, against 1.060-1.064x for S = 4 on
  128x128; level at 65536. And 64x128 runs faster on 4x2 warps than on 2x4 at
  every shape measured.
- **32 shapes with 66 or more CTAs**: no split was faster than unsplit beyond the
  run's own noise (26 slower, 0.78-0.99x; 6 level).
- **Decode**: splits were 1.04-1.08x at M = 1 and 8 with N = 14336 (B from
  DRAM), and 1.22-1.24x at N = K = 4096 with B in the L2, where the unsplit
  kernel's compensated sum costs most. A split of at most 32 rows adds in plain
  float32; not taken (accuracy unmeasured).
- **The empty `_reduce`** costs up to about 4% on the shortest kernels (1024³,
  32 x 4096 x 4096), 2% at 512 x 512 x 4096.

## Accuracy (step 5)

`accuracy/accuracy_auto.py` is `../f16_blocked_2026-10-25/accuracy/accuracy_blocked.py`
with three shapes added to its EXTRA list (seeds 106-108) and the "in 1 split"
header read; the nine distributions, seeds, float64 reference and metrics are
unchanged. `accuracy/gen_acc.sh` builds, at the seven shapes above 32 rows with
fewer than 66 unsplit CTAs: `Y_mma` (the chain, the control), `Y_unsplit` (the
default), `Y_auto` (the rule) and the split the first version of the rule, a
depth threshold of 2048 or no intensity bound would take. `accuracy/gate.txt`
(`gate.py`): with E = C - R, R the float64 product (32 sampled outputs per case
within 2.3e-16 S of `math.fsum`) and S = |A| @ |B|, rel_F = ||E||_F / ||R||_F
and max_norm = max |E| / S.

- **The rule's kernel is at or below cuBLAS in both, in all 63 cases** (27 of
  them its three splits; the other 36 are the unsplit default, as before).
  cuBLAS's rel_F over the splits': 6.5-25x on mixed signs, 9.3-39x on same
  signs, 2.4-20x under cancellation; max_norm 5.6-21x, 8.6-33x, 2.2-12.5x.
- **Not on every output**: the splits are farther from R than cuBLAS on up to
  6.6% of the mixed-sign outputs, none of the same-sign ones and 3.5-25% of the
  cancellation ones (most on the paired inputs).
- **Against the unsplit default**: rel_F 0.985-1.006x; max_norm 0.93-1.18x,
  above it on the cancellation inputs (paired at 512 x 512 x 16384: 1.18x).
  4 splits on 128x128 have the lower max_norm at 65536 (cuBLAS's 23-25x on mixed
  signs against 2 splits' 18-21x); both are below cuBLAS everywhere.
- The chain (`Y_mma`) is above cuBLAS in all 63, as recorded before.

## From DRAM (`speed/ad_dram`)

`run_speed.sh` with `DRAM=1`: each call reads the next of R copies of A and B,
R x (A + B) at least 144 MB (`sbench.py rot=R`), at the shapes whose operands
fit in the L2. The `auto2` column is the revised rule's kernel, `auto` the
first version's.

- 512 x 512 x 16384: the rule's 2 splits 1.06-1.11x unsplit, 1.02-1.07x cuBLAS.
- 256 x 256 x 16384: the rule's 4 splits 2.26-2.32x unsplit but 0.93-0.96x
  cuBLAS; 8 splits (2048 deep, below `SPLIT_K_MIN_DEPTH`) 1.01-1.04x. In the
  L2 the 4 were 1.7% ahead of the 8. Not changed: one shape, the two regimes
  disagreeing.
- 64 x 4096 x 4096: every split 0.98x unsplit or less (in the L2 1.10x).
- Decode at N = K = 4096: splits 0.96-1.01x (in the L2 1.22-1.24x).

## Compensated block folds (`F16Accumulation::BlockedKahan`, opt-in)

`Y_F16_ACCUMULATE=blocked_kahan` adds each block sum into the accumulator as
`Kahan` adds a tile sum (its rounding error summed apart, added once after the
K loop where the sum is finite): four float32 operations a block instead of
one, a third register per accumulator (140 -> 168 at 64x128 4x2, no spills).
The K loop's SASS is unchanged (65 instructions, 16 HMMA, no FADD).
`accuracy/acc_bk.json` (all nine distributions) and `acc_bk2.json` (normal,
offset, paired; blocks of 64 and 128 too); `speed/bk` (L2) and `speed/bk_dram`
(DRAM). Share of outputs farther from R than cuBLAS's, on the paired inputs,
and speed against the rule's kernel as it is:

| | blocks of 256 (now) | compensated 256 | compensated 128 | compensated 64 |
| :--- | :--- | :--- | :--- | :--- |
| 512 x 512 x 16384, paired | 25.2% | 21.4% | 15.3% | 10.8% |
| 512 x 512 x 65536, paired | 17.6% | 11.2% | 7.7% | 5.3% |
| 256 x 256 x 16384, paired | 23.8% | 21.8% | 15.8% | 11.4% |
| speed at 512 x 512 | 1 | 0.91-0.96x | 0.90-0.94x | 0.84-0.87x |
| speed at 256 x 256 | 1 | 0.97-1.00x | 0.93-0.99x | 0.87-0.98x |

Compensated blocks of 64 also take the N(0, 1) shares from 1.8-6.2% to
0.6-2.1% and cuBLAS's rel_F over Y's from 6.8-25x to 20-81x (blocks of 64 and
128 were measured on normal, offset and paired only). The per-output
share on cancelling inputs follows the error ratio, (2/pi) arctan(1/r) for
independent normal errors; what remains after compensation is the truncation
inside each block's chained mmas, which only shorter blocks reduce. At
512 x 512 x 16384 the cost takes the split from 1.04-1.08x cuBLAS's speed to
0.98-1.01x (256) or 0.91-0.92x (64). Not a default.

Kept as an opt-in knob (the default unchanged). Tests:
`tests/ptx_f16_accumulate.rs` checks its structure (each block sum folded once,
compensated, on the inner loop's exit; each compensation zeroed before the loop
and added once after it, behind the finite check) and on the GPU exact integer
products, a case the plain fold rounds away (A of ones, B's first block 2^23 and
31 blocks of 0.5: 8388608 blocked, 8388624 compensated, exact 8388623.5) and a
+Inf row staying +Inf; `tests/ptx_f16_split_k.rs` that each compensated split's
slot is the unsplit compensated kernel on its K range, bit for bit.

## Mutants (step 6)

`mutants/mutants.py`: 39 mutants of the rule (bounds, intensity, tile list,
tie order, shape check), of its tile's precedence (cache, measurement,
borrowing), of the `_splitk` form (grammar, the unsplit kernel writing C, the
empty reduction, the header), of the type checker and the CLI note, and of
`blocked_kahan`; each against `tests/f16_split_k_rule.rs`,
`tests/ptx_f16_split_k.rs` and `tests/ptx_f16_accumulate.rs` with the GPU
tests. 37 killed (`mutants.txt`, the tests each failed in `killed_by.txt`); the
2 that survive drop the tile-fits-M-x-N filter, which can change no choice
while the intensity bound is 96 or more (`split_k_choice`'s doc comment). The
restored sources pass and hash as before (`sources_before.sha256`).

## Checks so far

- `tests/f16_split_k_rule.rs` (the rule, each bound on both sides, a grid of
  shapes), `tests/ptx_f16_split_k.rs` and `tests/ptx_f16_accumulate.rs` pass,
  the GPU tests included (22 tests).
- `check/regen.txt`: all 60 committed `tests/*.ptx` regenerate the same under
  this compiler and HEAD's. `check/docgate.txt`: docgate's README checks give
  the same result on both trees.
