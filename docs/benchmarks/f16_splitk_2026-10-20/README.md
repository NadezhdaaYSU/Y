# Split-K F16 GEMM, 2026-10-20

RTX 4070 Ti SUPER (66 SMs, 48 MB L2), driver 615.71.09, `ptxas` from CUDA
13.4, PyTorch 2.13.0+cu130. Every timed run below was contended by the user's rule
(GPU utilisation over 5% before the run, with the desktop in use and no game
running); `env*.txt` beside each result logs it.

## What changed

An `@tile(M, N, K)` kernel named `<name>_splitk<S>` (S from 1 to 64) with
four parameters `(A, B: GlobalMemory<F16>, C, Partial: GlobalMemory<F32>)`
is a split-K GEMM:

- the main entry runs K range `z = %ctaid.z` of every tile (K / S deep, a
  multiple of 8) and writes its M x N float32 product to `Partial[z]`;
- a second entry, `<name>_reduce`, computes `C = ((P[0] + P[1]) + P[2]) + ...`
  with `add.rn.f32`, in slot order for every element, with no atomics;
- every split adds each mma's result in float32 (`Y_F16_ACCUMULATE=mma`
  chains instead);
- the PTX header states both launches; the type checker and the emitter apply
  the same two functions (`PtxEmitter::gemm_split_k`, `gemm_split_k_shape`),
  and a malformed suffix, or the suffix on another shape, is refused - the
  four parameters are also the bias form's.

The autotuner keys a split kernel's tile by the whole K and the split count
(`AUTOTUNE_F16_SPLITK<S>_MxNxK_<gpu>`; unsplit keys are unchanged), measures
the split kernel with its reduction (`tune_gemm_f16_split`,
`CudaContext::time_launch_sequence`), and adds the large-square tiles that fit
to the candidates (`generate_split_candidates`): 512 x 512's own bucket has no
128x128 tile, and that is the tile that measured fastest at long K.

## Correctness

`check/check.py` (`check/gen_check.sh`), at 512 x 512 x 4096 and the ragged
200 x 136 x 4000 (each split 1000 deep: a partial K-tile per split), every
output NaN-filled first, each check with its control:

- small integers equal float64, for S = 1 and 4, both accumulation modes;
- `_splitk1` gives the unsplit float32 kernel's bits;
- slot z gives, bit for bit, the unsplit float32 kernel run on contiguous
  copies of A's and B's K range z (control: slot 0 differs from slice 1);
- C is the in-order float32 sum of the slots read back (control: the reverse
  order differs in 111,458 and 11,697 outputs);
- 20 relaunches give the same bits.

`tests/ptx_f16_split_k.rs` keeps these on the device (128 x 128 x 4096 in 4,
72 x 136 x 1000 in 5), plus a same-sign precision test with its control (the
chained splits' mean error is visibly negative, float32's below 1e-6), the
name grammar and refusals, the profile key, the tuner measuring a split kernel
end to end, and the timed launch sequence running the reduction.
`mutants/`: 16 mutants of the change (slot addresses, offsets, A's row
stride on both copy paths, the default mode, the grammar, the type check, the
tuner key, candidates and launches), all killed; the 16th survived the first
file and is killed by the timed-sequence check added for it.

All 60 committed `tests/*.ptx` regenerate byte for byte from this compiler and
from 799c500's (`check/regen_*.txt`; `test_drift` differs from its commit
under both, as before).

## Accuracy

`accuracy_split.py` (`accuracy/gen_acc.sh` builds its kernels), 512 x 512 x K
for K = 4096, 16384 and 65536, on the nine input distributions, seeds and
float64 reference of `../f16_accuracy_2026-10-20/accuracy.py`, imported
unchanged (each case records the SHA-256 of A, B and every C). With
R = A @ B in float64 (sampled outputs checked against `math.fsum`) and
S = |A| @ |B|:

- rel_F = ‖C − R‖_F / ‖R‖_F;
- max_abs = max_ij |C − R|_ij;
- max_norm = max_ij |C − R|_ij / S_ij;
- mean_norm = mean_ij (C − R)_ij / S_ij, the signed bias;
- per output, against cuBLAS: the fractions of (i, j) where |C − R|_ij is
  smaller, equal or larger than cuBLAS's.

Columns: `Y_f32` (unsplit, float32 accumulation), `Y_f32_s<S>` (split in S),
`Y_mma_s<S>` (split in S, each split chained in the tensor core's
accumulator: the control that separates splitting from float32 accumulation),
cuBLAS (`torch.mm(a, b, out_dtype=torch.float32)`), SGEMM (float32 FMAs on
the CUDA cores on the same f16 values, TF32 off) and R rounded to float32.
`acc_2` (`tables_2.md`; `ratios_2.md` from `ratios.py`) repeats `acc_1`
with `Y_mma_s2` added; every C the two runs share has the same SHA-256.

**Split float32 has the lower rel_F than cuBLAS in 27 of 27 cases at every
S, and the lower max_norm in 27 of 27 for S ≥ 4.** At S = 2 and unsplit
it is 25 of 27: both lose max_norm on the paired inputs at K = 16384 and
65536 (cuBLAS's over Y's 0.76-0.87). cuBLAS's error over Y's, rel_F /
max_norm, ranges over each group:

| Y | K | mixed sign (5) | same sign (2) | cancellation (2) |
| :--- | ---: | :--- | :--- | :--- |
| split 2 | 4096 | 15-16 / 9.1-12 | 165-168 / 37-42 | 2.0-9.3 / 2.2-6.0 |
| split 4 | 16384 | 11 / 7.9-8.7 | 164-166 / 38-39 | 1.5-5.7 / 1.2-3.3 |
| split 4 | 65536 | 22-23 / 12.5-18 | 388-404 / 91 | 1.5-12 / 1.4-6.1 |
| split 16 | 65536 | 42-43 / 36-39 | 852-858 / 222-230 | 2.9-22 / 4.0-15 |

It is not more accurate on every output. Per output against cuBLAS, split 4
is farther from R on 2.4-4.8% of the outputs of the mixed-sign inputs, none
of the same-sign ones, and 5.2-36% of the cancellation ones (the paired
inputs: 25% at K = 4096, 36% at 16384 and 65536); split 16 on 1.4-2.9% and
3.2-24%.

**The splits do not remove the bias each mma's truncation leaves.** On the
same-sign inputs mean_norm is -7.0e-8 to -7.8e-8 in every float32 column, at
every S and K; cuBLAS's is -2.2e-5 to -2.0e-4 and SGEMM's -1.3e-9 to
-5.4e-9. So against SGEMM, the float32 control, split 4 has the lower rel_F
on mixed-sign and cancellation inputs at every K but is 0.58x on same-sign
inputs at K = 65536 (2.62e-7 against 1.52e-7 on |N(0, 1)|); split 8 is
level there (0.96-1.00x) and split 16 is 1.22-1.29x.

**cuBLAS at K = 16384 and 65536 is Y's chained split 8, bit for bit,** in all
18 cases: cuBLAS splits K in 8 (`splitKreduce_kernel` follows a CUTLASS or
`ampere_s16816gemm` 128x128 kernel), chains each split, and adds the eight
partials in order in float32. At K = 4096 cuBLAS runs one kernel
(`ampere_s16816gemm_fp16_128x64_ldg8_stages_64x4_nn`, no reduction kernel),
and its bits are Y's chained split 2 in all 9 cases (`acc_2`): two chained
halves of K added in float32.

The split columns' bits do not depend on the tile: split 4 with 64x64 and
with 128x128 tiles gave the same C in every case (`Y_f32_s4_t128`).

## Speed

`final/` (`gen_final.sh` builds the kernels, `run_final.sh` runs them,
`final_table.py` writes `table.md`): `sweep/sbench.py` at 512 x 512 x K,
every column first checked exact against float64 on small integers, on a
NaN-filled output, and again after timing; each column a CUDA graph of 50, 20
or 6 calls, replayed in 21 interleaved rounds with a rotated order after 3
warm-up rounds; the minimum per call, cuBLAS's minimum over Y's (above 1, Y is
faster). Two runs, each timed on the small integers and on f16 N(0, 1) inputs
(normal inputs lower the clock). The unsplit kernels take the re-tuned
profile's tile; the split ones are pinned with `Y_CTA_OVERRIDE`. Contended:
the desktop was in use (`env.txt`: 1-19% utilisation in the idle seconds
before each run, no game process).
Ranges over the four:

| K | Y | tile, warps | us per call | cuBLAS / Y |
| ---: | :--- | :--- | ---: | ---: |
| 4096 | unsplit, chained (default) | 64x64, 2x2 | 27.5-27.6 | 1.082-1.087 |
| 4096 | unsplit, float32 | 64x64, 2x2 | 28.4-28.7 | 1.047-1.052 |
| 4096 | split 2, chained | 64x64, 4x2 | 28.2-28.5 | 1.052-1.059 |
| 4096 | split 2, float32 | 64x64, 4x2 | 28.9-29.5 | 1.017-1.034 |
| 4096 | split 4, float32 | 64x128, 4x2 | 30.4-30.5 | 0.980-0.985 |
| 4096 | cuBLAS | | 29.8-30.0 | |
| 16384 | unsplit, chained (default) | 64x64, 2x2 | 104.8-106.2 | 1.008-1.019 |
| 16384 | unsplit, float32 | 64x64, 2x2 | 108.6-111.8 | 0.956-0.981 |
| 16384 | split 4, chained | 128x128, 4x2 | 101.9-102.2 | 1.046-1.048 |
| 16384 | split 2, float32 | 64x64, 4x2 | 104.6-108.6 | 0.986-1.020 |
| 16384 | split 4, float32 | 128x128, 4x2 | 105.8-106.1 | 1.008-1.009 |
| 16384 | split 8, float32 | 128x128, 4x2 | 110.4-110.7 | 0.965-0.967 |
| 16384 | cuBLAS | | 106.6-107.1 | |
| 65536 | unsplit, chained (default) | 64x64, 2x2 | 416.4-435.4 | 0.929-0.962 |
| 65536 | unsplit, float32 | 64x64, 2x2 | 435.8-459.1 | 0.878-0.919 |
| 65536 | split 4, chained | 128x128, 4x2 | 387.9-397.7 | 1.014-1.040 |
| 65536 | split 4, float32 | 128x128, 4x2 | 404.1-412.3 | 0.980-0.991 |
| 65536 | split 8, float32 | 128x128, 4x2 | 413.0-421.2 | 0.957-0.970 |
| 65536 | split 16, float32 | 128x128, 4x2 | 431.6-440.1 | 0.916-0.928 |
| 65536 | cuBLAS | | 400.7-404.5 | |

**Faster than cuBLAS with the lower rel_F, max_abs and max_norm in 9 of 9
distributions:** at K = 4096 the unsplit float32 kernel (1.05x) and split 2
(1.02-1.03x, lower errors still), and at 16384 split 4 (1.008-1.009x).
Per output they are still farther from R than cuBLAS on 2.7-4.8% of the
mixed-sign outputs and 28-36% of the paired ones (Accuracy above).
**Not at 65536:** split 4 in float32 is 1-2% slower than cuBLAS. Splitting removes the chain's slowdown there (chained split 4 is
1.01-1.04x), and float32 accumulation then costs about 4% (2.5-3.5% at
K = 4096), as it does unsplit; more splits cost more than they balance. Each
timed call includes the reduction's launch (S x 1 MB of partials read once).

The grid is the reason K = 65536 wants splitting at all: 512 x 512 in 128x128
tiles is 16 CTAs on 66 SMs. Split 4 is 64 CTAs (97% of the SMs); split 8, 128
CTAs in two waves of 66 and 62, gains no balance and doubles the partials.

## What did not work

**Stream K** (`streamk/`, the patch kept as `streamk.patch`): P CTAs, each an
equal share of all tiles' K-steps, crossing tile boundaries, pieces summed by
the same kind of fixed-order reduction. It was correct (every piece bit for
bit equal to the unsplit kernel on its K range; `check_stream.py`) and slower:
at 512 x 512 x 65536, P = 66, 565 us against even split's 405 (float32), 471
against 387 (chain). `ncu` (`prof/`): DRAM read 294 MB against 135 MB (the
operands are 128 MB), L2 hit rate 61% against 81%. An even split keeps every
CTA at the same K, so the four tiles sharing A's rows read each slice
together; stream K spreads them over all 65536 of K. Numbering K-blocks first
(B blocks, unit = K-block x tile) helped and was still slower: B = 4 419.7 us
(217 MB from DRAM), B = 8 518, B = 16 850. The two idle SMs of a 64-CTA grid
cost 3%; keeping them busy cost more in DRAM traffic.

**Reordering the float32 adds** in the PTX (after each mma, after the K step,
one group late) gave byte-identical SASS. Float32 accumulation's cost at the
128x128 tile (ncu, 512 x 512 x 65536 in 4): the SM clock 2.703 -> 2.655 GHz
(power), the tensor pipe 49.5% -> 47.0% active, registers 148 -> 218.

**The tuner's choice at long K is not resolved.** Three `--autotune` runs of
512 x 512 x 65536 in 4 (`tuner/`) put 128x128x32 4x2 first by minimum every
time (413-424 us) and 64x128x32 2x4 second, but the medians sat 7-11% above
the minima (power-throttled rounds on random data), so 5-19 candidates fell
inside the tie band and the structural tie-break (four warps, deeper K tile)
picked a 4-warp tile at 443-445 us, a different one each run. The tables
above pin the split tiles with `Y_CTA_OVERRIDE`.
