# F16 GEMM accuracy: three error metrics, nine input distributions, 2026-10-20

RTX 4070 Ti SUPER, driver 615.71.09, PyTorch 2.13.0+cu130 (CUDA 13.0),
Triton 3.7.1, NumPy 2.5.1; Y's compiler at c67119a under the re-tuned
profile. Nothing here is timed. The results reproduce bit for bit. Two runs
(`acc_1`, `acc_2`) drew the same inputs and gave the same output bits in every
column of all 99 cases. Two earlier runs of the 90 cases without 8192^3 match
them too (`repro_vs_first.txt`). A game (Sober) held the GPU throughout; it
changes timing, not results.

**Why this record exists.** The README said float32 accumulation made Y's
same-sign error "34-1290x below cuBLAS's". That was one statistic on one
distribution. The statistic was the mean *signed* normalized error, which
measures bias. The distribution was |N(0, 1)|. The range also left out M = 1,
where cuBLAS's mean signed error was the smaller. This record defines the
metrics and covers mixed signs, cancellation, magnitudes and long K. It also
compares the kernels output by output.

## Reference

A and B are f16. The product of two f16 numbers is exact in float64: its
significand has at most 22 bits and its exponent stays in range (the smallest
is 2^-48). So R = A.double() @ B.double() (cuBLAS's DGEMM) differs from the
exact sum only by float64 summation error.

This is checked on 32 sampled outputs per case, 3,168 in all. At each one, R is
compared with `math.fsum` of the same products, which returns the exact sum
correctly rounded. The largest deviations of R are:

- 2.3e-16 of S_ij;
- 5.0e-13 of |exact|;
- 6.9e-7 of the smallest error any column has at that output.

## Metrics

C is a kernel's float32 output, E = C - R, and

$$S_{ij} = \sum_k |a_{ik}|\,|b_{kj}| \;\ge\; |R_{ij}| .$$

The metrics:

$$\mathrm{rel}_F = \frac{\|C - R\|_F}{\|R\|_F}, \qquad
\mathrm{max}_{\mathrm{abs}} = \max_{ij} |C_{ij} - R_{ij}|, \qquad
\mathrm{max}_{\mathrm{norm}} = \max_{ij,\; S_{ij} > 0} \frac{|C_{ij} - R_{ij}|}{S_{ij}}$$

Two further statistics:

- **Mean signed normalized error,** mean(E_ij / S_ij), the bias. On
  same-sign inputs S = R, so it is the mean relative error.
- **Per-output comparison:** the fractions of outputs whose |E_ij| is smaller
  than, equal to, or larger than cuBLAS's, and the fraction bitwise equal.

**References near zero.** No error is divided by |R_ij|:

- **The relative error isn't used.** Under cancellation, a float32 sum can be
  off by about 2^-24 S_ij with every step correctly rounded. |E_ij| / |R_ij|
  then mostly measures the input's conditioning, κ_ij = S_ij / |R_ij|, rather
  than the kernel. An output's relative error is at most κ_ij times its
  normalized error.
- **When S_ij is zero.** S_ij = 0 only when every product is zero, and then
  R_ij is exactly 0. Such outputs are left out of max_norm and must be exactly
  0; `zero_scale_nonzero` counts any that aren't. None occurred, and no
  reference was exactly zero.
- **The Frobenius denominator.** ||R||_F is asserted positive.

Each case also records two measures of its conditioning:
κ_F = ||S||_F / ||R||_F and the median of κ_ij.

**Floor.** RN32 is R rounded to float32, the best output any kernel with a
float32 output could give. Its rel_F and max_norm must be at most 2^-24, and
are: the worst is 5.96e-8. `check_metrics.py` checks the metric code on
crafted outputs whose values are known by hand. `mutants.sh` runs it against
four mutants of the metric code, all caught, and one control it cannot see
(`mutants.txt`).

## Inputs and seeds

`accuracy.py`'s `DISTS`, in this order (d = 0..8). The κ_F column is the range
over the eleven shapes:

| d | name | A | B | tests | κ_F |
| :--- | :--- | :--- | :--- | :--- | ---: |
| 0 | normal | N(0, 1) | N(0, 1) | mixed sign | 10-163 |
| 1 | uniform_pm1 | U(-1, 1) | U(-1, 1) | mixed sign | 12-192 |
| 2 | abs_normal | \|N(0, 1)\| | \|N(0, 1)\| | same sign | 1 |
| 3 | uniform01 | U(0, 1) | U(0, 1) | same sign | 1 |
| 4 | offset | 1 + 0.01 N(0, 1) | N(0, 1) minus its column mean | cancellation | 1.3e3-2.0e4 |
| 5 | paired | [U \| U], U ~ N(0, 1) | [V ; fl16(-V + 2^-10 W)], V = fl16(N(0, 1)), W ~ N(0, 1) | heavy cancellation: the second half of K cancels the first | 1.5e4-2.3e5 |
| 6 | wide_range | ±(1 + U(0, 1)) 2^e, e uniform in -10..10 | the same | products from 2^-20 to 2^22 | 2.4-35 |
| 7 | small | N(0, 1) 2^-12 | N(0, 1) 2^-12 | magnitude: up to 19.8% of inputs f16 subnormals | 10-162 |
| 8 | large | N(0, 1) 2^7 | N(0, 1) 2^7 | magnitude: inputs up to 724 | 10-163 |

**Shapes** (s = 0..10): 256^3, 1024^3, 4096^3, M = 1, 8 and 32 with
N = K = 4096, 512 x 512 x K for K = 1024, 4096, 16384 and 65536, and 8192^3.

**Seeds.** Case (d, s) uses seed 1000 (d + 1) + s with
`numpy.random.RandomState(seed)`, whose stream NumPy keeps fixed across
versions. It draws A, then B (`paired`: U, V, W), in float64; NumPy rounds them
to f16, to nearest even.

**Exact inputs.** `acc_1.json` records, for every case:

- its seed;
- the SHA-256 of A's and of B's f16 bytes (row-major, little-endian);
- each operand's fractions of zeros and subnormals and its maximum magnitude;
- the SHA-256 of every column's output.

The arrays themselves, up to 64 MB per operand, are not committed.
`accuracy.py <bench_ab dir> <ptx dir> out.json --dump DIR normal:256` writes a
case's A and B as .npy, and the hashes identify them exactly.

## Columns

- **Y_mma:** each mma chained into its accumulator. This is Y's default above
  32 rows.
- **Y_f32:** each mma starts from zero and is added with `add.rn.f32`. This is
  Y's default up to 32 rows, and `Y_F16_ACCUMULATE=f32` selects it everywhere.
- **cuBLAS:** `torch.mm(a, b, out_dtype=torch.float32)`, with
  `allow_fp16_reduced_precision_reduction` off.
- **Triton:** `../f16_square_2026-10-18/bench_ab.py`'s autotuned `tl.dot`
  kernel, a chain with a float32 accumulator. It runs where 256 divides M and
  N and 64 divides K.
- **SGEMM:** `torch.mm(a.float(), b.float())` with TF32 off: float32 FMAs on
  the CUDA cores.
- **RN32:** the floor described above.

`gen.sh` compiles every kernel three ways: with no knob, with
`Y_F16_ACCUMULATE=mma` and with `=f32`. `gen.log` records that:

- each no-knob kernel is byte-identical to the committed `tests/*.ptx`;
- each no-knob kernel equals the mma-mode kernel above 32 rows and the
  f32-mode kernel up to 32;
- every kernel assembles for sm_89.

`ptx/` keeps the kernels not committed in tests/. To rerun, put tests'
`gemm_f16_{256,1024,4096,8192}.ptx` in `ptx/mma/` and
`gemm_f16_decode_m{1,8,32}.ptx` in `ptx/f32/`.

The cuBLAS kernels, from the profiler:

| shape | cuBLAS kernel | Y's chain bitwise equal to it |
| :--- | :--- | :--- |
| 256^3 | `cutlass_80_wmma_tensorop_s161616gemm_f16_32x32_128x2` | yes |
| 1024^3, 4096^3 | `ampere_s1688gemm_fp16_128x128_ldg8_stages_32x1` | yes |
| 512 x 512 x 1024 | `ampere_s16816gemm_fp16_64x64_ldg8_stages_64x5` | yes |
| M = 8 | `ampere_s16816gemm_fp16_128x64_ldg8_stages_64x3` | yes |
| M = 32 | `cutlass_80_wmma_tensorop_s161616gemm_f16_32x32_128x2` | yes |
| M = 1 | `gemvx`, no tensor cores | 0-0.24% of outputs |
| 512 x 512 x 4096 | `ampere_s16816gemm_fp16_128x64_ldg8_stages_64x4` | 0-1.0% |
| 512 x 512 x 16384 | `cutlass_80_tensorop_s16816gemm_f16_128x128_32x4` + `splitKreduce` | 0-0.07% |
| 512 x 512 x 65536 | `ampere_s16816gemm_fp16_128x128_ldg8_stages_64x3` + `splitKreduce` | 0-0.02% |
| 8192^3 | `ampere_s16816gemm_fp16_256x128_ldg8_stages_32x3` | 0-0.19% |

The two single kernels at 512 x 512 x 4096 and 8192^3 give bits that differ
from a chained sum and are 2-3x more accurate. How they order their sums was
not established.

## Results

The full per-case values of every metric and column are in `tables.md`.

**Y against cuBLAS.** The table gives cuBLAS's error over Y's, as rel_F /
max_norm, so a value above 1 means Y's error is the smaller. Each cell is a
range over the group's distributions. The last column is the share of outputs
where Y's |error| is larger than cuBLAS's.

- Mixed sign: normal, uniform_pm1, wide_range, small, large.
- Same sign: abs_normal, uniform01.
- Cancellation: offset, paired.

| shapes | Y | mixed sign | same sign | cancellation | outputs where Y is worse |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 256^3, 1024^3, 4096^3, 512 x 512 x 1024 | default (chain) | equal bits | equal bits | equal bits | 0 |
| 8192^3 | default (chain) | 0.34 / 0.31-0.38 | 0.31-0.37 / 0.32-0.37 | 0.34-0.35 / 0.32-0.36 | 80-100% |
| 512 x 512 x 4096-65536 | default (chain) | 0.13-0.51 / 0.10-0.53 | 0.12-0.49 / 0.12-0.49 | 0.014-0.55 / 0.024-0.55 | 71-100% |
| M = 8, 32 | default (f32) | 22-23 / 9.7-15 | 193-204 / 45-50 | 18 / 8.2-12 | 0-2.5% |
| M = 1 | default (f32) | 1.7-1.9 / 0.71-2.5 | 0.93-0.95 / 0.82-0.93 | 1.9-2.9 / 1.8-2.0 | 21-48% |
| the eight shapes above 32 rows | `Y_F16_ACCUMULATE=f32` | 4.7-23 / 2.5-14 | 26-202 / 11-40 | 1.04-18 / 0.83-9.7 | 0-45% |

**Where cuBLAS also chains K in the tensor core, Y's default gives cuBLAS's
bits.** That holds for all nine distributions. Triton's output equals Y's
chain bit for bit in all 72 cases it ran.

**Where cuBLAS does something else above 32 rows, it is more accurate than
Y's default chain.** That holds on every distribution and all three metrics:

- **512 x 512 x K ≥ 4096:** 2-8x more accurate, and up to 70x in rel_F on
  `paired` at K = 65536. At K = 16384 and 65536 cuBLAS splits K.
- **8192^3:** about 3x more accurate.

M = 1, where cuBLAS runs a GEMV, falls under float32 accumulation, below.

**Float32 accumulation (Y_f32) has the lower rel_F in 97 of the 99 cases and
the lower max_abs and max_norm in 94.** The exceptions:

- **M = 1, same-sign inputs:** cuBLAS's GEMV uses float32 FMAs and no tensor
  cores. It is 5-7% lower in rel_F, 8-22% in max_abs and 7-18% in max_norm.
- **M = 1, `wide_range`:** cuBLAS's max_norm is 0.71x Y_f32's.
- **`paired` at 512 x 512 x 16384 and 65536:** cuBLAS's split K is
  0.83-0.87x Y_f32's in max_abs and max_norm. In rel_F the two are level:
  Y_f32's is 0.96x and 0.94x cuBLAS's.

**Y_f32 isn't more accurate on every output.** It is worse than cuBLAS on:

- 47-48% of outputs at M = 1 with same-sign inputs;
- 21-30% at M = 1 on the other distributions;
- 32-45% for `paired` at 512 x 512 x K ≥ 4096;
- 1-10% on mixed signs elsewhere;
- no outputs on same-sign inputs anywhere but M = 1.

**Nor is it float32 summation.** Within each mma, the tensor core still
truncates as it adds the 16 products. That leaves a mean signed error of
-7.0e-8 to -8.0e-8 on same-sign inputs at every K, a bias that float32 FMAs
do not have. Against the float32 control:

- **Same sign:** SGEMM's rel_F is lower in 18 of 22 cases, at 0.15-0.96x
  Y_f32's. At 512 x 512 x 65536, SGEMM's 1.5e-7 is 0.15x of Y_f32's 9.9e-7.
- **Mixed sign:** SGEMM's rel_F is lower in only 6 of 55 cases, at
  0.59-0.96x Y_f32's: five of them at 512 x 512 x 65536, and `wide_range` at
  256^3. In the other 49, Y_f32's is 1.03-3.9x lower. That fits its two-level
  sum, 16 products in the tensor core and then a float32 add, though the
  cause was not isolated.

**Y_f32's error grows with K.** On |N(0, 1)|, rel_F goes from 1.0e-7 at K = 256
to 9.9e-7 at 65536. Only the bias stays at about -7.7e-8.

**The old figure reproduces as a bias ratio, with an exception.** The README's
"34-1290x" was the ratio of mean signed normalized errors on |N(0, 1)|. Here
it is 34-1300x at every shape but M = 1. At M = 1, cuBLAS's mean signed error
is +3.4e-9, 24x smaller than Y_f32's -8.0e-8.

**Cancellation.** At κ_F up to 2.3e5:

- No kernel's max_norm exceeds its own maximum on N(0, 1); Y_f32's is
  7.9e-8 at most.
- rel_F, which divides by ||R||_F, does grow. It reaches 1.2e-3 for Y_f32
  and for cuBLAS's split K, both at `paired` 512 x 512 x 65536, and 1.6e-3
  for SGEMM at `paired` 8192^3. That leaves about three correct digits.
- The chains do worse: cuBLAS's reaches 5.4e-3 at 4096^3, and Y's default
  8.7e-2 at 512 x 512 x 65536.
- The floor, RN32, is 2.6e-8 at most.

**Magnitudes.** Inputs scaled by 2^-12, with up to a fifth of them f16
subnormals, or by 2^7 give the same ratios against cuBLAS as N(0, 1). Their
cuBLAS / Y_f32 rel_F is 1.87-22.4 and 1.9-22.5, against 1.9-22.4.

Per distribution, as ranges over the eleven shapes (`analyze.py`; ratios are
cuBLAS / Y_f32 and SGEMM / Y_f32):

| distribution | κ_F | Y_f32 rel_F | cuBLAS / Y_f32: rel_F | max_abs | max_norm | outputs where Y_f32 is worse | outputs where Y's chain = cuBLAS bits | SGEMM / Y_f32 rel_F |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| normal | 10.2-163 | 9.89e-08-1.15e-06 | 1.9-22.4 | 1.65-14.9 | 1.63-14.9 | 0.0129-0.267 | 0.000179-1 | 0.697-3.93 |
| uniform_pm1 | 12.1-192 | 9.79e-08-1.15e-06 | 1.91-22.5 | 2.16-13.3 | 2.22-13.2 | 0.0128-0.269 | 0.000168-1 | 0.7-3.93 |
| abs_normal | 1 | 1.01e-07-9.92e-07 | 0.949-194 | 0.78-50.5 | 0.816-49.6 | 0-0.469 | 0-1 | 0.154-3.9 |
| uniform01 | 1 | 9.32e-08-8.39e-07 | 0.934-204 | 0.921-52.7 | 0.932-49.6 | 0-0.477 | 0-1 | 0.155-3.88 |
| offset | 1.29e3-2.04e4 | 5.56e-06-6.7e-05 | 2.93-17.7 | 2.02-11.2 | 2.04-11.2 | 0.0219-0.207 | 1.14e-05-1 | 1.17-3.87 |
| paired | 1.46e4-2.31e5 | 8.86e-05-0.00115 | 1.04-18.4 | 0.828-11.8 | 0.831-11.9 | 0.0211-0.453 | 0-1 | 0.971-3.92 |
| wide_range | 2.4-35.3 | 1.05e-07-1.15e-06 | 1.66-21.8 | 0.829-12.7 | 0.706-13.5 | 0.0132-0.296 | 0.00013-1 | 0.592-3.21 |
| small | 10.3-162 | 9.47e-08-1.15e-06 | 1.87-22.4 | 2.55-12.2 | 2.53-12.4 | 0.0123-0.27 | 0.000172-1 | 0.695-3.94 |
| large | 10.1-163 | 9.78e-08-1.15e-06 | 1.9-22.5 | 1.54-13.8 | 1.52-13.5 | 0.0124-0.269 | 0.00016-1 | 0.698-3.93 |

## Files

- **`accuracy.py`:** the harness. Its docstring states the definitions.
- **`check_metrics.py`:** the metric check, on CPU.
- **`mutants.sh`, `mutants.txt`:** the metric check against its mutants.
- **`analyze.py`:** checks that the two runs agree, checks the floor and the
  reference, and writes `tables.md` and `tables_summary.json`.
- **`tallies.txt`:** the group ranges and counts used above.
- **`gen.sh`, `gen.log`:** compile the kernels and record what each equals.
- **`queue.sh`:** runs both passes.
- **`acc_{1,2}.json`, `.log`:** the raw results of both runs.
- **`gpu.txt`:** the GPU and driver.
