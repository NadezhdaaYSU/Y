# Y's BF16 GEMM against cuBLAS, 2026-11-07

`@tile(M, N, K) kernel g(A, B: GlobalMemory<BF16>, C: GlobalMemory<F32>)`:
A `[M][K]` and B `[K][N]` row-major, C f32. This is the F16 tensor-core GEMM
with `.bf16` mmas. The baseline is cuBLAS's bf16 GEMM with f32 output,
`torch.mm(A, B, out_dtype=torch.float32)`: the same layout, bf16 inputs, f32
accumulation, f32 output.

## Setup

- RTX 4070 Ti SUPER (66 SMs), driver 615.78.08, torch 2.13.0+cu130, CUDA
  runtime 12.9. TF32 off.
- Y compiled against this machine's `.ysu_hw_profile`. Its tile is the F16
  GEMM's: the profile's measured tile for the shape, or the nearest measured
  one.
- The GPU was idle (the desktop compositor only) for both runs, each a fresh
  process. Clock, power and temperature at the end are on each log's last line.

## Method (`bf16_bench.py`, `run.sh`)

At every shape, before timing:
- **exactness:** small integers in [-8, 8], where every product and partial
  sum is exact, so C must be the integer product bit for bit (C is poisoned
  with NaN first). Every row passed;
- **bit identity:** on normal random inputs, the share of outputs whose bits
  equal cuBLAS's;
- **accuracy:** each one's largest `|C - exact| / sum|a||b|`, against a
  float64 product of the same bf16 values.

Timing: per-call time, the minimum over 15 rounds, the two kernels' order
shuffled every round, each a batch of 3 to 500 calls.

## Results (two runs; speed as the range over both)

| M × N × K | cuBLAS µs | TFLOPS | Y's tile | Y / cuBLAS | same bits | max error, Y | max error, cuBLAS |
|---|---|---|---|---|---|---|---|
| 256 × 256 × 256 | 3.6 | 9 | 32x64x32 2x2w s4 | 1.36 | 100.0% | 1.8e-07 | 1.8e-07 |
| 512 × 512 × 512 | 5.8 | 46 | 64x64x32 4x2w s4 | 1.07–1.08 | 100.0% | 3.1e-07 | 3.1e-07 |
| 1024 × 1024 × 1024 | 29.0 | 74 | 64x128x32 2x2w s3 | 1.05–1.06 | 100.0% | 4.5e-07 | 4.5e-07 |
| 2048 × 2048 × 2048 | 208.8 | 82 | 128x128x32 2x2w s2 | 0.99–1.01 | 100.0% | 6.5e-07 | 6.5e-07 |
| 4096 × 4096 × 4096 | 1593.0 | 86 | 128x128x32 2x2w s2 | 0.98–0.99 | 100.0% | 1.0e-06 | 1.0e-06 |
| 8192 × 8192 × 8192 | 12677.5 | 87 | 128x128x32 2x2w s2 | 1.00–1.01 | 0.2% | 1.3e-06 | 4.3e-07 |
| 8192³, since the tile fit (below) | 12515.4 | 88 | 128x128x32 4x2w s3 | 1.013–1.018 | 0.5% | 4.5e-08 | 4.2e-07 |
| 1000 × 1000 × 1000 | 29.0 | 69 | 64x128x32 2x2w s3 | 0.93–0.96 | 24.6% | 4.2e-07 | 4.2e-07 |
| 4000 × 4000 × 4000 | 1559.2 | 82 | 128x128x32 2x2w s2 | 0.98 | 100.0% | 1.0e-06 | 1.0e-06 |
| 768 × 768 × 768 | 22.3 | 41 | 64x128x32 2x2w s3 | 1.02 | 100.0% | 4.2e-07 | 4.2e-07 |
| 1024 × 1024 × 8192 | 198.2 | 87 | 64x128x32 2x2w s4 | 0.99 | 0.1% | 3.4e-08 | 1.2e-06 |
| 32 × 4096 × 4096 | 33.0 | 32 | 32x64x64 2x2w s3 | 1.50–1.51 | 0.3% | 8.1e-09 | 6.7e-07 |
| 32 × 11008 × 4096 | 176.0 | 16 | 32x64x32 2x2w s3 | 1.15–1.16 | 0.3% | 6.7e-09 | 7.1e-07 |
| 32 × 4096 × 11008 | 160.1 | 18 | 32x64x32 2x2w s3 | 1.01 | 0.2% | 3.9e-09 | 5.4e-07 |
| 32 × 14336 × 4096 | 204.8 | 18 | 32x64x32 2x2w s3 | 1.02 | 0.3% | 6.0e-09 | 6.6e-07 |
| 32 × 4096 × 14336 | 206.8 | 18 | 32x64x32 2x2w s3 | 1.00 | 0.2% | 3.8e-09 | 7.6e-07 |
| 128 × 4096 × 4096 | 52.7 | 82 | 64x64x32 2x2w s4 | 0.98–0.99 | 0.4% | 4.4e-08 | 8.2e-07 |
| 128 × 11008 × 4096 | 173.7 | 66 | 64x64x32 2x2w s4 | 0.95–0.99 | 1.4% | 4.1e-08 | 2.3e-07 |
| 128 × 4096 × 11008 | 175.0 | 66 | 64x64x32 2x2w s4 | 1.05–1.08 | 0.1% | 3.3e-08 | 1.2e-06 |
| 128 × 14336 × 4096 | 243.3 | 62 | 64x64x32 2x2w s4 | 0.96–0.99 | 0.4% | 4.4e-08 | 8.9e-07 |
| 128 × 4096 × 14336 | 226.8 | 66 | 64x64x32 2x2w s4 | 1.08 | 0.1% | 2.6e-08 | 1.3e-06 |
| 512 × 4096 × 4096 | 205.6 | 84 | 64x128x32 2x4w s4 | 0.98–0.99 | 0.4% | 4.0e-08 | 8.7e-07 |
| 512 × 11008 × 4096 | 614.8 | 75 | 64x128x32 2x4w s4 | 1.03–1.05 | 0.4% | 5.1e-08 | 1.0e-06 |
| 512 × 4096 × 11008 | 544.2 | 85 | 64x128x32 2x4w s4 | 0.95–0.98 | 0.1% | 3.4e-08 | 1.3e-06 |
| 512 × 14336 × 4096 | 805.1 | 75 | 64x128x32 2x4w s4 | 1.09–1.11 | 0.4% | 4.6e-08 | 9.2e-07 |
| 512 × 4096 × 14336 | 697.5 | 86 | 64x128x32 2x4w s4 | 0.95–0.96 | 0.1% | 3.2e-08 | 1.7e-06 |
| 2048 × 4096 × 4096 | 804.7 | 85 | 128x128x32 2x2w s2 | 0.99–1.00 | 100.0% | 9.3e-07 | 9.3e-07 |
| 2048 × 11008 × 4096 | 2192.2 | 84 | 128x128x32 2x2w s3 | 1.05 | 100.0% | 1.0e-06 | 1.0e-06 |
| 2048 × 4096 × 11008 | 2096.1 | 88 | 128x128x32 2x2w s3 | 0.98 | 100.0% | 1.4e-06 | 1.4e-06 |
| 2048 × 14336 × 4096 | 2771.6 | 87 | 128x128x32 2x2w s3 | 1.00 | 100.0% | 1.1e-06 | 1.1e-06 |
| 2048 × 4096 × 14336 | 2725.7 | 88 | 128x128x32 2x2w s3 | 0.99–1.00 | 100.0% | 2.0e-06 | 2.0e-06 |

## What it shows

- **Speed: 0.93-1.51x cuBLAS.** 21 of 30 shapes are at 0.98x or better in
  both runs. The losses are 2-7%:
  - 4096³ and 4000³, 0.98x;
  - the ragged 1000³, 0.93-0.96x;
  - 128 × 11008 × 4096 and 128 × 14336 × 4096, 0.95-0.99x;
  - 512 × 4096 × 11008 and 512 × 4096 × 14336, 0.95-0.98x.
- **Bit identity.** Where Y chains the tensor core's accumulator, as cuBLAS
  does, every output is bit-identical to cuBLAS's. That is every shape with
  100% above: 256³ to 4096³ and 2048 rows at every K.
- **Accuracy.** Elsewhere Y's default accumulation differs on purpose
  (§9.12 of the manual):
  - at 32 rows the compensated sums (`kahan`) are 80-200x closer to the
    float64 product;
  - at 128-512 rows with K of 4096 and up, the blocked sums are 6-53x closer;
  - at 8192³ cuBLAS's largest error was 3x smaller (4.3e-7 against 1.3e-6).
    cuBLAS splits K there. Y chained, because this machine's profile (dated
    2026-10-17, before the re-tune) gave 8192³ 2x2 warps, which have no
    room for block sums. The analytic model gives the same.

## Since 2026-11-07: the tile fit

From K = 8192 the compiler runs such a tile on twice the warps and at
least 3 stages (`../f16_block_sums_2026-11-07/`). Three rows above had such
a tile, and now sum in blocks. Measured with `bf16_bench.py` as before:
three runs at 8192³ (the third is `run3_8192_block_sums.log`) and two at
2048 rows (`run1_2048_block_sums.log`, `run2_2048_block_sums.log`):

| M × N × K | Y's tile now | Y / cuBLAS | same bits | max error, Y | max error, cuBLAS |
|---|---|---|---|---|---|
| 8192 × 8192 × 8192 | 128x128x32 4x2w s3 | 1.013-1.018 | 0.5% | 4.5e-08 | 4.2e-07 |
| 2048 × 4096 × 11008 | 128x128x32 4x2w s3 | 0.993 | 0.1% | 3.5e-08 | 1.8e-06 |
| 2048 × 4096 × 14336 | 128x128x32 4x2w s3 | 0.993 | 0.1% | 3.2e-08 | 1.6e-06 |

At 8192³ the profile's 2x2/s2 line is fitted to 4x2/s3; at 2048 rows the
analytic model's 2x2/s3 tile. Every other row is as above: its K is below
8192, or its tile already held the block sums. The two 2048-row shapes
gave cuBLAS's bits before and are now 50x below its largest error, at the
speed they had (0.98-1.00x).
