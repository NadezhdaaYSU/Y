# Square F16 GEMM against cuBLAS and Triton, 2026-10-18

The README's "Square FP16 GEMM vs cuBLAS" table was measured on 2026-08-09,
before the K-loop change of `8d605e9` (`../f16_gemm_loop_2026-10-17/`) and
the tiles re-measured in `f5ae72b` (`../f16_tiles_2026-10-17/`). It is
measured again here, at the README's six sizes, with the harness of the
K-loop record.

Y's kernels were compiled at `bc0bbc3` (`Y` sha256 a41995ae...) from
`tests/gemm_f16_<N>.ysu`, against a copy of the main checkout's
`.ysu_hw_profile`, whose six tiles for these shapes are in `tiles.txt`
(CTA M, N, K; warps M, N; stages). The PTX is byte for byte the committed
`tests/gemm_f16_<N>.ptx`, so it is not copied here.

`bench_ab.py` (unchanged from the K-loop record) checks every column equal
to float64 `A @ B` first (small integers, so exact), then times 17 rounds,
interleaved, of back-to-back calls, and drops the first two rounds; the
figures are the minimum of the other 15. cuBLAS is `torch.mm` with a float32
output, as Y's C is float32. (The commit that added the table it replaces,
`520d2b9`, does not record how it was measured.) Triton is the K-loop
record's kernel with its autotuner's choice at each size.

RTX 4070 Ti SUPER, driver 615.71.09, CUDA 13.4 `ptxas`, PyTorch
2.13.0+cu130, Triton 3.7.1. Two runs, each its own process (`run1.json`,
`run2.json`, and their logs), each started once the GPU's mean utilisation
over 10 samples was at most 5% with no game running: 4.0% and 3.2%
(`queue.log`). The single reading `bench_ab.py` takes at its start, after
CUDA is initialised, was 15% and 28%; the idle desktop reads 8-11%. The SM
clock, read after every sample, was 2505-2775 MHz. `table.py` prints the
table below from the two JSON files.

| M=N=K | Y us | cuBLAS us | Triton us | Y vs cuBLAS | Y vs Triton | Y TFLOPS | median/min Y, cuBLAS |
|---|---|---|---|---|---|---|---|
| 256 | 2.91 / 3.00 | 3.91 / 3.92 | 9.43 / 7.31 | 1.344x / 1.308x | 3.242x / 2.437x | 11.5 / 11.2 | 1.102, 1.005 / 1.057, 1.004 |
| 512 | 6.88 / 6.67 | 5.89 / 5.86 | 9.51 / 7.06 | 0.856x / 0.879x | 1.382x / 1.058x | 39.0 / 40.2 | 1.096, 1.103 / 1.025, 1.026 |
| 1024 | 29.75 / 29.79 | 28.81 / 28.79 | 30.12 / 30.19 | 0.969x / 0.967x | 1.013x / 1.013x | 72.2 / 72.1 | 1.001, 1.003 / 1.032, 1.046 |
| 2048 | 207.02 / 206.38 | 204.71 / 203.64 | 208.49 / 206.66 | 0.989x / 0.987x | 1.007x / 1.001x | 83.0 / 83.2 | 1.070, 1.050 / 1.005, 1.001 |
| 4096 | 1580.42 / 1584.58 | 1544.19 / 1545.22 | 1594.24 / 1590.86 | 0.977x / 0.975x | 1.009x / 1.004x | 87.0 / 86.7 | 1.002, 1.020 / 1.059, 1.021 |
| 8192 | 12489.22 / 12384.77 | 12429.84 / 12392.50 | 14182.30 / 14252.03 | 0.995x / 1.001x | 1.136x / 1.151x | 88.0 / 88.8 | 1.024, 1.042 / 1.053, 1.060 |

Run 1 / run 2 in every cell; "vs" is the other column's time over Y's, so
above 1 is Y faster.

- From 1024 to 8192 Y is 0.97-1.00x cuBLAS, and at 8192 the two runs
  disagree on the order (0.995x, 1.001x): a tie. At 2048 to 8192, whose
  tiles `f5ae72b` left alone, Y's minima are within 1.6% of the K-loop
  record's two runs (2026-10-17, started at 0.9% and 0.6%) and cuBLAS's
  within 0.3%. At 512 and 1024 Y's (6.67-6.88 and 29.75-29.79 us) are
  within 4.1% and 0.6% of the tiles record's four runs of the same tiles
  (6.61-6.83 and 29.61-29.77).
- At 512 Y is 0.86-0.88x cuBLAS (6.67-6.88 against 5.86-5.89 us).
- At 256 and 512 Y's samples are spread (median 1.06-1.10x its minimum), and
  Triton's minimum moved from 9.43 to 7.31 and from 9.51 to 7.06 us between
  the runs, so its ratios there carry little weight; the K-loop record saw
  Triton's samples at 512 sit at 9.4-9.6 us with one or two near 8.3.
- 88.0 and 88.8 TFLOPS at 8192 were measured with the SM at 2520-2760 MHz,
  above the 2.61 GHz boost clock at which the README's 88.1 TFLOPS peak is
  stated: at 2760 MHz that peak is 93.2.
