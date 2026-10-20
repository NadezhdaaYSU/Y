# Quiet-GPU measurements after the swizzle, 2026-10-20

RTX 4070 Ti SUPER, driver 615.71.09, `ptxas` from CUDA 13.4, PyTorch
2.13.0+cu130, Triton 3.7.1. No game was running (`games.sh`, `queue.txt`); the
10-sample mean utilisation before a run was 1.0-3.6% for ten of 16 runs and
9.6-37.4% for six, each read right after the previous run; the two SwiGLU
runs logged one sample each (14% and 1%; `queue.txt` also holds the two
attempts that stopped, below). `queue.sh` ran the accuracy,
512 x 512 x K, bias + ReLU, decode and square runs in order under one lock;
`retune/queue.sh` and the SwiGLU runs followed. Every timing column is checked
first (exact on small integers, or as each harness says), and timed in
interleaved rounds ranked by minimum.

Kernels: "base" is 81c055f's `tests/*.ptx`, "swz" 9d64faa's (the swizzle);
the decode kernels are renamed to the harnesses' names as in
`../decode_dram_2026-10-18/`. Kernels not committed are kept here.

## Accuracy against cuBLAS (`accuracy/`)

`accuracy.py`: Y (9d64faa's kernels; `accuracy/ptx/` for 512 x 512 x K, the
tuner's own tiles), cuBLAS (`torch.mm`, float32 out) and Triton
(`../f16_square_2026-10-18/bench_ab.py`'s autotuned kernel, where its block
sizes divide the shape) against float64, on N(0, 1) and |N(0, 1)| f16 inputs.
Each output's error is scaled by `sum_k |a| |b|` (for same-sign inputs, the
relative error). `acc_1.json` holds, per shape, distribution and column, the
mean and maximum |error|, the mean signed error, the share of outputs equal to
the correctly rounded float32 of the true sum and the mean float32 ulps from
it, the share bitwise equal to cuBLAS, and the cuBLAS kernel the profiler saw.

| shape | Y, N(0,1) mean abs | cuBLAS | Y, same sign mean | cuBLAS | Y = cuBLAS bits, N(0,1) / same sign |
| :--- | ---: | ---: | ---: | ---: | :--- |
| 256^3 | 3.4e-08 | 3.4e-08 | -2.6e-06 | -2.6e-06 | 1.000 / 1.000 |
| 512^3 | 4.6e-08 | 4.6e-08 | -5.7e-06 | -5.7e-06 | 1.000 / 1.000 |
| 1024^3 | 6.4e-08 | 6.4e-08 | -1.2e-05 | -1.2e-05 | 1.000 / 1.000 |
| 2048^3 | 9.0e-08 | 9.0e-08 | -2.5e-05 | -2.5e-05 | 1.000 / 1.000 |
| 4096^3 | 1.3e-07 | 1.3e-07 | -5.0e-05 | -5.0e-05 | 1.000 / 1.000 |
| 8192^3 | 1.8e-07 | 6.1e-08 | -1.0e-04 | -3.2e-05 | 0.002 / 0.000 |
| 1 x 4096 x 4096 | 1.2e-07 | 1.1e-08 | -5.0e-05 | +1.4e-09 | 0.003 / 0.000 |
| 4 x 4096 x 4096 | 1.3e-07 | 1.3e-07 | -5.0e-05 | -5.0e-05 | 1.000 / 1.000 |
| 8 x 4096 x 4096 | 1.3e-07 | 1.3e-07 | -5.0e-05 | -5.0e-05 | 1.000 / 1.000 |
| 32 x 4096 x 4096 | 1.3e-07 | 1.3e-07 | -5.0e-05 | -5.0e-05 | 1.000 / 1.000 |
| 512 x 512 x 1024 | 6.4e-08 | 6.4e-08 | -1.2e-05 | -1.2e-05 | 1.000 / 1.000 |
| 512 x 512 x 4096 | 1.3e-07 | 6.4e-08 | -5.0e-05 | -2.5e-05 | 0.008 / 0.000 |
| 512 x 512 x 16384 | 2.5e-07 | 3.3e-08 | -2.0e-04 | -2.5e-05 | 0.001 / 0.000 |
| 512 x 512 x 65536 | 5.0e-07 | 6.4e-08 | -8.2e-04 | -1.0e-04 | 0.000 / 0.000 |

Triton's output equals cuBLAS's where Y's does and Y's where cuBLAS's does
not. Y's same-sign error is -1.2e-5 per 1024 of K at every shape. cuBLAS ran,
at 8192^3, `ampere_s16816gemm_fp16_256x128_ldg8_stages_32x3_nn`; at M = 1 a
GEMV kernel; at 512 x 512 x 65536 a split-K kernel and `splitKreduce_kernel`
(`acc_1.log` names every one).

## 512 x 512 x K (`ksweep/`)

`../smem_swizzle_2026-10-20/ksweep/ktiles.py`, the tuner's 512 tile pinned
(64x64x32, 4x2 warps, 4 stages), padded (`Y_SMEM_SWIZZLE=0`) and swizzled,
CUDA-graph replays of 50 launches. Minimum us, runs 1 / 2:

| K | padded | swizzled | cuBLAS | cuBLAS / swizzled |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 4.42 / 4.35 | 3.63 / 3.76 | 4.41 / 4.34 | 1.21 / 1.15 |
| 512 | 7.02 / 6.81 | 5.64 / 5.52 | 6.21 / 5.85 | 1.10 / 1.06 |
| 1024 | 11.05 / 11.30 | 8.47 / 8.71 | 8.98 / 9.25 | 1.06 / 1.06 |
| 2048 | 19.55 / 19.84 | 14.38 / 14.40 | 17.88 / 18.17 | 1.24 / 1.26 |
| 4096 | 37.37 / 37.35 | 26.83 / 26.38 | 29.98 / 29.85 | 1.12 / 1.13 |
| 8192 | 73.04 / 73.00 | 51.33 / 51.36 | 55.23 / 55.07 | 1.08 / 1.07 |

## Fused GEMM + bias + ReLU (`bias_relu/`)

`bench_bias.py`: Y's `gemm_f16_bias_relu_<N>` (base and swz) against
`relu(F.linear(A, B^T, bias))` in FP16 (cuBLASLt's bias epilogue, FP16 out,
half Y's output bytes; the README's earlier baseline) and
`relu(addmm(bias, A, B, out_dtype=float32))` (Y's bytes). All exact. Minimum
us, run 1 (run 2 within 0.4%):

| N | base | swz | FP16 linear + relu | FP32 addmm + relu | base / swz | FP16 / swz | FP32 / swz |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 6.8 | 5.6 | 8.2 | 10.7 | 1.213 | 1.450 | 1.908 |
| 1024 | 29.8 | 28.2 | 31.7 | 38.3 | 1.056 | 1.124 | 1.359 |
| 2048 | 206.7 | 204.7 | 207.4 | 230.6 | 1.010 | 1.013 | 1.127 |
| 4096 | 1579.0 | 1580.0 | 1634.3 | 1913.9 | 0.999 | 1.034 | 1.211 |
| 8192 | 12419.0 | 12445.1 | 12812.0 | 13651.6 | 0.998 | 1.029 | 1.097 |

## Decode GEMMs from DRAM (`decode_dram/`)

`../decode_dram_2026-10-18/bench_cold.py`, four 33.5 MB copies of B ("cold")
and one ("warm"), base and swz. Minimum us, run 1 / 2:

| M | base cold | swz cold | cuBLAS cold | base warm | swz warm | cuBLAS warm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 55.48 / 55.45 | 56.18 / 55.11 | 55.77 / 55.80 | 19.69 / 19.71 | 18.26 / 18.07 | 26.69 / 26.72 |
| 4 | 55.80 / 55.49 | 54.69 / 54.89 | 56.02 / 56.09 | 21.60 / 21.71 | 18.15 / 18.28 | 54.24 / 53.71 |
| 8 | 55.54 / 55.55 | 55.82 / 55.79 | 57.34 / 57.37 | 23.38 / 23.31 | 18.97 / 18.89 | 50.49 / 50.51 |
| 32 | 56.42 / 56.32 | 54.82 / 55.13 | 55.54 / 55.93 | 30.55 / 30.49 | 20.54 / 20.55 | 23.78 / 23.83 |

From DRAM the three tie, bandwidth-bound (within 1.9-3.3% at each M); in the
L2 the swizzle is 8-49% faster and Y 1.16-2.99x cuBLAS.

## Square sizes, a third run (`square/`)

`bench_ab.py` as in `../smem_swizzle_2026-10-20/ab/`: base / swz 1.071, 1.216,
1.058, 1.013, 1.001 and 0.996 from 256^3 to 8192^3, and 1.096, 1.199, 1.215 and
1.465 at M = 1, 4, 8 and 32; cuBLAS / swz 1.456, 1.081, 1.052, 1.002, 0.980,
1.008, and 1.475, 2.981, 2.656, 1.127.

## Tiles under the swizzle (`retune/`)

`queue.sh`: candidate tiles (`<N>/<cta_m_cta_n_cta_k_warps_m_warps_n_stages>/`,
each with `ptxas -v`: no spills) through `bench_ab.py`, two runs a size, every
column exact. The current tile's time over each candidate's (above 1 the
candidate is faster), run 1 / 2, best three:

| N | current | candidate | current / candidate |
| ---: | :--- | :--- | ---: |
| 1024 | 64x128x32, 2x2, 3 | 64x128x32, 2x2, 4 | 1.001 / 1.002 |
| 2048 | 128x128x32, 2x2, 2 | 128x128x32, 2x2, 3 | 1.016 / 1.016 |
| 2048 | | cuBLAS | 0.998 / 0.995 |
| 4096 | 128x128x32, 2x2, 2 | 128x128x32, 2x2, 3 | 1.007 / 1.008 |
| 4096 | | 128x128x32, 4x2, 3 | 1.005 / 1.006 |
| 8192 | 128x128x32, 2x2, 2 | 128x128x32, 4x2, 3 | 1.013 / 1.015 |
| 8192 | | 128x128x32, 2x2, 4 | 1.006 / 1.006 |

cuBLAS stays 1.018-1.021x faster than the current 4096 tile and 1.010-1.013x
than the best candidate. The tiles were not changed in this commit.

## Fused SwiGLU (`swiglu/`)

`bench_swiglu.py`: `gemm_f16_swiglu_<N>`, padded (9d64faa) and swizzled (this
commit, which also gets 4 stages where the padded tiles fitted 3), against Y's
unfused path (two Y GEMMs, `F.silu(g) * u`) and PyTorch's (two `torch.mm`,
float32 out). The two fused kernels give the same bits; every column is
within 1e-4 of float64 (each column's error, as a share of the largest
output, is 2.0e-6 at 512, 4.4e-6 at 1024, 1.0e-5 at 2048 and 1.5e-5 at 4096). Minimum
us, run 1 / 2:

| N | padded | swizzled | Y unfused | PyTorch | padded / swizzled | unfused / swizzled | PyTorch / swizzled |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 34.03 / 34.08 | 32.29 / 32.30 | 15.14 / 15.15 | 16.00 / 16.00 | 1.054 / 1.055 | 0.469 / 0.469 | 0.496 / 0.495 |
| 1024 | 63.58 / 63.68 | 59.65 / 59.49 | 65.05 / 65.19 | 67.84 / 67.94 | 1.066 / 1.070 | 1.091 / 1.096 | 1.137 / 1.142 |
| 2048 | 470.95 / 472.44 | 444.88 / 444.70 | 492.08 / 493.38 | 463.20 / 462.66 | 1.059 / 1.062 | 1.106 / 1.109 | 1.041 / 1.040 |
| 4096 | 3650.56 / 3662.34 | 3438.08 / 3439.10 | 3673.60 / 3672.58 | 3587.90 / 3590.05 | 1.062 / 1.065 | 1.068 / 1.068 | 1.044 / 1.044 |

The first two attempts at these runs stopped at 2048 (`ZeroDivisionError`):
the tolerance was then 1e-5, which no column met there, PyTorch's included.
