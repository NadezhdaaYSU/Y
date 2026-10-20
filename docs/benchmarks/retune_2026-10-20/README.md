# Re-tuning the F16 GEMM tiles after the swizzle, 2026-10-20

RTX 4070 Ti SUPER, driver 615.71.09, `ptxas` from CUDA 13.4, PyTorch
2.13.0+cu130, Triton 3.7.1. The committed `tests/gemm_f16_*.ptx` come from the
`AUTOTUNE_F16_<M>x<N>x<K>_<GPU>` lines of the profile they were generated
under (the plain and bias + ReLU kernels of one shape share a line). Those
lines were measured with the padded shared-memory layout; the swizzle
(`../smem_swizzle_2026-10-20/`) frees shared memory and moves the trade-offs.
`composite/autotune_before.txt` and `autotune_after.txt` are the lines before
and after (the profile itself is machine-specific and not kept).

**Contention.** Discord's GPU process started at 14:16:46 and was found at
15:03 holding 1.4 GB with the GPU at 10-37% while none of these jobs ran.
`tuner/` (14:40) and both comparisons (14:56, 14:59) ran beside it;
`candidates/` (14:07-14:09) before it. Effects under 1% below are within its
noise; the runs are interleaved and agree with each other.

## The autotuner (`tuner/`)

`run.sh`: `Y <shape>.ysu --emit-ptx --autotune-force` for every committed F16
GEMM shape into a copy of the profile. The autotuner times its candidates with
B streamed from DRAM (several weight copies against the 48 MB L2; `*.log`).
`profile.diff`: it changed 256, 512, 1024, 2048, 4096, 8192, 16384, M = 8 and
the rectangular and uneven-grid shapes, picking 4 stages from 1024 up.

`tuner_ab/`: its kernels against the current ones, two runs each, every column
exact first: `bench_ab.py` (L2-resident, the README's tables),
`bench_cold_sq.py` (square shapes with B streamed from DRAM, as the autotuner
times them; adapted from `../decode_dram_2026-10-18/bench_cold.py`),
`bench_bias.py` and `bench_cold.py` for M = 4 and 8. Current time over the
autotuner's kernel's (above 1, the autotuner's is faster), run 1 / 2:

| shape | L2-resident | B from DRAM |
| :--- | ---: | ---: |
| 256^3 | 1.026 / 1.026 | - |
| 512^3 | 1.008 / 1.006 | - |
| 1024^3 | 0.992 / 0.991 | 0.993 / 0.993 |
| 2048^3 | 1.033 / 1.034 | 1.034 / 1.034 |
| 4096^3 | 0.995 / 0.996 | 0.996 / 0.991 |
| 8192^3 | 0.996 / 0.995 | 1.005 / 0.990 |
| 512 x 1024 x 256 | 1.064 / 1.065 | - |
| 1280 x 1024 x 1024 | 1.085 / 1.086 | - |
| M = 8 (decode, from DRAM / L2) | - | 1.017 / 1.018 (DRAM), 1.012 / 0.997 (L2) |

So its tiles lost at 1024 and 4096 in both regimes. `candidates/` had already
measured, L2-resident, 3 stages at 128x128x32 against the then-current tiles
for the bias + ReLU kernels (1.017x at 2048, 1.008x at 4096, 1.009-1.013x with
4x2 warps at 8192) and at 16384^3 (4x2 warps, 3 stages: 1.018x); see also
`../quiet_2026-10-20/retune/` for the plain kernels.

## The tiles committed (`composite/`)

The autotuner's lines where its kernels won in both regimes (256, 512, 2048,
M = 8, the rectangular and uneven-grid shapes), 1024's as it was, and the
sweeps' 3-stage tiles at 4096 (2x2 warps), 8192 and 16384 (4x2 warps).
`check_all.log`: each of the 13 kernels that changes gives the previous one's
output bit for bit on random inputs (the K-step order is the same; cta_k is 32
throughout), and both are exact on integers. `bench.sh`, as `tuner_ab/` plus
16384^3, two runs. Current time over the new kernel's:

| shape | L2-resident | B from DRAM | cuBLAS / new, L2-resident |
| :--- | ---: | ---: | ---: |
| 256^3 | 1.029 / 1.027 | - | 1.497 / 1.498 |
| 512^3 | 1.009 / 0.998 | - | 1.082 / 1.077 |
| 1024^3 | 1.000 / 1.000 (same kernel) | 1.000 / 1.000 | 1.051 / 1.054 |
| 2048^3 | 1.033 / 1.033 | 1.034 / 1.034 | 1.037 / 1.037 |
| 4096^3 | 1.008 / 1.008 | 1.008 / 1.008 | 0.987 / 0.989 |
| 8192^3 | 0.994 / 0.997 | 0.997 / 1.012 | 1.033 / 1.006 |
| 16384^3 | 1.018 / 1.018 | - | 1.015 / 1.015 |
| 512 x 1024 x 256 | 1.062 / 1.054 | - | 1.382 / 1.376 |
| 1280 x 1024 x 1024 | 1.085 / 1.086 | - | 1.096 / 1.094 |
| M = 8 | 1.014 / 1.013 | 1.016 / 1.016 | 2.825 / 2.831 |

Bias + ReLU, current over new: 1.008 / 1.007 at 512, 1.000 / 0.999 at 1024,
1.036 / 1.035 at 2048, 1.008 / 1.008 at 4096 and 1.012 / 1.015 at 8192. At
8192 the plain kernel is level within its 1.4-4% run-to-run spread and the
bias + ReLU one, on the same tile, 1.2-1.5% faster.
