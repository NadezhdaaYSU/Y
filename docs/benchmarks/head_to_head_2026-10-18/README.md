# Head to head, 2026-10-18

Two runs of `python/examples/run_head_to_head_comparison.py` on an RTX 4070 Ti
SUPER, at `3048a13`, a clean tree: driver 615.71.09, PyTorch 2.13.0+cu130,
Triton 3.7.1. `report.md`, `results.json` and `stdout.log` are what the first
run wrote; `run2/` holds the second, started when the first finished. `pmon.log`
in each is `nvidia-smi pmon -s u`, sampled every second through the run.

Each run started when the GPU's 10-sample mean utilisation was at most 5% and
no game was running: 2.8% and 1.4%, neither marked contended. The desktop
compositor, the Steam client (no game), Krita, Discord and Chromium held contexts;
`pmon` saw them use the SMs in single samples, at 1-16%. Every column passed
its check in both runs. The SM clock read 2445-2775 MHz after the samples.
Every graph-mode minimum of the second run is within 3.5% of the first's,
except the SwiGLU row, whose three columns all ran 6-7% faster.

New since `../head_to_head_2026-10-13/`: the FP8 rows have a "Y e4m3" column,
the same function from three Y kernels (quantise A, quantise and transpose B,
the `@tile` GEMM over e4m3 bytes; `../fp8_e4m3_2026-10-18/`), as PyTorch's
column is its quantisation kernels and `torch._scaled_mm`. Both Y FP8 GEMMs
now add each mma's result in float32; Triton's kernel does not (see that
record).

First run, microseconds; minimum of 15 rounds replayed from a CUDA graph,
except the launch row (median of plain launches; launches are bound by the
CPU). For the SwiGLU module the Triton column is `torch.compile` and the
PyTorch column eager. In the FP8 rows the ratios are Y e4m3's.

| row | Y | Y e4m3 | Triton | PyTorch | Y / Triton | Y / PyTorch |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| vector add | 1.05 | | 1.05 | 1.10 | 1.00 | 0.95 |
| vector add, launched (median) | 1.79 | | 5.72 | 2.06 | 0.31 | 0.87 |
| SwiGLU | 1.23 | | 1.22 | 2.62 | 1.01 | 0.47 |
| RMSNorm | 1.22 | | 1.32 | 8.87 | 0.92 | 0.14 |
| FP8 GEMM 1024^3 | 26.17 | 24.11 | 32.04 | 44.58 | 0.75 | 0.54 |
| FP8 GEMM 2048^3 | 191.49 | 171.10 | 199.68 | 275.46 | 0.86 | 0.62 |
| FP8 GEMM 4096^3 | 1486.85 | 1086.46 | 1576.96 | 1801.22 | 0.69 | 0.60 |
| FP8 GEMM 8192^3 | 11256.83 | 7390.21 | 15628.90 | 10384.03 | 0.47 | 0.71 |
| SwiGLU module, 2^22 | 80.55 | | 80.07 | 90.21 | 1.01 | 0.89 |

The SwiGLU row's 1.01 and the SwiGLU module's 1.01 are inside the columns'
spread: in the second run they read 1.00 (1.14 against 1.14) and 0.99 (80.21
against 80.65). Cold compilation of one kernel, Y: 0.6 ms to PTX plus 10.7 ms
for the driver to load it; Triton's first call with an empty cache, 505 ms.

Against `../head_to_head_2026-10-13/`:
- At 8192 Y's FP8 GEMM took 1.12x `_scaled_mm`'s time (11640 against 10427
  us); through the Y e4m3 column it takes 0.71x, 7390 against 10384. Every
  FP8 row's fastest column is now Y e4m3.
- The fused FP8 GEMM, which adds in float32 now, is within 2% of its earlier
  times at 1024-4096 and 3% faster at 8192 (11257 against 11640).
- A Y launch takes 1.79 us (median), from 2.99: the native launcher of
  `../launch_2026-10-17/`. PyTorch's takes 2.06.
- The elementwise rows moved by at most 0.11 us, the SwiGLU module by 0.5
  (81.08 to 80.55).
