# Head to head, 2026-10-13

Two runs of `python/examples/run_head_to_head_comparison.py` on an RTX 4070 Ti
SUPER, at `c8eea77`. `report.md`, `results.json` and `stdout.log` are what the
first run wrote; `run2_contended/` holds the second, started three minutes
later. `pmon.log` in each is `nvidia-smi pmon -s u`, sampled every second
through the run.

The script ran with `run.diff` applied, which is why the reports say "dirty
tree". The diff is committed with these files: each row's samples now start
from warmed clocks, and the SM clock is read after every sample
(`sm_clock_mhz` in `results.json`; 2430-2775 MHz over both runs). The FP8 tile
came from a hardware profile without measured `AUTOTUNE` lines, as in
`../fp8_gemm_epilogue_2026-10-13/`.

No game was running. The desktop compositor, Discord and Chromium were the
GPU's other clients.
- The first run started with the GPU 4.2% busy and is not marked contended.
- The second is: two of its ten starting samples read 75%, the other eight
  1-4%.
- Every column passed its check in both runs, and every graph-mode minimum of
  the second run is within 3.2% of the first.
- Launch-mode minima differ by up to 44% between the runs, while their
  medians agree within 3.6%. Launches are bound by the CPU, so the launch
  figures below are medians.

First run, microseconds; minimum of 15 rounds replayed from a CUDA graph,
except the launch row. For the SwiGLU module the Triton column is
`torch.compile` and the PyTorch column eager.

| row | Y | Triton | PyTorch | Y / Triton | Y / PyTorch |
| :--- | ---: | ---: | ---: | ---: | ---: |
| vector add | 1.04 | 1.05 | 1.08 | 0.99 | 0.97 |
| vector add, launched (median) | 2.99 | 5.75 | 2.10 | 0.52 | 1.42 |
| SwiGLU | 1.12 | 1.12 | 2.42 | 1.00 | 0.46 |
| RMSNorm | 1.24 | 1.34 | 8.90 | 0.92 | 0.14 |
| FP8 GEMM 1024^3 | 26.60 | 31.98 | 44.41 | 0.83 | 0.60 |
| FP8 GEMM 2048^3 | 192.10 | 199.48 | 268.89 | 0.96 | 0.71 |
| FP8 GEMM 4096^3 | 1473.54 | 1570.82 | 1800.19 | 0.94 | 0.82 |
| FP8 GEMM 8192^3 | 11639.78 | 15700.99 | 10427.39 | 0.74 | 1.12 |
| SwiGLU module, 2^22 | 81.08 | 80.41 | 89.87 | 1.01 | 0.90 |

The SwiGLU module's 1.01 is inside both columns' spread (medians 81.53 and
81.19). Through `y_inductor` it is one Y kernel, `silu` then the multiply,
bit for bit eager's. Cold compilation of one kernel, Y: 0.6 ms to PTX plus
10.2 ms for the driver to load it. Triton's first call with an empty cache:
513 ms.

Against `../head_to_head_2026-10-11/`, every Y column changed:
- RMSNorm was 1.40 us against Triton's 1.33.
- A Y launch was 6.6 us.
- The FP8 GEMM took 2.1-2.6x Triton's time and 1.5-3.3x
  `torch._scaled_mm`'s.
- The SwiGLU module was within 2% of eager, since only its multiply was a Y
  kernel; `torch.compile` was 11-13% faster.
