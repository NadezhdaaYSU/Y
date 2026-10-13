# Head to head: Y vs Triton vs PyTorch

NVIDIA GeForce RTX 4070 Ti SUPER (sm_89), driver 615.71.09, CUDA 13.0, PyTorch 2.13.0+cu130, Triton 3.7.1, Python 3.14.7. Y c8eea77699 (dirty tree), `liby.so` sha256 73f3a3aa869d, target sm_89. 20261013T145922Z.

> **Contended run.** Another process was using the GPU (mean utilisation 17.0% before starting; other processes on it: ['2093, /usr/bin/kwin_wayland, 30 MiB']). These timings are not a clean measurement.

Microseconds per call: minimum / median over 15 interleaved rounds. *launch*: back-to-back calls from Python; *graph*: the same calls replayed from a CUDA graph. A column that failed its check is not timed.

## Vector add (100,000 f32)

IEEE addition: every column must equal the correctly rounded float32 sum, exactly.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | exact | 2.94 / 2.96 | 1.06 / 1.08 |
| Triton | exact | 5.73 / 5.80 | 1.05 / 1.09 |
| PyTorch | exact | 2.06 / 2.09 | 1.11 / 1.11 |

SM clock read after each sample: 2490-2490 MHz (median 2490).

## SwiGLU (100,000 f32)

Y uses ex2.approx and rcp.approx; the tolerance is relative to the largest output.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | err 6.7e-08 | 2.95 / 2.98 | 1.10 / 1.11 |
| Triton | err 6.7e-08 | 4.85 / 5.78 | 1.11 / 1.15 |
| PyTorch | err 5.7e-08 | 4.60 / 4.64 | 2.38 / 2.39 |

SM clock read after each sample: 2490-2490 MHz (median 2490).

## RMSNorm (128 x 1024 f32)

Y uses rsqrt.approx; the tolerance is relative to the largest output.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | err 9.4e-08 | 3.18 / 3.19 | 1.21 / 1.27 |
| Triton | err 1.2e-07 | 4.29 / 5.11 | 1.32 / 1.36 |
| PyTorch | err 1.1e-07 | 18.43 / 18.49 | 8.87 / 9.22 |

SM clock read after each sample: 2490-2490 MHz (median 2490).

## FP8 GEMM 1024^3 (f32 in, e4m3 tensor cores, f32 out)

Every column quantises float32 A and B to e4m3 and returns float32; PyTorch's time includes the quantisation and B's transpose, as Y's and Triton's include theirs.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | exact | 27.55 / 27.60 | 26.59 / 26.66 |
| Triton | exact | 32.72 / 33.02 | 31.88 / 32.09 |
| PyTorch | exact | 47.82 / 48.54 | 44.48 / 45.19 |

SM clock read after each sample: 2760-2775 MHz (median 2760).

## FP8 GEMM 2048^3 (f32 in, e4m3 tensor cores, f32 out)

Every column quantises float32 A and B to e4m3 and returns float32; PyTorch's time includes the quantisation and B's transpose, as Y's and Triton's include theirs.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | exact | 192.41 / 193.31 | 190.67 / 191.34 |
| Triton | exact | 200.70 / 202.34 | 199.56 / 199.99 |
| PyTorch | exact | 276.83 / 277.40 | 268.80 / 270.22 |

SM clock read after each sample: 2580-2775 MHz (median 2580).

## FP8 GEMM 4096^3 (f32 in, e4m3 tensor cores, f32 out)

Every column quantises float32 A and B to e4m3 and returns float32; PyTorch's time includes the quantisation and B's transpose, as Y's and Triton's include theirs.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | exact | 1478.50 / 1489.92 | 1469.44 / 1483.55 |
| Triton | exact | 1591.26 / 1610.75 | 1574.62 / 1585.15 |
| PyTorch | exact | 1756.90 / 1781.76 | 1795.04 / 1812.22 |

SM clock read after each sample: 2760-2760 MHz (median 2760).

## FP8 GEMM 8192^3 (f32 in, e4m3 tensor cores, f32 out)

Every column quantises float32 A and B to e4m3 and returns float32; PyTorch's time includes the quantisation and B's transpose, as Y's and Triton's include theirs.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | exact | 11404.93 / 11658.24 | 11394.02 / 11640.48 |
| Triton | exact | 15363.07 / 15742.53 | 15297.31 / 15616.00 |
| PyTorch | exact | 10383.36 / 10568.70 | 10384.19 / 10527.46 |

SM clock read after each sample: 2490-2760 MHz (median 2760).

## SwiGLU module through a graph compiler (2^22 f32)

Y column: y_inductor (silu and the multiply are one Y kernel, bit for bit eager's). Triton column: torch.compile, whose Inductor backend generates Triton. PyTorch column: eager. y_inductor: 1 Y kernel(s), 2 op(s) lowered   y_kernel_0: silu -> mul | launches 405, fallbacks {}

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | err 8.1e-08 | 80.55 / 82.35 | 80.49 / 81.29 |
| Triton | err 9.9e-08 | 77.77 / 80.12 | 80.23 / 80.74 |
| PyTorch | err 8.1e-08 | 90.18 / 90.78 | 90.37 / 91.44 |

SM clock read after each sample: 2775-2775 MHz (median 2775).

## Cold compilation (one RMSNorm kernel, each sample a fresh process)

- Y `compile_to_ptx`: median 0.6 ms; loading the PTX (the driver compiles it): median 10.0 ms.
- Triton first call with an empty cache: median 507.8 ms (second call 0.05 ms).

Raw samples: `results.json`. Written by `python/examples/run_head_to_head_comparison.py`.
