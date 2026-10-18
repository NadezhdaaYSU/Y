# Head to head: Y vs Triton vs PyTorch

NVIDIA GeForce RTX 4070 Ti SUPER (sm_89), driver 615.71.09, CUDA 13.0, PyTorch 2.13.0+cu130, Triton 3.7.1, Python 3.14.7. Y 3048a134ff, `liby.so` sha256 669278cc7995, target sm_89. 20261018T005252Z.

GPU utilisation before starting: 2.8% (mean of 10 samples); other processes holding a context: ['2094, /usr/bin/kwin_wayland, 33 MiB'].

Microseconds per call: minimum / median over 15 interleaved rounds. *launch*: back-to-back calls from Python; *graph*: the same calls replayed from a CUDA graph. A column that failed its check is not timed.

## Vector add (100,000 f32)

IEEE addition: every column must equal the correctly rounded float32 sum, exactly.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | exact | 1.77 / 1.79 | 1.05 / 1.05 |
| Triton | exact | 5.67 / 5.72 | 1.05 / 1.08 |
| PyTorch | exact | 2.05 / 2.06 | 1.10 / 1.13 |

SM clock read after each sample: 2475-2475 MHz (median 2475).

## SwiGLU (100,000 f32)

Y uses ex2.approx and rcp.approx; the tolerance is relative to the largest output.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | err 6.7e-08 | 1.84 / 1.85 | 1.23 / 1.23 |
| Triton | err 6.7e-08 | 5.70 / 5.73 | 1.22 / 1.23 |
| PyTorch | err 5.7e-08 | 4.61 / 4.66 | 2.62 / 2.63 |

SM clock read after each sample: 2475-2475 MHz (median 2475).

## RMSNorm (128 x 1024 f32)

Y uses rsqrt.approx; the tolerance is relative to the largest output.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | err 9.4e-08 | 1.83 / 1.89 | 1.22 / 1.22 |
| Triton | err 1.2e-07 | 5.06 / 5.10 | 1.32 / 1.33 |
| PyTorch | err 1.1e-07 | 18.29 / 18.35 | 8.87 / 9.00 |

SM clock read after each sample: 2445-2475 MHz (median 2445).

## FP8 GEMM 1024^3 (f32 in, e4m3 tensor cores, f32 out)

Every column quantises float32 A and B to e4m3 and returns float32; PyTorch's time includes the quantisation and B's transpose, as Y's and Triton's include theirs. Y quantises inside its GEMM; Y e4m3 is three Y kernels (quantise A, quantise and transpose B, the GEMM over e4m3 bytes), as PyTorch's column is its quantisation kernels and _scaled_mm.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | exact | 26.87 / 26.90 | 26.17 / 26.21 |
| Y e4m3 | exact | 25.70 / 25.75 | 24.11 / 24.14 |
| Triton | exact | 32.87 / 32.99 | 32.04 / 32.14 |
| PyTorch | exact | 47.64 / 47.71 | 44.58 / 44.61 |

SM clock read after each sample: 2760-2775 MHz (median 2775).

## FP8 GEMM 2048^3 (f32 in, e4m3 tensor cores, f32 out)

Every column quantises float32 A and B to e4m3 and returns float32; PyTorch's time includes the quantisation and B's transpose, as Y's and Triton's include theirs. Y quantises inside its GEMM; Y e4m3 is three Y kernels (quantise A, quantise and transpose B, the GEMM over e4m3 bytes), as PyTorch's column is its quantisation kernels and _scaled_mm.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | exact | 192.69 / 193.44 | 191.49 / 192.29 |
| Y e4m3 | exact | 169.45 / 169.96 | 171.10 / 171.29 |
| Triton | exact | 202.66 / 205.64 | 199.68 / 200.61 |
| PyTorch | exact | 282.89 / 285.60 | 275.46 / 276.75 |

SM clock read after each sample: 2715-2775 MHz (median 2715).

## FP8 GEMM 4096^3 (f32 in, e4m3 tensor cores, f32 out)

Every column quantises float32 A and B to e4m3 and returns float32; PyTorch's time includes the quantisation and B's transpose, as Y's and Triton's include theirs. Y quantises inside its GEMM; Y e4m3 is three Y kernels (quantise A, quantise and transpose B, the GEMM over e4m3 bytes), as PyTorch's column is its quantisation kernels and _scaled_mm.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | exact | 1485.98 / 1499.14 | 1486.85 / 1494.14 |
| Y e4m3 | exact | 1067.01 / 1070.53 | 1086.46 / 1088.45 |
| Triton | exact | 1601.54 / 1616.80 | 1576.96 / 1589.25 |
| PyTorch | exact | 1765.54 / 1783.81 | 1801.22 / 1812.48 |

SM clock read after each sample: 2715-2760 MHz (median 2715).

## FP8 GEMM 8192^3 (f32 in, e4m3 tensor cores, f32 out)

Every column quantises float32 A and B to e4m3 and returns float32; PyTorch's time includes the quantisation and B's transpose, as Y's and Triton's include theirs. Y quantises inside its GEMM; Y e4m3 is three Y kernels (quantise A, quantise and transpose B, the GEMM over e4m3 bytes), as PyTorch's column is its quantisation kernels and _scaled_mm.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | exact | 11263.30 / 11550.72 | 11256.83 / 11522.18 |
| Y e4m3 | exact | 7380.70 / 7416.83 | 7390.21 / 7465.57 |
| Triton | exact | 15333.34 / 15963.14 | 15628.90 / 15758.50 |
| PyTorch | exact | 10576.48 / 10681.09 | 10384.03 / 10627.94 |

SM clock read after each sample: 2595-2745 MHz (median 2685).

## SwiGLU module through a graph compiler (2^22 f32)

Y column: y_inductor (silu and the multiply are one Y kernel, bit for bit eager's). Triton column: torch.compile, whose Inductor backend generates Triton. PyTorch column: eager. y_inductor: 1 Y kernel(s), 2 op(s) lowered   y_kernel_0: silu -> mul | launches 425, fallbacks {}

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | err 8.1e-08 | 81.23 / 81.68 | 80.55 / 80.70 |
| Triton | err 9.9e-08 | 80.46 / 81.91 | 80.07 / 80.46 |
| PyTorch | err 8.1e-08 | 89.48 / 90.01 | 90.21 / 90.70 |

SM clock read after each sample: 2520-2520 MHz (median 2520).

## Cold compilation (one RMSNorm kernel, each sample a fresh process)

- Y `compile_to_ptx`: median 0.6 ms; loading the PTX (the driver compiles it): median 10.7 ms.
- Triton first call with an empty cache: median 504.6 ms (second call 0.05 ms).

Raw samples: `results.json`. Written by `python/examples/run_head_to_head_comparison.py`.
