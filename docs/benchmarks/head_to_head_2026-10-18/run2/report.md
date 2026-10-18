# Head to head: Y vs Triton vs PyTorch

NVIDIA GeForce RTX 4070 Ti SUPER (sm_89), driver 615.71.09, CUDA 13.0, PyTorch 2.13.0+cu130, Triton 3.7.1, Python 3.14.7. Y 3048a134ff, `liby.so` sha256 669278cc7995, target sm_89. 20261018T005322Z.

GPU utilisation before starting: 1.4% (mean of 10 samples); other processes holding a context: ['2094, /usr/bin/kwin_wayland, 33 MiB'].

Microseconds per call: minimum / median over 15 interleaved rounds. *launch*: back-to-back calls from Python; *graph*: the same calls replayed from a CUDA graph. A column that failed its check is not timed.

## Vector add (100,000 f32)

IEEE addition: every column must equal the correctly rounded float32 sum, exactly.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | exact | 1.66 / 1.75 | 1.05 / 1.05 |
| Triton | exact | 5.68 / 5.71 | 1.05 / 1.06 |
| PyTorch | exact | 1.68 / 2.04 | 1.10 / 1.10 |

SM clock read after each sample: 2475-2475 MHz (median 2475).

## SwiGLU (100,000 f32)

Y uses ex2.approx and rcp.approx; the tolerance is relative to the largest output.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | err 6.7e-08 | 1.75 / 1.77 | 1.14 / 1.14 |
| Triton | err 6.7e-08 | 5.65 / 5.67 | 1.14 / 1.15 |
| PyTorch | err 5.7e-08 | 4.59 / 4.61 | 2.45 / 2.46 |

SM clock read after each sample: 2475-2475 MHz (median 2475).

## RMSNorm (128 x 1024 f32)

Y uses rsqrt.approx; the tolerance is relative to the largest output.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | err 9.4e-08 | 1.75 / 1.91 | 1.22 / 1.23 |
| Triton | err 1.2e-07 | 5.06 / 5.09 | 1.32 / 1.33 |
| PyTorch | err 1.1e-07 | 18.25 / 18.31 | 8.85 / 8.89 |

SM clock read after each sample: 2475-2475 MHz (median 2475).

## FP8 GEMM 1024^3 (f32 in, e4m3 tensor cores, f32 out)

Every column quantises float32 A and B to e4m3 and returns float32; PyTorch's time includes the quantisation and B's transpose, as Y's and Triton's include theirs. Y quantises inside its GEMM; Y e4m3 is three Y kernels (quantise A, quantise and transpose B, the GEMM over e4m3 bytes), as PyTorch's column is its quantisation kernels and _scaled_mm.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | exact | 26.88 / 26.90 | 26.15 / 26.17 |
| Y e4m3 | exact | 25.67 / 25.71 | 24.09 / 24.13 |
| Triton | exact | 32.81 / 32.94 | 31.97 / 32.13 |
| PyTorch | exact | 47.39 / 47.50 | 44.48 / 44.54 |

SM clock read after each sample: 2475-2475 MHz (median 2475).

## FP8 GEMM 2048^3 (f32 in, e4m3 tensor cores, f32 out)

Every column quantises float32 A and B to e4m3 and returns float32; PyTorch's time includes the quantisation and B's transpose, as Y's and Triton's include theirs. Y quantises inside its GEMM; Y e4m3 is three Y kernels (quantise A, quantise and transpose B, the GEMM over e4m3 bytes), as PyTorch's column is its quantisation kernels and _scaled_mm.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | exact | 192.70 / 193.54 | 191.39 / 192.49 |
| Y e4m3 | exact | 169.50 / 170.47 | 170.73 / 171.29 |
| Triton | exact | 202.19 / 203.13 | 199.40 / 200.40 |
| PyTorch | exact | 282.21 / 284.49 | 277.41 / 278.89 |

SM clock read after each sample: 2775-2775 MHz (median 2775).

## FP8 GEMM 4096^3 (f32 in, e4m3 tensor cores, f32 out)

Every column quantises float32 A and B to e4m3 and returns float32; PyTorch's time includes the quantisation and B's transpose, as Y's and Triton's include theirs. Y quantises inside its GEMM; Y e4m3 is three Y kernels (quantise A, quantise and transpose B, the GEMM over e4m3 bytes), as PyTorch's column is its quantisation kernels and _scaled_mm.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | exact | 1474.56 / 1490.94 | 1479.65 / 1493.95 |
| Y e4m3 | exact | 1066.98 / 1089.54 | 1084.42 / 1087.49 |
| Triton | exact | 1577.98 / 1624.06 | 1570.82 / 1587.20 |
| PyTorch | exact | 1773.57 / 1818.62 | 1792.00 / 1803.26 |

SM clock read after each sample: 2760-2760 MHz (median 2760).

## FP8 GEMM 8192^3 (f32 in, e4m3 tensor cores, f32 out)

Every column quantises float32 A and B to e4m3 and returns float32; PyTorch's time includes the quantisation and B's transpose, as Y's and Triton's include theirs. Y quantises inside its GEMM; Y e4m3 is three Y kernels (quantise A, quantise and transpose B, the GEMM over e4m3 bytes), as PyTorch's column is its quantisation kernels and _scaled_mm.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | exact | 11254.78 / 12037.12 | 11651.74 / 12048.70 |
| Y e4m3 | exact | 7370.75 / 7805.82 | 7452.26 / 7875.58 |
| Triton | exact | 16014.34 / 16741.38 | 15632.38 / 16632.83 |
| PyTorch | exact | 10529.54 / 11024.38 | 10332.16 / 10991.62 |

SM clock read after each sample: 2610-2685 MHz (median 2670).

## SwiGLU module through a graph compiler (2^22 f32)

Y column: y_inductor (silu and the multiply are one Y kernel, bit for bit eager's). Triton column: torch.compile, whose Inductor backend generates Triton. PyTorch column: eager. y_inductor: 1 Y kernel(s), 2 op(s) lowered   y_kernel_0: silu -> mul | launches 425, fallbacks {}

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | err 8.1e-08 | 80.55 / 81.40 | 80.21 / 80.70 |
| Triton | err 9.9e-08 | 81.43 / 81.92 | 80.65 / 80.70 |
| PyTorch | err 8.1e-08 | 89.69 / 90.51 | 89.97 / 90.26 |

SM clock read after each sample: 2775-2775 MHz (median 2775).

## Cold compilation (one RMSNorm kernel, each sample a fresh process)

- Y `compile_to_ptx`: median 0.6 ms; loading the PTX (the driver compiles it): median 10.9 ms.
- Triton first call with an empty cache: median 512.7 ms (second call 0.05 ms).

Raw samples: `results.json`. Written by `python/examples/run_head_to_head_comparison.py`.
