# Head to head: Y vs Triton vs PyTorch

NVIDIA GeForce RTX 4070 Ti SUPER (sm_89), driver 615.71.09, CUDA 13.0, PyTorch 2.13.0+cu130, Triton 3.7.1, Python 3.14.7. Y c8eea77699 (dirty tree), `liby.so` sha256 73f3a3aa869d, target sm_89. 20261013T145618Z.

GPU utilisation before starting: 4.2% (mean of 10 samples); other processes holding a context: ['2093, /usr/bin/kwin_wayland, 75 MiB', '2732, /home/yumin/.config/discord/app-1.0.161/Discord --type=gpu-process --ozone-platform=wayland --render-node-override=/dev/dri/renderD128 --crashpad-handler-pid=2656 --enable-crash-reporter=633b85dd-f46f-41fa-8579-4e9267aed935?no_channel --user-data-dir=/home/yumin/.config/discord --gpu-preferences=UAAAAAAAAAAgAQAEAAAAAAAAAAAAAGAAAQAAAAAAAAAAAAAAAAAAAAIAAAAAAAAAAAAAAAAAAAAYAAAAAAAAABgAAAAAAAAAAAAAAAAAAAAIAAAAAAAAAAgAAAAAAAAA --shared-files --field-trial-handle=3?i?960368201953347638?6761096197306292725?262144 --enable-features=DocumentPolicyIncludeJSCallStacksInCrashReports --disable-features=AllowAggressiveThrottlingWithWebSocket?DropInputEventsWhilePaintHolding?HardwareMediaKeyHandling?IntensiveWakeUpThrottling?LocalNetworkAccessChecks?MediaSessionService?ScreenAIOCREnabled?SpareRendererForSitePerProcess?TraceSiteInstanceGetProcessCreation?UseEcoQoSForBackgroundProcess?WaylandWpColorManagerV1?WinRetrieveSuggestionsOnlyOnDemand --variations-seed-version --pseudonymization-salt-handle=7?i?16684512138348829905?9225956593316396854?4 --trace-process-track-uuid=3190708988185955192, 1059 MiB'].

Microseconds per call: minimum / median over 15 interleaved rounds. *launch*: back-to-back calls from Python; *graph*: the same calls replayed from a CUDA graph. A column that failed its check is not timed.

## Vector add (100,000 f32)

IEEE addition: every column must equal the correctly rounded float32 sum, exactly.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | exact | 2.48 / 2.99 | 1.04 / 1.15 |
| Triton | exact | 5.17 / 5.75 | 1.05 / 1.16 |
| PyTorch | exact | 1.65 / 2.10 | 1.08 / 1.19 |

SM clock read after each sample: 2475-2475 MHz (median 2475).

## SwiGLU (100,000 f32)

Y uses ex2.approx and rcp.approx; the tolerance is relative to the largest output.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | err 6.7e-08 | 2.32 / 2.99 | 1.12 / 1.12 |
| Triton | err 6.7e-08 | 5.71 / 5.76 | 1.12 / 1.12 |
| PyTorch | err 5.7e-08 | 4.63 / 4.69 | 2.42 / 2.43 |

SM clock read after each sample: 2475-2475 MHz (median 2475).

## RMSNorm (128 x 1024 f32)

Y uses rsqrt.approx; the tolerance is relative to the largest output.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | err 9.4e-08 | 2.21 / 3.23 | 1.24 / 1.25 |
| Triton | err 1.2e-07 | 4.25 / 5.13 | 1.34 / 1.36 |
| PyTorch | err 1.1e-07 | 18.51 / 18.60 | 8.90 / 8.97 |

SM clock read after each sample: 2475-2475 MHz (median 2475).

## FP8 GEMM 1024^3 (f32 in, e4m3 tensor cores, f32 out)

Every column quantises float32 A and B to e4m3 and returns float32; PyTorch's time includes the quantisation and B's transpose, as Y's and Triton's include theirs.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | exact | 27.61 / 27.71 | 26.60 / 26.69 |
| Triton | exact | 32.82 / 33.07 | 31.98 / 32.21 |
| PyTorch | exact | 47.43 / 48.20 | 44.41 / 44.59 |

SM clock read after each sample: 2475-2475 MHz (median 2475).

## FP8 GEMM 2048^3 (f32 in, e4m3 tensor cores, f32 out)

Every column quantises float32 A and B to e4m3 and returns float32; PyTorch's time includes the quantisation and B's transpose, as Y's and Triton's include theirs.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | exact | 193.13 / 194.04 | 192.10 / 192.38 |
| Triton | exact | 202.14 / 205.07 | 199.48 / 200.09 |
| PyTorch | exact | 272.18 / 275.02 | 268.89 / 271.26 |

SM clock read after each sample: 2460-2460 MHz (median 2460).

## FP8 GEMM 4096^3 (f32 in, e4m3 tensor cores, f32 out)

Every column quantises float32 A and B to e4m3 and returns float32; PyTorch's time includes the quantisation and B's transpose, as Y's and Triton's include theirs.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | exact | 1473.54 / 1505.12 | 1473.54 / 1496.83 |
| Triton | exact | 1585.63 / 1644.26 | 1570.82 / 1593.34 |
| PyTorch | exact | 1785.86 / 1805.31 | 1800.19 / 1813.47 |

SM clock read after each sample: 2430-2760 MHz (median 2430).

## FP8 GEMM 8192^3 (f32 in, e4m3 tensor cores, f32 out)

Every column quantises float32 A and B to e4m3 and returns float32; PyTorch's time includes the quantisation and B's transpose, as Y's and Triton's include theirs.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | exact | 11672.58 / 11841.54 | 11639.78 / 11868.42 |
| Triton | exact | 15793.50 / 16252.93 | 15700.99 / 16108.29 |
| PyTorch | exact | 10591.23 / 10729.15 | 10427.39 / 10818.56 |

SM clock read after each sample: 2715-2760 MHz (median 2760).

## SwiGLU module through a graph compiler (2^22 f32)

Y column: y_inductor (silu and the multiply are one Y kernel, bit for bit eager's). Triton column: torch.compile, whose Inductor backend generates Triton. PyTorch column: eager. y_inductor: 1 Y kernel(s), 2 op(s) lowered   y_kernel_0: silu -> mul | launches 425, fallbacks {}

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | err 8.1e-08 | 81.66 / 82.90 | 81.08 / 81.53 |
| Triton | err 9.9e-08 | 80.55 / 83.12 | 80.41 / 81.19 |
| PyTorch | err 8.1e-08 | 89.87 / 90.65 | 89.87 / 90.49 |

SM clock read after each sample: 2490-2490 MHz (median 2490).

## Cold compilation (one RMSNorm kernel, each sample a fresh process)

- Y `compile_to_ptx`: median 0.6 ms; loading the PTX (the driver compiles it): median 10.2 ms.
- Triton first call with an empty cache: median 513.1 ms (second call 0.05 ms).

Raw samples: `results.json`. Written by `python/examples/run_head_to_head_comparison.py`.
