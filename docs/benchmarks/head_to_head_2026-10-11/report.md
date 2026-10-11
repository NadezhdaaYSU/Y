# Head to head: Y vs Triton vs PyTorch

NVIDIA GeForce RTX 4070 Ti SUPER (sm_89), driver 615.71.09, CUDA 13.0, PyTorch 2.13.0+cu130, Triton 3.7.1, Python 3.14.7. Y 38b412a3a3 (dirty tree), `liby.so` sha256 099bb8729720, target sm_89. 20261011T193121Z.

GPU utilisation before starting: 4.0% (mean of 10 samples); other processes holding a context: ['2104, /usr/bin/kwin_wayland, 20 MiB', '2815, /home/yumin/.config/discord/app-1.0.161/Discord --type=gpu-process --ozone-platform=wayland --render-node-override=/dev/dri/renderD128 --crashpad-handler-pid=2663 --enable-crash-reporter=633b85dd-f46f-41fa-8579-4e9267aed935?no_channel --user-data-dir=/home/yumin/.config/discord --gpu-preferences=UAAAAAAAAAAgAQAEAAAAAAAAAAAAAGAAAQAAAAAAAAAAAAAAAAAAAAIAAAAAAAAAAAAAAAAAAAAYAAAAAAAAABgAAAAAAAAAAAAAAAAAAAAIAAAAAAAAAAgAAAAAAAAA --shared-files --field-trial-handle=3?i?16386779438504641612?3556371226154673380?262144 --enable-features=DocumentPolicyIncludeJSCallStacksInCrashReports --disable-features=AllowAggressiveThrottlingWithWebSocket?DropInputEventsWhilePaintHolding?HardwareMediaKeyHandling?IntensiveWakeUpThrottling?LocalNetworkAccessChecks?MediaSessionService?ScreenAIOCREnabled?SpareRendererForSitePerProcess?TraceSiteInstanceGetProcessCreation?UseEcoQoSForBackgroundProcess?WaylandWpColorManagerV1?WinRetrieveSuggestionsOnlyOnDemand --variations-seed-version --pseudonymization-salt-handle=7?i?11698405955585533742?12954668196313725525?4 --trace-process-track-uuid=3190708988185955192, 1494 MiB'].

Microseconds per call: minimum / median over 15 interleaved rounds. *launch*: back-to-back calls from Python; *graph*: the same calls replayed from a CUDA graph. A column that failed its check is not timed.

## Vector add (100,000 f32)

IEEE addition: every column must equal the correctly rounded float32 sum, exactly.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | exact | 6.62 / 6.73 | 1.07 / 1.07 |
| Triton | exact | 4.82 / 5.78 | 1.07 / 1.08 |
| PyTorch | exact | 2.02 / 2.05 | 1.10 / 1.10 |

## SwiGLU (100,000 f32)

Y uses ex2.approx and rcp.approx; the tolerance is relative to the largest output.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | err 6.7e-08 | 6.72 / 6.81 | 1.13 / 1.14 |
| Triton | err 6.7e-08 | 5.72 / 5.75 | 1.13 / 1.14 |
| PyTorch | err 5.7e-08 | 3.69 / 4.68 | 2.43 / 2.44 |

## RMSNorm (128 x 1024 f32)

Y uses rsqrt.approx; the tolerance is relative to the largest output.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | err 9.4e-08 | 5.72 / 7.01 | 1.40 / 1.42 |
| Triton | err 1.2e-07 | 3.87 / 5.05 | 1.33 / 1.34 |
| PyTorch | err 1.1e-07 | 16.59 / 18.64 | 8.87 / 8.98 |

## FP8 GEMM 1024^3 (f32 in, e4m3 tensor cores, f32 out)

Every column quantises float32 A and B to e4m3 and returns float32; PyTorch's time includes the quantisation and B's transpose, as Y's and Triton's include theirs.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | exact | 82.48 / 83.68 | 81.87 / 82.08 |
| Triton | exact | 32.67 / 33.12 | 31.86 / 32.00 |
| PyTorch | exact | 47.62 / 48.06 | 44.22 / 44.38 |

## FP8 GEMM 2048^3 (f32 in, e4m3 tensor cores, f32 out)

Every column quantises float32 A and B to e4m3 and returns float32; PyTorch's time includes the quantisation and B's transpose, as Y's and Triton's include theirs.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | exact | 418.27 / 421.21 | 413.21 / 441.46 |
| Triton | exact | 200.93 / 202.73 | 198.54 / 199.34 |
| PyTorch | exact | 270.22 / 272.61 | 276.82 / 282.51 |

## FP8 GEMM 4096^3 (f32 in, e4m3 tensor cores, f32 out)

Every column quantises float32 A and B to e4m3 and returns float32; PyTorch's time includes the quantisation and B's transpose, as Y's and Triton's include theirs.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | exact | 3749.89 / 3779.58 | 3730.43 / 3988.48 |
| Triton | exact | 1575.94 / 1593.44 | 1566.72 / 1576.51 |
| PyTorch | exact | 1753.09 / 1777.76 | 1796.10 / 1800.19 |

## FP8 GEMM 8192^3 (f32 in, e4m3 tensor cores, f32 out)

Every column quantises float32 A and B to e4m3 and returns float32; PyTorch's time includes the quantisation and B's transpose, as Y's and Triton's include theirs.

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | exact | 33799.17 / 34664.45 | 33523.71 / 34445.31 |
| Triton | exact | 15277.70 / 16059.42 | 15538.18 / 16108.54 |
| PyTorch | exact | 10328.42 / 10792.96 | 10321.92 / 10783.74 |

## SwiGLU module through a graph compiler (2^22 f32)

Y column: y_inductor (silu runs eagerly, the multiply is a Y kernel). Triton column: torch.compile, whose Inductor backend generates Triton. PyTorch column: eager. y_inductor: 1 Y kernel(s), 1 op(s) lowered   y_kernel_0: mul | launches 0, fallbacks {}   not lowered: silu (call_function `silu`): call_function `silu` is not lowered

| column | check | launch min / median | graph min / median |
| :--- | :--- | ---: | ---: |
| Y | err 8.1e-08 | 91.92 / 92.60 | 91.82 / 92.40 |
| Triton | err 9.9e-08 | 80.70 / 83.24 | 80.31 / 81.29 |
| PyTorch | err 8.1e-08 | 90.45 / 91.13 | 90.16 / 91.09 |

## Cold compilation (one RMSNorm kernel, each sample a fresh process)

- Y `compile_to_ptx`: median 0.6 ms; loading the PTX (the driver compiles it): median 10.0 ms.
- Triton first call with an empty cache: median 500.3 ms (second call 0.05 ms).

Raw samples: `results.json`. Written by `python/examples/run_head_to_head_comparison.py`.
