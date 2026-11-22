# Exact training (T2) on an H100, 2026-11-22

## What was asked

Does the exact training of `docs/benchmarks/exact_train_lm_2026-11-21/`,
run on its fast path (compiled, one CUDA graph a step), give the same bits
on an H100 as on the development machine's CPU and RTX 4070 Ti SUPER, and
how fast is it there against float32?

## The run

`modal_train.py` on Modal: one H100 80GB HBM3 (sm_90, driver 620.70.06),
torch 2.13.0+cu126, Triton 3.7.1, Y built from the repository at 7e58f6d
(`liby.so` sha256 `b264cfca...`, the same library as the 4070 Ti runs).
`machine.txt` has the details. Five commands, logs and records here:

- `h100_exact_y_g12b8_fast.json`: the 5000-step T2 record, Y's int8 GEMM,
  12-bit gradients and int8 elsewhere (the default), `--graph --compile`.
- `h100_float32_fast.json`, `h100_float32_eager.json`: the float32 twin,
  compiled and graphed, and eager.
- `h100_speed_tiny.json`, `h100_speed_medium.json`: `tools/exact_train_speed.py`,
  each exact mode's weights checked against a CPU run made in the same
  container (Modal's CPU), then the step time.

Cost: $1.34 (and $0.02 for a first try whose image lacked `transformers`).

## Same bits

`compare.txt`: the H100's 5000-step record has the weights of the CPU record
(AMD Ryzen 9 9950X, float64 BLAS) and of the 4070 Ti record at all 10
checkpoints, and the same target probabilities at all 5000 steps; final
weights `8a7018b19d9b7a00`, held-out loss 1.688295 on all three. The float32
twin, the control, differs between the 4070 Ti and the H100 from step 1.

At width 512 (4 layers, 4,096 tokens, 12.8M parameters), the size at which
Inductor's store-then-reload race showed on the 4070 Ti, the H100's weights
after 3 compiled steps are `726658118d294e45`: the digest of Modal's CPU, of
the development CPU and of the 4070 Ti.

## Speed

Steady-state ms per step. Both arms get the same treatment; float32 is
PyTorch's default (no TF32).

| Model | GPU | exact, compiled + graphed | float32, eager | float32, compiled + graphed | ratio, same treatment |
|---|---|---|---|---|---|
| T2 (width 128, 1,024 tokens) | H100 | 0.71 | 2.78 | 0.38 | 1.9x |
| T2 | RTX 4070 Ti SUPER | 0.99 | 1.08 | 0.40 | 2.5x |
| width 512, 4 layers, 4,096 tokens | H100 | 11.64 | 11.59 | 8.79 | 1.32x |
| width 512, 4 layers, 4,096 tokens | RTX 4070 Ti SUPER | 43.8 | 24.6 | 18.5 | 2.4x |

The H100 runs float64 at half its float32 rate, the 4070 Ti at 1/64, and on
the 4070 Ti the gap at width 512 was all in the non-GEMM kernels (38.4 ms
against float32's 4.4, with the exact GEMMs 2.5x faster than float32's). The
H100 closes most of it, as that predicts; no H100 profile was taken, so the
split there is not measured. The T2 rows are launch-bound: float32 eager is
slower than the exact fast path on the H100, because the host behind it
issues launches more slowly than the development machine.

Compiling is slow on Modal's CPU: the first exact step took 226 s at T2's
size (23 s when Inductor's cache was warm, in the same container) and 539 s
at width 512, against 81 and 146 s on the development machine.

## Float32 activations, and a bf16 baseline (`act32/`)

`modal_train.py --stage act32`, Y at 24da895: the same training with
`Ops(act=torch.float32)` (every sum still an exact integer sum, elementwise
IEEE float32, products' exact integers rounded once to float32, quantizers on
24-bit mantissas in int32; parameters and Adam float64). A different model
from the float64 one, so its own CPU reference, made on a CPU-only Modal
container (24 CPUs, 3,148 s for 5000 steps).

**Same bits**: the H100's record has the CPU record's weights at all 10
checkpoints and its target probabilities at all 5000 steps
(`act32/compare.txt`, final weights `502c0f1cba2ee8f4`). At width 512 the
H100's weights after 3 steps are the CPU's (`e7714afaa0a4fb6b`); so are the
float64 model's (`726658118d294e45`) in the same run.

**Accuracy**: held-out 1.6963 after 5000 steps, against the float64 exact
model's 1.6883 and float32's 1.6933 to 1.7060 across runs: no measurable
cost, on one seed.

**Speed** (H100, ms a step, everything compiled and graphed). `bf16` is the
float twin with its products in bf16 (float32 parameters and elementwise
operations, the usual mixed-precision training):

| Model | exact, float32 activations | exact, float64 | float32 | bf16 |
|---|---|---|---|---|
| T2 (width 128) | 0.76 | 0.80 | 0.41 | 0.27 |
| width 512, 4 layers, 4,096 tokens | 9.57 | 11.70 | 8.89 | 2.32 |

At width 512 float32 activations bring the exact step to 1.08x float32's
time. **bf16 is 4.1x faster than the exact step there**, and 3.8x faster
than float32: its products run on the H100's bf16 tensor cores, float32's
do not (TF32 off). The exact step's linear layers are int8 tensor-core
products already; its attention products are int8 values multiplied in
float32 without tensor cores, and its quantizers and elementwise work are
the rest. No H100 profile was taken, so the split is not measured. Against
bf16, not float32, is where the exact step is slow.

Cost: $2.64 (H100 $2.2, the CPU container about $0.4).

## Where the time goes, and the float32 formula (`profile/`, `tune/`, `f32formula/`)

`tools/exact_train_profile.py` at width 512, H100, float32 activations,
compiled (GPU time a step, no CUDA graph):

| | before (24da895) | float32 formula (5f9d9c1) | bf16 |
|---|---|---|---|
| Inductor's elementwise and quantizer kernels | 7.09 ms | 5.30 ms | 1.32 ms |
| attention's float32 products (cuBLAS, no tensor cores) | 1.27 ms | 1.29 ms | |
| Y's int8 GEMM (51 calls) | 1.23 ms | 1.24 ms | |
| bf16 products (cuBLAS) | | | 0.88 ms |
| all | 9.62 ms | 7.86 ms | 2.20 ms |

The float32 formula: a float32 quantizer computes round(x * 2^a * 2^b) in
float32, the same function as the integer one on the float's bits (tested,
and the CPU reproduces the H100's earlier digests), in a few float32
operations instead of about thirty integer ones. Step time with it,
compiled and graphed, `f32formula/`: **8.03 ms against float32's 8.91 and
bf16's 2.33**, the CPU's weights after 3 steps (`e7714afaa0a4fb6b`). The
exact step is now faster than float32 here, and 3.4x bf16's time.

Inductor's coordinate-descent and pointwise autotuning (`tune/`) cut the
GPU time 9.62 to 9.10 ms on the older code (5%) and made CUDA graph capture
fail, so they are not used.

What is left, by kernel (`f32formula/h100_profile32f_medium_exact32.txt`):
the attention softmax and the quantizing of P (two kernels over 32768 rows
of 256, 0.86 ms), the other attention-sized quantizers and epilogues, and
attention's float32 products: about 3 ms of the 7.86 together. Adam's
integer square root over 12.8M parameters is 0.26 ms.

Cost of these three runs: $4.70.

## Not covered

- One H100, one run of each. Step times are wall clock over 500 (T2) and 50
  (width 512) steps after the first 30 and 3, one run each, not a minimum
  over rounds.
- Width 512 was checked for 3 steps, not trained.
