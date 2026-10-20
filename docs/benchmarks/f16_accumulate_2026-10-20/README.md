# F16 GEMM accumulation in float32, 2026-10-20

RTX 4070 Ti SUPER, driver 615.71.09, `ptxas` from CUDA 13.4, PyTorch
2.13.0+cu130, Triton 3.7.1. No game running; utilisation before each run in
`queue.txt` (1.0-3.5% mean for three, 11.3% and a 21% sample for two read
right after the previous run).

The tensor core adds an mma's products into its accumulator with less than
float32 rounding. Y's F16 GEMM chained its whole K loop in it, `mma d, a, b,
d`, as cuBLAS's and Triton's tensor-core kernels do: on same-sign inputs every
result sits below float64, by about 1.2e-5 per 1024 of K
(`../quiet_2026-10-20/accuracy/`). With `Y_F16_ACCUMULATE=f32` each mma starts
from a zero register and four `add.rn.f32` fold its result in, as the FP8
GEMM has done since 2026-10-18 (`emit_fp8_mma_into`). Kernels: "mma" is
b3402e4's (`tests/*.ptx`, the decode ones renamed); "f32" is
`speed/f32/` (and `accuracy/ptx/` for 512 x 512 x K), the same compiler with
the knob set. `ptxas -v`: the 128x128 tiles go from 203 registers to the 255
cap and spill 16 bytes; 64x64 tiles from 70 to 86; the decode tiles from 64-80
to 104-118; the fused SwiGLU kernel, capped at 128, spills 40 bytes.

## Accuracy (`accuracy/`)

`../quiet_2026-10-20/accuracy/accuracy.py` on the f32 kernels; cuBLAS's
columns are as in that run. Mean |error| on N(0, 1) inputs and mean signed
error on |N(0, 1)|, each scaled by `sum |a| |b|`:

| shape | f32, N(0,1) | cuBLAS | f32, same sign | cuBLAS | f32 max, same sign | correctly rounded, f32 / cuBLAS |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| 256^3 | 7.2e-09 | 3.4e-08 | -7.7e-08 | -2.6e-06 | 3.1e-07 | 0.258 / 0.043 |
| 512^3 | 6.2e-09 | 4.6e-08 | -7.7e-08 | -5.7e-06 | 4.6e-07 | 0.216 / 0.020 |
| 1024^3 | 5.7e-09 | 6.4e-08 | -7.7e-08 | -1.2e-05 | 7.2e-07 | 0.173 / 0.009 |
| 2048^3 | 5.3e-09 | 9.0e-08 | -7.7e-08 | -2.5e-05 | 9.3e-07 | 0.133 / 0.004 |
| 4096^3 | 5.1e-09 | 1.3e-07 | -7.7e-08 | -5.0e-05 | 1.4e-06 | 0.099 / 0.002 |
| 8192^3 | 5.0e-09 | 6.1e-08 | -7.7e-08 | -3.2e-05 | 2.0e-06 | 0.072 / 0.003 |
| 1 x 4096 x 4096 | 5.0e-09 | 1.1e-08 | -7.5e-08 | +1.4e-09 | 8.4e-07 | 0.110 / 0.048 |
| 4 x 4096 x 4096 | 5.2e-09 | 1.3e-07 | -7.7e-08 | -5.0e-05 | 1.1e-06 | 0.102 / 0.003 |
| 8 x 4096 x 4096 | 5.1e-09 | 1.3e-07 | -7.7e-08 | -5.0e-05 | 1.0e-06 | 0.099 / 0.002 |
| 32 x 4096 x 4096 | 5.1e-09 | 1.3e-07 | -7.6e-08 | -5.0e-05 | 1.1e-06 | 0.100 / 0.002 |
| 512 x 512 x 1024 | 5.6e-09 | 6.4e-08 | -7.7e-08 | -1.2e-05 | 6.7e-07 | 0.173 / 0.009 |
| 512 x 512 x 4096 | 5.1e-09 | 6.4e-08 | -7.7e-08 | -2.5e-05 | 1.1e-06 | 0.100 / 0.004 |
| 512 x 512 x 16384 | 5.0e-09 | 3.3e-08 | -7.6e-08 | -2.5e-05 | 2.2e-06 | 0.053 / 0.004 |
| 512 x 512 x 65536 | 4.9e-09 | 6.4e-08 | -7.9e-08 | -1.0e-04 | 4.9e-06 | 0.027 / 0.001 |

The same-sign error no longer grows with K. Against cuBLAS it is 34-1290x
smaller on same-sign inputs and 5-25x on N(0, 1) ones everywhere but M = 1,
where cuBLAS runs a GEMV kernel without tensor cores: +1.4e-9 on same-sign
inputs there, and float32 accumulation still 2.1x smaller on N(0, 1).

**Correction (2026-10-20, `../f16_accuracy_2026-10-20/`).** The paragraph
above compares means. On same-sign inputs that is the mean signed error, the
bias, which stays at about -7.7e-8. The error's spread does grow with K: on
|N(0, 1)| inputs the relative Frobenius error goes from 1.0e-7 at K = 256 to
9.9e-7 at 65536.

The later record also measures the relative Frobenius, maximum absolute and
maximum normalized errors. It covers nine distributions, including two
cancellation-heavy ones, and compares the kernels output by output. Float32
accumulation is not more accurate everywhere:

- **M = 1, same-sign inputs:** cuBLAS's GEMV is 5-7% lower in Frobenius error
  and is the closer one on 47-48% of the outputs.
- **Paired cancellation at long K:** cuBLAS's split K has the lower maximum
  error.

## Speed (`speed/`, `longk/`)

`../f16_square_2026-10-18/bench_ab.py` (`speed/`) and
`../smem_swizzle_2026-10-20/ksweep/ktiles.py` (`longk/`, CUDA graphs), every
column exact on small integers in both modes. The f32 kernel's time over the
mma one's, run 1 / 2 (below 1, f32 is faster):

| shape | f32 / mma |
| :--- | ---: |
| 256^3 | 1.031 / 1.029 |
| 512^3 | 1.089 / 1.087 |
| 1024^3 | 1.049 / 1.049 |
| 2048^3 | 1.061 / 1.062 |
| 4096^3 | 1.054 / 1.054 |
| 8192^3 | 1.066 / 1.053 |
| M = 1, N = K = 4096 | 0.974 / 0.979 |
| M = 4 | 0.975 / 0.978 |
| M = 8 | 0.952 / 0.955 |
| M = 32 | 1.004 / 1.004 |
| 512 x 512 x 1024 | 1.117 / 1.099 |
| 512 x 512 x 4096 | 1.023 / 1.029 |
| 512 x 512 x 16384 | 1.036 / 1.035 |
| 512 x 512 x 65536 | 1.050 / 1.060 |

So float32 accumulation is the default for GEMMs of at most 32 rows, where it
is faster or level, and the knob selects it (or the chain, `=mma`)
everywhere. The decode kernels `tests/gemm_f16_decode_m{1,4,8,32}.ptx` are
the ones `speed/f32/` measured (byte for byte); no other committed kernel
changes. At 512 x 512 x 65536 cuBLAS (split K) is the faster in either mode:
400-405 us against 419-421 (mma) and 442-444 (f32).

## Tests and mutants

`tests/ptx_f16_accumulate.rs`: the mode at M = 1, 8, 32, 33 and 256 with no
knob, each knob forcing the other mode, a misspelt knob refused; and on the
GPU, at 8 x 256 x 4096 and 128 x 128 x 4096 in both modes, exact integer
products and same-sign mean error against a float64 CPU sum: below -2e-5 for
the chain (the control) and within 1e-6 for float32. `mutants/run.sh`: six
mutants, each fails the file (`summary.txt`). The structural check first
required `add.rn.f32`, so a round-toward-zero mutant failed on it before the
precision bound could (`summary_before_relaxing.txt`); it accepts any f32
add now, and the mutant fails on the bound: -5.7e-6 on average.
