# A second GPU: Tesla T4 (2026-11-11)

Y's exact path had only ever run on one GPU, the RTX 4070 Ti SUPER (sm_89, Ada).
This is the first run on another: a **Tesla T4** (sm_75, Turing, 2018), two of
them in a free Kaggle notebook, driver 580.178.04. Qwen2.5-0.5B-Instruct,
`tools/exact_cross_device.py`, 5 cases x 160 tokens, the same workload hash as
the reference records. Software matched to theirs: torch 2.13.0 (the CUDA 12.6
build, which still supports Turing), transformers 5.15.0; Python 3.13.16 against
3.14.8. Y came from the wheel (`tools/wheel_glibc228/`) of `84a4429`; the record
tool from the repository at `426dd88` with its `python/` removed, so `y_lang`
was the wheel's.

## Same tokens

`compare.txt`, rerun on the development machine from the files here:

| comparison | exact | stock (control) |
|---|---|---|
| Zen 5 CPU vs T4, eager (`../exact_y_gemm_2026-11-07/cpu_zen5_0p5b.json`) | **0/5 changed** | 5/5 changed |
| RTX 4070 Ti SUPER vs T4, eager (`../exact_y_gemm_2026-11-07/gpu_sm89_triton_eager.json`) | **0/5 changed** | 5/5 changed |
| Zen 5 CPU, integer norm sum, vs T4 eager (`../exact_norm_sum_2026-11-10/cpu_0p5b.json`) | **0/5 changed** | (no stock arm) |
| same CPU record vs T4 compiled, every switch | **0/5 changed** | (no stock arm) |
| 4070 Ti, compiled, every switch (`../exact_attention_staging_2026-11-11/gpu_all_compiled.json`) vs T4 the same | **0/5 changed** | (no stock arm) |

Exact digest `30eec169d2f38da4` on every device and build; stock bf16 on the T4
`33e852ed3e1b8fd7`, different from both the CPU's and the 4070 Ti's. Each T4
record decoded every case twice and got the same tokens (the self-check); the
integer exp's domain digest on the T4 is the CPU's, `0x7170cd39b442d506`.

"Every switch" is `Y_EXACT_V_INT8=1 Y_EXACT_NORM=triton Y_EXACT_ATTN=y
Y_EXACT_KV_PACK=inplace`, compiled: the attention kernel
(`tests/exact_decode_attention.ysu`) and `exact_pv` compiled by Y for sm_75 and
run there, the Triton norm, the in-place cache. The linear layers took
`torch._int_mm` (cuBLASLt), which the notebook checked is exact on the T4:
Triton cannot build the default int8 GEMM below sm_80, and Y's own GEMM needs
sm_80's `mma.sync` and `cp.async`. Asked for it (`Y_EXACT_GEMM=y`), the
compiler refused by name (`t4_ygemm.log`).

## What it found first

The first two runs failed, both on Y's side:

1. **The default GEMM.** Triton 3.7 cannot compile `w8a8_gemv`'s int8
   `tl.dot` for sm_75 (`'arith.extf' op operand #0 must be floating-point-like`,
   in `TritonGPUAccelerateMatmul`), and the exact path picked that kernel on
   every CUDA device, so it died converting the model. `378dc9f`: below sm_80
   the layers take `torch._int_mm` or float64, which give the same bits
   (`python/y_lang/tests/test_exact_gemm_choice.py`).
2. **Z3.** The compiled run's `exact_pv` kernel carries an `@invariant`, the
   compiler proves it with Z3 at compile time, and Kaggle had none: Y refused
   to compile the kernel unverified, as it must. `84a4429`: the wheel depends
   on `z3-solver`, and `y_lang` finds the `z3` it installs beside the
   interpreter (`python/tests/test_solver_lookup.py`).

## Files

- `y_cross_gpu.ipynb`, written by `make_notebook.py`: the notebook as run;
  `notebook_output.txt`, its whole output.
- `t4_eager.json` (stock and exact), `t4_all_compiled.json` (exact), and each
  record's log; `t4_ygemm.log`, the refusal.
- `compare.txt`: the comparisons above.

Pushed and fetched with the Kaggle CLI (`kaggle kernels push --accelerator
NvidiaTeslaT4`; Kaggle has retired the P100, so Pascal is not covered). Eager
took 4.4 minutes, compiled 37.6 (mostly compiling).

## Not covered

One NVIDIA vendor line (Turing, Ada) and the CPU; one model family; 0.5B.
Qwen2.5-7B on the T4 and Y's Python test suites there are next.
