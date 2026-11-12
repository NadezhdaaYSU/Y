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

## Qwen2.5-7B on the T4 (round 2)

`make_notebook_round2.py`; `t4_7b.json`, `t4_7b.log`. Exact arm, eager,
`--low-memory`, against the Zen 5's 7B record
(`../exact_norm_sum_2026-11-10/cpu_zen5_eager_7b.json`): **0/5 cases changed**,
digest `7700e3e28f1d21d1` on both, weight fingerprint `fa21999abf0abea9` on
both, the T4 reproducing itself; 6.0 minutes. No stock arm: stock 7B in bf16
does not fit the T4's 15 GB, so the control at 7B is Finding 12's (CPU against
the 4070 Ti, stock 5/5).

## An Intel CPU (round 3)

`make_notebook_round3.py`; `kaggle_cpu.json`, `kaggle_cpu.log`. The same 0.5B
workload on the Kaggle machine's own CPU, an `Intel(R) Xeon(R) CPU @ 2.00GHz`
with 4 cores (`lscpu`, in the notebook's output), torch's CPU kernels:

| comparison | exact | stock (control) |
|---|---|---|
| AMD Ryzen 9 9950X (Zen 5) vs Intel Xeon | **0/5 changed** | 5/5 changed |
| Tesla T4 vs Intel Xeon | **0/5 changed** | 5/5 changed |

Exact `30eec169d2f38da4` again; stock on the Xeon `7adabf35b44847f3`, unlike
the Zen 5's and the T4's. 94 minutes on 4 cores. The record tool named every
CPU `cpu:x86_64` and refused the Zen 5 pair as one device; the comparison above
is `--allow-same-device`, for two CPUs of different vendors on different
hosts. CPU records now carry `cpu_model`, and two CPUs count as two devices
only when both name different models (`--check-gate`, scenarios 10-11;
`python/tests/test_cross_device_gate.py`).

## Y's test suites on the T4

Both Python suites, from the repository's sources with the wheel's `liby.so`.
A clean checkout without a build fails 13 tests of `python/tests` on any
machine (the wheel and library-age tests need cargo and a fresh `liby.so`);
those are set aside. On the T4 (`round2_suite_y_lang_tests.log`)
`python/y_lang/tests` failed 35 more, for three reasons, none of them a
kernel computing a wrong value:

- **An import that fails below Python 3.14.** `y_lang/ops/block.py` used
  `Tuple` in annotations without importing it; 3.14 evaluates annotations
  lazily, 3.13 (Kaggle's) when the function is defined, so `import y_lang.ops`
  raised `NameError`. Fixed, and `python/tests/test_annotations_resolve.py`
  asks for every annotation in the package, which fails on any version.
- **Tests that built for sm_80 and launched.** An sm_80 module does not load on
  sm_75 (CUDA error 218). Tests that launch now build for the card
  (`y_lang.torch_interop.device_target`); hand-written PTX fixtures have their
  `.target` lowered to the card's; the `cp.async` test skips below sm_80.
- **A test that confused Triton.** `test_exact_gemm_choice.py` patched the
  capability torch reports to fake an sm_89 card; Triton took its target from
  that and built every later kernel in the process for sm_89, which then did
  not load. It patches Y's rule instead now.

Round 3 (`round3_suite_y_lang_tests.log`): 3 left, all three the PTX
fixtures, then retargeted. Round 4 (`round4_suite_*.log`, with `3bbc082`):
`python/y_lang/tests` 190 run, **0 failing**, 53 skipped (they need sm_80 or
sm_89: tensor-core and FP8 kernels, which say so); `python/tests` only the
12 a clean unbuilt checkout fails anywhere.

## Overnight (2026-11-12)

`make_overnight_jobs.py` wrote five Kaggle jobs run in parallel:

| job | result |
|---|---|
| 0.5B, all 17 cases of `exact_ragged_batch`, T4 (`t4_0p5b_full.*`) vs a Kaggle CPU, `Intel(R) Xeon(R) CPU @ 2.20GHz` (`kaggle_cpu_0p5b_full.*`, 414 minutes) | exact **0/17 changed**, digest `23e14cd41b101a6e`; stock 17/17 changed |
| 7B, compiled, every switch (`t4_7b_all_compiled.*`) vs the Zen 5's eager 7B record | exact **0/5 changed**, `7700e3e28f1d21d1` (45 minutes, mostly compiling) |
| Qwen2.5-1.5B, T4, both arms (`t4_1p5b.*`) | recorded (exact `22178ff4d2aede57`); its Kaggle CPU twin ran out its 12 hours on 4 cores; compared below |
| Qwen2.5-3B, T4 (`t4_3b_without_low_memory.log`) | stock recorded; exact ran out of the T4's memory, being built without `--low-memory` (a harness error); its CPU twin, out of time; rerun below |
| the Rust suites, `cargo test`, Kaggle CPU, no GPU | see below |

The record tool's `cpu_model` (above) is in the Xeon record.

## 1.5B and 3B (2026-11-12)

The CPU twins ran on the development machine's Zen 5 instead (`cpu_zen5_1p5b.*`,
4 minutes; `cpu_zen5_3b.*`, 9 minutes). The 3B T4 record was made again with
`--low-memory` (`t4_3b.*`, 9.3 minutes; job `y-t4-3b` in
`make_overnight_jobs.py`).

| comparison | exact | stock (control) |
|---|---|---|
| Qwen2.5-1.5B, Zen 5 vs T4 (`compare_1p5b.txt`) | **0/5 changed** | 5/5 changed |
| Qwen2.5-3B, Zen 5 vs T4 (`compare_3b.txt`) | **0/5 changed** | 5/5 changed |
| Qwen2.5-3B, Zen 5 built with `--low-memory` vs T4 (`compare_3b_lowmem.txt`) | **0/5 changed** | 5/5 changed |

The two Zen 5 3B records, one per builder, are the same in every field the
comparison reads (tokens `f349e1e481daccd2` exact, `c1389ce3339bc3d1` stock).

**The first 3B comparison refused its control, wrongly.** The T4's stock
weights fingerprinted `4548a1af4d9f1c10` against the CPU's `de4101d908728218`,
so `--compare` called the stock arm inconclusive. They were the same weights.
Qwen2.5-3B ties `lm_head` to the embedding, and the low-memory build on a GPU
keeps the embedding on the host and gives the head its own copy on the device,
which `named_parameters` then listed a second time. Recomputed on the CPU with
the head listed separately, the fingerprint is `4548a1af4d9f1c10` exactly. The
fingerprint now counts a tied head once while it equals the embedding
(`tools/exact_cross_device.py`, `python/tests/test_weight_fingerprint.py`,
which fails on two mutants: the head always counted, and the head always
dropped). Existing records keep their fingerprints: a tie that stays shared was
already listed once. The T4 record was made again with the fixed tool.

## The Rust suites on Kaggle's CPUs

`cargo test --no-fail-fast` of `4d67971` on a Kaggle CPU session (Intel Xeon
@ 2.20GHz: AVX2 and FMA, no AVX-512), no GPU, then the 20 failing targets
again in a triage session (`cargo_intel_avx2_*.log`; Y's runtime named by
`Y_RUNTIME_C`, the build being outside the checkout): 1381 of 1421 passed. Of
the 40 failures none is a wrong value:

- 16 need `clang`, which the machine did not have: Y's LLVM backend links with
  it and says so (`clang not found or failed to execute`). Installed, they
  pass. They fail rather than report a skip.
- 28 with clang installed (24 of the 40, and 4 more the missing clang had
  stopped earlier) expect Y's exact CPU GEMM (`vpdpwssd`, AVX-512 VNNI) to be
  substituted. On a CPU without AVX-512 VNNI the compiler declines it, says
  why ("this machine does not have AVX-512 VNNI, so `vpdpwssd` would fault"),
  and keeps the scalar exact lowering, as it should; the tests assume a host
  that has it, like the development machine's Zen 5.

Both are the suites' assumptions about the machine, and on a laptop without
AVX-512 (most Intel parts) they would show the same failures.

Those tests now report a skip on such a machine, naming what it lacks, and
fail under `Y_VERIFICATION_STRICT=1` (`tests/common/verification.rs`:
`exact_cpu_gemm_host`, `avx512_host`, `clang_host`). The development machine
played that host for the whole suite (`Y_NO_AVX512_VNNI=1`, a `PATH` without
clang, and without the CUDA toolkit): 1420 passed and 3 failed. One was
`the_predicate_agrees_with_the_machine`, which reads the real CPU and refuses
the override by design. The other two were the `exact_pv` device tests
(`tests/ptx_exact_pv.rs`): with a GPU but no `ptxas` the translation validator
cannot run. Those two now skip without `ptxas` and `nvdisasm`, or on a card
other than sm_89, which is the only one the validator licenses; a T4 running
`cargo test` would have failed them the same way.

## Not covered

One NVIDIA vendor line (Turing, Ada); x86 CPUs from AMD (Zen 5) and Intel (two
Xeon models); one model family, Qwen2.5 at 0.5B, 1.5B, 3B and 7B. Y's own
GEMM and the Triton int8 GEMM need sm_80, so on the T4 the linear layers ran on
cuBLASLt (`torch._int_mm`).
