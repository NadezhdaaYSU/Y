# Y

Y is a systems language and compiler for hardware-aware numerical code. One
Rust compiler (`src/`) takes `.ysu` programs to native x86-64 code through
LLVM, to NVIDIA PTX, to an in-process CPU JIT, and to R1CS circuits for
zero-knowledge proofs. A Python package (`y_lang`) launches Y kernels on
PyTorch tensors and carries a reproducible-inference path for Hugging Face
Qwen2 models.

The project is single-developer research software. Every claim below links
to the record it comes from, with the hardware, the raw data and the
conditions. What does not work is listed as carefully as what does.

## What it is for

| Use | State | Where |
|---|---|---|
| **Reproducible LLM inference.** The same tokens on a CPU and a GPU, and whatever else is in the batch. Every reduction whose order can change is done in integers. | A PyTorch + Triton path; the linear layers and p·V can run on Y's own kernels. Qwen2.5 0.5B and 7B gave the same tokens on a CPU and an RTX 4070 Ti SUPER (0 of 5 cases changed; stock bf16 changed 5 of 5), 0.5B also on a Tesla T4, and the same tokens in 16 batch compositions (stock changed 16 of 16). | [bit-identical decode](docs/bit_identical_decode.md), [Tesla T4 record](docs/benchmarks/exact_cross_gpu_t4_2026-11-11/README.md) |
| **A machine-checked proof of that inference path.** Coq theorems: every valid execution of a decode step gives the same next token and cache rows, for Qwen2.5 0.5B to 7B; and a row of a ragged (left-padded) batch decodes as its prompt alone. | Proved, conditional on a stated trust boundary (for ragged batches, also on the run's values being finite). The model, each prefill row included, is checked against the code bit for bit on a Zen 5 CPU, a Xeon and a Tesla T4. | [what is proved and assumed](docs/exact_decode_theorem.md) |
| **GPU kernels at vendor-library speed.** Tensor-core GEMMs written in Y. | Measured on one RTX 4070 Ti SUPER: FP32 0.96–1.20x cuBLAS, bit-identical to an in-order `fmaf` reference (on a Tesla T4 0.96–1.20x where its tiles fill the SMs, 0.23–2.6x where they do not); FP16 0.99–1.50x; BF16 0.93–1.51x; int8 1.03x cuBLASLt at 4096³. Where Y loses is measured too. | [GPU backend](NOTEBOOK.md#gpu-nvidia-ptx-backend), [where it loses](NOTEBOOK.md#where-the-gpu-backend-loses) |
| **Kernels with proofs.** Coq proofs of what a kernel's schedule and value are, tied to the emitted code; a PTX-to-SASS translation validator. | 30 proof files, none admitted, checked in the test suite. Each proof says what it does not cover. | [proof inventory](docs/exact_decode_proof_inventory.md), [proof-carrying kernels](docs/proof_carrying_kernels.md), [translation validation](docs/ptxas_translation_validation.md) |
| **Safety checks at compile time.** `@safe` blocks, `@invariant`s discharged by Z3, interval arithmetic, exactly-once async tokens, `@ZeroDrift` exact accumulation. | Real; a check that cannot be discharged is an error. | [safety and verification](NOTEBOOK.md#safety-and-verification) |
| **Zero-knowledge circuits.** R1CS and Groth16, interoperating with circom and snarkjs, plus a circom 2.x front end. | Compiles circomlib 31/31 and circom-ecdsa 17/17. Needs `--features zk`. | [ZK backend](NOTEBOOK.md#zero-knowledge-backend-r1cs--groth16), [circom front end](docs/circom_frontend.md) |
| **A CPU JIT** for ordinary Y host functions, embeddable from Rust, C and Python. | Real on Linux x86-64, through LLVM ORC. | [CPU JIT](docs/cpu_jit.md) |
| **Source-level debugging** of host programs in gdb and of GPU kernels in cuda-gdb. | Real. | [debugging](NOTEBOOK.md#debugging-y-programs) |

Y is not a general-purpose replacement for CUDA C++ or PyTorch. It runs on
Linux on x86-64; the GPU results above were measured on NVIDIA sm_89 and
sm_75 cards. Open defects are in [known issues](docs/known_issues.md).

## Quick start

Requirements: a Rust toolchain and `clang`. `z3` is needed by any program
with an `@invariant` (the compiler refuses one it cannot check). `coqc`,
`ptxas` and a CUDA driver enable the proof and GPU gates.

```bash
cargo build --release
./target/release/Y tests/hello.ysu                   # native binary, through LLVM
./target/release/Y tests/hello.ysu --emit-cpu        # prints scalar Rust; compiles nothing
./target/release/Y tests/paged_decode_attention_128_32_8_16.ysu --emit-ptx
```

The Python package, as a wheel that carries the compiler library:

```bash
python3 -m pip wheel ./python --no-deps -w dist
python3 -m pip install dist/y_lang-*.whl transformers
```

```python
from y_lang.exact import load, generate

model, tokenizer = load("Qwen/Qwen2.5-0.5B-Instruct", device="cuda")
tokens = generate(model, tokenizer, "The capital of France is", max_new_tokens=32)
```

The model is quantised (int8 weights; quantised activations and K/V cache),
so its tokens are not the bf16 model's: the claim is "deterministic", not
"identical to bf16". Accuracy is measured in
[bit-identical decode](docs/bit_identical_decode.md). More in
[`python/README.md`](python/README.md).

## Backends

| Flag | Output | State |
|---|---|---|
| *(none)* | native executable, through LLVM IR and `clang` | the default |
| `--emit-llvm` | LLVM IR | real |
| `--emit-ptx` | NVIDIA PTX | real; gated on `ptxas` assembling it |
| `--jit` | runs host functions in the in-process CPU JIT | real; Linux x86-64 |
| `--target=r1cs` | R1CS; `--witness` and `--emit-verifier` add a witness and a Groth16 Solidity verifier | real; build with `--features zk` |
| `--emit-native` | standalone x86-64 ELF | a straight-line subset over `I32` and `bool`; refuses the rest by name |
| `--emit-cpu` | scalar host Rust source, printed for you to compile | not a build step; it emits no SIMD |

Every flag, with its limits:
[the whole command-line surface](NOTEBOOK.md#the-whole-command-line-surface).

## Checking the claims

```bash
cargo test --release
cargo test --release -p y-gpu        # the GPU library's own tests
cargo test --release --features zk   # with the ZK backend
PYTHONPATH=python python3 -m unittest discover -s python/tests -p "test_*.py"
PYTHONPATH=python python3 -m unittest discover -s python/y_lang/tests -p "test_*.py"
python3 tools/verify.py              # SMT, Coq, translation and PTX gates, strictly
```

**A gate whose tool is missing skips and still reports `ok`.** A green run
without `coqc`, `z3`, `ptxas`, `clang` or a GPU has not run those checks; the
list of which gate needs what is in
[Building](NOTEBOOK.md#building). Benchmarks keep their raw outputs and
toolchain identities under `docs/benchmarks/` and `benchmarks/`; the
[benchmarks index](README_BENCHMARKS.md) lists them.

## Repository layout

```
src/                       Rust bootstrap compiler
  lexer.rs parser.rs ast.rs        front end
  type_checker.rs                  safety blocks, Z3 invariants, interval arithmetic
  linear_tracker.rs                async token single-consumption
  guarantees.rs                    what the front end proved, checked or trusted
  sentinel.rs ysu_gpu_probe.rs     hardware probe
  autotuner.rs empirical_autotune.rs cuda_runtime.rs
  bank_conflict.rs                 shared-memory swizzle solver
  llvm_emitter.rs                  LLVM IR (default backend)
  debug_info.rs                    DWARF for -g: debugging Y programs in gdb
  ptx_emitter.rs                   NVIDIA PTX
  cpu_emitter.rs cpu_gemm.rs       --emit-cpu, and the AVX-512 exact GEMM
  cpu_jit.rs                       in-process CPU JIT (LLVM ORC)
  native_emitter.rs                standalone ELF
  zero_drift.rs                    @ZeroDrift representation selection
  exact_gemm_certificate.rs        the .v a compilation emits with its kernel
  exact_attention_certificate.rs   the same, for the GPU attention kernel
  exact_attention.rs fixed_exp.rs  exact int8 attention PTX + integer exp2
  zk_field.rs                      BN254 Fr, Montgomery form
  zk_emitter.rs zk_witness.rs      R1CS emission and witness solving
  zk_solidity.rs                   Groth16 on-chain verifier
  circom_{lexer,ast,parser,lower}.rs   circom 2.x front end
  c_api.rs                         C ABI; the crate also builds as a cdylib
  ir_grapher.rs coprocessor_scheduler.rs rt_core_emitter.rs
                                   a scheduling simulation, not a code path
  ypm.rs ysu_gpu_probe.rs          separate executables, not modules

crates/y-gpu/   GPU runtime library
self_hosted/    compiler phases rewritten in Y (.ysu); not the default build path
proofs/         Rocq (Coq) proofs; ExactGemmSchedule.v, AttentionSchedule.v and
                ExactDecodeConstants.v are GENERATED
tests/          test programs, benchmarks, PTX assembly gates
python/         the `y_lang` package: bindings to liby.so, torch interop,
                y_lang.exact (reproducible inference), and its tests
tools/          measurement and analysis harnesses, run by hand
  ptxas_tval/        PTX-vs-SASS translation validator
  exact_gemm_bench/  the three-arm GEMM benchmark
  ydb/               Y-aware gdb commands and PTX/SASS line maps
circomlib/      vendored circomlib (upstream 2.0.5, GPL-3.0; see NOTICE)
docs/           the language manual, design notes, benchmark records
NOTEBOOK.md     the dated research notebook (the former README)
```

## Documentation

- [Language manual](docs/y_language_documentation.md)
- [Known issues](docs/known_issues.md): open defects, read before relying on a feature
- [NOTEBOOK.md](NOTEBOOK.md): the full dated record of results, failures and
  corrections, which this README summarises
- [Bit-identical decode](docs/bit_identical_decode.md) and
  [the decode-step theorem](docs/exact_decode_theorem.md)
- [The process: taking a kernel from fast to verified](docs/verified_kernel_process.md)
- [Adaptive GPU JIT](docs/adaptive_jit.md), [CPU JIT](docs/cpu_jit.md)

## Licence

Business Source License 1.1 ([LICENSE](LICENSE)). Personal, educational,
research and internal evaluation use is permitted. Use in a commercial
product or service, or as a hosted service, needs a commercial licence from
the licensor; contact details are in [LICENSE](LICENSE). On 2030-01-01 the
licence changes to Apache 2.0. Third-party files keep their own licences
([NOTICE](NOTICE)).

Author: Umut Korkmaz (YSU). Built with LLM assistance for implementation; the
architecture and design decisions are the author's.
