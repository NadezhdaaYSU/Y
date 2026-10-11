# Known issues and limits

Reviewed against the working tree on 2026-10-10 after shortening `CLAUDE.md`,
and on 2026-10-11 against every report in the archived backlog.

## Open

Reviewed 2026-10-11 against every report in [the archived backlog](claude-history/UNRESOLVED.md),
most reproduced against the compiler built from the tree rather than read from
source; each entry there now carries its status. What is still open:

### A wrong answer under a precondition nothing checks

- **Exact attention's `KFix`** (archive 1639 (c)) and **`exact_pv`'s licences
  on `T`, `D`, `Q` and the extents** (archive 1692 (a)) are kernel parameters.
  `KFix = 0` makes the softmax uniform and `KFix >= 2^31` makes a key vanish;
  nothing at the launch boundary checks either, and `temperature_fixed_point`
  is a host helper a caller may skip. The certificates name them as trust
  items. The remedy is the exact GEMM's: check them when the kernel starts.
  That changes PTX that is proved and validated, so it is its own increment.

### Benchmarks

- `python/examples/run_head_to_head_comparison.py` is not a fair comparison:
  its "FP8 GEMM" rows time Y's FP8 kernel against FP16 matmuls in the Triton and
  PyTorch columns, and its block-scan row times Y through a CUDA graph and the
  others as plain launches. Do not quote its report until both are fixed.
- `python/examples/kernels/flash_attention.ysu` does not compile (`Pipeline` is
  not a type); the scripts report that rather than stop.

### Refused loudly, not implemented

- No conversion between integers and floats (§20.4); no type aliases or generic
  functions; `for i in 0..v.size` (a field access as a range bound) does not
  parse; `@ghost let` does not parse, and a `@ghost` block is emitted, not
  stripped (§9.13); `@require` on a `fn` and `&&`/`||` inside it are refused;
  `@align` is refused (§9.4).
- `--emit-cpu` refuses struct literals, zero-initialised arrays and the string
  built-ins; `--emit-native` is a straight-line `I32`/`bool` subset.
- `eval_interval` models no `|`, `^`, `<<`, `>>` or `%`, so an index bounded
  only through one is refused as unprovable (archive 1857).

### Documentation not yet audited

- The §9 directive fragments, §13's 22 migration-guide rows, §20.5's claim that
  `TF32` is a fragment type, and the profile-key table (it names
  `TENSOR_F16_LATENCY` and `WARP_SHUFFLE_LATENCY`, which the profile does not
  contain, and its "how it affects codegen" column is unverified). Three §9
  passages were wrong when checked on 2026-10-11 (§9.3, §9.4, §9.12).

### Test-suite hygiene

- About 37 test files run the compiler in the repository directory. Where the
  repository has no `.ysu_hw_profile` they probe and write one there (atomically
  now), and their output depends on this machine's CPU probes (archive 2214
  (a)-(d), 2437 (a)).
- Without z3 the tests whose programs carry an `@invariant` fail, by design: an
  invariant that cannot be checked fails the build (archive 2214 (f)).

### Debugger

- A data-carrying enum's payload prints as raw words (§38.6); the runtime has no
  debug information, by design; a breakpoint in a kernel the LLVM backend
  replaces with the packed GEMM is hit only on the fallback path.
- `ydb verify`: an invariant proof's listed assumptions are a superset; the
  default build substitutes the exact GEMM but writes no certificate (only
  `--emit-llvm` does); standing validation results are not re-run;
  `GlobalMemory<T>` resolves to `Unknown` in its messages (archive 2420).

### Performance, the proof programme and the translation validator

- Exact GEMM: no hybrid M-then-K split for short-M, short-K shapes (archive
  1515); about 30% of its ISA ceiling against OpenBLAS's ~63%, and nothing stops
  a caller asking for 32 threads on a 16-core part (archive 1566).
- The validator and proof-scope items in archive entries 587-2049 are research
  queue, not wrong answers: 923 of the 925 emitted `mma.sync` are floating point
  and carry no proof; the f16 GEMMs carry no exactness argument; validation is
  at `-O1`, not the shipped `-O3` (unroll matching); the multi-back-edge lift;
  `cvt.rn.f16.f32` unmodelled; `FMUL`/`FMIN` commutativity unassumed; the solver
  wall on carry chains (`bn254_fr_mul_fast`); `mutgate` covers 3 of 14
  harnesses; `barregion.muls` counts no division. Their counts were not
  re-measured here; the current state is in
  [the validator documentation](ptxas_translation_validation.md) and Codex's
  2026-10-08 review.

## Fixed

| Issue | Fix and regression gate |
| --- | --- |
| Python GPU PTX cache omitted compiler identity | The key holds a SHA-256 of the loaded `liby.so`, its path, the target, the hardware profile's hash and the `Y_`/`YSU_` environment. A rebuilt or re-pointed library refuses cache hits until Python restarts. The loaded mapping is matched by path and inode, not device: btrfs reports different devices to `stat()` and `/proc/self/maps`, and comparing devices refused every real library. `python/tests/test_ptx_cache.py`. |
| LLVM data-carrying enums | Constructors store each field in its own slot with its declared type, `match` tests the tag and binds typed payloads per arm, and `x.data.Variant._N` reads and writes a field. Payloads are scalars or references, at most 8; aggregate, generic and Q payloads and ambiguous constructor names are refused by name. `tests/llvm_enum_payloads.rs`. |
| Ordinary LLVM Q formats were computed as integers | `let x: Q16.16 = 1.5` stored 1 and `x > 1.0` was false. A Q value is `value * 2^frac` in 8/16/32/64-bit storage; literals, `*` and `/` round to nearest with ties away from zero; overflow and division by zero trap; struct fields keep the scale; arrays of a Q format, `match` on a Q value and Q arguments to built-ins are refused by name. Manual §20.3; `tests/llvm_fixed_point.rs` (independent oracle, four formats, -O0 and -O2). |
| C runtime widths disagreed with the LLVM declarations | Not only `ystr_len`: seven functions were declared to return `ptr`/`i64` and defined to return `int32_t` (the caller read a register half the callee never set), and four `i64` arguments were received as `int32_t`, so `print_int(5000000000)` printed 705032704. `c_src/runtime.c` now matches the declarations and the JIT's runtime. `tests/runtime_abi_agreement.rs` compares every declaration with clang's type of the definition; handle arguments stay `int32_t` (the pool is `MAP_32BIT`). |
| A name defined twice was not refused | Two `fn f`, two `Type::m`, or a function colliding with an enum constructor: the type checker kept the second; the default build failed inside clang with no reason and `--emit-llvm`/`--emit-cpu` wrote output that does not compile. The front end refuses it by name on every backend. `tests/duplicate_definitions.rs`. |
| `@cache_policy(..., reuse_count=N)` was silently dropped | No backend lowers a reuse count (`createpolicy` takes a fraction of the lines); it is refused by name now. `tests/ptx_cache_policy.rs`. |
| PTX let-bound multiply/add contraction (archived report, project lines 2143/2151) | Confirmed live: `let p: F32 = a * b; p + c` emitted `mul.f32` + `add.f32`, which `ptxas` assembled to ONE `FFMA` under PTX stating two roundings. An unfused float multiply is `mul.rn` now, which `ptxas` does not contract. All 66 corpus kernels emit byte-identical PTX before and after (no shipped kernel takes that path). `tests/ptx_let_bound_rounding.rs` checks the PTX, the SASS and the value on the device (0 rounded twice, 2^-24 fused). |
| Solver discovery depended on the working directory | The project-venv candidates (`venv/bin/z3`, `.venv/bin/z3`, `z3/build/z3`) resolved only against the working directory, so `Y` run elsewhere - or `liby.so` in a Python process started elsewhere - refused every invariant although the repository's own solver existed. They are also looked for beside the compiler's own file (found from the process's mappings, since `current_exe()` names the interpreter inside Python), in its directory and up to three parents. Missing solvers still fail invariant verification. `tests/safe_invariant_enforcement.rs` (a copy of `Y` at the deepest Cargo layout finds a stub solver; one level deeper it does not), `python/tests/test_solver_lookup.py`; tests that need no solver run a copy of the compiler with nothing beside it. |
| `for` headers were evaluated differently by each backend | `for i in 0..20 step k { k = k + 1; }` ran 5 iterations on the LLVM backend and the JIT (the step was re-read after every body) and 20 on the GPU. `--emit-cpu` replaced every non-literal step with 1, re-read the bound every iteration, and let Rust type the variable from a `U32` bound (3e9 ran 3e9 times where every other backend compares its bits as an I32 and runs none); `U8`/`U16`/`U32`/`I64`/`F64`/`bool` parameters were written under their Y names and did not compile. `step 0` was refused only by PTX. Now every backend evaluates `start`, `end`, `step` once, in that order, with an I32 variable compared in 32 bits; `step 0` is refused by the front end. Manual §13.10; `tests/for_header_semantics.rs` (LLVM, JIT, `--emit-cpu` through rustc, and a kernel on the device). |
| Unsigned integers were refused in invariant proofs | Every unsigned variable under an `@invariant`, and so every loop with a `U32` bound, failed with "does not have a supported signed integer type", from when the backends disagreed on unsigned operators (archive lines 2437, 2453). They agree now (`llvm_unsigned_ops`, `llvm_integer_widths`, `ptx_integer_datapath` on the GPU, `for_header_semantics`), and `src/type_checker.rs` models U8..U64 exactly in `0..=2^n-1` with an unsigned quotient. Where backends could read operands differently it is conservative: a literal beside an unsigned value must lie in its range, mixed signedness or unsigned widths are proved within `0..=2^31-1`-style agreement ranges (a `U32` 4294967295 and an `I32` -1 compare equal on the machine), negation of an unsigned value is refused, and a `U32` loop bound is read as the I32 its bits spell. I8/I16/U8/U16/U64 locals are now tracked by the loop proofs. `tests/smt_unsigned.rs`. |
| `@bounds` the compiler could not prove was taken on trust | `@bounds(0, 3) let i: I32 = get(1000000);` compiled, every proof using `i`'s range rested on the annotation, and the index it fed skipped its run-time check: `arr[i] = 42` wrote a million elements past a four-element array under `@safe` and exited 0. An exact-GEMM operand outside its `@bounds` overflowed the int32 accumulator into a wrong answer under a certificate claiming exactness. Now an unprovable range is checked when the program runs: LLVM and the JIT print the value and exit 1, PTX traps, `--emit-cpu` panics, `--emit-native` refuses; the exact GEMM scans both operands before computing; `ydb verify` reports `run-time` and no proof rests on the bound. `tests/bounds_runtime_check.rs`. |
| `@bounds` in a circuit ignored `min` and could not be proved | The R1CS backend decomposed the value into `bit_len(max)` bits: `min` was ignored (`@bounds(10, 20)` admitted 0..9), negative ranges refused their own values, a non-constant bound was skipped without a word, and the bit wires had no witness recipe, so no circuit carrying `@bounds` was provable for any input. Now `value - min` and `max - value` each decompose into `bit_len(max - min)` bits through `emit_num2bits` (witness recipes included), and a non-constant bound is refused. Manual §24.4; `tests/zk_bounds_constraint.rs`. |
| `@ZeroDrift` accumulators wrapped, saturated and trusted their `@bounds` | The representation is chosen to hold the declared `@bounds`, and nothing tested it: 20 terms of 100 under `@bounds(-1024.0, 1024.0)` reached 2000 and carried on. The accumulate was a plain `add` (two I64 terms of 5e18 wrapped negative), LLVM's `fptosi` of an oversized term is poison (one term of 1e12 gave a total of 0, exit 0), and PTX's conversion saturates (a NaN became `i64::MIN`, measured). Now the term is tested before conversion, the accumulate is overflow-checked, and the declared range is tested after every store; each stops the program (exit 1 with a message on LLVM and the JIT, a trap on the GPU). The manual's §9.6 claimed Kahan compensation and §14.8 a compile-time `D0001` error; neither existed. `tests/zero_drift_runtime_checks.rs`. |
| `--emit-cpu` wrote untyped `let`s | Rust picked each local's type from how it was used: `let x: U8 = 200; x = x + 100; return x / 2;` gave 150 where the LLVM backend's `u8` wraps and gives 22, and `I64 + I32` did not compile. Now a numeric local carries its declared type; a `let`, assignment, `op=`, `return` or argument converts with `as` (a literal is written as one of the target type); a mixed operation computes in Y's common type (the wider, unsigned at equal width, a float over an integer); and typed integer `+ - *` are the wrapping operations, as Y's integers wrap on every backend (Rust's operators panic in a debug build and refuse a constant overflow). `tests/cpu_emitter_types.rs` runs each case on the LLVM backend and through rustc. |
| `y_inductor` kept a multiply apart from the add it feeds | The rule existed because ptxas contracted the pair; once the PTX emitter rounded a `let`-bound product first, the fused kernel matched eager bit for bit and the rule only cost kernels. Removed after measuring: fused took 0.38-0.87x of the separate kernels' time at 2^22 and 2^24 elements in every model and run (2^20 was noise-dominated). `benchmarks/inductor_fusion/`; `test_inductor.py` asserts the fusion and the bit-identity. |
| Void `ysu_main` exit status | The CLI's AOT builds give a `main` with no return type an `i32` return of 0; at `-O0` it exited with the last callee's `eax` (43 in the test). Library and JIT callers keep the source's `void`. `tests/llvm_void_main_status.rs`. |
| A host `fn main` was called without its parameters, and its float result was not read (archive 2243 (f)) | The C runtime calls `int32_t ysu_main(void)` and exits with its result; native `_start` calls `main` with no arguments. `fn main(x: I32) -> I32 { return x + 5; }` exited 5 on both (`x` was whatever its register held) and `fn main() -> F64 { return 7.0; }` exited 0 (the result sat in `xmm0`). Both are refused on the default build, `--emit-llvm` and `--emit-native`; an integer, `bool` or absent result still reaches the exit status. The JIT and circuits, whose `main` does take arguments, are unchanged. Manual §38.4; `tests/host_main_contract.rs`. |
| `Y` with no input file compiled a built-in program that no longer parsed (archive 2278 (iii)) | It printed a syntax error about source the user never wrote and exited 1. It is a usage error now (exit 2). `tests/unknown_flags_are_rejected.rs`. |
| `--emit-cpu` dropped struct-field attributes (archive 2151 (b), 1533 (e)) | A struct passed by reference was written with `pub a: i32` for `@atomic a: I32`; `@gpu_uncached` likewise. Refused by name now. Manual §9.3, §9.5; `tests/field_attributes_are_lowered_or_refused.rs`. |
| `@align(N)` claimed an alignment nothing provided | Found in this review. The LLVM backend put `align N` on every access of the field while the struct kept it at its natural offset (`%S = type { i32, i32, i32 }` put an `@align(64)` second field at offset 4) and its storage was an `alloca` at natural alignment - undefined behaviour - and the padding the manual promised was never emitted. Refused by the parser now; the two fixtures that used it already padded by hand. Manual §9.4; `tests/field_attributes_are_lowered_or_refused.rs`. |
| `@tile` on a loop or a `fn` changed nothing (archive 2151 (c)) | On a `for`, the LLVM and PTX backends wrote it into their output as a comment (the PTX one twice); on a `fn`, its one reader excluded the function from a helper analysis. Both refused now; `@tile(M, N, K)` on a `kernel` is unchanged. `tests/train_spec.ysu` lost the attribute and `tests/train_spec.ptx` was regenerated (only the two comment lines moved); `tests/tile_test.ysu`, a demo of both refused forms, was removed. Manual §9.12; `tests/directives_are_consumed_or_refused.rs`. |
| The GPU probe was not found from Python (archive parent 2170) | With no profile in the working directory the compiler probes and writes one there. It looked for `ysu_gpu_probe` beside `current_exe()`, which inside Python is the interpreter, so a `compile_to_ptx` outside the repository wrote `GPU_NAME=Unknown GPU`, `SM_VERSION=0.0` on a machine with an RTX 4070 Ti SUPER, and every later compile there read it. It is looked for beside the loaded `liby.so` now, as the solver is. `python/tests/test_gpu_probe_lookup.py`. |
| `.ysu_hw_profile` writes were not atomic (archive 2214 (b)) | The probe wrote it with `fs::write` (truncate, then write), so a compile starting in that window read a partial profile, which still EXISTS and so is not re-probed. The autotuner wrote-then-renamed through one fixed temp name that two writers share. Both use `sentinel::write_profile_atomically` now (a per-process, per-call temp name, then `rename`). Unit test `a_reader_never_sees_a_partial_profile`. |
| `--emit-llvm` and the default build chose `@ZeroDrift` representations from different inputs (archive 2198, 2214 (e)) | Only `--emit-llvm` loaded the profile's drift costs - GPU timings - so with `I64` recorded as cheapest it reported `acc: I32 -> I64` while the binary was built with `Q16.16`. Both LLVM paths take the narrowest sufficient representation now and both print the choice. `tests/zero_drift_llvm_paths_agree.rs`. |
| `--emit-cpu` struct-field arithmetic was untyped | A residual of the `--emit-cpu` typing fix above: a field's type was unknown, so `p.a + 100` on a `U8` field used Rust's `+` (a debug build panicked where Y wraps to 44), and `p.a + p.b` across widths did not compile. Fields of struct parameters are typed now. `tests/cpu_emitter_types.rs`. |
| The Python example benchmarks reported figures they had not measured (archive parent 2170) | `Benchmarks.py` wrote a "results" report from a fixed template - every figure, the GPU and the software versions were literals (a 0.18 µs cache lookup where this machine measures ~90 µs; a FlashAttention kernel "emitted" that does not compile). `benchmark_triton_parity_features.py` loaded `libY.so` (the library is `liby.so`), declared a four-argument function with three, and on failure appended a fixed "0.001 µs (cached)" as a result. Three scripts compiled `flash_attention.ysu` first and stopped there; `run_head_to_head_comparison.py` also compiled an unused kernel the PTX backend refuses, so it raised at import. Each reports what it measured now, a failure as a failure. |
| Chapter 10's gate counted IR clang rejects as compiling (archive 2266 (e)) | Example 17 put PTX's `nanosleep.u32` in a host `chisel` block; `--emit-llvm` wrote it out as x86 inline assembly, clang rejects it and the default build fails, and the example carried no note. The gate builds each host example's module with `clang -c` now, and Example 17 has its note. `tests/manual_examples.rs`. |
| The `@ghost` refusal said `@ghost let` works | Its hint named `@ghost let ..` as a working form; it is a syntax error (§9.13). Corrected. |

## Verification limits

`tools/ptxas_tval/README.md` states that licensed validation requires explicit
matching sm89 targets and that general unrolled-loop correspondence remains
unsupported. Assembly on another architecture is assembly evidence. Preserve
`UNPROVED`/refusal results; do not turn incomplete coverage into a correctness claim.


## Remaining reported issues

[UNRESOLVED.md](claude-history/UNRESOLVED.md) indexes every line explicitly marked
unfixed in both original files, including duplicate and retracted reports. Each
entry retains its original snapshot line number and investigation context.
The [archive index](claude-history/INDEX.md) covers other findings that did not
use those markers. Original text is preserved verbatim with recorded hashes.

Each entry there carries its status as of 2026-10-11 (fixed, with the test that
pins it; open, and listed above; retracted; or research). A status is a claim
about the tree it was checked against: before related changes, search the
backlog and source, reproduce when necessary, and record the result here.
