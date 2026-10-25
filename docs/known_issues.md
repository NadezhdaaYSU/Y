# Known issues and limits

Reviewed against the working tree on 2026-10-10 after shortening `CLAUDE.md`,
and on 2026-10-11 against every report in the archived backlog.

## Open

Reviewed 2026-10-11 against every report in [the archived backlog](claude-history/UNRESOLVED.md),
most reproduced against the compiler built from the tree rather than read from
source; each entry there now carries its status. What is still open:

### Checked by name only

- `@divergence(uniform)` is checked by looking for identifiers containing
  `threadIdx` or `laneId`, so `@divergence(uniform) if thread_idx_x() < 20`
  passes. It changes no PTX (the LLVM backend attaches metadata to the
  branch), and since 2026-10-12 it does not admit a shuffle or a barrier
  into a branch the threads can disagree on; see the fixed entry below.

### Benchmarks

- `python/examples/benchmark_3d_block_ptr.py` has been run only on a busy GPU
  since its rewrite (2026-10-11); no figure from it is published.
- A Y launch from Python costs 1.72 us of CPU time per call through the
  native launcher, against PyTorch's 2.02 and Triton's 5.6
  (`docs/benchmarks/launch_2026-10-17/`); it was 6.6, then 2.9, which is
  still the cost where the launcher cannot be compiled (no C compiler or no
  `Python.h`; `launcher_status()` says). Performance, not correctness.
- `python/examples/kernels/flash_attention.ysu` does not compile (`Pipeline` is
  not a type); the scripts report that rather than stop.
- The head-to-head's Triton FP8 kernel (`tl.dot` on e4m3) chains its K loop
  in the tensor core's accumulator, as Y's FP8 GEMM did until 2026-10-18: a
  median relative error of 6.6% on positive inputs at K = 4096, where Y's
  FP8 GEMMs give 6.5e-5 and `_scaled_mm` 5.7e-4. Its inputs are small
  integers, exact either way, so the comparison is of time at unequal
  precision.

### Precision

- The F16 GEMM (`@tile`, f16 in, f32 out) chains its K loop in the tensor
  core's accumulator above 32 rows. That accumulator adds with less than
  float32 rounding, truncating, so same-sign results land below float64 by
  about 1.2e-5 per 1024 of K. Exactness tests on small integers cannot see it
  (`docs/benchmarks/f16_xbar_2026-10-18/precision/`).

  Measured on 2026-10-20 over nine input distributions and eleven shapes
  (`docs/benchmarks/f16_accuracy_2026-10-20/`). The metrics are the relative
  Frobenius error, the maximum absolute error, and the maximum error over
  S_ij = sum_k |a_ik| |b_kj|. Where cuBLAS and Triton also chain (256^3 to
  4096^3, 512 x 512 x 1024), the three give the same bits. Elsewhere cuBLAS is
  more accurate than the chain on every distribution and metric:
  - 8192^3: about 3x.
  - 512 x 512 x 4096 to 65536: 2-8x, and up to 70x in relative Frobenius
    error on the long-K cancellation case. cuBLAS splits K at 16384 and
    65536.
  - 64 x 4096 x 4096: 2-9x.

  **Status (2026-10-25): fixed where the block sums fit.** From K = 4096 the
  sum is blocked by default (`docs/benchmarks/f16_blocked_2026-10-25/`):
  chained 256 of K at a time into a block sum added in float32. Its relative Frobenius and maximum
  normalized errors are below cuBLAS's in all 117 cases where it differs from
  the chain, at all three shapes above and at 512 x 4096 x 14336 among
  others. Tiles holding 128 accumulators a thread (64 x 64 warp tiles, such
  as 4096^3's) keep the chain, whose bits are cuBLAS's at the shapes measured
  with them. Per output it is still farther from R than cuBLAS on up to 27%
  of the outputs under cancellation. It costs 0-4% of the chain's speed
  (median 1% over 22 LLM layer shapes); at 512 x 512 x 16384 that is
  0.97-0.985x cuBLAS's speed where the chain was 1.01x, and at 65536
  0.91-0.94x (the blocked split-K kernel runs at 1.01-1.04x).

  A kernel named `<name>_splitk<S>` splits K (2026-10-20,
  `docs/benchmarks/f16_splitk_2026-10-20/`), each split in float32, the
  partials added in order. It is below cuBLAS in Frobenius error in 27 of
  27 cases at 512 x 512 x 4096 to 65536, and faster at 4096 and 16384, but
  at 65536 it takes 404-412 us against cuBLAS's 401-404. Chained
  (`Y_F16_ACCUMULATE=mma`) it takes 388-398 us and is half as accurate as
  cuBLAS. cuBLAS's own bits there are Y's chained split 8, and at 4096
  chained split 2. Since 2026-10-25 a split of more than 32 rows longer
  than a block sums blocks by default: in 4 splits it is below cuBLAS in
  all 18 cases at 16384 and 65536 and runs at 1.01-1.04x cuBLAS's speed
  (`docs/benchmarks/f16_blocked_2026-10-25/`).

  Since 2026-10-25 a kernel named `<name>_splitk` chooses the split itself
  (opt-in, `docs/benchmarks/f16_auto_splitk_2026-10-25/`): 2 splits on 64x128
  tiles at 512 x 512 x 16384 and 65536, 1.02-1.07x and 1.01-1.06x cuBLAS's
  speed (with A and B in the L2 or from DRAM), below cuBLAS's error in all
  18 cases there (and in the 9 of its
  third split, 256 x 256 x 16384). Still open:
  - A plain three-parameter GEMM is not split: its hosts launch it with no
    workspace on a z = 1 grid. The compiler names the split it would take;
    the adaptive JIT, which owns its launches and buffers, is where a default
    could apply it.
  - At 256 x 256 x 16384 with A and B streamed from DRAM the rule's 4 splits
    run at 0.93-0.96x cuBLAS, where 8 (2048 deep, below the rule's bound)
    run at 1.01-1.04x; with A and B in the L2 the 4 are 1.7% ahead.
  - Decode splits at N = 14336 ran 1.04-1.08x faster from DRAM. The rule does
    not take them: a split of at most 32 rows adds in plain float32, not the
    compensated sum, and its accuracy against cuBLAS is unmeasured.
  - Under cancellation up to 25% of the splits' outputs are still farther
    from R than cuBLAS's. `Y_F16_ACCUMULATE=blocked_kahan` with
    `Y_F16_BLOCK_K=64` takes that to 5-11%, at 0.84-0.87x the speed.

  GEMMs of at most 32 rows add each mma's result in float32 instead,
  compensated by default since 2026-10-24 (below). `Y_F16_ACCUMULATE=f32`
  selects the plain float32 adds everywhere; on square shapes they take
  1.03-1.09x the time.
  - Against cuBLAS it has the lower Frobenius error in 97 of 99 cases. It is
    not more accurate everywhere.
  - At M = 1, cuBLAS's GEMV (float32 FMAs, no tensor cores) is 5-7% lower in
    Frobenius error on same-sign inputs, and its output is the closer one on
    47-48% of the outputs.
  - On the paired cancellation case at K = 16384 and 65536, cuBLAS's split K
    has the lower maximum normalized error.
  - Each mma still truncates its 16 products: a bias of -7.0e-8 to -8.0e-8
    on same-sign inputs at any K, which float32 FMAs (cuBLAS SGEMM) do not
    have. SGEMM has the lower Frobenius error in 18 of the 22 same-sign
    cases.

  The compensated adds, once per K-tile, are the default at M <= 32 outside
  split K since 2026-10-24, and `Y_F16_ACCUMULATE=kahan` selects them
  everywhere (`docs/benchmarks/f16_kahan_2026-10-20/`). Their Frobenius and
  maximum normalized errors are below cuBLAS's and SGEMM's in all 63 cases
  measured, M = 1 included. At M <= 32 and N = K = 4096 they cost nothing
  measurable when B streams from DRAM and 1.06-1.20x the plain adds' time
  when B is L2-resident, still 0.39-0.92x cuBLAS's. At M = 32 with K = 11008
  or 14336 they take 1.02-1.03x the plain adds' time from DRAM, and at
  32 x 4096 x 14336 that is 0.99x cuBLAS where the plain adds were 1.01x
  (`docs/benchmarks/f16_kahan_default_2026-10-24/`). On 512 x 512 tiles they run at
  0.72-0.75x cuBLAS, so split K keeps the plain adds. The bias stays, and
  per output the result is still farther from R than cuBLAS's GEMV on 6-21%
  of the outputs at M = 1.

### Python packaging

- `python/pyproject.toml` cannot be built into a wheel: its `readme =
  "../README.md"` lies outside the project, and setuptools refuses it
  (checked 2026-10-17). The package is used through `PYTHONPATH`.

### `y_lang.ops`

- Nothing in it runs a kernel the Y compiler emitted. `fp8_gemm`'s CUDA C++
  path (`tests/y_tensor_core_gemm.cu` through CuPy) cannot run with CuPy 14:
  NVRTC cannot find `crt/mma.h` through the pip CUDA headers, and
  `Stream.from_external` no longer takes a raw pointer. It raises on 1- and
  2-byte CUDA operands now. If it is made to run, the kernel has never been
  checked against a reference.
- `int4_weight_only_gemm` takes unpacked values (one per byte), not the packed
  layout its name suggests; `paged_attention` is a Python loop with a host
  sync per block. The other helpers are single PyTorch calls.

### Refused loudly, not implemented

- The CPU interpreter (`y_lang.interpreter`) does not model the tensor-core
  instructions (`mma`, `ldmatrix`, `wmma`), the packed FP8/FP16 conversions,
  `vote`, `.param`-space calls, `bar.sync` with a thread count, `bar.arrive`,
  `barrier.sync` without `.aligned`, or any modifier outside its per-instruction
  list (`.relu`, `.xorsign`, `.shared::cluster` ...). It refuses each by name,
  so the GEMM and attention kernels built on them are refused there. Its
  `.approx` functions match the GPU's only within the GPU's documented error,
  not bit for bit; every other modelled instruction form matches bit for bit
  (`python/y_lang/tests/test_interpreter_ops.py`).

- No conversion between integers and floats (§20.4); no type aliases or generic
  functions; `for i in 0..v.size` (a field access as a range bound) does not
  parse; `@ghost let` does not parse, and a `@ghost` block is emitted, not
  stripped (§9.13); `@require` on a `fn` and `&&`/`||` inside it are refused;
  `@align` is refused (§9.4).
- `--emit-cpu` refuses struct literals, zero-initialised arrays and the string
  built-ins; `--emit-native` is a straight-line `I32`/`bool` subset.
- `--emit-cpu` does not lower kernels (found 2026-10-18): every
  `GlobalMemory<T>` parameter is written `*mut f32` whatever `T` is, an element
  read is the element's ADDRESS cast to the value's type (`let v: I32 = X[1]`
  became `((X.add(1)) as i32)`, which rustc accepts), and an element store is
  `Out.add(0) = v`, which rustc rejects. A kernel therefore does not compile
  as Rust, since it does nothing without a store. `GlobalMemory<E4M3>` is
  refused by name.
- `eval_interval` models no `|`, `^`, `<<`, `>>` or `%`, so an index bounded
  only through one is refused as unprovable (archive 1857).

### Documentation not yet audited

- The §9 directive fragments, §13's 22 migration-guide rows, §20.5's claim that
  `TF32` is a fragment type, and the profile-key table (it names
  `TENSOR_F16_LATENCY` and `WARP_SHUFFLE_LATENCY`, which the profile does not
  contain, and its "how it affects codegen" column is unverified). Three §9
  passages were wrong when checked on 2026-10-11 (§9.3, §9.4, §9.12).

### Autotuning

- The autotuner's in-process tile cache (`AUTOTUNE_CACHE` in
  `src/autotuner.rs`) is keyed by M, N and K alone. Within one process the
  first compile of a shape decides the tile of every later one, whatever its
  precision, GPU or `.ysu_hw_profile` (whose measured tiles, and whose
  shared-memory limit, are read from the working directory). The tile is a
  performance choice, but the emitted PTX then depends on what the process
  compiled first. `python/y_lang/tests/test_launch.py` pins its tiles with
  `Y_CTA_OVERRIDE` and its profile with a temporary one for this reason.
- A shape with no measured tile borrows the nearest measured one
  (`load_persisted_nearest`), whatever the grid size then does. At
  1280x1024x1024, 1024^3's 3-stage tile measured 6.4% slower than its
  2-stage tile (`docs/benchmarks/f16_tiles_2026-10-17/`). By shared memory,
  the 3-stage tile (41,472 bytes) fits 2 CTAs per SM, so that shape's 160
  CTAs take two waves; the 2-stage tile (27,648 bytes) fits 3, one wave. The
  committed artifact at that shape (`gemm_f16_uneven_grid`) now has a tile
  measured for it.

### Test-suite hygiene

- About 37 test files run the compiler in the repository directory. Where the
  repository has no `.ysu_hw_profile` they probe and write one there (atomically
  now), and their output depends on this machine's CPU probes (archive 2214
  (a)-(d), 2437 (a)).
- Without z3 the tests whose programs carry an `@invariant` fail, by design: an
  invariant that cannot be checked fails the build (archive 2214 (f)).

### Debugger

- A float kernel argument past float32's range raises `OverflowError` when
  launched on the GPU, and becomes inf under the CPU interpreter
  (`Y_INTERPRETER=1`), which converts it with numpy.
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
- A `/` or `%` by a literal in a kernel's own source (`a / 8` on `U32`)
  still reaches `ptxas` as a division by a register holding the literal
  (`mov.u32 %r13, 8; div.u32 %r15, %r10, %r13`). The pass that rewrites the
  emitter's own divisions by constants (2026-10-19) sees immediates only,
  so `ptxas` still divides these at run time with a reciprocal sequence.
  Performance, not correctness (`docs/benchmarks/const_div_2026-10-19/`).
- `ptxas` puts three dead `@!PT LDS RZ, [RZ]` before a `cp.async` that
  follows a `bar.sync`, and after nothing else probed. The F16 GEMM's K loop
  issues the next tile's copies after its barrier every K-tile: at 512^3 that
  is 48 dead loads a warp, 7.9% of the 311K instructions it executes
  (`docs/benchmarks/const_div_2026-10-19/probes/`). Every form of the copy
  probed gets them, from `ptxas` 12.8 and 13.4, on sm_80 to sm_89
  (`docs/benchmarks/smem_swizzle_2026-10-20/probes_dead_lds/`).
- The F16 GEMM's and the fused SwiGLU kernel's A and B tiles are swizzled
  since 2026-10-20 because the padded rows made every `cp.async` copy read
  1.5 times the L2 sectors it needs (`docs/benchmarks/smem_swizzle_2026-10-20/`).
  The FP8 GEMM (f32 inputs) copies raw f32 K-tiles into dense stages, and
  Nsight Compute puts its copies at the ideal 16 sectors an instruction at
  1024^3 and 4096^3 (`docs/benchmarks/retune_2026-10-20/fp8_sectors/`); the
  e4m3-input GEMM was written with a chunk swizzle. The int8 GEMM was not
  measured.
- The committed F16 GEMM tiles were re-tuned on 2026-10-20 after the
  swizzle, but not by the autotuner alone: `--autotune-force`, which times
  with B streamed from DRAM, picked tiles that were 0.4-0.9% slower than the
  current ones at 1024^3 and 4096^3 in both regimes, and 4 stages where an
  L2-resident sweep found 3 faster from 4096^3 up. The profile they are
  generated from takes the autotuner's tiles where they won in both, the
  sweep's at 4096^3-16384^3, and 1024^3's as it was
  (`docs/benchmarks/retune_2026-10-20/`). Those runs shared the GPU with
  Discord's GPU process; effects under 1% there are within its noise.
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
- `python3 tools/ptxas_tval/docgate.py` reports 15 figures in the validator
  documentation and the README that disagree with its measurement (checked
  2026-10-17 at `ff3526f`): the loop census's buckets and its
  `loopval`/`nestval` split; the multi-back-edge kernels (33 of 48 published,
  29 measured) and liftgap's levels (115 published; 95 measured on 5e391da,
  87 since the fused SwiGLU kernel's 64x64 tile took two loop levels from
  each of its four kernels, and 84 since the F16 GEMM tiles were re-tuned,
  both 2026-10-20); the
  unroll
  census (24 `MATCHED` and 18 `REFUSED` published, 27 and 15 measured); and the
  `-O1` and `-O3` frontier censuses, stamped on an older tree
  (`python3 frontier.py`, minutes, restamps them). The F16 GEMM K-loop change
  moved the unroll census to 25 and 17: `ptxas` now unrolls the whole K loop
  of `gemm_f16_512` and `gemm_f16_bias_relu_512`, and `unroll.py` refuses a
  PTX loop it cannot pair with a SASS one. The FP8 GEMM's float32
  accumulation (2026-10-18) moved it to 26 `MATCHED`, and its "no loop"
  bucket from the 21 published to 20, the fourteenth figure: `ptxas` no
  longer unrolls the whole K loop of `gemm_fp8_256`. The paged decode
  attention KV loop (2026-10-18) added a fifteenth: the loop census's bucket
  for the SASS branch form `BRA.DIV ~URZ` (1 published) is now measured by no
  kernel, because `paged_decode_attention_128_32_8_16`, refused there before,
  is now refused as "PTX loop body has more than one branch" (6 measured
  before and 7 now, 2 published): the lazy rescale is a branch inside the
  loop. The kernel was refused before and is refused now. The unroll census
  and `gap.py` give the attention kernels the verdicts they gave before; the
  unroll census's reason now names an irreducible SASS nest of 3 levels
  rather than 2. The F16 GEMM's barrier between a K-tile's last two steps
  (2026-10-18) moves no gated figure, and `gap.py` gives every changed kernel
  the opcode gap it gave before, but `ptxas` now issues the copies of
  `gemm_f16_1024`, `gemm_f16_bias_relu_1024` and the four decode kernels as
  `LDGSTS.E.BYPASS.LTC128B.128`, so the validator documentation's opcode
  list for all 23 F16 GEMMs and its `gemm_f16_1024` row, which name
  `LDGSTS.E.BYPASS.128`, hold for 17 of them. The fused RMSNorm's unrolled groups
  (2026-10-19) take `rmsnorm_residual_4096` out of the loop census, and
  docgate reports 18. Three figures newly disagree: "48 kernels / 0
  validated" (47 measured), the loop census's bucket of 1 for "PTX loop
  cross-thread effect: st.shared.v4.f32" (measured by no kernel now), and
  liftgap's "SEQUENTIAL depth 1" (25 published, 24 measured). The unroll
  census's "no loop" figure (21 published) now matches. The figures that
  already disagreed move by one or two: nestval answers 29, the
  multi-back-edge kernels are 28 of 47, liftgap 28 kernels / 95 levels, and
  the unroll census 25 `MATCHED`. **docgate's own control fails too:**
  `perturbed_unroll('rmsnorm_residual_4096')` doubles that kernel's SASS loop
  body and expects `UNROLLED`, and the kernel has no loop left to double, so
  the control reads `None`. It needs another subject in
  `tools/ptxas_tval/docgate.py`; `naive_gemm_f32`, `gemm_f16_4096` and
  `gemm_f16_decode_m4` each read `UNROLLED` under the same perturbation
  (checked 2026-10-19 by calling `perturbed_unroll` on the rebuilt corpus).
- The RoPE kernels' launch guards and 64-bit row index (2026-10-19) move no
  gated figure: `build_corpus.sh`, `regress.sh`, `fpgate.py` and `unroll.py`
  give what they gave on 532338f, and docgate's 48 failures differ only in
  their tree stamps. They do widen the gap. `gap.py`'s first refusal for
  `rope_64/128/256` is now `trap` in PTX and `BPT.TRAP` in SASS, with 6 PTX
  and 15-16 SASS opcodes in its gap (were 4 and 10-11). `depth.py` counts
  25, 25 and 23 unknown ops on each side (were 16, 18 and 16 in PTX, 15, 17
  and 15 in SASS). So the validator documentation's "16-18 unknown PTX ops
  each" for these kernels (*A first-refusal census names one opcode*) no
  longer holds.

## Fixed

| Issue | Fix and regression gate |
| --- | --- |
| `rope_<head_dim>` wrote part of each row at a head_dim not a multiple of 64, and its README comparison with FlashInfer was of half FlashInfer's work | A row's lanes took whole pairs, with only a `debug_assert` on the width, which release builds drop: the parent's `rope_96` wrote 64 of each row's 96 elements and `rope_32` none (`docs/benchmarks/rope_2026-10-19/old_bugs/`). Such a head_dim is refused now. The kernel takes a fourth parameter, `Rows`, and puts 8 rows in a 256-thread CTA (16 at head_dim 64). It traps on a block of another size, on a grid that does not reach `Rows`, and on a negative `Rows`. The three-parameter form is refused, since unmatched it would compile as an empty kernel. The benchmark scripts gave `apply_rope_pos_ids` X as both q and k, where Y rotated X once, and timed Python launches, so the README's 1.85x and 1.92x were of launches and of unequal work. With the same work, in graph replays, that kernel took 1.47-2.49x FlashInfer's time at head_dim 128 inside the L2. `python/y_lang/tests/test_rope.py` (head_dim 64-512, partial CTAs, each trap in a child process, the refusals); each of nine mutants fails it. |
| `rmsnorm_residual_<d>` could not be built from hidden 12288 up, and wrote part of a row at a width not a multiple of 8 | It staged h through `d * 4` bytes of static shared memory, which `ptxas` refuses above 48 KB ("uses too much shared data"), and moved whole 16-byte groups of 8 f16 with no check of the width, so hidden 100 compiled and left the last 4 elements of every row unwritten. h now stays in registers (no shared memory but the 8-to-32-float cross-warp sum), a width that is not a multiple of 8 is refused, and a launch with another block size than the PTX header states traps; the warp count changed with it, from 8 to 16 at 4096 (`docs/benchmarks/rmsnorm_residual_2026-10-19/`). `python/y_lang/tests/test_rmsnorm_residual.py` (hidden 256-65536, the trap in a child process, the refusal); each of nine mutants fails it. |
| Y's FP8 GEMM summed in the tensor core's accumulator | The `@tile` FP8 GEMM chained every `mma.sync.m16n8k32` of its K loop in the mma's own accumulator, which on an RTX 4070 Ti SUPER adds an mma's products at about 11 bits, truncating: on positive inputs at K = 4096 the median relative error was 6.6% (`torch._scaled_mm`: 5.7e-4). Every FP8 test and benchmark used small integers, whose partial sums are exact, so none could see it. Each mma now starts from zero and is added into the accumulator with `add.rn.f32`: 6.5e-5, at no measured cost (`docs/benchmarks/fp8_e4m3_2026-10-18/`). The new e4m3-input GEMM does the same. `test_partial_sums_are_added_in_float32` in `test_fp8_gemm.py` and `test_fp8_e4m3.py`; each fails on the chained form (7.1e-2). |
| The CLI reported an F16 tile for every `@tile` kernel | It asked `Autotuner::autotune` for an F16 tile for the int8 and FP8 GEMMs too, which choose their own tiles, printed it as theirs, and under `--autotune` measured F16 candidates for them. It reports F16 kernels only now. `the_cli_reports_an_autotuned_tile_for_f16_only` (`tests/ptx_e4m3.rs`). |
| Y's F16 tile GEMMs could not run from Python; a missing kernel name ran `main` | `TorchKernel.launch` passed no dynamic shared memory, so every kernel staging through `.extern .shared` died with an illegal address; it now passes what the kernel's PTX states and raises the function's limit above 48 KB. The CPU interpreter still gave that array 0 bytes, under `Y_INTERPRETER=1` and in `debug_interpret`, so such a kernel stopped at its first store there, reported out of bounds; `CPUInterpreter.run` now takes the size from the PTX as well (`DynamicSharedMemory` in `test_interpreter.py`). A kernel name the module lacked fell back to an entry named `main`; it is refused. The launch path also went from 6.6 to 2.9 us of CPU time per call (Triton 5.7, PyTorch 2.0; `docs/benchmarks/launch_2026-10-12/`). `python/y_lang/tests/test_launch.py` (F16 GEMMs below and above 48 KB exact against float64, the current and an explicit stream on a non-blocking stream, the interpreter switch); each of seven mutants fails it. |
| Y's FP8 GEMM took 2.1-2.6x as long as Triton's | A fused Triton kernel doing the same work (f32 in, e4m3 tensor cores, f32 out) was 2.1-2.6x faster on 2026-10-11. Four changes: B staged transposed with a register prefetch and grouped CTAs (`docs/benchmarks/fp8_gemm_2026-10-12/`), `ldmatrix` fragments and a tile per shape (`fp8_gemm_tiles_2026-10-12/`), raw tiles copied with `cp.async` (`fp8_gemm_async_2026-10-12/`), and C written through shared memory with 16-byte stores, where a quarter of the kernel's time had gone on 4-byte stores (`fp8_gemm_epilogue_2026-10-13/`). Y takes 0.33-0.96x Triton's time at every size measured from 256 to 8192, 0.96x at 2048 the closest. `python/y_lang/tests/test_fp8_gemm.py` (every tile the rule picks, exact, ragged edges, scales whose product is not 1, compute-sanitizer memcheck). |
| A shuffle or a barrier only some threads reached was emitted | `warp_reduce_sum`, `warp_reduce_max` and `shfl_sync_bfly` emit `shfl.sync` over all 32 lanes, and `barrier_sync` emits `bar.sync` for the whole block, wherever they were written: `if t < 20 { warp_reduce_sum(x) }` compiled, and the GPU leaves it undefined (the shuffle reads lanes that are not there; a barrier may hang). The PTX backend now works out how far each variable is the same across threads (`src/uniformity.rs`: block-wide, per warp, or per thread; every memory read per thread) and refuses a collective instruction - any `shfl`/`vote`/`mma` and other `.sync` or `.aligned` form, or `bar`/`barrier` - under a branch, loop or earlier `return` the threads can disagree on, naming the line, the branch and the value that decides it. It refuses some correct kernels: a branch on a thread index divided by 32 is per warp only for some block shapes, and is refused. A `chisel` block's PTX is not checked, and one that branches counts as a `return` some threads take. `tests/ptx_divergent_collectives.rs` (19 refused forms; the rewrites run on the device and are exact), `uniformity`'s unit tests, `test_the_compiler_refuses_the_same_shuffle_written_with_a_builtin`. |
| The CPU interpreter interpreted nothing | `CPUInterpreter.run` printed its arguments and returned `SUCCESS`; with `Y_INTERPRETER=1` every `TorchKernel.launch` printed "Executed safely under software bounds validation" and wrote no output; `debug_interpret("dummy", ...)` succeeded with no kernel; the C entry point `y_interpret_kernel` only type-checked. It now runs the kernel's PTX on the CPU in SIMT lockstep and stops at out-of-bounds and misaligned accesses, `trap`, divergent barriers and shuffles, conflicting stores and unmodelled instructions, naming the thread and PTX line. Each of the 518 instruction forms it models agrees with the GPU bit for bit over its operands' edge values and random bits, NaN payloads and signed zeros included, except that `.approx` functions agree within the GPU's documented error. The first version got 142 of those forms wrong, 137 of them without an error: canonical and propagated NaNs, `.ftz` tininess, signed zeros in `min`/`max`/`.sat`, NaN-to-integer conversions, `cvt.rni.f32.f32`, `sub.sat`, signed carries, `div.approx` and `sin`/`cos.approx` argument reduction. It also read a `cp.async` source size held in a register as the full copy, which raised spurious out-of-bounds errors and copied data the GPU zero-fills, and zero-extended `ld.global.s32` into a 64-bit register. `python/y_lang/tests/test_interpreter.py` (22 tests) and `test_interpreter_ops.py`; each of 41 mutants of the interpreter, and one returning the debug-mode launch to a no-op, fails a test. |
| `benchmark_3d_block_ptr.py` compared unequal work | Y's kernel wrote 32,768 of the 8,388,608 elements it named and Triton's 524,288; neither output was checked, the script printed "FASTER" whatever the ratio, and it deleted `~/.ysu/cache` on import. Both kernels now cover the volume and must equal `2x + 1` exactly before they are timed; a busy GPU is reported. |
| The second-copy gate failed in a checkout holding evidence snapshots | `no_gated_document_has_a_second_copy` flagged Codex's frozen "before" copies of `README.md`. It now skips what git ignores (`build_artifacts/`, like `target/`) and a copy whose current SHA-256 an ancestor's `evidence-manifest.json` records; an edited, unrecorded or untracked copy still fails it. `only_sealed_or_ignored_copies_are_exempt`. |
| The parity tests never ran | `python/y_lang/tests/test_triton_parity_features.py` defined bare functions, which `unittest` discovery does not collect; a `TestCase` runs them now (the suite went from 26 to 30 tests, and a broken `block_sort` fails it). |
| `rmsnorm_v4` assumed a 1024-wide row, `eps = 1e-5` and 256 threads | The mean was `sum * 0.0009765625` and the second reduction read eight warp partials, whatever the kernel declared: at `eps = 0.5` the output was 2.7% off, a 512-wide row was scaled by `sqrt(2)`. It takes `(x, w, out, hidden_dim, eps)` now, reduces over every warp, and traps on a launch that is not one row per block in whole warps. `tests/ptx_rmsnorm_v4.rs` (five shapes against float64 on the device, three trapping launches, five mutants). |
| `rmsnorm_v4` could race inside a loop | Thread 0 handed the scale to the other warps through the shared word warp 0's partial for the next row goes to, so in a `for` or `while` a warp a row ahead could overwrite it before the others read it (not observed: it needs one warp a whole row ahead of another). Each warp now reduces the warp partials itself, bit for bit what thread 0 computed, with one barrier, and a second one inside a loop. `tests/ptx_rmsnorm_v4.rs` (rows through a `for` and a `while` on the device, the barrier counts); each of five mutants fails it. |
| The head-to-head benchmark measured something else in every Y column | Y's kernels were CUDA-graph replays and the others plain launches; the "Block Scan" Y column was `torch.cumsum`; the "FP8 GEMM" Y column was `y_lang.ops.fp8_gemm`, which on this machine was always `torch.mm` on FP16 copies (its CuPy path failed and the failure was swallowed), against FP16 matmuls; the elementwise launches wrote 352 floats past their buffers. The manual published these as 1.03x-6.75x speedups. The script now checks every column against a reference (exactly for addition and the FP8 GEMMs), times all three both ways, and records the toolchain and the GPU's state; the manual's tables are withdrawn (§33.8, §34.6). |
| `y_lang.ops` substituted or guessed | `fp8_gemm` raises when its CuPy path cannot run instead of returning `torch.mm`'s product; `sparse_24_gemm` (which ignored its 2:4 metadata) is refused; the docstrings and manual §34.2/§34.5 say what each helper runs. `python/y_lang/tests/test_ops_fail_closed.py`. |
| Exact attention's `KFix` was checked only by a host helper (archive 1639 (c)) | The accumulating entries took any `u32` temperature: `KFix = 0` launched and returned a uniform softmax, and `KFix >= 2^31`, which the signed `mul.wide.s32` reads negative, returned one where most keys weigh zero. `exact_attention::temperature_fixed_point` refused both, for callers that used it. `attn_accum` and `attn_accum_naive` now check the parameter as the multiply reads it and trap outside `1..2^31`; the certificate's item is checked rather than trusted. `tests/attention_kfix_is_checked_at_launch.rs` (both entries at `0`, `1`, `2^19 + 1`, `2^31 - 1`, `2^31`, `2^32 - 1` on the device, in-range weights against the host exp). |
| `exact_pv`'s licences on `T`, `D`, `Q` and the extents (archive 1692 (a)) | Fixed in `3e67cd9`; listed open here by mistake on 2026-10-11. `verified_exact_pv::ExactPvShape` refuses non-positive dimensions, element counts past `i32::MAX` and a contraction past the i64 accumulator licence; `check_launch_buffers` refuses null, misaligned, short and overlapping buffers; `proofs/ExactPvLaunchContract.v` derives the dot-product theorem's premises from the checked domain. Launching the PTX through `load_ptx` claims none of this. `tests/exact_pv_launch_contract.rs`, the driver-spy tests in `src/cuda_runtime.rs`. |
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
