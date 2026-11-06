# The staged int8 GEMM against cuBLASLt, 2026-11-06

Y's int8 tensor-core GEMM (`@tile(M, N, K)` over `GlobalMemory<I8>`, C int32)
stages its K tiles in shared memory since this change
(`src/ptx_emitter/int8_staged_gemm.rs`). This is the record of what it measures
against cuBLASLt and against the kernel it replaced.

## Setup

- RTX 4070 Ti SUPER (66 SMs, 48 MB L2, 285 W limit), driver 615.78.08 (the
  PTX is JIT-compiled by the driver through CuPy's `RawModule`), CUDA runtime
  12.9, torch 2.13.0+cu130, CuPy 14.1.1, Linux 7.3.0-5-cachyos, rustc 1.97.1.
- Y: the working tree of this change, on top of 8e3e2ce, PTX emitted with the
  machine's `.ysu_hw_profile` (the int8 tile does not read it). "old" is the
  kernel this replaced, emitted by a compiler built from 8e3e2ce (`YBIN_OLD`)
  and launched as its header said: 32-thread blocks, one per warp tile, `C`
  zeroed.
- cuBLASLt is `torch._int_mm(A, B.t())`, B passed as the transposed view of
  the `[N][K]` buffer, which it takes without a copy.
- The GPU was otherwise idle except for the desktop compositor. The clock,
  power and temperature at the end of each run are on the log's last line.

## Method (`int8_bench.py`, `run.sh`)

**Correctness before timing.** For every kernel at every shape:
- the output is compared with `torch._int_mm`, and with a float64 product of
  the same int8 values, which is exact here (`|sum| <= K * 2^14 < 2^53`);
- `C` is filled with 0xA5 before the launch, so a store that is missing shows;
- the staged kernel is also run with K split over `gridDim.z = 3` into a
  zeroed `C`, and on a grid of 3 CTAs.

Every row in the logs ends `bits ok`.

**Timing.** Per-call time is the minimum over 15 rounds, and the order of the
kernels is shuffled every round. Each round launches each kernel 3 to 200
times back to back.

There are two regimes:
- `run1.log`, `run2.log`: every call reads the same A and B, which stay in L2
  where they fit;
- `cold1.log`, `cold2.log` (`--cold`, the 24 LLM shapes): every call reads
  its own copy of A and B, rotating over 128 MB of copies, so the operands
  come from DRAM as a model's weights do.

The two runs of each are separate processes.

`run0_before_the_fit_condition.log` is the first run, made before the tile
rule had its "fits inside the matrix" condition (below). Its forced-tile and
old-kernel columns are valid. Its rule column is the old rule.

## Results, operands in L2 (two runs; the fastest forced tile in bold)

"rule" is the tile the compiler picks. The next three columns force each tile
with `Y_INT8_TILE`. Each cell is the range over the two runs of cuBLASLt's
time divided by Y's: above 1 means Y is faster. The cuBLASLt time is the
lower of the two runs, and the TOPS the higher.

| M × N × K | cuBLASLt µs | TOPS | rule's tile | rule | 128×128 | 64×128 | 64×64 | old kernel |
|---|---|---|---|---|---|---|---|---|
| 256 × 256 × 256 | 3.2 | 10 | 64x64 | 1.43–1.56 | 0.78–0.85 | 1.12–1.21 | **1.43–1.55** | 1.04–1.12 (16x8) |
| 512 × 512 × 512 | 4.2 | 64 | 64x64 | 1.23 | 0.74 | 1.10 | **1.23** | 0.43 (64x64) |
| 1024 × 1024 × 1024 | 14.3 | 150 | 128x128 | 1.35 | **1.35** | 1.24 | 1.00 | 0.50–0.51 (64x64) |
| 2048 × 2048 × 2048 | 61.8 | 278 | 128x128 | 1.02 | **1.02** | 0.81–0.82 | 0.66 | 0.40–0.42 (64x64) |
| 3072 × 3072 × 3072 | 213.2 | 272 | 128x128 | 1.09–1.10 | **1.10–1.11** | 0.87 | 0.69 | 0.39–0.40 (64x64) |
| 4096 × 4096 × 4096 | 445.1 | 309 | 128x128 | 1.03 | **1.02–1.03** | 0.81 | 0.62 | 0.42 (64x64) |
| 8192 × 8192 × 8192 | 3325.6 | 331 | 128x128 | 0.97–1.00 | **1.00** | 0.74–0.77 | 0.59–0.60 | 0.23 (64x64) |
| 4000 × 4000 × 4000 | 444.4 | 288 | 128x128 | 0.94 | **0.94** | 0.69 | 0.55–0.56 | 0.20–0.21 (32x32) |
| 1008 × 1008 × 1024 | 14.1 | 147 | 128x128 | 1.34 | **1.33–1.34** | 1.23–1.24 | 1.00 | 0.21 (16x16) |
| 4096 × 4096 × 4000 | 446.6 | 300 | 128x128 | 0.93–0.94 | **0.93–0.94** | 0.67 | 0.53–0.54 | 0.42 (64x64) |
| 4096 × 4096 × 4064 | 452.2 | 302 | 128x128 | 0.94–0.95 | **0.95** | 0.70 | 0.54–0.55 | 0.43–0.44 (64x64) |
| 256 × 256 × 65536 | 197.6 | 44 | 64x64 | 1.77 | 0.52 | 1.02 | **1.77** | 0.54–0.55 (16x8) |
| 1024 × 1024 × 16384 | 143.0 | 240 | 128x128 | 1.38–1.39 | **1.38** | 1.05 | 0.73 | 0.49–0.50 (64x64) |
| 32 × 4096 × 4096 | 14.7 | 73 | 64x64 | 1.35–1.39 | 0.57 | 1.06 | **1.35–1.39** | 0.40 (32x64) |
| 32 × 11008 × 4096 | 29.9 | 96 | 64x64 | 1.11–1.15 | 0.59–0.60 | **1.12–1.14** | 1.12–1.14 | 0.66–0.69 (32x64) |
| 32 × 4096 × 11008 | 35.4 | 81 | 64x64 | 1.24 | 0.54 | 1.04 | **1.22–1.24** | 0.37 (32x64) |
| 32 × 14336 × 4096 | 100.5 | 37 | 64x64 | 1.00–1.01 | 0.99–1.00 | **1.01–1.02** | 1.00–1.01 | 0.36 (32x64) |
| 32 × 4096 × 14336 | 104.0 | 36 | 64x64 | 1.01 | 0.90 | 0.93 | **1.01** | 0.23–0.24 (32x64) |
| 32 × 1024 × 4096 | 14.5 | 18 | 64x64 | 1.68 | 0.56–0.57 | 1.06 | **1.68** | 0.71–0.72 (16x8) |
| 128 × 4096 × 4096 | 20.3 | 212 | 64x128 | 1.00–1.01 | 0.76 | **1.00–1.01** | 0.82–0.83 | 0.35 (64x64) |
| 128 × 11008 × 4096 | 55.4 | 208 | 128x128 | 1.01–1.02 | **1.01** | 0.86–0.88 | 0.85–0.86 | 0.49–0.52 (64x64) |
| 128 × 4096 × 11008 | 52.1 | 222 | 64x128 | 0.96–1.00 | 0.77 | **0.97–1.00** | 0.77 | 0.25–0.27 (64x64) |
| 128 × 14336 × 4096 | 112.7 | 133 | 128x128 | 0.99–1.00 | 0.99–1.00 | **1.02** | 0.98–0.99 | 0.41–0.42 (64x64) |
| 128 × 4096 × 14336 | 110.7 | 136 | 64x128 | 0.92–0.93 | 0.90–0.91 | 0.92–0.93 | **0.98–0.99** | 0.23 (64x64) |
| 128 × 1024 × 4096 | 14.8 | 72 | 64x64 | 1.65 | 0.56 | 1.04 | **1.65** | 0.35 (16x8) |
| 512 × 4096 × 4096 | 76.5 | 225 | 128x128 | 1.36 | **1.36** | 1.06 | 0.83–0.84 | 0.53–0.54 (64x64) |
| 512 × 11008 × 4096 | 169.9 | 272 | 128x128 | 0.97 | **0.97** | 0.86 | 0.68 | 0.32–0.34 (64x64) |
| 512 × 4096 × 11008 | 201.7 | 229 | 128x128 | 1.33–1.34 | **1.32–1.35** | 1.02–1.03 | 0.80–0.81 | 0.47–0.48 (64x64) |
| 512 × 14336 × 4096 | 230.7 | 261 | 128x128 | 1.11–1.13 | **1.12–1.13** | 0.90–0.91 | 0.71–0.72 | 0.20–0.21 (64x64) |
| 512 × 4096 × 14336 | 265.9 | 226 | 128x128 | 1.38–1.39 | **1.38–1.39** | 1.01–1.04 | 0.79–0.80 | 0.48–0.49 (64x64) |
| 512 × 1024 × 4096 | 20.1 | 214 | 64x128 | 1.01–1.03 | 0.74–0.76 | **1.00–1.03** | 0.83–0.86 | 0.35–0.37 (64x64) |
| 2048 × 4096 × 4096 | 227.7 | 302 | 128x128 | 1.02 | **1.02** | 0.80 | 0.63 | 0.41–0.42 (64x64) |
| 2048 × 11008 × 4096 | 607.2 | 304 | 128x128 | 1.04 | **1.04** | 0.81 | 0.63 | 0.32–0.33 (64x64) |
| 2048 × 4096 × 11008 | 568.6 | 325 | 128x128 | 0.99–1.00 | **1.00** | 0.73–0.74 | 0.58–0.59 | 0.34 (64x64) |
| 2048 × 14336 × 4096 | 773.1 | 311 | 128x128 | 1.01–1.02 | **1.02** | 0.79 | 0.61–0.62 | 0.17–0.18 (64x64) |
| 2048 × 4096 × 14336 | 730.7 | 329 | 128x128 | 0.99–1.00 | **0.99–1.00** | 0.73 | 0.58–0.59 | 0.34–0.35 (64x64) |
| 2048 × 1024 × 4096 | 76.1 | 226 | 128x128 | 1.35–1.36 | **1.35–1.36** | 1.05–1.07 | 0.83 | 0.54–0.55 (64x64) |

## Results, operands from DRAM (two runs)

| M × N × K | cuBLASLt µs | TOPS | rule's tile | rule | 128×128 | 64×128 | 64×64 |
|---|---|---|---|---|---|---|---|
| 32 × 4096 × 4096 | 29.4 | 36 | 64x64 | 1.04 | 0.98 | 1.02 | **1.04** |
| 32 × 11008 × 4096 | 75.8 | 38 | 64x64 | 0.99–1.00 | 0.97 | **1.00** | 0.99 |
| 32 × 4096 × 11008 | 78.0 | 37 | 64x64 | 1.01 | 0.87–0.88 | 0.90–0.91 | **1.01** |
| 32 × 14336 × 4096 | 101.5 | 37 | 64x64 | 0.99–1.00 | 0.99–1.00 | **1.01–1.02** | 1.00 |
| 32 × 4096 × 14336 | 104.4 | 36 | 64x64 | 1.01 | 0.89 | 0.91–0.92 | **1.01** |
| 32 × 1024 × 4096 | 15.1 | 18 | 64x64 | 1.33 | 0.57–0.58 | 0.77 | **1.33** |
| 128 × 4096 × 4096 | 30.6 | 140 | 64x128 | 1.01 | 0.98 | 1.01 | **1.02** |
| 128 × 11008 × 4096 | 86.6 | 133 | 128x128 | 1.03 | **1.03** | 0.96–0.97 | 1.02–1.03 |
| 128 × 4096 × 11008 | 82.7 | 140 | 64x128 | 0.91 | 0.88–0.89 | 0.91–0.92 | **0.99** |
| 128 × 14336 × 4096 | 114.7 | 131 | 128x128 | 1.01–1.02 | 1.01–1.02 | **1.03** | 0.99 |
| 128 × 4096 × 14336 | 109.5 | 137 | 64x128 | 0.93 | 0.89–0.90 | 0.94 | **0.99** |
| 128 × 1024 × 4096 | 15.3 | 70 | 64x64 | 1.27–1.28 | 0.57 | 0.73–0.74 | **1.27–1.28** |
| 512 × 4096 × 4096 | 77.3 | 222 | 128x128 | 1.34 | **1.33–1.34** | 1.04–1.05 | 0.82 |
| 512 × 11008 × 4096 | 170.2 | 271 | 128x128 | 0.97–0.98 | **0.97–0.98** | 0.86 | 0.68 |
| 512 × 4096 × 11008 | 203.0 | 227 | 128x128 | 1.34–1.35 | **1.34–1.35** | 1.03–1.04 | 0.80 |
| 512 × 14336 × 4096 | 230.3 | 261 | 128x128 | 1.12–1.13 | **1.12** | 0.90 | 0.71 |
| 512 × 4096 × 14336 | 263.1 | 229 | 128x128 | 1.37–1.38 | **1.37–1.38** | 1.03–1.04 | 0.79–0.80 |
| 512 × 1024 × 4096 | 20.5 | 209 | 64x128 | 0.93 | 0.74–0.75 | **0.93** | 0.83 |
| 2048 × 4096 × 4096 | 227.0 | 303 | 128x128 | 1.01–1.02 | **1.01** | 0.79 | 0.62–0.63 |
| 2048 × 11008 × 4096 | 608.9 | 303 | 128x128 | 1.05 | **1.05** | 0.81 | 0.63 |
| 2048 × 4096 × 11008 | 566.9 | 326 | 128x128 | 0.99–1.00 | **0.99–1.00** | 0.73–0.74 | 0.58–0.59 |
| 2048 × 14336 × 4096 | 771.1 | 312 | 128x128 | 1.01–1.02 | **1.01** | 0.79 | 0.60–0.61 |
| 2048 × 4096 × 14336 | 730.1 | 329 | 128x128 | 0.99–1.05 | **1.00–1.04** | 0.73–0.77 | 0.58–0.61 |
| 2048 × 1024 × 4096 | 76.7 | 224 | 128x128 | 1.33–1.34 | **1.33–1.34** | 1.04–1.05 | 0.81–0.82 |

## What the record shows

- **Exact everywhere.** Every kernel at every shape, in all five runs, gave
  bits identical to cuBLASLt and to the float64 product, unsplit, split over
  z = 3, and on 3 CTAs.
- **Square sizes:** 1.02–1.10x at 2048³–4096³, 0.97–1.00x at 8192³, and
  1.2–1.8x at the smaller and K-heavy shapes. Against the kernel it replaced
  that is 2.2–6.4x from 512³ up, and 1.4x at 256³.
- **LLM shapes** (M = 32–2048 against Llama's 4096 / 11008 / 14336 and
  1024 × 4096): over both regimes, 83 of 96 results are 0.98x or better.
  The range is 0.91–1.68x.
- **Losses:**
  - K % 64 == 32: 0.93–0.95x. Every second row starts 32 bytes into a
    64-byte segment. K % 32 == 0 is required, so this is the only misaligned
    case.
  - 128 × 4096 × 11008 and × 14336: 0.91–0.94x from DRAM. The 64×64 tile
    gives 0.99x there, but from L2 it is 17–24% slower than the rule's
    64×128 at K = 4096 and 11008. The rule does not see where B lives, so it
    was left as it is.
  - 512 × 11008 × 4096: 0.97–0.98x.
  - 512 × 1024 × 4096 from DRAM: 0.93x (1.00–1.03x from L2).
  - 8192³: 0.97–1.00x. Run 2 measured the same kernel 0.969x in its rule
    column and 1.001x in its 128×128 column; that 3% spread is the noise of
    this long kernel at the power limit.
- **The tile rule.** It takes the largest of 128×128 and 64×128 that fits
  inside the matrix and leaves at least 64 output tiles, else 64×64. In
  `run0`, without the fit condition, 32 × 11008 × 4096 took 128×128 (86 tiles,
  three quarters of each past M) and measured 0.60x. With the condition it
  takes 64×64, 1.11–1.15x. Over both L2 runs the rule's tile is within 3% of
  the fastest forced tile everywhere except 128 × 4096 × 14336, where it is
  6% slower (0.92–0.93x against 0.98–0.99x).

## How it got there (development runs, not in these logs)

Single runs at 4096³ during development, with the harness above but a fixed
order, so they are indicative only:

| step | Y / cuBLASLt |
|---|---|
| the replaced kernel (no shared memory) | 0.40x |
| staged, every output reduced with `red.global.add.s32` | 0.74x |
| an unsplit launch stores, a split one reduces | 0.97x |
| the K loop in CUTLASS's multistage order | 1.03x |

## Checks beyond the benchmark

- `tests/int8_gemm_launch_contract.rs` covers the following:
  - the declared block works, and blocks of 32, 64 and 256 trap;
  - ten grids give the same matrix;
  - a 2^32-CTA grid traps;
  - every tile, the forced odd geometries included, matches a CPU reference
    on a ragged shape;
  - the split-K epilogue gives the same answer at every z;
  - the Coq files state the schedule the emitter makes.
- `compute-sanitizer` memcheck and racecheck are clean on ten runs (the same
  file). The buffers are exactly the matrices' size, so any read or write
  past an edge is caught.
- Mutants (`int8_mutate.py`, `mutants.log`): 17 of 17 killed by the five
  int8 test files. They cover:
  - the copy swizzle and K offset;
  - the `cp.async` wait depth;
  - each barrier, and a barrier one substep late;
  - the store/reduce choice, both ways;
  - the tile stride, the stripe count, and the last launch group's size;
  - B's fragment halves;
  - both traps;
  - the copy and store edge predicates.

  M5 (no barrier between output tiles) is killed only by the racecheck test.
  The log's "by" lists name only the first failing test file's failures,
  because `cargo test` stops at the first test binary that fails. The tile
  id's wrap check is not mutated: only a grid of nearly 2^32 CTAs reaches
  it, more than a test can launch. `proofs/Int8GemmSchedule.v` proves that
  it detects the wrap.

## Reproduce

```sh
cargo build --release
# a compiler from 8e3e2ce for the old kernel, then:
YBIN_OLD=/path/to/old/Y Y_PROFILE=$PWD/.ysu_hw_profile WORKDIR=/tmp/i8 \
  bash docs/benchmarks/int8_gemm_2026-11-06/run.sh run1.log
Y_PROFILE=$PWD/.ysu_hw_profile WORKDIR=/tmp/i8c \
  bash docs/benchmarks/int8_gemm_2026-11-06/run.sh cold1.log --cold
```
