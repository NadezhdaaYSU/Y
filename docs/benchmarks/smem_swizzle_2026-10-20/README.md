# Swizzled shared-memory tiles for the F16 GEMM, 2026-10-20

RTX 4070 Ti SUPER, driver 615.71.09, `ptxas` from CUDA 13.4, PyTorch
2.13.0+cu130, Triton 3.7.1. Nsight Compute's raw CSV pages print counts with
the locale's `.` as the thousands separator (`2.263.040` is 2,263,040).

## Where the 512^3 gap was

After `../const_div_2026-10-19/` the F16 GEMM at 512^3 still took 1.12-1.14x
cuBLAS's time, and no tile closed it (`../f16_square_2026-10-18/512/`).

**It grows with K.** `ksweep/ksweep.py` times M = N = 512 over K with the
tuner's 512 tile pinned (`Y_CTA_OVERRIDE=64,64,32,4,2,4`; `ksweep/v/`, whose
K = 512 kernel is 81c055f's `tests/gemm_f16_512.ptx` but for its name): each
column a CUDA graph of 50 launches, so neither side's host launch cost is
timed, 21 interleaved rounds after 3, minimum, every output checked exact
first. Two runs, `ks_1` and `ks_2`, started at 3.5% and 16.0% mean
utilisation with Steam running (`queue.txt`): contended.

| K | Y (us) | cuBLAS (us) | cuBLAS / Y | cuBLAS kernel |
| ---: | ---: | ---: | ---: | :--- |
| 256 | 3.88 / 4.20 | 4.10 / 4.25 | 1.057 / 1.012 | 64x64, K step 64, 5 stages |
| 512 | 6.72 / 6.87 | 5.98 / 5.99 | 0.890 / 0.872 | same |
| 1024 | 11.60 / 11.59 | 9.45 / 9.32 | 0.815 / 0.804 | same |
| 2048 | 19.89 / 19.62 | 18.56 / 18.20 | 0.933 / 0.928 | 128x64, K step 64, 4 stages |
| 4096 | 37.05 / 37.04 | 29.86 / 30.83 | 0.806 / 0.832 | same |
| 8192 | 72.86 / 72.90 | 55.38 / 55.19 | 0.760 / 0.757 | CUTLASS 128x128 split-K + reduction |

Y is the faster at K = 256 and falls behind as K grows against the same
cuBLAS kernel: the gap is in the cost of a K-tile, not in the prologue,
epilogue or launch. ptxas unrolls Y's K loop fully at K = 256 and 512 and
keeps it rolled, one K-tile an iteration, from 1024. (`ksweep/v/` keeps the
sources and the K = 256, 2048 and 8192 kernels; the 512, 1024 and 4096 ones
are `ksweep/t/64_64_32_4_2_4/`'s.)

**It is in every tile.** `ksweep/ktiles.py` (the same harness, several
columns) at K = 512, 1024 and 4096 for six tiles around Y's, cuBLAS's own
shape (64x64x64, 2x2 warps, 5 stages) among them (`tiles_1`, `tiles_2`, at
15.3% and 8.5%: contended). At K = 1024 all six took 1.21-1.30x cuBLAS's time
(10.66-11.43 us against 8.79 and 8.83).

**It is L2 traffic.** `ksweep/prof/ncu.sh` profiles the tuner's tile and
cuBLAS's shape as Y emits them, and cuBLAS, at 512x512x1024 with no cache
flush (`metrics.csv`, source counters `sass_1/3/5.csv`, at a 6-60%
utilisation, `env.txt`). The loads have no bank conflicts, but both Y kernels
read 786,432 L2 sectors where cuBLAS reads 524,288, the minimum for 64 CTAs of
64x64 (each reads 64 x 1024 of A and 1024 x 64 of B once). The source counters
put it on every `cp.async`: 24 sectors an instruction against an ideal 16,
A's and B's alike; cuBLAS's take 16. An L2 throughput profile
(`sol1024.ncu-repz`, `sol_details.csv`; utilisation not logged) puts Y at
64-73% of the L2's peak over the kernel and cuBLAS at 54%.

**It is the padding.** `sectors/probe.py` copies a 64 x 1024 f16 matrix
through shared memory in K-tiles with Y's thread-to-chunk mapping (A's: 8 rows
of 64 bytes a warp; B's: 4 rows of 128), varying one thing at a time; every
copy is read back and checked, and a variant reading the wrong chunk fails the
check. One CTA, so the counts do not depend on contention (`metrics.csv`):

| destination rows | A-like L2 sectors | B-like L2 sectors |
| :--- | ---: | ---: |
| Y's: padded to 80 / 144 bytes, `cp.async` with Y's src-size operand | 6,144 | 3,072 |
| padded, without the src-size operand | 6,144 | 3,072 |
| dense, 64 / 128 bytes, with and without it | 4,096 | 2,048 |
| dense with an XOR swizzle | 4,096 | 2,048 |

The emitter padded every A and B row by 8 elements so that `ldmatrix` has no
bank conflicts; nobody had measured what it costs the copies. Without it
(`Y_SMEM_PAD=0`, `pad_1`, `pad_2`, contended: 29.5% and 17.5%) the copies read
the minimum but `ldmatrix` conflicts: 2,263,040 bank conflicts and 2,670,592
shared load wavefronts against 394,816 (`prof/pad0_metrics.csv`), and the
kernels were 1.3-1.9x slower. A layout needs both.

## The change

`ptx_emitter::SmemTile` describes a tile's rows. `SmemTile::swizzled(cols)`
packs them at `cols` and stores chunk `c` of row `r` as chunk `c ^ ((r >>
shift) & mask)`: rows of 8 or more 16-byte chunks XOR the row's low 3 bits,
rows of 4 (2), two (four) rows to a 128-byte line, XOR bits 1-2 (bit 2). Any 8
rows from a multiple of 8 then put a chunk in 8 different 16-byte slots, and a
warp's 32 copies fill whole 128-byte lines. `emit_tensor_core_gemm_kernel`
(plain, bias + ReLU, decode, uneven-grid and rectangular F16 GEMMs) uses it for
A and B wherever both rows are a power of two of at least two chunks, on
every path that writes or reads them: the synchronous copy, the per-chunk and
hoisted `cp.async` copies, and the `ldmatrix` addresses of both compute
blocks. A lane's `ldmatrix` rows are `lane & 15` plus multiples of 16 and its
chunk an even number plus `lane >> 4`, so the XOR is one per-lane term per
tile, computed once. A hoisted copy needs a thread's chunks a whole number of
swizzle periods (8 rows) apart, or it falls back to the per-chunk copy.
`Y_SMEM_SWIZZLE=0` selects the padded layout and `Y_SMEM_PAD` still sets its
pad. The SwiGLU kernel and the FP8 and int8 GEMMs have their own copy code and
keep their layouts; whether theirs costs the same was not measured.

With `Y_SMEM_SWIZZLE=0` the compiler regenerates all 27 committed GEMM kernels
byte for byte (`check/regen.sh`, with the profile they were generated under:
the `xbar_ptx/final` profile of `../f16_xbar_2026-10-18/`). With the swizzle,
19 change and 8 (SwiGLU, FP8) do not; shared memory a CTA falls by the padding
(`gemm_f16_512`: 38,912 to 32,768 bytes). `check/check_all.py` runs each of the
19 against its committed kernel: bit for bit on random normal inputs, and both
exact on small integers against float64 (against cuBLAS at 16384^3, exact
there because every partial sum is an integer below 2^24); all pass
(`check_on.log`).

## Measured

**Every shape.** `ab/bench_ab.py` (the harness of `../f16_square_2026-10-18/`:
back-to-back launches timed with events, 15 interleaved rounds, minimum, every
column exact first): `base` is 81c055f's `tests/*.ptx`, `swz` this commit's
(decode kernels renamed to the harness's names, as in
`../decode_dram_2026-10-18/`). Two runs on a quiet GPU, 4.2% and 0.6% mean
utilisation with no game running (`ab/queue.txt`).

| shape | base / swz | cuBLAS / swz | cuBLAS / base | Triton / swz | spread (median / min) swz, base |
| :--- | ---: | ---: | ---: | ---: | :--- |
| 256^3 | 1.062 / 1.060 | 1.294 / 1.484 | 1.218 / 1.400 | 3.080 / 3.351 | 1.09-1.10 |
| 512^3 | 1.190 / 1.205 | 1.076 / 1.084 | 0.904 / 0.899 | 1.448 / 1.777 | 1.01-1.02 |
| 1024^3 | 1.063 / 1.065 | 1.054 / 1.050 | 0.991 / 0.986 | 1.101 / 1.104 | 1.00-1.01 |
| 2048^3 | 1.006 / 1.013 | 0.995 / 1.002 | 0.989 / 0.990 | 1.008 / 1.017 | 1.00-1.08 |
| 4096^3 | 1.000 / 1.001 | 0.980 / 0.981 | 0.980 / 0.980 | 1.013 / 1.010 | 1.01 |
| 8192^3 | 0.998 / 0.998 | 1.010 / 1.010 | 1.012 / 1.012 | 1.114 / 1.116 | 1.005-1.009 |
| M = 1, N = K = 4096 | 1.092 / 1.089 | 1.482 / 1.481 | 1.356 / 1.360 | - | 1.00 |
| M = 4 | 1.199 / 1.177 | 2.974 / 2.942 | 2.480 / 2.499 | - | 1.00-1.02 |
| M = 8 | 1.217 / 1.221 | 2.668 / 2.671 | 2.192 / 2.187 | - | 1.00-1.01 |
| M = 32 | 1.447 / 1.460 | 1.144 / 1.133 | 0.791 / 0.776 | - | 1.00-1.04 |

Above 1, the second column is faster. The decode rows are L2-resident (the
same inputs every call); from DRAM they are bandwidth-bound
(`../decode_dram_2026-10-18/`) and were not re-measured. At 8192^3 the
swizzled kernel was 0.2% slower in both runs, inside that row's 0.5-0.9%
spread; at 4096^3 the two are level. Both sizes use 128x128 tiles, which read half
the L2 bytes an MMA that 64x64 tiles do.

**512 x 512 x K.** `ksweep/` `swz_1`, `swz_2` (contended: 35.8% and 16.2%),
minimum us:

| K | tuned tile, padded | swizzled | cuBLAS's shape, padded | swizzled | cuBLAS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 6.68 / 6.35 | 5.33 / 5.02 | 7.10 / 6.80 | 5.91 / 5.68 | 5.66 / 5.51 |
| 1024 | 10.84 / 11.01 | 8.33 / 8.16 | 11.35 / 11.49 | 9.19 / 9.48 | 8.89 / 8.87 |
| 4096 | 37.21 / 37.15 | 26.39 / 26.43 | 36.88 / 36.98 | 29.20 / 29.12 | 29.91 / 29.88 |

`prof/swz_metrics.csv` (512x512x1024): both swizzled kernels read 524,288
sectors with no load bank conflicts; the tuner's tile makes 407,552 shared
load wavefronts (394,816 padded), cuBLAS's shape 274,432, cuBLAS's own count.

## Tests and mutants

`tests/ptx_smem_swizzle.rs`:
- the layout for rows of 2-32 chunks: each row a permutation of its own
  chunks, 8 rows from a multiple of 8 conflict-free, a warp's copies whole
  lines, period 8; the padded layout as the control (it fails only the copy
  check) and an unpacked one (it fails `ldmatrix`);
- with no knob the GEMM is swizzled, and `Y_SMEM_SWIZZLE=0` and
  `Y_SMEM_PAD=8` give the same padded module;
- seven GEMMs, each emitted padded and swizzled at a pinned tile, run on the
  GPU: bit for bit equal on random inputs and both exact on integers against
  the CPU. Each asserts the path it is for: four stages with the barrier
  between a tile's K steps, rows of 8 chunks on both sides, two stages with
  B rows of 16 chunks, the synchronous single-stage copy, an M tail (the
  per-chunk copy), 64 threads where only the swizzle period stops the hoist
  (the padded module hoists), and the bias + ReLU epilogue with B rows of 4.

`mutants/run.sh`: 13 mutants of the change, each rebuilt; each fails the file
and the control passes (`mutants/summary.txt`). Removing the period check
(M6) fails on the path assertion first; run with that assertion taken out it
also fails on the device (`M6_numeric_only.test.log`).

**Not the dead loads.** The three `@!PT LDS RZ, [RZ]` that `ptxas` places
before a `cp.async` following a `bar.sync` (`../const_div_2026-10-19/`) are
there for every form of the copy probed, guarded, with `.L2::128B`, `.ca`, a
src-size or an ignore-src operand, for sm_80 to sm_89 (6 for sm_90 and
sm_120), and from Triton's bundled `ptxas` 12.8 as from 13.4
(`probes_dead_lds/counts.txt`). Y's 512 kernel already has `ldmatrix`s between
its barrier and its copies. cuBLAS's SASS issues a copy two instructions after
a barrier without them; none of the PTX forms probed gives that through
either `ptxas`. They are left alone.

## Validator gates

`gates/run.sh` on 81c055f (`base`) and on this tree (`new`), `gates/summary.txt`:
`build_corpus.sh`, `regress.sh`, `fpgate.py` and `unroll.py` print the same on
both. `docgate.py` reports the same 18 disagreeing figures on both, but for
its tree stamps, with the README's staging census at 691 lines for
`gemm_f16_4096`. `gap.py` refuses every kernel at the same first operation on
both trees (132 rows); the SASS gap lists of `gemm_f16_1024`,
`gemm_f16_bias_relu_1024` and the four decode kernels gain uniform-datapath
opcodes (`UIADD3`, `ULOP3.LUT`, `ULEA`, `R2UR`, `USEL`, `UISETP`) and uniform
predicates (`@UP0`-`@UP2`), where `ptxas` moved the warp-uniform swizzle
terms, and the contaminated counts of `gemm_f16_bias_relu_64`,
`gemm_f16_bias_relu_8192` and `gemm_f16_uneven_grid` move by one or two
(`gates/pythongappy_*.log`).

## Not done

- The profile's tiles were measured with the padded layout; nothing was
  re-tuned. The swizzle frees the padding's shared memory (2,560 bytes a
  stage at 128x128x32), which can allow another stage or a second CTA.
- The SwiGLU kernel and the FP8 and int8 GEMMs have their own copy code and
  layouts; whether their copies read extra sectors was not measured.
- `Autotuner::score_candidate` sizes stages with the emitter's layout now
  (no committed kernel changed by it), but no test holds it to the emitter's
  own stage count.
