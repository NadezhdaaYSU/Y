# Division by a constant, 2026-10-19

RTX 4070 Ti SUPER, driver 615.71.09, `ptxas` from CUDA 13 (`ptxas -arch=sm_89`),
PyTorch 2.13.0+cu130, Triton 3.7.1.

## Where it was found

The F16 GEMM at 512^3 took 1.14-1.17x cuBLAS's time, and emitting cuBLAS's
own tile shape did not help (`../f16_square_2026-10-18/512/`). `profile512/`
is an Nsight Compute run with source counters (`ncu_src.sh`, started at a
3.2% mean with no game running, `queue.txt`): per-instruction executed counts
for Y's committed `gemm_f16_512` and cuBLAS's
`ampere_s16816gemm_fp16_64x64_ldg8_stages_64x5_nn`, both exact
(`ncu.out`), and `regions.py` splits them at the first and last HMMA
(`regions.txt`):

| | Y | cuBLAS |
| :--- | ---: | ---: |
| warp instructions executed | 450,048 | 343,296 |
| in the HMMA span | 292,864 | 169,984 |
| instructions a HMMA, HMMA span | 4.47 | 2.59 |
| IMAD | 111,872 | 3,840 |
| LDSM | 49,152 | 33,792 |
| dead `@!PT LDS RZ, [RZ]` | 24,576 | 0 |

Y's K loop is fully unrolled (766 of its 864 SASS lines execute once a warp),
yet inside it `ptxas` computed the stage index `(k + 3) % 4` at run time:
`IMAD.HI.U32` by a reciprocal, then a compare and a conditional subtract.

`probes/remprobe.ptx` is the reason, in isolation: `rem.u32 d, a, 4` and
`div.u32 d, a, 4`, an immediate and a power of two, assemble to a reciprocal
division (`I2F.U32.RP`, `MUFU.RCP`, `F2I`, two `IMAD.HI`, two correction
steps; `remprobe.sass.txt`). `ptxas` does not reduce a division by a constant,
and it expands the division after constant propagation, so a dividend known at
compile time is divided at run time as well. nvcc's output never shows it,
because NVVM reduces such divisions before it writes PTX. Y's emitter writes
them directly: 54 of its lines write a `div.u32` or `rem.u32`, most by a
constant, and 27 committed kernels held 8 to 57 such divisions each.
`probes/mh.ptx` is `x % 3` as a multiply: three SASS instructions
(`mh.sass.txt`).

## The change

`ptx_emitter::strength_reduce_constant_division`, run over every module
`emit_program` returns:

- a power of two: `div` is `shr`, `rem` is `and` with `c - 1` (u32, u64);
- 1: a `mov`;
- another u32 divisor: `mul.hi.u32` by `m` then `shr.u32` by `l`, the
  smallest `l` for which `m = ceil(2^(32+l) / c)` fits 32 bits and
  `m*c - 2^(32+l) <= 2^l` (Granlund and Montgomery 1994, Theorem 4.2, which
  makes it exact for every u32); a remainder subtracts `q * c`, and is left
  alone where its destination is its dividend;
- anything else (a register divisor, signed division, 0, a divisor with no
  32-bit multiplier such as 7 or 2^31 - 1) is left as it was. A guard stays on
  every instruction a guarded division becomes.

And `emit_gemm_plain_epilogue` marks its copy loop `.pragma "nounroll"`. Its
body divided by the chunks a row; without those divisions `ptxas -O3`
unrolled the loop four times, which moved ten kernels (the square GEMMs from
1024 up, the four decode kernels and `uneven_grid`) from `MATCHED` to
`UNROLLED` in the validator's loop census (`tools/ptxas_tval/unroll.py`, the
layer that decides whether `-O1` validation can speak for `-O3`) and grew
their SASS. With the pragma, every one of the 27 is back where it was in the
census.

`check/cdiv_pass.py` is an independent Python port. Applied to the 27
committed kernels it changes, it gives byte for byte what the new compiler
emits from their sources with the profile they were generated under (the
`xbar_ptx/final` profile of `../f16_xbar_2026-10-18/`), and the parent
compiler with that profile reproduces all 27 committed files byte for byte. `check/exhaustive.py`
checks the multiplier against `n // c` for all 2^32 dividends at 3 (the only
divisor in the committed kernels that is not a power of two, from three-stage
pipelines), 5, 6 and 24 (`exhaustive.log`).

`check/census.txt` is the static SASS of the 27 kernels before and after
(`census.sh`): reciprocal setups (`I2F`) fall from 5-10 to 1-2 a kernel, the
rest being the grid swizzle's divisions by `%nctaid`, which are run-time
values, and every kernel is smaller (`gemm_f16_512`: 864 to 608 SASS lines;
`gemm_f16_decode_m1`: 496 to 400). `census_unrolled.txt` is the same without
the pragma, where `ptxas` unrolled the epilogue and several kernels grew
(`gemm_f16_4096`: 832 to 1024).

## Measured

`../f16sq`'s `bench_ab.py`, interleaved rounds, minimum of 15, every column
checked exact first. The parent's kernels are 93778db's
`tests/gemm_f16_<N>.ptx` and, for the decode shapes,
`../decode_dram_2026-10-18/ptx/` (the committed decode kernels under the
names the harness uses).

- `final_ab/` run 1: the parent against this change before the epilogue
  pragma ("unrolled": the square kernels then regenerated, and
  `final_ab/decode_unrolled/`). Started at a 4.2% mean (`queue.txt`); this run
  used `run.sh` with two columns, `base` and `new`. Its second run never found
  the GPU quiet in two hours.
- `final_ab/` runs 3 and 4: three columns, the parent, this change
  ("final": `tests/` and `final_ab/decode_final/`) and "unrolled", started
  by `contended.sh` without waiting for a quiet GPU: a game was running, at a
  52.2% and a 27.2% mean (`queue.txt`). Contended.
- `pow2_ab/`: the parent against only the power-of-two half of the rewrite
  (`pow2div.py`), square sizes, two runs (4.0% and 1.6%).

The parent's time over the other's (above 1, the other is faster):

| shape | final, runs 3 / 4 (contended) | unrolled, run 1 (quiet) / 3 / 4 | power of two only, quiet |
| :--- | ---: | ---: | ---: |
| 256^3 | 1.037 / 1.035 | 1.008 / 1.009 / 1.006 | 1.007 / 1.007 |
| 512^3 | 1.023 / 1.029 | 1.018 / 1.009 / 1.018 | 1.016 / 1.012 |
| 1024^3 | 1.007 / 1.008 | 1.012 / 1.008 / 1.009 | 1.008 / 1.008 |
| 2048^3 | 1.004 / 1.004 | 1.003 / 1.004 / 1.003 | 1.004 / 1.012 |
| 4096^3 | 1.004 / 1.003 | 1.003 / 1.003 / 1.002 | 1.003 / 1.002 |
| 8192^3 | 1.001 / 1.020 | 1.003 / 1.003 / 0.993 | 1.003 / 1.003 |
| M=1, N=K=4096 | 1.017 / 1.020 | 1.011 / 1.014 / 1.017 | - |
| M=4 | 1.004 / 1.004 | 1.007 / 1.001 / 1.000 | - |
| M=8 | 1.005 / 1.004 | 1.004 / 1.003 / 1.003 | - |
| M=32 | 1.003 / 1.004 | 1.003 / 0.999 / 1.002 | - |

The final kernels are faster than the parent's at every shape in every run,
0.1-3.7%; against the unrolled ones, in the same two runs, they are level
from 1024 up and on the decode shapes (0.998-1.004, and 1.027 once at 8192) and 1.1-2.9% faster at
256 and 512. cuBLAS's time over the final 512 kernel's is 0.876 and 0.886
(0.892 for the unrolled one in the quiet run); the gap is smaller, not
closed.

## What is left at 512^3

`profile512_final/` is the source-counter run of the final `gemm_f16_512`,
under contention (a 6.9% mean, `--clock-control none`; the instruction counts
do not depend on it, the stall samples do and are not used): Y executes
311,296 warp instructions, against cuBLAS's 343,296 and its own 450,048
before, 192,000 of them in the HMMA span (cuBLAS 169,984; before 292,864),
IMAD 48,128 (was 111,872). What remains:

- `probes/barrier_*.ptx`: `ptxas` puts three dead `@!PT LDS RZ, [RZ]`
  before a `cp.async` that follows a `bar.sync`, and only then
  (`barrier_probes.txt`: not after a `cp.async.wait_group` alone, nor after
  an `ldmatrix` or a shared load). Y's K loop waits, synchronizes and issues
  the next tile's copies every K-tile, so the 512 kernel executes 48 of them a
  warp, 24,576 in all, 7.9% of its instructions; cuBLAS executes none.
- Y's 4x2 warps on a 64x64 tile give each warp 16x32, which costs 0.70 LDSM
  a HMMA against cuBLAS's 0.50 on 2x2 warps of 32x32; Y's 2x2 shape was
  slower as Y emits it (`../f16_square_2026-10-18/512/`).

## Tests and mutants

`tests/ptx_constant_division.rs`:
- the multiplier against `n / c` for every divisor up to 100000 at boundary
  and random dividends;
- every rewritten form, and every form that must be left alone, as text;
- a module with nothing to rewrite comes back byte for byte;
- the original and the rewritten module of a kernel dividing 2^16 full-range
  and boundary u32 and u64 inputs by 36 constants (guarded, aliased and
  fallback cases among them), both run on the GPU against the CPU;
- the emitted 512 GEMM holds no reducible division, with the parent's
  kernel as the control.

- the plain epilogue's copy loop stays rolled: the SASS holds exactly the
  PTX's global stores (2 for `gemm_f16_4096`; 16 without the pragma).

`mutants/run.sh`: ten mutants of the change, each rebuilt; each fails the file
and the control passes (`mutants/summary.txt`; `summary_before_pragma.txt` is
the first nine against the source before the pragma).

## Validator gates

Against the parent (93778db): `build_corpus.sh`, `regress.sh`, `fpgate.py`
and `unroll.py` give what they gave; `gap.py`'s SASS gaps for the GEMMs lose
`MUFU.RCP` and `F2I.FTZ.U32.TRUNC.NTZ`, and the PTX gaps of the three-stage
kernels gain `mul.hi.u32`. docgate's one new disagreement, the README staging
census's line count for `gemm_f16_4096` (672, now 674, the two pragma lines),
is corrected in the README; docgate then prints what it printed on the
parent, but for its tree stamps. Without the pragma, the same gates found the ten
kernels above `UNROLLED`, and docgate 19 disagreeing figures for 18.
