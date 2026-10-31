# `drift_add_at`: exact forces on atoms (2026-10-28)

`lj_exact` (`lj.ysu`) computes Lennard-Jones pair forces (sigma = epsilon = 1),
one thread per pair. Each pair hands `+f*d` to atom i and `-f*d` to atom j,
per component: six `drift_add_at` calls on one `@ZeroDrift` Q32.32 partial,
each adding it to a 128-bit total with a 64-bit atomic (and its carry into the
high word) and setting it back to zero. Two ways to end:
- `lj_exact`: `drift_finish`. Every block takes a ticket, and the last block
  converts all 3 x atoms totals to f32 alone.
- `lj_closed` + `lj_convert`: `drift_close` takes the tickets and marks the
  totals complete, and a second kernel's `drift_convert` converts them on every
  SM (528 blocks of 256 here), after checking that mark.

Atoms sit on a cubic lattice (spacing 1.1, each coordinate jittered by up to
0.05); every pair closer than 2.5 is one thread. Baselines (`baselines.cu`,
CUDA C with Y's own instructions for the arithmetic: the same fused
multiply-adds and `div.approx.f32`, so every variant sums the same f32 terms):
- `f32 atomics`: `atomicAdd(float)`; the buffer zeroed first.
- `f64 atomics`: each term widened, `atomicAdd(double)`; zeroed first.
- `fixed unchecked`: Q32.32 in 64-bit integer atomics, as AMBER's SPFP does it,
  with no range or term checks; zeroed first, converted to f32 after.

Each timed unit produces f32 forces from nothing (the Y kernels leave their
buffers zero). The `no pairs` rows run the Y kernels with no pairs: every block
takes its ticket, and the totals (all zero) are converted.

RTX 4070 Ti SUPER, driver 615.71.09, PyTorch 2.13.0+cu130, CUDA 13.4. Blocks
of 256. The clocks are brought up for 1.5 s first. Each timing is 10 launches
back to back, divided by 10, interleaved over 60 rounds; ranked by the
minimum. No game ran before, between or after the timings (`games.sh`).

`run_2026-10-28_close_convert/`: utilisation 0% before the run, and
`nvidia-smi pmon` showed no other process. Minimum (median), us:

| atoms | pairs | Y finish | Y close + convert | f32 atomics | f64 atomics | fixed unchecked |
|---|---|---|---|---|---|---|
| 32,768 | 785,833 | 223.0 (224.4) | 157.6 (157.9) | 57.7 (57.9) | 99.4 (99.7) | 101.9 (102.1) |
| 262,144 | 6,532,504 | 1804.8 (1813.4) | 1202.7 (1203.1) | 450.2 (450.6) | 805.6 (806.1) | 813.5 (814.2) |

| atoms | Y finish, no pairs | Y close + convert, no pairs |
|---|---|---|
| 32,768 | 85.4 (86.4) | 18.2 (18.3) |
| 262,144 | 668.2 (678.6) | 57.1 (57.4) |

At 262,144 atoms:
- `drift_finish` takes 4.01x the f32 atomics, 2.24x the f64 atomics and 2.22x
  the unchecked fixed point. 37% of it is the last block converting 786,432
  totals alone (0.85 ns each; reading and zeroing 16 bytes and writing 4 for
  each, 28 MB at 42 GB/s).
- `drift_close` + `drift_convert` take 2.67x the f32 atomics, 1.49x the f64
  atomics and 1.48x the unchecked fixed point. The conversion on every SM takes
  57 us instead of 668. What remains is the hand-overs: each needs the atomic's
  old value for its carry, and an index test.

**Later (2026-10-31).** The hand-overs' wait for the old value was not the cost: L2 was. With each total's two words split (low words side by side, high words after them), `drift_close` + `drift_convert` takes 884 us instead of 1215, 1.10x f64 atomics: `../zero_drift_layout_2026-10-31/`.

**Correctness, checked before timing.** Over 5 block sizes (64 to 1024) and
3 pair orders (as built, reversed, shuffled):
- `lj_exact`, and `lj_closed` followed by `lj_convert` (converting at 1 x 1024,
  528 x 256, one thread per total, 7 x 33 and 66 x 1024 in turn), stored, in 15
  of 15 configurations at both sizes, the bits of an exact host sum of the same
  terms. The terms come from `lj_terms`, Y's arithmetic storing them; the
  baselines' `lj_terms_cuda` stores the same bits. Each term is rounded to
  Q32.32, ties away from zero, summed in integers, and read back through f64.
  After timing the outputs were still that, and both buffers zero.
- f32 atomics gave 15 different results in 15 configurations. Against S_k,
  the exact sum of component k's terms rounded once to f64 (`math.fsum`),
  at 262,144 atoms: max_k |got_k - S_k| = 1.5e-5, which is 6.0e-7 of
  max_k |S_k| (25.3). At both sizes 59-79% of the components differ from
  f32(S_k).
- f64 atomics gave one result, equal to S_k in every component in every
  configuration: here every sum fits f64's 53 bits, so no addition rounds and
  the order cannot matter. That holds for this input, not in general (below).
- The Y kernels differ from f32(S_k) in 2,749 of 786,432 components (0.35%):
  they sum the terms rounded to Q32.32, and a term below 2^-32 loses bits.
- The unchecked fixed point gave one result, not equal to Y's raw sums: it
  rounds ties to even, Y away from zero, and a term of magnitude 2^-10 to
  2^-9 is a tie half the time.

**A stress test: denser lattices** (`stress_forces.py`,
`stress_2026-10-28/`). The same kernels and 15 configurations on 32,768
atoms, the lattice spacing shrinking from 1.1 to 0.7: atoms come closer, the
largest terms grow from 24 to 35,000, and one component's sum mixes them with
very small ones. Y's kernel here is `lj_exact_wide` (bounds +-2e9).

| spacing | pairs | closest pair | Y exact | f64: distinct results | f64: components != S_k (most in one configuration) | f64 rounded to f32: distinct |
|---|---|---|---|---|---|---|
| 1.1 | 785,833 | 1.001 | 15/15 | 1 | 0 | 1 |
| 1.0 | 1,153,901 | 0.901 | 15/15 | 2 | 1 | 1 |
| 0.95 | 1,194,148 | 0.851 | 15/15 | 5 | 7 | 1 |
| 0.9 | 1,217,314 | 0.801 | 15/15 | 9 | 11 | 1 |
| 0.85 | 1,413,551 | 0.751 | 15/15 | 14 | 61 | 1 |
| 0.8 | 1,871,128 | 0.701 | 15/15 | 15 | 247 | 1 |
| 0.75 | 2,340,389 | 0.651 | 15/15 | 15 | 686 | 1 |
| 0.7 | 2,669,194 | 0.601 | 15/15 | 15 | 1,343 | 1 |

From spacing 1.0 down, f64 atomics give different bits for different block
sizes and pair orders, and from 0.8 down a different result in every
configuration. Rounded to f32, they still gave one result at every spacing:
two f64 sums can round to different f32 values only when the exact sum lies
within their difference of a rounding boundary, which none of the 98,304
components did here. f32 atomics gave 15 different results at every spacing.

**The emitted code** (`lj.ptx`, assembled for sm_89):
- `lj_exact`: 7 `ATOMG.E.ADD.64` (six hand-overs and the ticket), 6
  `RED.E.ADD.64` (the high words), 2 `BAR.SYNC` and 2 `MEMBAR.SC.GPU`. The
  finish's loop is 31 instructions per total, with its two 64-bit loads not
  overlapped across iterations.
- `lj_closed`: the same hand-overs and ticket, and 3 fences (the last before
  the tag). `lj_convert`: 130 instructions, one ticket, and 7 traps (a negative
  count, the tag, the count, 64 bits, the two bounds, a stale ticket).

**Checks of the code itself.**
- `sanitizer.txt`: compute-sanitizer's racecheck, synccheck, memcheck and
  initcheck report nothing on 4 launches of `lj_exact` and 4 pairs of
  `lj_closed` and `lj_convert` (`sanitize.py`).
- `mutants.txt`: 34 mutants of `drift_add_at` / `drift_finish` and 29 of
  `drift_close` / `drift_convert`, all killed. The four fence mutants give
  right answers on this GPU and are caught by the emitted-shape test. Five are
  caught only by the CPU interpreter, which refuses an access outside a
  buffer: a finish converting one total too many, and four checks that
  another check stops first on the GPU (a count stored for another count, a
  converter's stale ticket, a negative count in either kernel).

**Earlier runs, kept and not used above.**
- `run_2026-10-28/`: `lj_exact` and the baselines only. The desktop loaded the
  GPU at the start (`nvidia-smi pmon` at idle clocks: plasmashell 44%, VS
  Code's GPU process 49%); the times agree with the table within 0.6%.
- `run_2026-10-28_ieee_baselines/`: the baselines used IEEE division and no
  fused multiply-add (`baselines_ieee.cu`), so they summed slightly different
  terms from Y's. The timings agree with the table.

Usage: `python3 bench_forces.py lj.ptx baselines.ptx <out dir> [games script]`
and `python3 stress_forces.py lj.ptx baselines.ptx <out dir>`, with `lj.ptx`
from `Y lj.ysu --emit-ptx -o lj.ptx` and `baselines.ptx` from
`nvcc -ptx -arch=sm_89 -o baselines.ptx baselines.cu`, with its `.version`
lowered from nvcc 13.4's 9.4 to 7.8, the floor its instructions need on sm_89
(`tests/committed_ptx_artifacts.rs`; the SASS is the same).
