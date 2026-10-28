# `drift_add_at`: exact forces on atoms (2026-10-28)

`lj_exact` (`lj.ysu`) computes Lennard-Jones pair forces (sigma = epsilon = 1),
one thread per pair. Each pair hands `+f*d` to atom i and `-f*d` to atom j,
per component: six `drift_add_at` calls on one `@ZeroDrift` Q32.32 partial,
each adding it to a 128-bit total with a 64-bit atomic (and its carry into the
high word) and setting it back to zero. `drift_finish` takes one ticket per
block, and the last block converts all 3 x atoms totals to f32 and leaves the
buffer zero.

Atoms sit on a cubic lattice (spacing 1.1, each coordinate jittered by up to
0.05); every pair closer than 2.5 is one thread. Baselines (`baselines.cu`,
CUDA C with Y's own instructions for the arithmetic: the same fused
multiply-adds and `div.approx.f32`, so every variant sums the same f32 terms):
- `f32 atomics`: `atomicAdd(float)`; the buffer zeroed first.
- `f64 atomics`: each term widened, `atomicAdd(double)`; zeroed first.
- `fixed unchecked`: Q32.32 in 64-bit integer atomics, as AMBER's SPFP does it,
  with no range or term checks; zeroed first, converted to f32 after.

Each timed unit produces f32 forces from nothing (the Y kernel leaves its own
buffer zero). `Y finish only` is `lj_exact` with no pairs: every block takes
its ticket and the last converts 3 x atoms zero totals.

RTX 4070 Ti SUPER, driver 615.71.09, PyTorch 2.13.0+cu130, CUDA 13.4. Blocks
of 256. The clocks are brought up for 1.5 s first. Each timing is 10 launches
back to back, divided by 10, interleaved over 60 rounds; ranked by the
minimum. No game ran before, between or after the timings (`games.sh`).

**Contended at the start, by the desktop.** Before `run_2026-10-28/`,
`nvidia-smi pmon` (maximum of 5 samples, at idle clocks) showed plasmashell
at 44% and VS Code's GPU process at 49%; after it, kwin_wayland at 1%. The
first run (`run_2026-10-28_ieee_baselines/`, plasmashell 6% and kwin_wayland
2% before) gives the same Y and baseline times within 0.4%.

`run_2026-10-28/`, minimum (median), us:

| atoms | pairs | Y exact | Y finish only | f32 atomics | f64 atomics | fixed unchecked |
|---|---|---|---|---|---|---|
| 32,768 | 785,833 | 222.7 (223.5) | 84.9 (85.9) | 57.1 (57.4) | 98.8 (99.1) | 101.1 (101.3) |
| 262,144 | 6,532,504 | 1800.2 (1811.3) | 668.0 (678.5) | 448.2 (450.0) | 801.2 (805.5) | 808.9 (813.7) |

At 262,144 atoms Y takes 4.02x the f32 atomics, 2.25x the f64 atomics and
2.23x the unchecked fixed point. The finish is 37% of it: one block converts
786,432 totals (0.85 ns each), reading and zeroing 16 bytes and writing 4
for each, 28 MB in all, at 42 GB/s. Without the finish, the hand-overs cost
about 1.4x the unchecked fixed point: each needs the atomic's old value for
its carry, and an index test.

**Correctness, checked before timing.** Over 5 block sizes (64 to 1024) and
3 pair orders (as built, reversed, shuffled):
- `lj_exact` stored, in 15 of 15 configurations at both sizes, the bits of an
  exact host sum of the same terms. The terms come from `lj_terms`, Y's
  arithmetic storing them; the baselines' `lj_terms_cuda` stores the same
  bits. Each term is rounded to Q32.32, ties away from zero, summed in
  integers, and read back through f64. After timing the output was still
  that, and the buffer zero.
- f32 atomics gave 15 different results in 15 configurations. Against S_k,
  the exact sum of component k's terms rounded once to f64 (`math.fsum`),
  at 262,144 atoms: max_k |got_k - S_k| = 1.5e-5, which is 6.0e-7 of
  max_k |S_k| (25.3). At both sizes 59-79% of the components differ from
  f32(S_k).
- f64 atomics gave one result, equal to S_k in every component in every
  configuration: here every sum fits f64's 53 bits, so no addition rounds and
  the order cannot matter. That holds for this input, not in general.
- `lj_exact` differs from f32(S_k) in 2,749 of 786,432 components (0.35%):
  it sums the terms rounded to Q32.32, and a term below 2^-32 loses bits.
- The unchecked fixed point gave one result, not equal to Y's raw sums: it
  rounds ties to even, Y away from zero, and a term of magnitude 2^-10 to
  2^-9 is a tie half the time.

**The emitted code** (`lj.ptx`, assembled for sm_89): 7
`ATOMG.E.ADD.64` (six hand-overs and the ticket), 6 `RED.E.ADD.64` (the high
words), 2 `BAR.SYNC` and 2 `MEMBAR.SC.GPU`. The finish's loop is 31
instructions per total, with its two 64-bit loads not overlapped across
iterations.

**Checks of the code itself.**
- `sanitizer.txt`: compute-sanitizer's racecheck, synccheck, memcheck and
  initcheck report nothing on 4 launches of `lj_exact` (`sanitize.py`).
- `mutants.txt`: 34 mutants of the new code. 33 were killed by tests that
  run kernels. The 34th (no fence before the ticket) gives right answers on
  this GPU and is caught by the emitted-shape test, as `drift_add_to`'s was.
  One mutant (a finish converting one total too many) is caught only by the
  CPU interpreter, which refuses the read past the buffer.

**Earlier run, kept and not used above.** `run_2026-10-28_ieee_baselines/`:
the baselines used IEEE division and no fused multiply-add
(`baselines_ieee.cu`), so they summed slightly different terms from Y's. The
timings agree with the table.

Usage: `python3 bench_forces.py lj.ptx baselines.ptx <out dir> [games script]`,
with `lj.ptx` from `Y lj.ysu --emit-ptx -o lj.ptx` and `baselines.ptx` from
`nvcc -ptx -arch=sm_89 -o baselines.ptx baselines.cu`.
