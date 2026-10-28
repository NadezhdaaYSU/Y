# Where a `@ZeroDrift` loop stops, before and after (2026-10-28)

**What changed.** Every `@ZeroDrift` check (a term that does not fit, an
accumulate that overflows, a value outside `@bounds`) used to end in a trap
branch right after the term. ptxas does not unroll a loop with such an exit,
so each term waited for its loads before the next term's loads were issued.
Now, inside a `for` whose step is 1 and whose body nothing outside the thread
can see, a failed check only sets a flag (`or.pred`), and one trap after the
loop stops the program. The same inputs stop, and nothing after the loop runs.
Any other loop still stops at the term (`src/ptx_emitter.rs`,
`drift_loop_defers`).

**How it was found** (`attribution/`, before any change to the compiler).
The known-issues entry had blamed the integer conversion. Hand-edited PTX of
the old `rowsum_zd` (`variants.py`, `var.ptx`; not compiler output) answered
it, on the 4,096 rows where latency limits the sum (`run1.json`; plasmashell
drawing at 24% before it, so contended):
- the two in-loop traps replaced by a flag and one trap after the loop: 1.63x
  f32, against 3.71x as compiled. ptxas unrolled it (14 loads instead of 2),
  it stored the same bits in 231,130 of 231,130 sums, and it stopped in the
  same four cases as the compiled kernel;
- no checks at all (not fail-closed, a reference only): 1.47x. So the
  conversion itself is a small part of the 3.71x, and the trap branch most
  of it.
Nsight Compute's stall samples (`stalls_*.csv`, from `prof_launch.py`) show
why `zd_sum` gained less: in the hand-edited `zd_sum_flag`
(`grid_sum_flag.ptx`, made by `flagify.py`) each of the four unrolled loads is
still waited on alone, 21% of the samples at each first use, behind the
`if i < n` branch. `f32_partials` predicates that test, issues its four loads
together, and does most of its waiting (75% of its samples) at the first add.

## Measured

`bench.py` compiles `rowsum.ysu` (the kernels of
`../zero_drift_conversion_2026-10-27/`) and `grid_sum.ysu` (`zd_sum` and
`f32_partials` of `../zero_drift_grid_sum_2026-10-27/`, plus `zd_split`) with
the compiler before the change (db0a0e8) and after it.

- **Machine:** RTX 4070 Ti SUPER, driver 615.71.09, PyTorch 2.13.0+cu130,
  CUDA 13.4.
- **Timing:** row sums as in the conversion benchmark (minimum of 25
  interleaved rounds of 20 launches after a 2 s clock ramp); grid sums as in
  the grid-sum benchmark (528 blocks of 256, minimum of 60 rounds of 10 after
  1.5 s).
- **Contended, by the desktop.** No game ran before or after. Before the run
  the card was at idle clocks with 22% utilisation: `kwin_wayland` 42% and
  Discord's GPU process 26% (`nvidia-smi pmon`, maximum of 5 samples); after,
  3% and 13%. The minima are reported, and both compilers' kernels were timed
  interleaved, so they shared the load. The old compiler's ratios reproduce
  the published 1.30x, 3.71x and 1.19x within 0.01.

`run_2026-10-28/`, minimum (median), us:

| row sum | 46,226 rows (bandwidth-bound) | 4,096 rows (latency-bound) |
|---|---|---|
| plain f32 | 305.6 (315.9) | 52.3 (52.4) |
| `@ZeroDrift`, before | 396.8 (410.9): 1.30x | 193.7 (201.3): 3.70x |
| `@ZeroDrift`, after | 299.5 (308.4): **0.98x** | 85.6 (85.6): **1.64x** |

| grid sum | 4 MB | 64 MB | 256 MB |
|---|---|---|---|
| `torch.sum` | 5.8 | 107.3 | 425.1 |
| f32 two-pass | 7.5 (1.29x) | 111.9 (1.04x) | 431.3 (1.01x) |
| `zd_sum`, before | 7.8 (1.33x) | 129.1 (1.20x) | 507.4 (1.19x) |
| `zd_sum`, after | 7.3 (1.25x) | 125.5 (1.17x) | 493.2 (1.16x) |
| `zd_split`, before | 7.3 (1.25x) | 125.2 (1.17x) | 492.2 (1.16x) |
| `zd_split`, after | **6.5 (1.11x)** | **118.5 (1.10x)** | **460.2 (1.08x)** |

(us, minimum, and the ratio to `torch.sum`; the medians are in `results.json`.)

**`zd_split` is the idiom.** `zd_sum` tests `if i < n` in every round, so the
conversion sits in a branch, and the loads still go out one at a time after
the change (above). `zd_split` loops over the rounds every thread takes
in full, with no test, and then the rest under the test. That loop has no
branch at all, and its four loads go out together.

**The same bits, checked before timing.**
- Row sums: both compilers' `rowsum_zd` equal the documented semantics in
  462,260 of 462,260 sums (46,226 rows x 10 orders).
- Grid sums: `zd_sum` and `zd_split`, from both compilers, store the exact
  total at 528 x 256, 37 x 1000 and 330 x 64, at all three sizes, and leave
  `ACC` zero; after timing, still so.

**The same stops, each in a child process.** Both compilers' `rowsum_zd` stop
on a term of 1e30, a NaN, a total out of range, and partial sums out of range
whose total is 0, and store 489.0 on good input; `zd_sum` and `zd_split` stop
on a NaN and store 25000.75 on good input.

**SASS** (`results.json`, `sass`):

| kernel | before: instructions, loads, traps | after |
|---|---|---|
| `rowsum_zd` | 71, 2, 2 | 312, 14, 1 |
| `zd_sum` | 339, 5, 4 | 585, 11, 4 |
| `zd_split` | 387, 7, 5 | 859, 19, 5 |

More instructions, because ptxas now unrolls the loop 4 times (with remainder
loops of 2 and 1), as it does the plain f32 loop (14 loads).

**The validator's gates** (`tools/ptxas_tval`, base db0a0e8 against this
tree): regress, fpgate and the corpus are the same, and docgate has the same
failures. `tests/train_spec.ptx` was regenerated: its loop now defers. The
validator does not model `or.pred`, so that kernel's opcode gap grows from 6
to 7. It was already outside the validator (a loop and traps), and unroll
still refuses it, for the same reason.

## Files

| File | What it is |
|---|---|
| `bench.py` | the benchmark |
| `run_2026-10-28.log`, `run_2026-10-28/results.json` | its output, with every timing |
| `run_2026-10-28/*_old.ptx`, `*_new.ptx` | both compilers' PTX |
| `rowsum.ysu`, `grid_sum.ysu` | the kernels |
| `mutants.txt` | the mutation test of the new code: 30 of 31 mutants fail a test, the 31st is equivalent |
| `attribution/rowsum_db0a0e8.ptx` | the old compiler's row sums, which the variants edit |
| `attribution/variants.py` → `var.ptx` | the flag and no-check row-sum variants, by hand |
| `attribution/bench_p4.py` → `run1.json` | their check, stop cases (`--trap`) and timing |
| `attribution/flagify.py` → `grid_sum_flag.ptx` | the old `grid_sum.ptx` with a flag copy of `zd_sum` |
| `attribution/prof_launch.py`, `stalls_*.csv` | one launch per kernel for `ncu --set full`, and the stall samples per instruction it gave |

Usage: `python3 bench.py <old Y> <new Y> <.ysu_hw_profile> <out dir> [games script]`;
`Y_PYTHON=<repo>/python GAMES=<games script> python3 attribution/bench_p4.py
attribution/rowsum_db0a0e8.ptx attribution/var.ptx <out json>`.
