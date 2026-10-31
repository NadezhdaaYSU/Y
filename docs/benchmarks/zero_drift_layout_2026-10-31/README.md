# `drift_add_at`'s totals, split in ACC, before and after (2026-10-31)

**What changed.** Total i used to be words 2i (low) and 2i + 1 (high) of
`ACC`: 16 bytes per total. Now its low word is word i and its high word word
n + i, so the low words, which every hand-over adds to, sit 8 bytes apart.
The header stays at word 2n and `ACC` keeps its size; `drift_add_to`'s single
total (n = 1) is laid out as before, and its code is byte-identical.
`drift_finish` and `drift_convert` read and clear the same two words.

**How it was found** (`attribution/`, before any change to the compiler).
E8's README blamed the hand-over's wait for its atomic's old value (the carry
into the high word) and its index test. Hand-edited copies of the compiled
`lj_closed` (`variants.py`, on `lj_d0d4611.ptx`; not compiler output) said
otherwise, at 262,144 atoms (`run1.json`; no game, the desktop at 3% before):
- the traps replaced by a flag and one trap before the close: 1202.3 us
  against 1202.6 as compiled;
- each carry worked out after all six atomics, so their round trips overlap:
  1219.6 us;
- no old value at all (`red`: not exact, a bound only; kernel alone, `ACC`
  zeroed first): 1150.6 us against 1186.1 for the compiled kernel timed the
  same way.

Nsight Compute (`l2_by_kernel.csv`, from `prof_h.py`): L2 is the bottleneck
of all three kernels, at 98% (`lj_closed`), 96% (`red`) and 90% (the
unchecked fixed point) of its peak throughput. Y's atomics made 16.7 M L2
requests against the baseline's 9.1 M, and 39.2 M sectors against 30.4 M; the
`red` copy made as many as Y. A warp's 32 hand-overs go to atoms 3 apart, and
with 16 bytes per total they touch about twice as many cache lines as with 8.
A copy with the split layout (`split.py`, `bench_split.py`) stored the exact
sums, read back on the host, and took 866.7 us instead of 1204.7 (kernel
alone; `run_split.json`).

## Measured

`bench.py` compiles `../zero_drift_forces_2026-10-28/lj.ysu` with the compiler
before the change (d0d4611) and after it, and times both against E8's CUDA
baselines (`../zero_drift_forces_2026-10-28/baselines.ptx`, with Y's own
arithmetic, so all sum the same f32 terms).

- **Machine:** RTX 4070 Ti SUPER, driver 615.78.08 (E8 ran on 615.71.09),
  PyTorch 2.13.0+cu130.
- **Timing:** minimum of 60 interleaved rounds of 10 launches, after 1.5 s
  of clock warm-up; blocks of 256. Each timed unit makes f32 forces from
  nothing (the Y kernels leave their buffers zero; the atomic baselines zero
  theirs first, and the fixed point converts after).
- **Contended, by the desktop.** No game ran before or after. Before the run
  the card was at idle clocks with 26% utilisation: `kwin_wayland` 22% and
  Chromium's GPU process 23% (`nvidia-smi pmon`, maximum of 5 samples); after,
  7% and 8%. The old compiler's ratios reproduce E8's within 1%, on the
  newer driver too.

`run_2026-10-31/`, minimum (median), us:

| 262,144 atoms (6.5 M pairs) | before | after |
|---|---|---|
| `drift_close` + `drift_convert` | 1215.0 (1238.3) | **883.8 (905.5)** |
| `drift_finish` | 1831.8 (1858.8) | 1467.4 (1507.2) |
| fixed point, unchecked | 813.1 (837.3) | |
| f64 atomics | 805.4 (827.8) | |
| f32 atomics | 447.6 (469.0) | |

| ratio, 262,144 atoms | before | after |
|---|---|---|
| close + convert / unchecked fixed point | 1.49x | **1.09x** |
| close + convert / f64 atomics | 1.51x | **1.10x** |
| close + convert / f32 atomics | 2.71x | 1.97x |
| finish / f64 atomics | 2.27x | 1.82x |

At 32,768 atoms, close + convert took 157.0 us before and 118.3 after (1.55x
and 1.17x the unchecked fixed point).

**The same bits, checked before timing.** Both compilers' `lj_exact` and
`lj_closed` + `lj_convert` stored the exact sums, bit for bit, for 3 block
sizes (64, 256, 1024) x 3 pair orders (as built, reversed, shuffled), at both
sizes, and left their buffers zero. After the timing, the buffers were still
zero and one more launch of each was exact.

**Checks of the code.** `mutants.txt`: 15 mutants of the new addresses and stores (each kernel's low or high word put elsewhere, or not cleared, and `drift_add_to`'s two words moved), all killed. Two new interpreter tests read `ACC` between the two kernels (low words, then high words) and hand the converter a crafted `ACC` whose high word only a converter reading word n + i sees.

## Files

| File | What it is |
|---|---|
| `bench.py` | the benchmark |
| `run_2026-10-31.log`, `run_2026-10-31/results.json` | its output, with every timing |
| `run_2026-10-31/lj_old.ptx`, `lj_new.ptx` | both compilers' PTX |
| `attribution/lj_d0d4611.ptx` | the old compiler's kernels, which the variants edit |
| `attribution/variants.py`, `bench_h.py` → `run1.json` | the flag, deferred-carry and no-old-value variants: checks, stop cases, timing |
| `attribution/split.py`, `bench_split.py` → `run_split.json` | the split layout by hand: host read-back and timing |
| `attribution/prof_h.py` → `l2_by_kernel.csv` | one launch per kernel for `ncu --set full`, and its L2 counts |
| `mutants.txt` | the mutation test |

Usage: `python3 bench.py <old Y> <new Y> <.ysu_hw_profile> <out dir> [games script]`.
The attribution scripts take `Y_PYTHON=<repo>/python`; `variants.py <lj.ptx> <out.ptx>`
and `split.py <lj.ptx> <out.ptx>` make the variant modules they time.
