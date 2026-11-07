# N odd on the 128- and 64-column tiles: edge copies, and K split in order (2026-11-07)

## What was slow

With N odd the FP32 GEMM was 0.77-0.98x cuBLAS on mid-size shapes, and
1023 x 1025 x 1027 0.80-0.86x. The odd-N stores (`../odd_n_stores/`) did
not touch the 128-column tile. An interleaved sweep around 1024^3
(`sweep.py`, three-parameter kernels) split the loss in two:
- **Odd N on its own costs about 12%.** With no ragged column and one wave,
  1024 x 897 x 1024 took 95-98 us against 87 us at N = 900, and was still
  1.01-1.03x cuBLAS. B is copied 4 bytes at a time and C stored 4 bytes at
  a time.
- **A one-column leftover costs a second wave.** At 1024 x 1025 x 1024 the
  128 whole tiles fill the card's 132 CTA slots, and the 16 remainder tiles
  for column 1024 run after them: 140 us against 80 us at N = 1024. An
  aligned 1024 x 1152 x 1024, with 144 whole tiles, took 136 us.

The workspace form (`W`) fixes the second at N % 4 == 0 by splitting K in
order (stream-K), but it was disabled at N % 4 != 0.

## The changes

**1. B past N is no longer copied from one address**
(`emit_f32_gemm_tile_set`). In a tile cut by N, a thread whose B chunk lies
past N used to copy the last chunk inside the row instead. Its values feed
only C columns that are not stored. At N odd the copies are 4 bytes wide
and cached in L1 (`cp.async.ca`). In a one-column edge tile, 127 of 128
threads then read one address per row, and those tiles' CTAs ran 1.8x as
long as the others (`diagnosis.txt`). Such a chunk now copies column
`col - N` of the same row, capped at the last chunk: a valid address, and a
different one per thread. Every output keeps its bits.

**2. Stream-K at N odd on the mid and small tiles**
(`f32_gemm_streamk_plan`, `emit_f32_gemm_streamk_kernel`).
- A split tile's continuation loads its accumulators from C one float at a
  time, under the guards the epilogue stores with. A row is only 4-byte
  aligned at N odd.
- Only on the 128- and 64-column tiles, whose plain path already copies B
  and stores C 4 bytes at a time. On the 256-column tiles the plain path's
  remainder tiles keep the aligned C pieces and the per-row B widths, which
  stream-K's single ragged set would lose.
- Taken where the model predicts at least 1.25x; at N % 4 == 0 the bar
  stays 1.04x. The model counts the work, not what a split costs at odd N:
  scalar handoffs, and segment prologues with twice the copies. Of 23 odd-N
  shapes on the mid tile:
  - the 9 given at least 1.25x ran at 0.78-0.97x the plain time;
  - of the 14 under it, 7 ran slower (up to 1.13x) and 6 faster.

  Nothing separates those 14 (`streamk_first_and_heldout.txt`,
  `streamk_fresh_validation.txt`). The last file is a set chosen after the
  bar was set: both of its shapes above the bar won, and of the 8 under it
  4 would have lost.

## Measurements

**Three-parameter form** (`sweep_run1.txt`, `sweep_run2.txt`): 83da64d's
kernel against the new one and cuBLAS (TF32 off), interleaved in one
process, 15 shuffled rounds, minimum per side. Every output of every kernel
equals the in-order `fmaf` oracle bit for bit.

| shape | old / cuBLAS | new / cuBLAS |
|---|---|---|
| 1024 x 1025 x 1024 | 0.757, 0.762 | 0.843, 0.861 |
| 1023 x 1025 x 1027 | 0.801, 0.793 | 0.901, 0.894 |
| 1025 x 1025 x 1025 | 0.687, 0.699 | 0.738, 0.747 |
| 2048 x 2049 x 1024 | 0.967, 0.980 | 1.056, 1.069 |
| 2047 x 2049 x 2051 | 0.910, 0.911 | 0.981, 0.979 |
| 1000 x 1001 x 1000 | 0.919, 0.930 | 0.931, 0.944 |
| 1024 x 1021 x 1024 | 0.977, 0.951 | 0.985, 0.965 |
| 1024 x 897 x 1024 | 1.028, 1.029 | 0.993, 1.005 |
| 768 x 1025 x 768, 1024 x 1028 x 1024, 4095 x 4097 x 512, 1024 x 30522 x 768, 1024^3 | | within 2% of old |
| 1024 x 50257 x 768 | 0.944, 0.891 | 0.906, 0.930 (old and new each moved 4% between runs, and both read below the 0.96-0.99x of `../odd_n_stores/`; these runs went back to back at the power cap, 271 W) |

1024 x 897 x 1024 is 2-3% slower in all four runs (the two here and two in
`zero_fill_ab.txt`). Its remainder tiles share the wave with the whole
tiles. The cause is not the extra B reads: zero-filling those copies with a
source size of 0, which reads nothing, measured the same
(`zero_fill_ab.txt`), so the simpler redirect was kept.

**Workspace form** (`streamk_run1.txt`, `streamk_run2.txt`;
`../streamk_bench.py` on the final build): the three-parameter kernel (Y3),
the workspace form (Y4) and cuBLAS, 15 shuffled rounds, minimum per side.
W was full of garbage, and every output equals the oracle on grids 1, 3,
P/2 and P, and again after the timing.

| shape | Y3 / cuBLAS | Y4 / cuBLAS | Y4 / Y3 time |
|---|---|---|---|
| 1023 x 1025 x 1027 | 0.890, 0.979 | 0.982, 0.996 | 0.906, 0.983 |
| 1024 x 1025 x 1024 | 0.904, 0.855 | 0.953, 0.930 | 0.949, 0.920 |
| 1025 x 1025 x 1025 | 0.746, 0.734 | 0.972, 0.983 | 0.767, 0.747 |
| 1100 x 1101 x 1100 | 0.770, 0.809 | 0.904, 0.945 | 0.852, 0.856 |
| 640 x 1025 x 2048 (66 CTAs) | 0.821, 0.820 | 1.040, 1.050 | 0.790, 0.781 |
| 1024 x 1153 x 1024 | 0.817, 0.800 | 0.920, 0.911 | 0.888, 0.878 |
| 512 x 1025 x 4096 (66 CTAs) | 0.741, 0.768 | 0.969, 0.982 | 0.764, 0.782 |
| 1000 x 1153 x 2000 | 0.862, 0.882 | 1.011, 0.987 | 0.852, 0.893 |
| 960 x 1281 x 960 | 0.787, 0.789 | 0.937, 0.935 | 0.840, 0.844 |
| 768 x 1025 x 768, 2048 x 1025 x 1024 (under the bar: W unused) | 1.05-1.11 | 1.05-1.11 | 0.99-1.02 |
| 1024 x 1028 x 1024, 768^3 (N % 4 == 0, unchanged) | 0.85-0.95 | 1.12-1.25 | 0.68-0.85 |

With `W`, the odd-N shapes the bar takes are 0.91-1.05x cuBLAS, from
0.73-0.98x without it. Y3 moves between runs here (1023 x 1025 x 1027:
0.890 and 0.979); it was 0.89-0.90x in both sweep runs.

## Stream-K needs an idle card

Stream-K assumes every CTA of its grid is resident at once. On this
desktop GPU, while other programs drew, the unchanged 1024 x 1028 x 1024
`W` kernel measured 94.6, 516.1, 148.2 and 94.9 us in successive runs,
where the plain 1024^3 kernel stayed at 83.5-84.3 us (`diagnosis.txt`).
Its minimum over rounds is what the tables report; a busy card can make it
several times slower than the three-parameter form.

## Tests

`tests/f32_simt_gemm.rs`:
- The plan test gains 1023 x 1025 x 1027 and 1025^3 (stream-K, the mid
  tile, a continuation that loads C a float at a time and no float4 access
  of C). It also gains 768 x 1025 x 768 and 2048 x 1025 x 1024 (under the
  bar) and 4095 x 4097 x 512 and 1024 x 50257 x 768 (the large tile), which
  must compile to the three-parameter kernel's SASS.
- Five odd-N stream-K shapes are checked element by element against the
  in-order oracle on seven grids each, W full of garbage:
  - 215 x 181 x 17 and 215 x 183 x 256 on the mid tile;
  - 197 x 181 x 1000 on the mid tile, ragged M with a K tail over 63 K
    tiles;
  - 35 x 221 x 33 and 60 x 221 x 300 on the small tile.
- compute-sanitizer memcheck and racecheck run four of them.
- `mutate.py`: 5 of 6 mutants killed (`mutants.txt`). Three are caught only
  by memcheck: the reload without its column guard, and the redirect
  missing column N or uncapped. The survivor, stream-K at N odd on the
  256-column tile, is unreachable: the large tile's own choice caps the
  model's gain near 1.11x.

`cta_times.py` patches a kernel to record each CTA's start, end and SM.
`cta_times.txt` ran the first stream-K build and the final one one after
the other, not interleaved, and the GPU's speed moved between them (the
unchanged-path 1024 x 1028 x 1024 kernel went from 101 to 125 us). Read
only the spread within each run: the first build's edge-tile CTAs ended at
173-198 us against 101-120 us for the rest; the final build has no such
group.

## Reproduce

`final.sh Y_OLD Y_NEW Y_SK_FIRST` (an idle GPU, about 35 minutes) writes
`sweep_run*.txt`, `streamk_run*.txt` and `cta_times.txt`:
- Y_OLD is 83da64d's compiler;
- Y_NEW is this commit's;
- Y_SK_FIRST is a build that splits K at N odd but still clamps B to the
  last chunk.

`plan_model.py` gives the model's predicted gain for any shape.
`mutate.py` runs the mutants.
