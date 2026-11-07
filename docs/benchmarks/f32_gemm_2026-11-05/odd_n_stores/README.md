# C stored in aligned pieces at N odd (2026-11-07)

## What was slow

With N odd the FP32 GEMM was 0.77-0.98x cuBLAS, and GPT-2's vocabulary
projection, 1024 x 50257 x 768, was 0.85x. The record put the loss on B's
4-byte copies. ncu (`prof.sh`, `ncu_before.txt`, base clocks) compared the
odd shapes against the nearest N that is a multiple of 4:

| kernel | ms | issue active | `lg_throttle` | L2 write sectors |
|---|---|---|---|---|
| 1024 x 50257 x 768 | 3.14 | 66.6% | 0.49 | 32.3M |
| 1024 x 50176 x 768 | 2.57 | 81.2% | 0.04 | 7.0M |
| 1024 x 32001 x 4096 | 9.43 | 77.8% | 0.09 | 21.0M |
| 1024 x 32000 x 4096 | 8.84 | 82.7% | 0.01 | 5.0M |

Two things point at the epilogue rather than at B:
- **C's writes were 5x its size in L2 sectors.** C is 6.4M sectors at
  50257 columns. Each warp store wrote 4 bytes a lane across 8 rows, about
  20 sectors, and each sector was written by four stores.
- **The gap did not scale with K.** A fit of `e * M*N + c * M*N*K` to the two
  gaps gives the stores (`e`) about 0.49 ms of the 0.57 ms at K = 768 and
  half of it at K = 4096. B's copies (`c`) are the rest.

## The change

A thread's float4 of C sits at place `(row * N + col) mod 4` in a 16-byte
line. Its rows are `rw` (mod 4), and its columns start at a multiple of 4.
So the place of element `e` is `(rw * N + e) mod 4`, which is known when
compiling. Each float4 now goes out in its widest aligned pieces:

| place of element 0 | stores |
|---|---|
| 0 | `st.global.v4.f32` |
| 2 | `st.global.v2.f32` + `st.global.v2.f32` |
| 1 or 3 | `st.global.f32` + `st.global.v2.f32` + `st.global.f32` |

That is 2.25 stores a float4 instead of 4.

Where it applies:
- whole 256-column tiles: the large tile and the `rows32x256` strips;
- ragged-M tiles too (each piece keeps the row guard).

Where it does not:
- a ragged column set keeps 4-byte stores;
- so does the 128-column mid tile, where the pieces measured slower. In 9
  runs `1023 x 1025 x 1027` took 1.016-1.090x the old time, mean 1.047. In
  the same runs `1000 x 1001 x 1000`, whose kernel is byte-identical to the
  old one, averaged 1.004 (`ab_mid_tile.txt`). Without a ragged M the mid
  tile was 0.97-1.01x. The same shape was why B's row widths are limited to
  the 256-column tiles too.

Every output keeps its bits: they are the same `fma` chains, only stored
differently.

## Measurements

**ncu after** (`ncu_after.txt`):
- 1024 x 50257 x 768: 3.14 -> 2.85 ms; `lg_throttle` 0.49 -> 0.23; L2
  write sectors 32.3M -> 18.0M; issue active 66.6% -> 73.5%.
- 1024 x 32001 x 4096: 9.43 -> 9.23 ms; 21.0M -> 11.5M sectors.
- The aligned shapes are unchanged.

**Timed** (`ab.py`, `ab_run1.txt`, `ab_run2.txt`): the old kernel (this
file reverted), the new one and cuBLAS (TF32 off), interleaved in one
process, 15 rounds shuffled, minimum per side. Every output of both
kernels equals the in-order `fmaf` oracle bit for bit, and they equal each
other. Two runs:

| shape | old / cuBLAS | new / cuBLAS | new / old time |
|---|---|---|---|
| 1024 x 50257 x 768 (GPT-2's vocabulary) | 0.846, 0.902 | 0.964, 0.986 | 0.877, 0.914 |
| 4096 x 50257 x 768 | 0.893, 0.863 | 0.962, 0.975 | 0.928, 0.885 |
| 1024 x 50259 x 768 (N = 3 mod 4) | 0.865, 0.863 | 0.930, 0.944 | 0.930, 0.914 |
| 4116 x 50257 x 768 (a 20-row strip) | 0.960, 0.913 | 1.006, 0.984 | 0.954, 0.928 |
| 1024 x 32001 x 4096 | 0.960, 0.971 | 0.976, 0.984 | 0.983, 0.987 |
| 4095 x 4097 x 512 | 0.769, 0.777 | 0.919, 0.917 | 0.837, 0.848 |
| 4096 x 4097 x 4096 | 0.985, 0.986 | 1.021, 1.021 | 0.965, 0.966 |
| 2047 x 2049 x 2051, 1023 x 1025 x 1027, 1 x 50257 x 768 (unchanged) | | | 0.994-1.000 |
| 1024 x 30522 x 768 (N even), 4096^3 (controls) | | | 0.996-1.002 |

`ab_all_tiles_first_version.txt` is the first version, which used the pieces
on every tile, against the same old kernel.

**What is left.** At 1024 x 50257 x 768 the odd kernel still writes 2.8x
C's sectors, and B's rows are still copied at 16, 8 and 4 bytes (11 copies
a thread a stage, against 4 at N = 50176). Both could go:
- stores aligned across lanes by warp shuffles, or staged through shared
  memory;
- B rows copied 16 bytes at a time from the aligned-down address and shifted
  in shared memory.

## Tests

`tests/f32_simt_gemm.rs`:
- `odd_n_stores_c_in_aligned_pieces_on_the_256_column_tiles` checks the
  store mix at N = 1 and 3 mod 4, and that the mid tile has no wide C store.
- The in-order oracle test gained 2048 x 4097 x 40 and 2060 x 4097 x 24
  (N = 1 mod 4, a row strip); 2047 x 4099 x 36 and 2048 x 12803 x 64/67
  already cover N = 3 mod 4 with ragged M and a K tail.
- compute-sanitizer memcheck and racecheck run both new shapes.
- `store_mutate.py`: 3 of 3 mutants killed (`mutants.txt`). The row guard
  dropped is caught only by memcheck.
