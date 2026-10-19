# `rope_<head_dim>` with several rows a CTA, 2026-10-19

The fused RoPE kernel (interleaved pairs, base 10000, `theta_i = pos *
10000^(-2i/d)`, `sin.approx`/`cos.approx`, f16 in and out). RTX 4070 Ti SUPER
(48 MB L2), driver 615.71.09, PyTorch 2.13.0+cu130, Triton 3.7.1, FlashInfer
0.6.16. Every run was started by `gpu_quiet_then.sh`
(`../paged_attention_2026-10-18/`) at a 10-sample mean of at most 5% with no
game running (each directory's `queue.txt`). Every column is checked against
float64 first (within 4e-3 of the largest output).

## What it was

One CTA of 32 threads a row, the grid one CTA a row. At most 24 CTAs fit an
SM, so the kernel ran at half occupancy, each CTA moving 128-512 bytes in and
out, and was bound by how fast CTAs launch rather than by memory: at 34 MB,
inside the L2, it moved 0.54 TB/s at head_dim 64, 1.07 at 128 and 2.03-2.05
at 256, where the new mapping moves 2.71-2.75 at all three.
At head_dim 64 a lane rotated one pair, one 4-byte load. Two more defects:
a head_dim that is not a multiple of 64 compiled and wrote part of each row,
checked only by a `debug_assert` that release builds drop. The parent's
`rope_96` writes 64 of every row's 96 elements and `rope_32` none
(`old_bugs/`: the sources, the parent compiler's PTX for them as
`rope_<d>_parent.ptx`, and `run_old.py` with its log, on 1000 rows).

## Discovery

`discovery/mk_rope.py` writes the kernel by hand with the emitter's exact
arithmetic and another mapping - `tpr` threads a row, `slots` rows a CTA,
`unroll` rows a thread, rows past a fourth parameter predicated off - and
`probe_rope.py` timed 141 of them (`make_variants.sh` regenerates them;
`variants.sha256` is what was measured) against the committed kernel and
Triton at 28 configurations, each checked bit for bit against the committed
kernel, also at rows + 3. These runs time back-to-back launches, so the
Triton columns at 2 MB measure its ~6 us Python launch; the final runs below
use graph replays.

- **Rows a CTA are what matters.** 64 threads on one row a CTA, the
  mapping the old launch could have kept, is level with the committed kernel
  (1.00-1.03x, `rope_d1.log`); 4 to 16 rows a CTA are 2.1-2.5x it at head_dim
  128 at 8-34 MB.
- **At least 8 bytes a lane.** At head_dim 64, the fastest variant with 16
  lanes a row, two pairs each, is 1.16-1.19x faster than the fastest with 32
  lanes and one pair at 8-34 MB (`rope_d64.log`).
- The mapping chosen (256 threads a CTA, one row a thread, `min(32, d / 4)`
  lanes a row) is within 1.0% of the fastest variant at 8-34 MB and from DRAM
  at all three head dims, and 2.6-8.4% behind it at 2 MB, where variants with
  two rows a thread lead.

## The change

- **`rope_<d>(X, Positions, Out, Rows: I32)`.** A CTA is 256 threads
  covering 16 rows at head_dim 64 and 8 above, `min(32, d / 4)` lanes a
  row, so a lane's pairs are one 64-bit load (64, 128) or one 128-bit load
  (256). The grid is at least `ceil(Rows / rows a CTA)` CTAs; rows past
  `Rows` are predicated off, loads zero-filled. Every load is issued before
  any arithmetic. The PTX header states the block and the rows a CTA.
- **Traps:** a block of another size, a grid that does not reach `Rows`, a
  negative `Rows` (sign-extended, so that check alone catches it). A larger
  grid is allowed. The row index is 64-bit.
- **Refusals:** the three-parameter form, which would otherwise compile as an
  empty kernel, and a head_dim that is not a multiple of 64.
- The arithmetic is unchanged: every column below that compares bits finds
  the new kernel's output equal to the parent's.

## Measured

`final/probe_final.py`, two runs (`final_f1.log` at 2.0% start,
`final_f2.log` at 2.6%), CUDA-graph replays, 15 interleaved rounds after 2,
minimum. Columns:

- **before:** `final/ptx/rope_<d>_before.ptx`, the parent's kernel: 532338f's
  `tests/rope_<d>.ptx`, from that commit's three-parameter `tests/rope_<d>.ysu`.
- **after:** `rope_<d>_after.ptx`, which is `tests/rope_<d>.ptx`.
- **Triton:** a kernel doing the same arithmetic with `tl.sin`/`tl.cos`, which
  lower to libdevice's precise `sinf`/`cosf` (no `sin.approx` in its PTX).
- **Triton fast:** the same with `libdevice.fast_sinf`/`fast_cosf`, which
  lower to `sin.approx`/`cos.approx` as Y does. Each Triton column is the
  faster of the two best of 28 (ROWS, num_warps) configurations in a graph
  screen at that size.
- **FlashInfer:** `apply_rope_pos_ids(q, k, pos)` with q and k the two halves
  of the same rows. It rotates both, so it does the same work as one Y launch
  over all the rows; positions repeat across the halves to match.
- **In place:** `apply_rope_pos_ids_inplace` against Y with `Out = X`. In
  place, the footprint is half, so these two columns are compared with each
  other.

Ratio of the column's time to after's, run 1 / run 2 (`final/table.py`,
`table.md`); above 1, after is faster:

| head_dim | rows | footprint | after (us) | before / after | Triton / after | Triton fast / after | FlashInfer / after | FlashInfer in place / after in place |
|---|---|---|---|---|---|---|---|---|
| 64 | 8192 | 2 MB | 1.59 / 1.58 | 3.013 / 3.032 | 1.547 / 1.544 | 1.151 / 1.146 | 1.201 / 1.215 | 1.234 / 1.215 |
| 64 | 32768 | 8 MB | 3.70 / 3.70 | 4.386 / 4.386 | 1.778 / 1.778 | 1.484 / 1.492 | 1.114 / 1.111 | 1.119 / 1.108 |
| 64 | 65536 | 17 MB | 6.64 / 6.64 | 4.727 / 4.729 | 1.809 / 1.812 | 1.580 / 1.590 | 1.051 / 1.057 | 1.054 / 1.056 |
| 64 | 131072 | 34 MB | 12.41 / 12.42 | 5.081 / 4.976 | 1.837 / 1.834 | 1.642 / 1.639 | 1.027 / 1.027 | 1.023 / 1.022 |
| 64 | 393216 | 102 MB | 165.23 / 164.98 | 1.170 / 1.207 | 1.004 / 1.004 | 1.001 / 1.002 | 1.000 / 1.002 | 0.941 / 0.853 |
| 128 | 4096 | 2 MB | 1.59 / 1.59 | 1.799 / 1.786 | 1.553 / 1.547 | 1.151 / 1.157 | 1.208 / 1.214 | 1.195 / 1.229 |
| 128 | 16384 | 8 MB | 3.69 / 3.71 | 2.341 / 2.337 | 1.740 / 1.739 | 1.474 / 1.477 | 1.114 / 1.111 | 1.109 / 1.111 |
| 128 | 32768 | 17 MB | 6.63 / 6.61 | 2.448 / 2.457 | 1.751 / 1.761 | 1.570 / 1.569 | 1.051 / 1.056 | 1.054 / 1.056 |
| 128 | 65536 | 34 MB | 12.43 / 12.46 | 2.532 / 2.539 | 1.780 / 1.782 | 1.632 / 1.624 | 1.024 / 1.020 | 1.018 / 1.023 |
| 128 | 196608 | 101 MB | 164.84 / 164.86 | 0.998 / 0.999 | 1.000 / 0.998 | 0.995 / 0.997 | 1.003 / 1.002 | 0.884 / 1.046 |
| 256 | 2048 | 2 MB | 1.43 / 1.44 | 1.350 / 1.340 | 1.657 / 1.639 | 1.280 / 1.271 | 1.364 / 1.347 | 1.343 / 1.336 |
| 256 | 8192 | 8 MB | 3.77 / 3.79 | 1.300 / 1.296 | 1.655 / 1.636 | 1.446 / 1.441 | 1.088 / 1.087 | 1.093 / 1.087 |
| 256 | 16384 | 17 MB | 6.64 / 6.66 | 1.318 / 1.318 | 1.706 / 1.715 | 1.553 / 1.553 | 1.051 / 1.047 | 1.048 / 1.047 |
| 256 | 32768 | 34 MB | 12.44 / 12.35 | 1.333 / 1.333 | 1.751 / 1.769 | 1.631 / 1.636 | 1.024 / 1.039 | 1.020 / 1.022 |
| 256 | 98304 | 101 MB | 163.30 / 163.44 | 1.000 / 0.999 | 1.006 / 1.005 | 1.002 / 1.003 | 1.006 / 1.006 | 0.950 / 0.948 |

- **Against the old kernel:** inside the L2, 3.0-5.1x faster at head_dim 64,
  1.79-2.54x at 128 and 1.30-1.35x at 256. From DRAM it is 1.17-1.21x at 64
  and level at 128 and 256.
- **Against FlashInfer, the same work:** 1.20-1.36x at 2 MB, 1.09-1.11x at
  8 MB, 1.05-1.06x at 17 MB and 1.02-1.04x at 34 MB; from DRAM level
  (1.000-1.006). The old kernel took 1.47-2.49x FlashInfer's time at head_dim
  128 inside the L2, 2.49-4.95x at 64 and 0.99-1.30x at 256.
- **In place:** the same ratios inside the L2 (1.02-1.34x). At the largest
  size, a footprint of about 50 MB, the L2's own size, FlashInfer is faster
  at head_dim 64 and 256 (1.06-1.17x and 1.05x), and at 128 the run decides
  (FlashInfer 1.13x, then Y 1.05x).
- **Against Triton:** 1.54-1.84x the precise kernel and 1.15-1.64x the fast
  one inside the L2 (the fast one: 1.15-1.28x at 2 MB, 1.62-1.64x at
  34 MB); from DRAM all level (0.995-1.006).

`unsigned_f1/` is a run of an earlier build of the change, whose `Rows`
checks were unsigned (a negative `Rows` was caught by the coverage check
instead of its own), and whose FlashInfer column was the in-place one.
`f0_flashinfer_build_failed.log`: FlashInfer's JIT build of its rope op
fails from a path with a space; the runs import it from a copy under the
scratch directory.

## Tests and mutants

`python/y_lang/tests/test_rope.py`, at head_dim 64, 128, 192 (three 32-bit
loads a lane), 256 and 512 (two 128-bit loads), each at 1, 7, 8, 9, 17 and
1003 rows, with NaN-filled outputs and NaN rows past `Rows`. It checks:

- every row against float64, and nothing written past `Rows`;
- three launches bit for bit alike, and a row's bits the same whatever the
  row count it is launched with;
- that each wrong launch (half the block, one CTA short, `Rows` = -1) traps,
  each in a child process after a right launch;
- that a grid three times too large is allowed and writes nothing more;
- the two refusals;
- the loads: one position load and the lane's pairs in loads of the
  expected width, every load before the first `sin.approx`, no branch.

Against the parent's compiler all five tests fail. It compiles the
four-parameter form as an empty kernel. `mutants/run.sh` applies nine
mutants of the new emitter code, rebuilding each, and each fails the file
(`mutants/summary.txt`; the control row passes):

- each of the three guards removed, each caught only by its own launch;
- stores unpredicated;
- half the rows a CTA;
- one warp a row at every head_dim;
- no head_dim refusal;
- the three-parameter form unmatched;
- a pair's byte offset halved.
