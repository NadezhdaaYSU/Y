# `drift_add_to`: an exact sum over the whole grid (2026-10-27)

`zd_sum` (`grid_sum.ysu`) sums n f32 values exactly with `drift_add_to`.
Each thread adds its share as a `@ZeroDrift` Q32.32 partial in 128 bits, a
full warp adds its partials with shuffles, one lane adds the sum to a 64-bit
atomic total (with its carry into the high word), and the last block to finish
checks and stores the total. Two baselines:
- `torch.sum`, PyTorch's f32 reduction;
- `f32 two-pass`: the same partition summed in plain f32, each thread's partial
  stored and then added by `torch.sum`.

RTX 4070 Ti SUPER, driver 615.71.09, PyTorch 2.13.0+cu130; 528 blocks
(8 per SM) of 256 threads. Each timing is 10 launches back to back, divided by
10, interleaved over 60 rounds; ranked by the minimum.

**Contended:** Discord and the compositor used 3-28% of the GPU before the run
(17% at its start). No game was running.

| n (f32) | bytes | `zd_sum` min (median) | `torch.sum` | f32 two-pass | `zd_sum` / `torch.sum` |
|---|---|---|---|---|---|
| 2^20 | 4 MB (in L2) | 9.9 (10.0) us | 7.4 (7.6) us | 9.7 (9.9) us | 1.34 |
| 2^24 | 64 MB | 128.9 (130.3) us | 107.3 (107.9) us | 112.1 (112.8) us | 1.20 |
| 2^26 | 256 MB | 508.0 (532.9) us | 425.6 (439.2) us | 431.4 (450.6) us | 1.19 |

At 256 MB `torch.sum` reads 631 GB/s and `zd_sum` 528 GB/s.

**Correctness, checked before timing.**
- Every `zd_sum` launch, at three shapes (528 x 256, 37 x 1000, 330 x 64),
  stored the exact total: each term rounded to Q32.32, ties away from zero,
  summed in Python integers, read back through f64.
- After timing, the output was still that total and `ACC` was zero.
- The float sums are not that total. At 2^20 the exact total is -37.95335007:
  `torch.sum` gives -37.95336151, the two-pass -37.95341492
  (`results.json`, `torch_sum_err` and `two_pass_err`).

**A first run timed one launch per interval** (`bench_single_launch.log`,
`results_single_launch.json`). The card stayed in a low clock state between
launches: `torch.sum` read 306 GB/s at 256 MB. That run is kept, and it is not
used above.

**Not attributed.** Where the extra ~20% goes is not measured. The loop is not
unrolled and loads one f32 per iteration, while `torch.sum` uses vector loads;
this is a hypothesis, not a profile.

`mutants.txt`: 23 mutants of the lowering. All fail a test.
- M3, which shuffles in warps the block does not fill, passes on this GPU,
  which happens to return something harmless for the missing lanes. Only the
  CPU interpreter and the PTX shape check catch it.

Reproduce: compile `grid_sum.ysu` with `--emit-ptx`, then run
`python3 bench.py grid_sum.ptx <out dir>`.
