# `drift_add_to`: an exact sum over the whole grid (2026-10-27, rerun 2026-10-28)

`zd_sum` (`grid_sum.ysu`) sums n f32 values exactly with `drift_add_to`.
Each thread adds its share as a `@ZeroDrift` Q32.32 partial in 128 bits, a
full warp adds its partials with shuffles, one lane adds the sum to a 64-bit
atomic total (with its carry into the high word), and the last block to finish
checks and stores the total. Two baselines:
- `torch.sum`, PyTorch's f32 reduction;
- `f32 two-pass`: the same partition summed in plain f32, each thread's partial
  stored and then added by `torch.sum`.

RTX 4070 Ti SUPER, driver 615.71.09, PyTorch 2.13.0+cu130; 528 blocks
(8 per SM) of 256 threads. The clocks are brought up for 1.5 s first. Each
timing is 10 launches back to back, divided by 10, interleaved over 60 rounds;
ranked by the minimum.

**Contended, by the desktop.** No game was running. Utilisation was 16%
before the run, at idle clocks (210 MHz), where little work fills much of the
time. Per process (`nvidia-smi pmon`, maximum of 5 samples, in
`results.json`):

| process | before | after |
|---|---|---|
| plasmashell | 21% | - |
| kwin_wayland | 14% | 11% |
| Discord's GPU process | 17% | 5% |

In the first attempt of the same night, only kwin_wayland was busy (22%), and
Discord showed 0%.

`run_2026-10-28/` (the table below):

| n (f32) | bytes | `zd_sum` min (median) | `torch.sum` | f32 two-pass | `zd_sum` / `torch.sum` |
|---|---|---|---|---|---|
| 2^20 | 4 MB (in L2) | 7.8 (7.9) us | 5.7 (5.9) us | 7.5 (7.6) us | 1.37 |
| 2^24 | 64 MB | 129.5 (130.3) us | 107.3 (107.8) us | 112.1 (112.7) us | 1.21 |
| 2^26 | 256 MB | 506.9 (515.0) us | 425.1 (428.1) us | 431.2 (435.0) us | 1.19 |

At 256 MB `torch.sum` reads 631 GB/s and `zd_sum` 530 GB/s.

**Correctness, checked before timing.**
- Every `zd_sum` launch, at three shapes (528 x 256, 37 x 1000, 330 x 64),
  stored the exact total: each term rounded to Q32.32, ties away from zero,
  summed in Python integers, read back through f64.
- After timing, the output was still that total and `ACC` was zero.
- The float sums are not that total. At 2^20 the exact total is -37.95335007:
  `torch.sum` gives -37.95336151, the two-pass -37.95341492
  (`results.json`, `torch_sum_err` and `two_pass_err`).

**Earlier runs, kept and not used above.**
- `bench.log` / `results.json`, 2026-10-27, no warm-up. The 64 MB and 256 MB
  ratios were the same (1.20, 1.19). The 4 MB row, timed first, read 1.34.
  This README called its 3-28% load "Discord and the compositor", which nobody
  had measured: the user saw Discord idle, and the per-process samples above
  replace that guess.
- `run_2026-10-28_no_warmup/`: the clocks were still rising (memory at 810 MHz)
  when the 4 MB row was timed. `torch.sum` took 28.9 us there; the large sizes
  agree with the table (1.22, 1.19).
- `bench_single_launch.log`, `results_single_launch.json`: one launch per
  interval. The card fell to a low clock state between launches (`torch.sum`
  306 GB/s at 256 MB).

**Not attributed.** Where the extra ~20% goes is not measured. The loop is not
unrolled and loads one f32 per iteration, while `torch.sum` uses vector loads;
this is a hypothesis, not a profile.

`mutants.txt`: 23 mutants of the lowering. All fail a test.
- M3, which shuffles in warps the block does not fill, passes on this GPU,
  which happens to return something harmless for the missing lanes. Only the
  CPU interpreter and the PTX shape check catch it.

Reproduce: compile `grid_sum.ysu` with `--emit-ptx`, then run
`python3 bench.py grid_sum.ptx <out dir>`.
