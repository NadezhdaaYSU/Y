# Paged decode attention: 16 lanes a token, a lazy rescale, a prefetch, 2026-10-18

The README's "Where the GPU backend loses" table listed one row: paged decode
attention with the KV cache in L2, 0.76-0.81x FlashInfer's
`BatchDecodeWithPagedKVCacheWrapper`. Its suspected cause, "one 32-lane
butterfly reduction per query head, so four per token", had not been
measured. This record measures it, changes the kernel's KV loop, and times
the result against the kernel it replaces and against FlashInfer.

RTX 4070 Ti SUPER (66 SMs, 48 MB L2), driver 615.71.09, CUDA 13.4 `ptxas`,
PyTorch 2.13.0+cu130, FlashInfer 0.6.16, CuPy 14.1.1. head_dim 128, 32 query
heads, 8 KV heads (GQA 4:1), page size 16, shuffled page table, NHD layout:
the README's configuration throughout unless a row says otherwise.

## What bound the old kernel

`ncu_old_b1c4096.txt` (Nsight Compute, no cache flush; its durations are of
replayed passes and are not used): at batch 1 / ctx 4096 the old 16-split
kernel executed 5.33 M warp instructions to FlashInfer's 4.28 M (1.25x), and
took 14.33 us of device time to FlashInfer's 10.98 (1.31x, `cupti_1.log`).
Both issued about half an instruction per cycle per scheduler, averaged over
the GPU (0.51 and 0.54; the last table below): the time ratio was the
instruction ratio.

`sass_loop_census.txt`: the old loop was 157 sm_89 SASS instructions per
token for one warp and the CTA's four query heads, 20 of them `SHFL` and 8
`MUFU`. All 32 lanes held one token, 4 f16 each, so every score took a
5-step butterfly; a token's K/V load waited on a page-table load every
second iteration, with nothing in flight ahead of it.

## The change

`emit_paged_decode_attention_kernel`'s KV loop:

- A token's row spans 16 lanes, 8 f16 each (one 128-bit load), so a warp
  works on two tokens per iteration and each score's butterfly is 4 steps
  over 16 lanes. `attn_loop_elems_per_lane` keeps 8 elements a lane only
  while Q and the accumulators of the CTA's query heads fit 64 registers: at
  GQA 8:1 it is 4 (32 lanes, the old layout), at head_dim 64 a token is 8
  lanes, at 256 it is 32.
- Each warp takes one contiguous block of the CTA's tokens, so its page-table
  entry changes once a page; the physical page is reloaded with a predicated
  load when the logical page changes.
- The softmax state is rescaled only when a score passes its reference max by
  more than 8 (`ATTN_LAZY_RESCALE`, nats): `(m, l, acc)` stays consistent
  for any reference `m`, and every weight is at most e^8. The decision is a
  warp-uniform `vote.any`.
- The next iteration's page lookup and K/V loads are issued before this
  iteration's arithmetic.
- A lane whose token lies past its warp's block reads the block's last token
  and its weight is zeroed; after the loop the warp's two token groups are
  folded with the same max-rescale-and-sum as the cross-warp merge.
- The loop tests its condition at the top (the shape the translation
  validator's loop recogniser reads); `ptxas` still emits it bottom-tested.

The split kernel's loop is now 239 SASS instructions an iteration, 60 of
them in the rescale block, so 179 for two tokens when no rescale is taken
(`sass_loop_census.txt`). The layout alone, before the lazy rescale and the
prefetch, was 216 (`exploration/ptx/layout_only_16_8.ptx`). Registers went
from 80 to 122 (`ncu_*.txt`), which at 8 warps is two CTAs an SM.

## Results

Two runs, `cupti_1.log` and `cupti_2.log`, each started once the GPU's mean
utilisation over 10 samples was at most 5% with no game running: 4.0% and
3.4% (`queue.log`, `queue_2.log`; `env.txt` and `env_2.txt` hold the single
reading taken just before each, 6% and 7%). The
utilisation printed per round in the logs is read straight after the
round's own warm-up matmuls, so it shows this benchmark's load, not
another process's. The SM clock was 2520-2790 MHz in every round, 2760-2775
in 62 of 84. Every Y kernel's output is checked against an f32 reference
before it is timed; FlashInfer's, which the timing harnesses do not check,
is within 3.6e-4 of the same reference on all six shapes (`check_fi.py`,
`check_fi.log`). Each figure is the minimum over 7 interleaved rounds of
the mean over 50 back-to-back calls (`cupti_attn.py`); `tables.py` prints
the tables below from the logs.

Main kernel, device time in us, run 1 / run 2:

| case | old 16 splits x 8 warps | old 4 splits x 8 warps | new 16 splits x 8 warps | new 4 splits x 8 warps | new 4 splits x 4 warps | new 32 splits x 4 warps | FlashInfer |
|---|---|---|---|---|---|---|---|
| b1c1024 | 5.90 / 5.33 | 11.83 / 12.26 | 4.92 / 4.90 | 6.10 / 6.19 | 8.52 / 8.24 | 4.99 / 4.90 | 6.59 / 6.03 |
| b1c4096 | 14.33 / 14.57 | 40.90 / 41.63 | 9.19 / 9.53 | 16.14 / 16.47 | 26.57 / 26.39 | 9.77 / 9.88 | 10.98 / 11.40 |
| b8c1024 | 27.58 / 27.69 | 27.02 / 26.60 | 29.86 / 28.99 | 17.96 / 17.24 | 15.82 / 15.76 | 26.49 / 26.05 | 19.50 / 19.36 |
| b8c4096 | 197.00 / 201.42 | 213.22 / 220.83 | 199.41 / 197.19 | 189.93 / 184.91 | 183.89 / 183.97 | 196.14 / 195.27 | 186.01 / 186.36 |
| b32c1024 | 206.77 / 200.98 | 194.88 / 194.35 | 203.21 / 211.66 | 189.55 / 193.95 | 187.13 / 192.01 | 221.66 / 215.78 | 183.40 / 182.71 |
| b32c4096 | 889.14 / 895.02 | 902.64 / 905.82 | 894.64 / 882.84 | 875.09 / 865.23 | 859.77 / 875.16 | 911.25 / 899.11 | 855.49 / 865.24 |

Whole call (`span`: first kernel start to last kernel end, so Y's gap
between its two launches is in it; FlashInfer also launches two kernels).
The README's configurations, 16 splits at batch 1 and 4 at batch 8 and 32,
8 warps; and 4 splits x 4 warps. FlashInfer over Y, >1 is Y faster:

| case | old | new | new 4x4 | FlashInfer | FI / old | FI / new | FI / new 4x4 |
|---|---|---|---|---|---|---|---|
| b1c1024 | 9.16 / 9.49 | 9.28 / 8.87 | 12.29 / 11.71 | 11.00 / 11.59 | 1.20x / 1.22x | 1.19x / 1.31x | 0.90x / 0.99x |
| b1c4096 | 18.21 / 18.38 | 13.22 / 13.64 | 29.78 / 29.37 | 14.05 / 14.45 | 0.77x / 0.79x | 1.06x / 1.06x | 0.47x / 0.49x |
| b8c1024 | 29.52 / 29.19 | 20.84 / 19.70 | 18.67 / 19.02 | 22.81 / 22.62 | 0.77x / 0.77x | 1.09x / 1.15x | 1.22x / 1.19x |
| b8c4096 | 216.16 / 223.79 | 193.32 / 188.38 | 186.93 / 186.95 | 189.78 / 193.79 | 0.88x / 0.87x | 0.98x / 1.03x | 1.02x / 1.04x |
| b32c1024 | 199.48 / 199.05 | 197.71 / 198.62 | 194.11 / 196.47 | 189.28 / 188.58 | 0.95x / 0.95x | 0.96x / 0.95x | 0.98x / 0.96x |
| b32c4096 | 908.30 / 911.52 | 881.07 / 871.17 | 865.41 / 881.07 | 861.47 / 871.18 | 0.95x / 0.96x | 0.98x / 1.00x | 1.00x / 0.99x |

The KV cache is 4, 17 and 34 MB in the first three rows, inside the L2, and
134-537 MB in the last three. In the first three the new kernel is
1.06-1.31x FlashInfer's speed where the old one was 0.77-1.22x. In the last
three, which are DRAM-bound, FlashInfer's own figures move by up to 2% between
the two runs here and by up to 4% in the README benchmark's, and Y's are
0.95-1.04x of it; at batch 32 / ctx 1024 Y is behind in all four of its
comparisons (0.95-0.98x), by 2-5%, about the size of that noise.

Instructions (Nsight Compute `smsp__inst_executed.sum`, main kernel) against
run 1's device time, at the SM clock of 2760 MHz most of its rounds ran at;
66 SMs x 4 schedulers:

| case | kernel | warp instructions | registers | device time | instructions / cycle / scheduler |
|---|---|---|---|---|---|
| b1c4096 | old 16x8 | 5.33 M | 80 | 14.33 us | 0.511 |
| b1c4096 | new 16x8 | 3.51 M | 122 | 9.19 us | 0.524 |
| b1c4096 | FlashInfer | 4.28 M | 60 | 10.98 us | 0.535 |
| b8c1024 | old 4x8 | 10.66 M | 80 | 27.02 us | 0.542 |
| b8c1024 | new 4x8 | 7.01 M | 122 | 17.96 us | 0.536 |
| b8c1024 | new 4x4 | 6.44 M | 122 | 15.82 us | 0.559 |
| b8c1024 | FlashInfer | 8.56 M | 60 | 19.50 us | 0.603 |

In the same configuration the new kernel executes 0.66x the old one's
instructions at both shapes and issues them at the same rate: the time went
down with the instruction count, not through more overlap. Per stage, in
the exploratory runs on a busy GPU (`exploration/`), main kernel at batch 1
/ ctx 4096, 16 x 8: the 16-lane layout 14.21 to 11.24 us
(`cupti_lpt_1.log`), the lazy rescale 10.96 to 10.50 (`cupti_lazy_1.log`),
the prefetch 10.60 to 9.68 (`cupti_pf_1.log`). Each pair is from one
contended run, and none is used in the tables above.

The README's own benchmark (`readme_bench_{old,new}.log` and `_2.log`, the
same snapshots, CUDA-event times of the whole call, the better of its two
committed split kernels, 16 x 8 and 4 x 8) gives the split kernel 1.11x / 1.11x, 1.15x / 1.13x and
1.13x / 1.13x FlashInfer on the first three rows (old: 1.08x / 1.03x, 0.81x
/ 0.81x, 0.78x / 0.76x) and 0.98x / 1.01x, 0.99x / 0.95x, 0.99x / 1.03x on
the last three (old: 0.94-0.99x).

Four warps beat eight at 4 splits for batch 8 / ctx 1024 by 1.14x / 1.09x
(main kernel); the committed 4-split kernel keeps 8 warps, and this record
does not change the warp counts the benchmark compiles.

## Correctness

`python/y_lang/tests/test_paged_decode_attention.py` runs 11 configurations
(both shapes; head_dim 64/128/256; GQA 1:1, 4:1, 8:1; page sizes 1, 7, 12,
16, 32; 4, 5, 6, 8 and 32 warps; 3, 4, 8 and 16 splits) on two length sets
(14 ragged lengths from 0 to 1000, and 4095/4096) and six score shapes
(random; scores rising 12 per token, so a rescale every iteration; rising
0.5; falling 0.5; one spike of +40; Q scaled by 40), each row against a
float64 reference within 3e-3 relative L2, Out and the split scratch
NaN-filled first. A second test runs four configurations under
compute-sanitizer memcheck with 512 bytes of padding after every
allocation, each tensor its own allocation. The kernel of 7e08614 passes the
same test.

`mutate.log`: 33 mutants of the new loop (`mutate_attn.py`), each built and
run against both tests. 28 fail them, by a wrong row, a non-finite row, an
illegal or misaligned address (three report it as the next module load's
error, 700 or 716), or a hang (counted after 300 s). Five pass, and each is a correct kernel: an unpredicated store of a
token group's state that every group then holds identically
(`smem_store_unpredicated`); a lane-local rescale decision, which is
uniform within a token group anyway (`vote_is_lane_local`); a page-table
reload every token (`page_cache_not_updated`); the 32-lane layout
everywhere (`policy_always_32_lanes`); and no prologue skip for a warp with
no tokens, whose clamped token is still a real one, so the fetch is wasted
but in bounds, as memcheck agrees (`prologue_skip_dropped`). That run listed
the last as unexpected; it was moved to the expected list after this
analysis, and the code comment beside the skip now says it only saves the
fetch. The layout policy is pinned by the Rust unit test
`loop_layout_follows_the_register_budget` instead: with
`policy_always_32_lanes` applied it and
`split_shape_merges_gqa_heads_and_emits_both_entries` fail, and both pass
once it is restored.

## Files

- `cupti_attn.py`: the timing (CUPTI device times, interleaved rounds,
  exactness first); `run_all.sh`, `run_2.sh`, `queue.sh`, `queue_2.sh`: how
  the two runs were started, from snapshots of both compilers so later edits
  could not change what was measured (`env*.txt` has their hashes).
  `tables.py`: the tables above. `check_fi.py`: FlashInfer's output.
- `ncu_all.sh`, `prof_attn.py`, `ncu_tab.py`: the Nsight Compute counts;
  `ncu/` holds the reports, `ncu_*.txt` the extracted rows.
- `sass_loop.py`, `sass_loop_census.txt`: the SASS census and the GQA 8:1
  register counts.
- `ptx/`: every timed kernel (`old_*` from 7e08614, `new_*` from this change)
  and the sources and pinned profile they were compiled from.
- `exploration/`: the runs that chose the design, ALL on a busy GPU (the
  desktop compositor at 25-57% at idle clocks, and Roblox from
  `cupti_pf_1.log` on); comparisons within a run are interleaved and ranked
  by minimum, and no figure in the tables above comes from them. `cupti_gen.py`
  timed other head dims and GQA ratios; `exploration/ptx/` holds the
  layout-only kernel and the GQA 8:1 kernels at 8 and 4 elements a lane
  (194 and 121 registers; with 16 splits at batch 8 / ctx 1024 the first
  took 1.47x the second's time, `cupti_gen_1.log`).
