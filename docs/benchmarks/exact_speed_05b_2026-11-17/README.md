# Qwen2.5-0.5B decode speed on the current code (2026-11-17)

How much slower is the exact path than stock, after the integer norm sum and
Y's kernels? The last figures were from 2026-11-02 (1.11x). This record also
found two bugs in `tools/exact_throughput.py` that affected every earlier run
that timed the fixed-order control next to the exact arm. They are fixed in
67b8fdc.

**Correction (2026-11-18).** A third bug in the same tool made some prefills
compute logits for every prompt position, so the tables under "Results"
below are off by 1% to 2%. The corrected RTX 4070 Ti SUPER figures are in
"Results after the third fix". On the Tesla T4, the exact path was also
slow for a reason in the code itself, now fixed (56847c5). See
`docs/benchmarks/exact_t4_profile_2026-11-17`.

## Results after the third fix (2026-11-18)

`generate()` passes `logits_to_keep=1` only when it can see that argument
in the model's `forward`. The tool's wrappers around `forward` hid it, so
the prefill computed the logits of all 28 prompt positions, not one per
sequence. `run.sh` ran at 90cd97f, where only stock was wrapped, so stock
did the extra work. `run_fixed.sh` ran at 67b8fdc, where both arms were
wrapped. 328c6aa fixes it, and `build()` now refuses an arm that hides
the argument.

`run_after_logits_fix.sh` ran all four setups again at 79c5cca and
f26a0a0 (the tool is the same in both). The order was A B, then a stop
for the GPU owner (`after_logits_fix/stopped_C_every_switch.010924.log`
is that partial run), then C D D C B A. Same `liby.so`. Steam (no game),
Discord and Chromium were open. Logs in `after_logits_fix/`, console in
`after_logits_fix_console.txt`.

| Setup | Stock | Exact | Exact against stock | Before the fix |
|---|---|---|---|---|
| A: default (127 levels), run 1 | 241.3 | 225.0 | 1.07x slower | 1.05x |
| A: default, run 2 | 246.8 | 231.5 | 1.07x slower | 1.05x |
| B: 511 levels, run 1 | 241.6 | 219.2 | 1.10x slower | 1.10x |
| B: 511 levels, run 2 | 244.1 | 221.3 | 1.10x slower | 1.10x |
| C: every switch, run 1 | 250.5 | 251.1 | 1.00x | 0.98x |
| C: every switch, run 2 | 245.5 | 249.1 | 0.99x (exact faster) | 0.99x |
| D: every switch and Y's GEMM, run 1 | 241.1 | 231.8 | 1.04x slower | 1.02x |
| D: every switch and Y's GEMM, run 2 | 244.7 | 234.5 | 1.04x slower | 1.04x |

Every exact run at 127 levels gives the 64-token reference with 40
distinct tokens, and at 511 levels with 48, as before. The fixed-order
control ran at 17.9 and 18.6 compiled, 13.5x and 13.3x slower than stock.

## Setup

- RTX 4070 Ti SUPER (sm_89), driver 615.78.08, 285 W limit. This GPU also
  runs the desktop. During the runs the Steam client (no game), Discord and
  Chromium were open. GPU use at the start of each run is in the console
  files.
- Qwen2.5-0.5B-Instruct at 7ae5576. `generate()`, greedy, batch 32, 64 new
  tokens, best of 5. The arms take turns inside each process. Stock is bf16
  with SDPA.
- `run.sh` (90cd97f): setups B, C and D, one process each, in the order
  A B C D D C B A. Its setup A runs are invalid (see below).
- `run_fixed.sh`: setup A again with the fixed tool, compiled twice, then
  once eager. The first of these ran with the fix as an uncommitted change.
  Its code was the same as 67b8fdc, which only adds a reworded comment. The
  other two ran on 67b8fdc.
- `liby.so` sha256 713be562…, built at 258e49f (`commit.txt`).

## Results (before the third fix)

Tokens per second per sequence. The ratio is stock time over exact time,
in the same process. These runs have the `logits_to_keep` bug described
above.

| Setup | Stock | Exact | Exact against stock |
|---|---|---|---|
| A: default (127 levels), run 1 | 239.3 | 227.0 | 1.05x slower |
| A: default, run 2 | 247.0 | 235.6 | 1.05x slower |
| B: 511 levels, run 1 | 240.2 | 219.2 | 1.10x slower |
| B: 511 levels, run 2 | 244.9 | 223.7 | 1.10x slower |
| C: every switch, run 1 | 241.9 | 247.1 | 0.98x (exact faster) |
| C: every switch, run 2 | 245.9 | 249.0 | 0.99x (exact faster) |
| D: every switch and Y's GEMM, run 1 | 244.7 | 238.8 | 1.02x slower |
| D: every switch and Y's GEMM, run 2 | 244.2 | 234.5 | 1.04x slower |

"Every switch" is `Y_EXACT_V_INT8=1 Y_EXACT_NORM=triton Y_EXACT_ATTN=y
Y_EXACT_KV_PACK=inplace`. D adds `Y_EXACT_GEMM=y`, the 7B configuration.
The default GEMM here is Triton's.

The fixed-order fp32 control (setup A):

| Mode | Stock | Exact | Control | Control against stock |
|---|---|---|---|---|
| Compiled, run 1 | 239.3 | 227.0 | 17.7 | 13.5x slower |
| Compiled, run 2 | 247.0 | 235.6 | 18.7 | 13.2x slower |
| Eager | 177.2 | 25.5 | 37.4 | 4.7x slower |

The control is fastest eager. Its eager 37.4 is 6.5x slower than compiled
stock (243 on average) and 6.2x slower than compiled exact (231). Those two
ratios compare different processes.

## Tesla T4 (Kaggle)

These runs had the `logits_to_keep` bug, and the exact path's int8 weights
in the order that kept `torch._int_mm` off the T4's tensor cores. With both
fixed, the default exact path was 1.03x slower than stock float16 on one T4
machine and 0.98x on another. See `docs/benchmarks/exact_t4_profile_2026-11-17`.

Two Kaggle jobs on a Tesla T4 (sm_75, driver 580.178.04), made by
`t4/make_t4_speed_job.py`. Same protocol, but the stock arm is float16,
because a T4 has no bf16 tensor cores (`--stock-dtype float16`). Triton
cannot build the int8 dot below sm_80, so the exact path's linear layers
fall back to `torch._int_mm`. `liby.so` came from the wheel dataset
(sha256 5c26fe725eb35a5d…). Job 1 ran the tool at 90cd97f and job 2 at
67b8fdc. Each setup ran once, and the two jobs got different T4 machines
(stock 64 against 75 tokens/s per sequence), so compare ratios, not speeds.

| Setup | Stock fp16 | Exact | Exact against stock |
|---|---|---|---|
| Default (127 levels), job 2 | 75.1 | 50.7 | 1.48x slower |
| 511 levels, job 1 | 64.1 | 32.2 | 1.99x slower |
| Every switch, job 2 | 74.5 | 54.4 | 1.37x slower |

The control ran at 7.0 compiled (job 2, 10.7x slower than stock) and 6.4
eager (job 1, 10.0x slower). Job 1's control figure is valid: the control
was built last, so it ran on its own attention, and the old tool always
ran it eager. Job 1's exact figure in that run is not valid (bug 1), so
that log is `t4/invalid_job1_A_127_control.log`.

Job 1's every-switch run failed (`t4/failed_job1_C_every_switch.log`). The
source tarball kept `python/y_lang/exact/exact_pv.ysu` as a symlink, and
Kaggle unpacked it as an empty file. Y refused to build a module with no
kernel, and the exact path refused to fall back. Job 2's tarball stores
both copies as real files, and its notebook stops if any `.ysu` file is
empty.

On the T4 the exact arm's reference has 40 distinct tokens, as on the RTX
4070 Ti SUPER. Stock float16 there has 45.

## The switches do not change the tokens

Every exact run gives the same 64-token reference: 40 distinct tokens and the
same hash, eager and compiled, with each switch alone and all together
(checked directly on 2026-11-17). Setup B (511 levels) is a different model,
so its text differs (48 distinct tokens).

## Two bugs in the tool, and the invalid runs

`invalid_A_127_control.*.log` were timed before the fix. Do not quote them.

1. **The exact arm ran with the control's attention.** Qwen2 reads its
   attention from a module global each time it runs. The tool builds every
   arm first and then times them. Building the control last left that
   global pointing at the control's fixed-order float attention, so the
   "exact" arm used it. Its reference text had 51 distinct tokens instead of
   the exact path's 40. Each arm now puts its own attention in place around
   every call.
2. **The control was never compiled.** It was returned as
   `torch.compile(m)`. Calling `generate()` on that runs the original
   module's forward, which is not compiled: dynamo compiled no frame. It now
   goes through the same compile path as the other arms (3 graphs).

So every earlier "compiled" figure for the control was eager. This includes
Finding 11 in `docs/bit_identical_decode.md` (36.0 compiled, 37.7 eager) and
its claim that compiling does not help the control. Compiled for real, the
control is about twice as slow. We did not find out why. Both bugs were
there from the commit that added the control (8f26cee, 2026-08-22). Finding
11 timed the exact arm and the control in one process. With bug 1, at least
one of the two then runs with the wrong attention, whatever the build order.
So its exact figure (238.5) cannot be trusted either.

The batch-invariance results are not affected. `exact_ragged_batch.py` and
`exact_cross_device.py` build, run and free one arm at a time.
