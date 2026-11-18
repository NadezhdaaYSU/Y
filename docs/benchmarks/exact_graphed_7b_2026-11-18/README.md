# 7B with graphed decode: a stopped, partial run (2026-11-18)

Not a result to quote against vLLM. The run was stopped at the user's
request after two of its six processes (the GPU is the user's desktop GPU,
and a 7B run takes most of its memory).

`run.sh` was to time Qwen2.5-7B-Instruct on the exact path only
(`Y_THR_ARMS=exact`, `--low-memory`, compiled, batch 32, 64 new tokens, best
of 5), one configuration per process, order A B C C B A:

- A: the 2026-11-08 setting (`Y_EXACT_GEMM=y Y_EXACT_V_INT8=1`, q/k/v
  merged), `--loop greedy`, as a bridge to that day's vLLM FP8 runs;
- B: the same, `--loop graphed`;
- C: every switch on top of Y GEMM, `--loop graphed`.

What finished (`runs/`, commit 7897048, Discord and a browser open, the GPU
24 to 50% busy from the desktop before each process):

| Process | tok/s/seq | Token checks |
|---|---|---|
| A, greedy (16:04) | 49.6 | greedy == generate on all 32 rows |
| B, graphed (16:16) | 52.3 | greedy == generate, graphed == greedy, all 32 rows |

`runs/stopped_C_graphed_every_switch.162458.log` is the cut-off third
process; it has no timing.

What this shows and what it does not:

- With Y's GEMM at 7B, each decode step as one CUDA graph gives the greedy
  loop's tokens on all 32 rows.
- B was 1.05x faster than A. These are two processes, so this carries the
  drift between processes.
- The bridge failed: A gave 49.6 tok/s/seq, against 54.5 for the same
  setting on 2026-11-08 (`../exact_greedy_loop_7b_2026-11-08/`), when only
  VS Code was open. So the machine was not in that day's state, and B
  cannot be set against that day's vLLM FP8 figures (58.4 and 59.1).
- vLLM was not run: it asks for 88% of the card's memory, and the desktop
  held 2.2 GB.
