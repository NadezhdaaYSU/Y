# Decode as one CUDA graph a step (2026-11-18)

## Why

A compiled decode step of the exact path launches about 600 kernels. On the
RTX 4070 Ti SUPER, with every switch on, its GPU work took 2.61 ms a step,
21% less than stock bf16 (3.32 ms). The whole step still took longer,
because the host could not launch the kernels as fast as the GPU ran them.
A CUDA graph launches a whole step at once.

Hugging Face's own route to CUDA graphs (its static cache) is blocked for the
exact path. So `python/y_lang/exact/graphed.py` (`GraphedGreedy`) builds a
decode step from the model's own modules, with shapes that never change:

- a cache of fixed buffers per layer, filled by the prefill and then written
  one row a step at a position held on the GPU;
- an attention mask built on the GPU from that position: a row sees its own
  prompt (not its left padding) and the tokens decoded so far, and nothing in
  the spare rows;
- the next token, the positions and the step counter, all moved forward on
  the GPU inside the graph.

The step is compiled once, recorded once per (batch, prompt length, new
tokens) and then replayed. `python/y_lang/tests/test_exact_graphed.py`
checks the tokens against `y_lang.exact.greedy`, bit for bit, with float
cache rows, packed rows and packed rows read by Y's attention kernel, with
and without left padding. Two controls (a mask that lets the step see the
padding, positions that do not move) must change the tokens, and do.

`tools/exact_throughput.py --loop graphed` runs every arm this way. Stock
gets the same graph, built from its own modules and compiled with PyTorch's
default Inductor settings. So this compares graphs with graphs, not the
exact path with graphs against stock without them.

Before timing, each run checks, for both arms, on all 32 rows: the greedy
loop gives `generate()`'s tokens, and the graphed loop gives the greedy
loop's tokens. If either check fails, the run stops and times nothing.

## Setup

- RTX 4070 Ti SUPER, the user's desktop. Discord and a browser were open;
  the GPU was 9 to 33% busy from the desktop before each run.
- Qwen2.5-0.5B-Instruct at 7ae5576, batch 32, 64 new tokens, compiled,
  best of 5, prefill included. Stock and exact are timed in turns inside one
  process.
- Commit 47913a7 plus the uncommitted files listed in `runs/commit.txt`
  (`graphed.py`, its test, and the `--loop graphed` change).
  liby.so sha256 `713be562...` (full hash in `runs/commit.txt`).
- `D`: exact at its defaults. `E`: exact with every switch
  (`Y_EXACT_V_INT8=1 Y_EXACT_NORM=triton Y_EXACT_ATTN=y Y_EXACT_KV_PACK=inplace`).
- `run.sh runs` (order D E E D) and `run.sh runs_greedy greedy "E D"`.

## Results, graphed loop (`runs/`)

Tokens per second per sequence, and the time of exact over the time of stock.

| Run | Stock graphed | Exact graphed | Exact time / stock time |
|---|---|---|---|
| D, defaults (15:02) | 295.3 | 296.0 | 1.00 |
| E, every switch (15:11) | 281.8 | 356.1 | 0.79 (1.26x faster) |
| E, every switch (15:21) | 297.5 | 378.1 | 0.79 (1.27x faster) |
| D, defaults (15:32) | 282.8 | 283.4 | 1.00 |

All four runs passed both token checks for both arms. The level moves by
about 5% from one process to the next (stock 282 to 298); the ratio inside
a process did not move (1.00 and 1.00; 0.79 and 0.79).

The logs print "0.79x slower". The tool had that word written in, so it
said "slower" for any ratio. The number is a time ratio: 0.79x the time is 1.26x
faster. The tool now prints `0.79x the time (1.26x faster)` (`time_ratio` in
`tools/exact_throughput.py`), and these logs are kept as they were written.

## Results, greedy loop (`runs_greedy/`)

The same runs without graphs: `--loop greedy` (`y_lang.exact.greedy`, no
sync between steps), same afternoon, same settings.

| Run | Stock greedy | Exact greedy | Exact time / stock time |
|---|---|---|---|
| E, every switch (15:41) | 281.5 | 294.6 | 0.96 (1.05x faster) |
| D, defaults (15:51) | 274.2 | 257.3 | 1.07 |

Both runs passed the greedy == `generate()` check for both arms. During the
D run the user's screen recorder (`gpu-screen-recorder`, which uses the
video encoder) was on; its first saved clip is from 15:53:48. Both arms ran
under it in turns. The ratio, 1.07, is the one recorded before with
`generate()` (`../exact_speed_05b_2026-11-17/`). The run was not repeated:
after the recorder stopped, Discord was using the video encoder and the GPU
was 40% busy, which is no cleaner. `run.sh` now lists the recorder among
the other programs it reports.

Graphs against no graphs, arm by arm: these numbers come from different
processes, so they carry the 5% drift between processes. With that in mind,
the graph made stock 0 to 8% faster (281.5 to 281.8 and 297.5 in the E
runs; 274.2 to 295.3 and 282.8 in the D runs), exact at its defaults 10 to
15% (257.3 to 296.0 and 283.4), and exact with every switch 21 to 28%
(294.6 to 356.1 and 378.1). The exact path gains more because more
of its step was host time.

## What this means

- With every switch, the exact path is now faster than stock, both with
  graphs, by 1.26x to 1.27x, and its tokens are the greedy loop's tokens
  bit for bit. The host cost that hid its smaller GPU work is gone.
- At its defaults, the exact path ties stock with graphs (1.00x). Its GPU
  work a step is about stock's, so removing the host cost helps both alike.
- Every switch stays opt-in for now. On the T4 every switch was 1.36x slower
  than stock with the old loop, because its `torch.library` ops cost 50 to
  92 us of host time a call on Kaggle's 2 GHz Xeon. A graph removes that host
  cost, so the T4 jobs `y-t4-graphed-e` (defaults) and `y-t4-graphed-f`
  (every switch) test it, each with the graphed and the greedy loop on one
  machine.

## Limits

- `GraphedGreedy` does greedy decoding only, for one input shape per
  recording, for models laid out as transformers' Qwen2, without
  sliding-window layers, with "eager" or "sdpa" attention. It refuses other
  cases instead of guessing.
- The prefill is not graphed. It runs as before and is included in the times.
- The first call for a shape compiles and records; that time is not in the
  table (each arm is warmed up before timing).
- Only 0.5B on one GPU so far. A 7B run against vLLM (which also uses CUDA
  graphs) has not been done.

## Prototype

Before `graphed.py`, `prototype/graph_both.py` gave (same GPU and model,
tokens per second per sequence):

| Arm | Equal rows | Left-padded rows |
|---|---|---|
| Stock bf16, greedy loop | 287.8 | 281.5 |
| Stock bf16, CUDA graph | 308.1 | 298.3 |
| Exact (every switch), greedy loop | 294.9 | 288.8 |
| Exact (every switch), CUDA graph | 383.9 | 377.7 |

`prototype/` holds those scratch scripts.

## Files

- `run.sh`: the timing runs. `runs/`, `runs_console.txt`: the graphed loop.
  `runs_greedy/`, `runs_greedy_console.txt`: the greedy loop.
- `prototype/`: the first scripts.
