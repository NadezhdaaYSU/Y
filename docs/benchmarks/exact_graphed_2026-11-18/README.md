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

## Tesla T4 (Kaggle)

`make_t4_graphed_job.py` (copied here), Tesla T4 with a 4-core Intel Xeon
at 2.00 GHz, torch 2.13.0+cu126, transformers 5.15.0, stock in float16,
liby.so from the y-lang-wheel dataset (sha256 `5c26fe725eb35a5d...`).
Source: the files of 6abf6cc (uploaded just before that commit), through
the y-lang-src dataset. Each job ran on one Kaggle machine, one process per
loop. Same settings as above.

| Job, setup, loop | Stock fp16 | Exact | Exact time / stock time |
|---|---|---|---|
| e, defaults, graphed | 88.6 | 105.7 | 0.84 (1.19x faster) |
| e, defaults, greedy | 93.5 | 93.4 | 1.00 |
| f, every switch, greedy | 90.0 | 69.2 | 1.30 |
| f, every switch, graphed | 83.8 | 129.4 | 0.65 (1.54x faster) |

Every process passed the token checks for both arms (greedy == `generate()`,
and graphed == greedy, on all 32 rows). Files: `t4_job_e/`, `t4_job_f/`.

- On the T4 the graph makes stock slower, by 5% (93.5 to 88.6) and 7% (90.0
  to 83.8). So stock's fastest loop there is greedy, and graphed exact
  against graphed stock flatters exact. Why stock loses was not measured. A
  guess: stock's step on the T4 waits on its GPU work, not on the host, so
  the graph saves little, while the graphed step attends over every row of
  its buffer (prompt plus 64, rounded up to 32) under a mask, and greedy
  only over the rows so far. Job h (next section) supports this: on the T4
  each spare row costs stock about 20 us a step, which accounts for most of
  the gap.
- Against stock's fastest loop, graphed exact is 1.13x faster at its
  defaults (105.7 against 93.5) and 1.44x faster with every switch (129.4
  against 90.0). Each pair is two processes on one machine. Job g times
  them in one process (below): 1.13x and 1.43x.
- With every switch, the graph makes the exact path 1.87x faster on the T4
  (69.2 to 129.4). Its `torch.library` ops cost 50 to 92 us of host time a
  call on this Xeon (`../exact_t4_profile_2026-11-17/`, job 2c), and the
  graph replays them without the host.

### Job g: graphed exact against greedy stock, in one process

`--loop graphed --stock-loop greedy` (61f08de): each arm runs its own
fastest loop on the T4, in turns inside one process. Another Kaggle machine
of the same kind (T4, 4-core Xeon at 2.00 GHz). Files: `t4_job_g/`.

| Setup | Stock fp16, greedy | Exact, graphed | Exact time / stock time |
|---|---|---|---|
| Defaults | 91.1 | 102.5 | 0.89 (1.13x faster) |
| Every switch | 92.7 | 132.8 | 0.70 (1.43x faster) |

Both processes passed the token checks: stock greedy == `generate()`, exact
greedy == `generate()` and exact graphed == exact greedy, on all 32 rows.
This matches the estimates from jobs e and f (1.13x and 1.44x).

## What the spare rows of the buffer cost

`capacity_sweep.py` times the graphed decode steps alone (graph replays,
no prefill) with the buffer at 92 rows (the prompt's 28 plus 64, no
rounding), 96 (what `GraphedGreedy` uses), 128 and 256. A step sees 60 rows
on average. RTX 4070 Ti SUPER, 182668c, batch 32, 64 new tokens, compiled,
best of 5 rounds, the sizes and arms in turns, no game running before or
after. Milliseconds per decode step (`capacity_sweep/`):

| Buffer rows | Stock bf16 | Exact, defaults | Exact, every switch |
|---|---|---|---|
| 92 | 3.084 | 3.087 | 2.251 |
| 96 | 3.106 | 2.986 | 2.251 |
| 128 | 3.229 | 3.115 | 2.304 |
| 256 | 4.524 | 3.700 | 2.521 |

Each arm gave the same tokens at every size (stock too, here). Going from
96 to 128 rows costs 2% to 4%, and 96 to 256 costs 12% (every switch) to
46% (stock). So at 64 new tokens, one graph per 32-row length bucket would
save a few percent at most on this GPU, and is not built. It would matter
more for long outputs.

The same sweep on the T4 (job h, `t4_job_h/`, stock in float16, another
Kaggle machine of the same kind):

| Buffer rows | Stock fp16 | Exact, defaults | Exact, every switch |
|---|---|---|---|
| 92 | 11.373 | 9.186 | 6.701 |
| 96 | 11.419 | 8.968 | 6.720 |
| 128 | 12.078 | 9.307 | 6.843 |
| 256 | 15.666 | 12.485 | 7.438 |

Same tokens at every size, each arm. From 96 to 128 rows a row costs stock
about 21 us a step, the exact default about 11 us and every switch about
4 us (the exact default at 92 rows was slower than at 96, so 96 is the
start). The greedy loop attends over 29 to 92 rows (60 on average) where the
graphed step attends over 96, so for stock the spare rows come to about
0.6 ms a step if that cost holds below 92 rows (not measured). That is
about the gap between stock's graphed and greedy loops on the T4: 11.3
against 10.7 ms a token in job e, 11.9 against 11.1 in job f (prefill
included). So most of stock's loss under the graph is attention over the
spare rows (PyTorch's mem-efficient SDPA on the T4), and one graph per
length bucket would mostly help stock. For every switch it would save
about 2%. Not built: the T4 comparison already uses stock's faster loop. The first try of this sweep failed to link
(`torch.compile` reached torch through the path with a space); the command
in the script's docstring is the one that works.

## Where a graphed step's GPU time goes

`profile_step.py` records a `GraphedGreedy` and profiles 16 decode steps run
through the compiled step directly (the kernels the graph replays). RTX 4070
Ti SUPER, aa7a8b8, batch 32, no game before or after (`profile_step/`).
Milliseconds a step:

| | Stock bf16 | Exact, defaults | Exact, every switch |
|---|---|---|---|
| Graphed step, wall | 3.251 | 3.144 | 2.427 |
| GPU kernels | 3.223 (470) | 3.030 (512) | 2.364 (502) |
| Linear layers | 2.274 | 1.303 | 1.313 |
| Attention | 0.771 | 0.393 or more | 0.369 |
| Inductor's fused kernels | 0.173 | 1.328 | 0.515 |
| Other (Triton norm) | 0.006 | 0.006 | 0.167 |

(Kernel counts in brackets.) The logs' own group totals put three
kernels under GEMM that belong to attention: PyTorch's mem-efficient SDPA
(`fmha_cutlassF`, 489 us, stock) and two cuBLAS strided-batched float
kernels (263 and 130 us, the exact default's Q.K^T and P.V). The table
moves them; the script now groups them as attention. The exact default's
attention may hold more than 0.393 ms, in fused kernels not listed.

- The wall time is within 4% of the GPU time: the graph left the GPU no
  idle time to speak of.
- Every switch saves the exact default's fused quantise kernels (1.328 to
  0.515 ms) and its float attention (0.393 to 0.369 ms, now Y's
  `exact_decode_attention`), and pays 0.167 ms for the Triton norm.
- The int8 linear layers move about 494 MB of weights a step (358 MB in
  the layers, 136 MB in `lm_head`). At 1.313 ms that is 376 GB/s, 56% of
  the card's 672 GB/s; `lm_head` alone reaches about 78%
  (`profile_step/w8a8_shape_sweep_m32.log`, `tools/w8a8_shape_sweep.py`),
  so the per-layer GEMMs run near 51% in place. Stock's bf16 linears move
  988 MB in 2.274 ms, 65%. If the layer GEMMs reached `lm_head`'s 78%, the
  step would lose about 0.37 ms (16%). That is the largest lever left on
  this GPU. The T4 does not run this kernel (its int8 GEMM is
  `torch._int_mm`); job i (`y-t4-graphed-i`) profiles the T4's step.

## What this means

- With every switch, the exact path is now faster than stock on both GPUs:
  with both arms graphed, 1.26x to 1.27x on the RTX 4070 Ti SUPER (where
  graphed is also stock's fastest loop, by 0 to 8%) and 1.54x on the T4;
  against stock's fastest loop on the T4 (greedy), 1.43x in one process
  (job g). Its tokens are the greedy loop's tokens bit for bit. The host
  cost that hid its smaller GPU work is gone.
- At its defaults, the exact path ties stock with graphs on the 4070 Ti
  (1.00x) and is faster on the T4 (1.19x against graphed stock, 1.13x
  against greedy stock in one process).
- Every switch is still opt-in. With graphs it is the faster setting on
  both GPUs; without graphs it is slower on the T4 (1.30x). The switches do
  not change the tokens (same 64-token reference and hash, each alone and
  all together, eager and compiled:
  `../exact_speed_05b_2026-11-17/`), so turning them on when decoding is
  graphed would be a choice about speed only. Not done here.

## Limits

- `GraphedGreedy` does greedy decoding only, for one input shape per
  recording, for models laid out as transformers' Qwen2, without
  sliding-window layers, with "eager" or "sdpa" attention. It refuses other
  cases instead of guessing.
- The prefill is not graphed. It runs as before and is included in the times.
- The first call for a shape compiles and records; that time is not in the
  table (each arm is warmed up before timing).
- Only 0.5B so far, on two GPUs. A 7B run against vLLM (which also uses
  CUDA graphs) has not been done: the local one was stopped after two
  processes (`../exact_graphed_7b_2026-11-18/`, partial).

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
- `t4_job_e/` to `t4_job_h/`: the T4 logs (each loop's log, and the
  notebook's console as `job_*_console.log`). `make_t4_graphed_job.py`
  writes those Kaggle jobs.
- `capacity_sweep.py`, `capacity_sweep/`: the buffer-size sweep.
- `profile_step.py`, `profile_step/`: the per-kernel profile of a graphed
  step, and the GEMM shape sweep's output.
- `prototype/`: the first scripts.
