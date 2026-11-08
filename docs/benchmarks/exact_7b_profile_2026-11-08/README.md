# Where Y's 7B decode step goes, beside vLLM's GEMMs (2026-11-08)

Why Y's exact path ran at 0.89x vLLM FP8 at 7B (`../vllm_fp8_7b_2026-11-08/`).
RTX 4070 Ti SUPER (sm_89), torch 2.13.0+cu130; vLLM 0.21.0 (torch 2.11).
Measured before q/k/v merging and the greedy loop
(`../exact_merged_7b_2026-11-08/`, `../exact_greedy_loop_7b_2026-11-08/`).

## The step, under `generate`

`profile_step.py` (`y_vint8.log`, `y_vint8.json`): Y GEMM + int8 V cache,
compiled, batch 32, one 16-token `generate` under torch.profiler. Wall
373.1 ms, GPU busy 339.5 ms (91%). `y_int8_mm` is 70.9% of the busy time
(3152 calls: 197 a forward). One device-to-host copy a token (19 in all):
`generate`'s finished-rows check. The profiler records every CPU-side op, so
its idle time overstates the timed runs' (the loop that removed the copy
gained 3%, not the ~10% this idle time suggested).

## The GEMMs alone

`gemm_shapes.py`: each decode linear at M = 32, weights from DRAM (copies
cycled past 4x the L2), CUDA events over 200 calls, best of 7, outputs
checked against a float64 product. Two rounds each (`gemm_y_*.json`, run in
this repo's venv; `gemm_vllm_*.json`, in vLLM's). Microseconds per call:

| shape (N x K) | Y int8_mm | cuBLASLt int8 | vLLM CUTLASS int8 | vLLM CUTLASS FP8 |
|---|---|---|---|---|
| q / o (3584 x 3584) | 22.7, 22.4 | 23.2, 23.2 | 27.5, 27.5 | 27.8, 27.8 |
| k / v (512 x 3584) | 6.9, 6.9 | 13.4, 13.3 | 9.5, 9.5 | 10.8, 10.8 |
| gate / up (18944 x 3584) | 121.1, 121.2 | 121.7, 122.4 | 124.0, 122.3 | 131.1, 130.9 |
| down (3584 x 18944) | 121.5, 122.5 | 129.3, 128.1 | 137.3, 137.5 | 136.6, 136.9 |
| lm_head (152064 x 3584) | 972.9, 977.7 | 957.1, 979.0 | 952.5, 957.4 | 951.5, 956.5 |
| q/k/v merged (4608 x 3584) | 28.8, 28.8 | 29.2, 29.2 | 39.1, 38.7 | 40.2, 40.2 |
| gate+up merged (37888 x 3584) | 242.3, 245.1 | 246.0, 244.6 | 242.3, 240.0 | 248.8, 245.0 |

Per decode step, as each engine calls them (round 1): Y unmerged 12.8 ms;
Y with q/k/v merged 12.6 ms; vLLM FP8 merged (q/k/v and gate+up, as vLLM
runs them) 13.7 ms. **Y's GEMMs are about 1 ms a step faster than vLLM's**,
so the gap is everything else: Y's other kernels (3.7 ms a step, see
`../exact_greedy_loop_7b_2026-11-08/`) and, under `generate`, idle time.

## CUDA graphs

`graphs_y_vint8.log`: `exact_throughput --graphs` (static cache +
`reduce-overhead`) at 7B dumped core in HF's `StaticCache.update`
(`cumulative_length.add_`, an in-place update inside the captured region),
as it did at 0.5B on 2026-08-18. Still blocked.
