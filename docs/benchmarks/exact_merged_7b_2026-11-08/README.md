# Merged projections at 7B (2026-11-08)

Qwen2.5-7B-Instruct, exact path on Y's int8 GEMM with the int8 V cache
(`Y_EXACT_GEMM=y Y_EXACT_V_INT8=1`), compiled, batch 32 x 64 new tokens,
`tools/exact_throughput.py`, best of 5. RTX 4070 Ti SUPER (sm_89), torch
2.13.0+cu130, transformers 5.15.0; vLLM 0.21.0 (torch 2.11) in its own venv.
Only VS Code was open (`run.out` lists the other GPU users it looked for, and
the GPU's utilisation before each process).

`y_lang.exact.model.merge_projections` replaces each layer's q/k/v
`ExactLinear`s with one, and optionally gate/up with one. That changes no bits
by construction: each output column is computed from its own weight row and
per-channel scale, and the activation's per-row quantisation depends only on
the input, which the merged parts share. `Y_EXACT_MERGE` picks `0`, `qkv` or
`all`.

## Same tokens

`gpu_sm89_y_vint8_merged_compiled.json` (merge `all`, `record_merged.log`):
the exact arm's token digest is `7700e3e28f1d21d1` and all 5 cases x 160
tokens equal the CPU record's
(`../exact_cross_device_7b_2026-11-04/cpu_zen5_eager.json`), 0 of 5 differ.
Its weight fingerprint (`108d47c28dae4291`, CPU `fa21999abf0abea9`) differs
because the fingerprint hashes parameter names and shapes, which merging
changes (`qkv_proj`, `gate_up_proj`). `exact_cross_device --compare` refuses
this pair for lacking a stock control arm, so the cases were compared
directly. The q/k/v-only record is in `../exact_greedy_loop_7b_2026-11-08/`.

## Speed

One configuration per process, in the order run (`run.sh`, `run.out`):

| process | tok/s/seq |
|---|---|
| vLLM FP8, CUDA graphs | 59.07 |
| Y, merge `0` | 52.1 |
| Y, merge `qkv` | 52.9 |
| Y, merge `all` | 52.1 |
| Y, merge `all` | 52.0 |
| Y, merge `qkv` | 53.1 |
| Y, merge `0` | 52.3 |
| vLLM FP8, CUDA graphs | 59.15 |

Each Y process's spread (median over best) was 0.1%.

- q/k/v merged: +1.5% over no merging (52.9/53.1 vs 52.1/52.3).
- q/k/v and gate/up merged: no gain over no merging, and it raises the peak
  memory of a long prefill: one `[rows][2 x 18944]` int32 output where there
  were two halves (870 MiB in the record's longest prefill). That is the allocation the
  first attempt's record died on (`apps_open/record_merged.log`), with apps
  holding 2.2 GB and that record run without `expandable_segments`, so
  3.5 GB of PyTorch's reservation sat unallocated.

So the default became `qkv`. Y is at 0.90x vLLM FP8 (53.0 / 59.1).

## apps_open/

The first attempt, with Discord and Chromium open: vLLM failed to start
twice (not enough free memory at 0.88 utilisation), the merge `all` record ran
out of memory (above), and Chromium used up to 65% of the GPU during the Y
runs. None of it is used above; it is kept because it is what happened.
