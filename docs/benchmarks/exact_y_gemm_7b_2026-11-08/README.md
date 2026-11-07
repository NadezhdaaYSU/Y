# Qwen2.5-7B on Y's int8 GEMM, 2026-11-08

`run.sh` (at 27cfc0c): the CPU==GPU record with every linear layer on Y's
int8 GEMM, then decode speed in three configurations, one per process, in
ABCCBA order. RTX 4070 Ti SUPER (sm_89, driver 13.0), PyTorch 2.13.0+cu130,
transformers 5.15.0. Every arm built by `y_lang.exact.large_model`
(`--low-memory`), compiled (`torch.compile(dynamic=True)`). Steam (no game),
Discord and Chromium were open for every run; `run.out` logs the GPU
utilisation sampled before each.

## The same tokens

`gpu_sm89_y_compiled.json`: the exact arm, `Y_EXACT_GEMM=y`, compiled, 5
cases x 160 tokens. Its token digest is `7700e3e28f1d21d1`, the CPU record's
(`../exact_cross_device_7b_2026-11-04/cpu_zen5_eager.json`): **0 of 5 cases
changed**, and the sanity gate and the same-card re-run passed. The stock
control is in that directory (5 of 5 changed); this record has no stock arm.

## Speed

`exact_throughput.py --compile --batch 32 --tokens 64 --reps 5`, best of 5,
tokens/s per sequence (`b7_*.log`):

| configuration | first | second | against Triton |
|---|---|---|---|
| Triton int8 GEMM (the default) | 43.5 | 44.4 | - |
| Y's int8 GEMM (`Y_EXACT_GEMM=y`) | 46.8 | 45.7 | 1.03-1.08x |
| Y's GEMM and the int8 V cache on Y's `exact_pv` (`Y_EXACT_V_INT8=1`) | 51.2 | 48.4 | 1.09-1.18x |

Ratios pair each run with the other configuration's run in the same half of
the ABCCBA sequence. Within a run the five repetitions spread 0.1-1.3%;
between processes the same configuration moved up to 6% (51.2 and 48.4),
more than the 2026-11-04 session's Triton arm alone (47.4 then, 43.5-44.4
here, with the desktop applications open). So the order is measured, the
size of each step to within several percent.

What this is not: a comparison with stock bf16 (31.5 tok/s/seq on
2026-11-04, another session) or with a non-deterministic int8 engine, which
has still not been measured at 7B.
