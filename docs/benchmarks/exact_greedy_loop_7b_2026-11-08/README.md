# A decode loop without per-step host syncs, at 7B (2026-11-08)

Qwen2.5-7B-Instruct, exact path on Y's int8 GEMM with the int8 V cache and
q/k/v merged (`Y_EXACT_GEMM=y Y_EXACT_V_INT8=1`, merge default `qkv`),
compiled, batch 32 x 64 new tokens, `tools/exact_throughput.py`, best of 5.
RTX 4070 Ti SUPER (sm_89), torch 2.13.0+cu130, transformers 5.15.0; vLLM
0.21.0. Only VS Code was open.

## Why

The profile of a 7B decode under Hugging Face's `generate`
(`../exact_7b_profile_2026-11-08/`) had the GPU idle for about 2 ms of each
19 ms step. `generate` checks every step whether all rows have finished
(`unfinished_sequences.max() == 0` and a Python `if`): one device-to-host copy
per token (19 in that 16-token profile), after which the GPU waits while the
CPU prepares the next step.

`y_lang.exact.greedy` decodes a fixed number of tokens with the argmax on the
device and nothing copied to the host until the end. It feeds the model what
`generate` feeds it (positions from the attention mask, a dynamic cache,
`logits_to_keep=1`) and applies no logits processor, as vLLM's
`SamplingParams(temperature=0)` applies none. `generate` applies the
repetition penalty from Qwen2.5-Instruct's generation config (1.05), so the
two are compared with `repetition_penalty=1.0`.

## Same tokens

- `exact_throughput --loop greedy` decodes the timed batch both ways before
  timing and refuses to time a mismatch: both greedy processes printed
  `greedy == generate(repetition_penalty=1.0) on all 32 rows`.
- `python/tests/test_exact_package.py` checks, on a tiny Qwen2 and a
  left-padded ragged batch, that `greedy` gives `generate`'s tokens and that
  every forward call receives `generate`'s `input_ids`, `position_ids`,
  `attention_mask` and `logits_to_keep`. The input check is there because the
  token check alone let two mutants through (decode positions off by one;
  positions not reset by padding): a tiny random model's argmax hardly
  depends on position. All 5 mutants tried are caught.
- `gpu_sm89_y_vint8_qkv_compiled.json` (`record_qkv.log`): the q/k/v-merged
  default, through `generate` as the recorder always runs it. Token digest
  `7700e3e28f1d21d1`; all 5 cases x 160 tokens equal the CPU record's
  (`../exact_cross_device_7b_2026-11-04/cpu_zen5_eager.json`).

## Speed

One configuration per process, in the order run (`run.sh`, `run.out`):

| process | tok/s/seq |
|---|---|
| vLLM FP8, CUDA graphs | 59.10 |
| Y, `generate` | 52.9 |
| Y, `greedy` | 54.5 |
| Y, `greedy` | 54.5 |
| Y, `generate` | 53.1 |
| vLLM FP8, CUDA graphs | 58.44 |

The loop gains 3.0% (54.5 vs 52.9/53.1). Y is at 0.93x vLLM FP8 (54.5 /
58.8), from 0.89x in `../vllm_fp8_7b_2026-11-08/`.

## Where the step goes now

`profile_greedy.py` (`profile_greedy.log`, `.json`), run from `tools/` like
the timed runs:

- One 16-token `greedy`: wall 331.6 ms, GPU busy 322.1 ms (97%). The idle time
  is gone.
- One decode step issued onto an idle GPU (outside the profiler): 4.05 ms of
  CPU, 17.0 ms until the GPU finishes. The GPU no longer waits for the CPU.
- The gain is smaller than the earlier profile's idle time (about 2 ms a
  step under `generate`) suggested. Probably the profiler, which records every
  CPU-side op, slowed the CPU and inflated that idle time; the timed runs run
  without it. Not measured further.
- Non-GEMM kernels: 3.7 ms of each decode step. The largest are the two
  residual-add + RMSNorm + quantise kernels per layer, 18 us each (1.0 ms a
  step), whose float64 sum and divide run at 1/64 rate on this GeForce part;
  next is `exact_pv`, 0.5 ms a step.
