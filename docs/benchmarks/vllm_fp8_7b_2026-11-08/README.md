# Qwen2.5-7B: Y's exact path against vLLM's FP8, 2026-11-08

The baseline every earlier 7B note said was missing: a production engine at
8 bits a weight that is NOT batch-invariant, on the same card. vLLM 0.21.0
(torch 2.11.0+cu130, `~/vllm_env`), `quantization="fp8"`: the bf16 checkpoint's
weights quantised to FP8 on load, activations per token, CUTLASS FP8 GEMMs,
FlashAttention. Y at 69243fd: `y_lang.exact`, Y's int8 GEMM and the int8 V
cache on `exact_pv` (`Y_EXACT_GEMM=y Y_EXACT_V_INT8=1`), compiled, no CUDA
graphs. RTX 4070 Ti SUPER (sm_89); nothing else open (`run.out` samples the
GPU before each process: 0-21% at idle clocks).

## Speed

The work `tools/exact_throughput.py` times: 32 copies of its prompt, 64 new
tokens, greedy, the wall time of one generate call including prefill, best of
5 after a warm-up. vLLM's prefix caching is off (it would compute the shared
prompt once where HF computes it 32 times) and EOS is ignored. One
configuration per process, in the order of `run.sh`: vLLM, Y, Y, vLLM, then
vLLM's other two modes.

| engine | tok/s/seq | Y against it |
|---|---|---|
| vLLM FP8, CUDA graphs (its default) | 58.4, 58.2 | **0.89x** |
| vLLM FP8, `enforce_eager` | 56.6 | 0.92x |
| **Y exact (Y GEMM + int8 V)** | **52.0, 52.0** | - |
| vLLM FP8, `VLLM_BATCH_INVARIANT=1` | 16.5 | **3.15x** |

Within each process the five repetitions spread 0.1% for Y (the harness's
measure) and 0.5-2.0% for vLLM (slowest over fastest, `times_s`); the
repeated processes agree to 0.4%.

## Does a prompt's output depend on its batch?

The target prompt decoded alone and in 5 other batch compositions (two, seven
and 31 other prompts after it, or in the middle of 11), 160 greedy tokens,
then alone again (`vllm_7b.py`; `y_compositions.py` reads the same prompts out
of it). The first token that differed from the target decoded alone:

| engine | +2 after | +7 after | +31 after | middle of 11 | changed |
|---|---|---|---|---|---|
| vLLM FP8, graphs (first process) | 5 | - | 3 | 3 | 3 / 4 |
| vLLM FP8, graphs (second process) | - | - | 3 | 3 | 2 / 4 |
| vLLM FP8, eager | - | - | 32 | 44 | 2 / 4 |
| vLLM FP8, batch-invariant | - | - | - | - | 0 / 4 |
| **Y exact** | - | - | - | - | **0 / 4** |

Every engine reproduced its own "alone" decode. The two vLLM graph processes
disagree with each other on the "+2 after" composition, and vLLM's eager and
graph modes decode the target to different text from the first tokens.

## What it says, and what it does not

- vLLM's deterministic mode costs it 3.5x here (58.3 to 16.5): with it, the
  FP8 linears stay on CUTLASS, attention runs with fixed splits, and its
  batch-invariant Triton matmul (`matmul_kernel_persistent`) serves the bf16
  matmuls. Y's exact path, deterministic, decodes 3.15x faster than that,
  and 0.89x of vLLM's non-deterministic default.
- Y's tokens are also the same on a CPU and a GPU at 7B
  (`../exact_y_gemm_7b_2026-11-08/`); vLLM's across devices was not tested.
- The quantisations differ: FP8 e4m3 with per-tensor weight scales against
  Y's int8 with per-channel weights and per-token activations. Neither's
  accuracy is measured here.
- The engines differ: vLLM's scheduler and CUDA graphs against Hugging Face
  `generate` with `torch.compile`. That is the comparison a user faces, not
  a kernel comparison.
- vLLM documents batch-invariant mode for bf16; with FP8 it ran and changed
  0/4 compositions here, which is all this measures about it.
