# `exact_pv` on Y's kernel, registered and launched like the int8 GEMM (2026-11-07)

With `Y_EXACT_V_INT8=1` the exact model stores V as int8 and computes
`p @ v` in attention with the compiler's kernel (`tests/exact_pv.ysu`, int64
accumulation, one launch) instead of the digit split's float32 matmuls.
`docs/bit_identical_decode.md` measured that route at 0.70x the digit split
compiled, on a GPU another process was using, and left it there.

## What the route cost, beside the kernel

`tools/batch_invariance_demo.py` had it as:
- a `torch.library.custom_op` taking the int64 weight and returning float64,
  with both casts inside the op;
- PTX built by running the compiler binary, which wrote the committed
  `tests/exact_pv.ptx` on every process's first call;
- launched through `ptx_bridge.launch`, which calls `cuCtxSynchronize`
  after every launch: 24 times a decode step at 0.5B, so the host could
  never queue ahead of the GPU.

Now:
- a `torch.library.Library` op (`torch.ops.y.exact_pv`): int32 in, int64
  out, with a fake for `torch.compile`. The casts are outside it, where
  Inductor fuses them into the pointwise on either side. Called directly
  when not compiling, as `int8_mm` is;
- compiled and launched through `y_lang`'s `TorchKernel`, the launcher Y's
  int8 GEMM uses: one C call, on torch's current stream, no
  synchronisation. Its PTX is the CLI's, byte for byte (compared against
  `tests/exact_pv.ptx`);
- offsets past 2^31, which the kernel forms in 32-bit arithmetic, are
  refused (they used to be truncated by `c_uint`).

The kernel is unchanged.

## Time per call (`dispatch_cost.py`, `dispatch_cost.log`)

0.5B decode shape: p [32, 2, 7, 64], v [32, 2, 64, 64]. 300 calls with no
synchronisation in the loop (the old launch synchronises by itself),
minimum of 7 batches, every output checked against a float64 matmul first.
One change at a time:

| route | eager | compiled, + one pointwise |
|---|---:|---:|
| `custom_op`, synchronising launch, casts inside (until today) | 36.9 us | 66.9 us |
| no synchronisation | | 61.2 us |
| `Library` instead of `custom_op` | | 54.1 us |
| casts outside, `TorchKernel` launch (now) | 12.1 us | 36.1 us |
| the digit split, float32 V (the default) | | 46.8 us |

Of the old body's 20 us without the synchronisation, the ctypes launch was
7.8 us and `torch.cuda.current_stream(dev)` 2.4 us (`TorchKernel` reads the
raw stream in one C call).

## Decode throughput (`e2e.sh`, `e2e_0p5b/`)

Qwen2.5-0.5B, the exact arm only, compiled, batch 32, 64 new tokens, best of
5, Triton GEMM (the default), one route per process in ABCCBA order, idle
GPU (2026-11-07, 19:38-20:21). The old route ran from a worktree at
`83da64d`, with the same build, GEMM tuning and compiler profile:

| `exact_pv` route | tok/s per sequence |
|---|---|
| A: digit split, float32 V (the default) | 218.1, 217.8 |
| B: int8 V, the route until today | 198.8, 189.1 |
| C: int8 V, the route now | 218.8, 224.0 |

The spreads over the 5 repetitions were 0.1-1.1%, except run 3 (C) at 3.9%.
- On an idle card the old route ran at 0.87-0.91x the digit split. The 0.70x
  in `docs/bit_identical_decode.md` was taken while another process held
  the GPU.
- The route now runs at 1.00-1.03x the digit split, and 1.10-1.18x the old
  one.

At T of 25-89 keys this is the int8 cache's worst case: the kernel's gain
over the digit split grows with T (5.04x at T = 4096 on the kernel alone,
`tools/exact_pv_bridge.py`), and the decode here never gets far.

`route_check.py` (`route_check.log`) builds the exact arm as
`exact_throughput.py --compile` does and generates 8 tokens. With
`Y_EXACT_V_INT8=1`:
- the V cache is int8;
- the registered op ran 192 times (24 layers x 8 forward passes) and the
  eager body never did;
- dynamo made 3 graphs with 0 breaks.

Without `Y_EXACT_V_INT8`, no kernel is loaded and the op never runs.

## Tokens: CPU against GPU (`record.sh`)

`gpu_sm89_vint8_compiled.json` (log beside it) is the 0.5B record of
`tools/exact_cross_device.py`, compiled, with the int8 V cache on this
kernel. It was compared with the CPU record of
`docs/benchmarks/exact_y_gemm_2026-11-07/cpu_zen5_0p5b.json`, whose V is
float32 because the CPU has no kernel route
(`compare_cpu_vs_vint8_compiled.txt`):
- exact changed 0/5 cases, with the same token digest, `30eec169d2f38da4`;
- stock bf16 changed 5/5 (the control fired).

The record passed the harness's gate and reproduced itself on a second run.

## Correctness

`tools/exact_selftest.py` has a new check, `check_pv_int8_kernel`, run when
`Y_EXACT_V_INT8=1`. The old `check_pv` hands `exact_pv` an int64 V, so it
checks the digit split only, and nothing checked the kernel route. The new
check compares against a float64 matmul (exact: |sum| <= T * 2^28 * 127 <
2^53):
- at T in {1, 63, 80, 512, 1039, 1040, 4097}, head_dim 64 (5 queries) and
  128 (1 query), p up to 2^28 inclusive, v down to -127;
- called directly, and through `torch.ops.y.exact_pv`;
- compiled with `fullgraph=True` (no graph break), at two key lengths;
- on a side stream, with the inputs made and the output read there.

A changed V value must change the output, and CPU tensors must be refused
by the CUDA-device check. `selftest_vint8.log` is the whole self-test in
that mode.

Mutants (`mutants.txt`), each killed:
- the launch on the default stream instead of torch's current one: the
  side-stream comparison fails;
- `_y_exact_pv` called directly under `torch.compile`: dynamo breaks the
  graph, and `fullgraph=True` fails;
- the fake returning float64: the compiled result differs;
- the kernel told T - 1 keys: the products differ;
- the CUDA-device check removed: CPU pointers reach the launch, and the
  process segfaults.
