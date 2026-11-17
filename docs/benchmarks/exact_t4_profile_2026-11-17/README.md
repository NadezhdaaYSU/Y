# Why the exact path is slow on a Tesla T4 (2026-11-17)

On a Tesla T4 the exact path ran 1.37x to 1.48x slower than stock float16
(`docs/benchmarks/exact_speed_05b_2026-11-17`). On an RTX 4070 Ti SUPER it
is 1.05x. This record finds where the T4's time goes.

## Job 1: the profile

`profile_05b.py` builds an arm with `tools/exact_throughput.py`'s own
`build()`, compiled, batch 32, and runs one `generate()` of 16 new tokens
under `torch.profiler`. `gemm_shapes.py` times `torch._int_mm` against a
float16 matmul at the model's shapes, 32 rows. Both ran on Kaggle
(`t4_job1/make_t4_profile_job.py`; the scripts as of 582b6a8, the tool at
67b8fdc). Tesla T4, driver 580.178.04, torch 2.13.0+cu126, a 4-core Xeon at
2.00 GHz.

Per new token, prefill included:

| Arm | Wall | GPU busy | Kernels | Matrix multiplies | Fused Inductor kernels |
|---|---|---|---|---|---|
| Stock float16 | 14.3 ms | 11.1 ms | 540 | 7.6 ms | 0.7 ms |
| Exact, default | 25.1 ms | 21.3 ms | 716 | 13.7 ms (int8) + 1.0 ms (float) | 5.4 ms |
| Exact, every switch | 23.5 ms | 19.3 ms | 605 | 13.6 ms (int8) + 0.1 ms (float) | 2.9 ms (+ 1.0 ms Y kernels) |

The GPU is busy for 78% to 85% of the wall time in every arm, so the step
is not mostly waiting on Kaggle's slow CPU. The int8 matrix multiplies are
the biggest cost. They take 13.7 ms, almost twice stock's 7.6 ms for all of
its matrix multiplies.

Every int8 multiply ran one kernel, `volta_igemm_int8_128x128_ldg4_nn`.
That is a kernel written for Volta, which has no int8 tensor cores. The T4
has them, but this kernel does not use them. At each shape alone
(`t4_job1/gemm_shapes.log`, median of 15 samples of 200 calls):

| Shape (32 rows) | `torch._int_mm` | float16 | int8 against float16 |
|---|---|---|---|
| q/k/v 896 to 1152 | 83.0 us | 25.9 us | 3.21x slower |
| o 896 to 896 | 41.9 us | 15.2 us | 2.75x slower |
| gate or up 896 to 4864 | 54.8 us | 43.1 us | 1.27x slower |
| down 4864 to 896 | 207.2 us | 47.6 us | 4.35x slower |
| lm_head 896 to 151936 | 1837.6 us | 1140.5 us | 1.61x slower |

All the int8 results were exact.

## The cause: the weight's storage order

The exact path stored each int8 weight as a row-major [K][N] matrix and
passed it to `torch._int_mm`. cuBLAS calls that form "NN". Its documentation
says its int8 tensor-core kernels need the "TN" form: the weight stored as
nn.Linear stores it, [N][K] with K contiguous, and passed as its transpose.
For NN, cuBLAS picks a kernel that does not use the tensor cores.

On the RTX 4070 Ti SUPER the same thing shows (`gemm_shapes.py`, 3 arms).
With NN, cuBLAS runs a `cutlass1x` `IgemmTraits` kernel. With TN, it runs
`cutlass_80_tensorop_i16832gemm_s8`, a tensor-core kernel, and `lm_head`
takes 392 us instead of 797 us. The 4070 Ti does not normally take this
path: its int8 multiplies run on the Triton kernel, which Triton cannot
build below sm_80.

The fix (`python/y_lang/exact/model.py`, `convert`): when the linear layers
will take `torch._int_mm` on a GPU, store each weight in nn.Linear's order.
`w8t` is still the same [K][N] matrix with the same values, now a view.
`python/y_lang/tests/test_exact_gemm_choice.py` checks the layout and that
the logits keep the same bits.

## Job 2a: the fix on the T4

`make_t4_layout_job.py` part a, the tool at 366bed6 (the fix, and
`exact_throughput.py` keeping `logits_to_keep`, see below). Same T4 type,
driver, torch and CPU as job 1. Files in `t4_job2a/`.

### Each shape alone

`gemm_shapes.py`, 32 rows, median of 15 samples of 200 calls:

| Shape | int8, old order (NN) | int8, new order (TN) | float16 | New int8 against float16 |
|---|---|---|---|---|
| q/k/v 896 to 1152 | 81.5 us | 15.1 us | 18.9 us | 0.80x |
| o 896 to 896 | 112.1 us | 19.8 us | 25.3 us | 0.78x |
| gate or up 896 to 4864 | 53.1 us | 29.2 us | 53.0 us | 0.55x |
| gate and up together, 896 to 9728 | 112.5 us | 58.2 us | 100.6 us | 0.58x |
| down 4864 to 896 | 203.4 us | 39.8 us | 58.6 us | 0.68x |
| lm_head 896 to 151936 | 1749.5 us | 803.5 us | 1467.0 us | 0.55x |

With the new order cuBLAS runs `cutlass_75_tensorop_i8816gemm_s8`, a
Turing tensor-core kernel, and int8 is now faster than float16 at every
shape. With the old order it ran `volta_igemm_int8_128x128_ldg4_nn`, as in
the model. All int8 results were exact. (The profiler listed no kernel
for the `lm_head` row; the times are CUDA events, as for the others.)

### The whole model

`layout_ab.py`: stock float16, the exact path as `convert` now builds it,
and the exact path with every weight copied back to the old order. One
process, compiled, batch 32, 64 new tokens, best of 5, taken in turns.

| Arm | Tokens per second per sequence | Against stock |
|---|---|---|
| Stock float16 | 69.4 | |
| Exact, new order | 67.3 | 1.03x slower |
| Exact, old order | 55.4 | 1.25x slower |

The fix makes the exact path 1.21x faster on the T4. Both exact arms give
the same token ids on all 32 rows, and the same 40 distinct reference
tokens as before.

### Where a decode step's time goes now

`profile_05b.py`, the difference between 16 and 48 new tokens, so the
prefill cancels. Per decode step (steps 17 to 48):

| | Stock float16 | Exact, new order |
|---|---|---|
| Wall time | 15.2 ms | 16.0 ms |
| GPU busy | 10.3 ms (68% of the wall) | 8.9 ms (55% of the wall) |
| Kernels | 517 | 728 |
| Matrix multiplies | 6.2 ms (float16) | 3.8 ms (int8) + 0.8 ms (float32, attention) |
| Attention kernels | 2.4 ms | (in the fused kernels) |
| Fused Inductor kernels | 0.5 ms | 2.9 ms |
| Copies, fills, other | 1.2 ms | 1.2 ms |

The exact path's GPU now does less work per step than stock's. What is
left of the gap is on the host. The exact step launches 211 more kernels,
and on this 2.00 GHz Xeon the GPU sits idle for 45% of the step. Stock is
partly host bound too (68% busy). So on this machine the next thing to cut
is the number of kernels, not their speed.

## Does `torch._int_mm` now beat Triton where Triton runs?

No. On the RTX 4070 Ti SUPER the exact path's linear layers run the Triton
int8 kernel. `gemm_path_ab.py` times that against the same model switched
to `torch._int_mm` in the TN form (`triton_ok` cleared, weights in
nn.Linear's order), next to stock bf16, in one process at 79c5cca. The GPU
was idle (6% use at the start, no game running).
`local_4070ti/gemm_path_ab.log`:

| Arm | Tokens per second per sequence | Against stock |
|---|---|---|
| Stock bf16 | 244.3 | |
| Exact, Triton (the default here) | 227.6 | 1.074x slower |
| Exact, `torch._int_mm` TN | 225.3 | 1.084x slower |

Triton stays 1% faster, so the default does not change. Both exact arms
give the same token ids on all 32 rows.
