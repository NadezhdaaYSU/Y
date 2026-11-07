# Y's BF16 GEMM against cuBLAS, 2026-11-07

`bf16_bench.py` checks each shape first: small-integer inputs must give the
exact product, and the share of outputs bit-identical to cuBLAS's is counted.
Then it times Y's `@tile` BF16 GEMM against cuBLAS's BF16 GEMM with f32 output
(`torch.mm(A, B, out_dtype=torch.float32)`), same layout.

**Speed: not measured yet.** The GPU was not free for it.

Correctness so far (`tests/ptx_bf16.rs`, `python/y_lang/tests/test_bf16.py`):
- exact on small integers at every shape tried, ragged ones included;
- the bias and split-K forms are exact;
- above 32 rows every output is bit-identical to cuBLAS's at 512³ to 4096³,
  and 1024x1024x4096, since both chain the tensor core's accumulator;
- at 32 rows the compensated sums are 10-400x closer to the float64 product
  than cuBLAS on positive inputs.
