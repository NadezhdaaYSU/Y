# Fusing a multiply into the add it feeds (`y_inductor`)

`y_inductor` used to keep a multiply out of the kernel of the add or subtract it
feeds, because `ptxas` contracted the pair into one fused multiply-add and the
result then differed from eager PyTorch. Since the PTX emitter rounds a
`let`-bound product first (`mul.rn`), the fused kernel matches eager bit for bit,
and this measured what keeping them apart cost before the rule was removed.

## Method

- Commit `b8f96ab`, where the rule (`inductor._fusable`) still existed.
  `bench_fusion.py` compiles each model twice with `torch.compile(backend=y_inductor)`:
  with the rule (`rule`) and with `_fusable` patched to always fuse (`fused`),
  and also runs it eagerly.
- Before timing, every output of both compiled variants is compared with
  eager, bit for bit; a mismatch aborts the run. A variant recompiled during
  timing (and so re-partitioned) aborts it too.
- Timing: CUDA events around 20 calls, 15 rounds interleaved across the three
  variants, the minimum per-call time over the rounds. Three independent runs
  (`run1.json` .. `run3.json`, which also hold each variant's kernel partition).
- Models over three `float32` CUDA tensors `a, b, c`: `MulAdd` returns
  `a * b + c, c - a * b` (the test's model), `Axpy` returns `a * b + c`, and
  `Chain` returns `relu(a * b + c) * b - c`.
- RTX 4070 Ti SUPER, driver 615.71.09, max SM clock 3120 MHz; PyTorch
  2.13.0+cu130; Python 3.14.7. The desktop compositor was the only other GPU
  client.

## Results

Fused time over the rule's, per run (microseconds in the JSON files):

| model | n | rule kernels | fused kernels | run 1 | run 2 | run 3 |
| --- | --- | --- | --- | --- | --- | --- |
| MulAdd | 2^20 | 4 | 2 | 0.68 | 0.67 | 0.42 |
| MulAdd | 2^22 | 4 | 2 | 0.86 | 0.86 | 0.87 |
| MulAdd | 2^24 | 4 | 2 | 0.68 | 0.68 | 0.67 |
| Axpy | 2^20 | 2 | 1 | 0.76 | 2.03 | 1.52 |
| Axpy | 2^22 | 2 | 1 | 0.87 | 0.87 | 0.87 |
| Axpy | 2^24 | 2 | 1 | 0.66 | 0.66 | 0.65 |
| Chain | 2^20 | 3 | 1 | 1.26 | 0.52 | 1.03 |
| Chain | 2^22 | 3 | 1 | 0.49 | 0.50 | 0.50 |
| Chain | 2^24 | 3 | 1 | 0.43 | 0.38 | 0.42 |

At 2^22 and 2^24 elements, where the kernels are memory-bound, fusing is
faster in every model and every run, by 13% to 62%, and the runs agree to
within a few percent. At 2^20 (4 MiB per tensor) the minimum itself moved by up
to 3x between runs - the rule's `Axpy` measured 37.2, 13.7 and 10.9 us - so
launch overhead and cache residency dominate and these runs show no reliable
difference in either direction. The fused results are bit-identical to eager
in every case, so the rule bought nothing and was removed.
