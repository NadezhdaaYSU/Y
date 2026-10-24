# Why the F16 GEMM is 0.99x cuBLAS at 4096^3, 2026-10-24

RTX 4070 Ti SUPER (66 SMs, 48 MB L2), driver 615.71.09, `ptxas` from CUDA
13.4, PyTorch 2.13.0+cu130. The desktop was in use (no game); every run
below is contended by the 5% rule. Y's kernel is the committed
`tests/gemm_f16_4096.ptx` (128x128x32 tile, 2x2 warps, 3 stages, chain
accumulation).

## The gap, by how it is timed

| harness | data | Y | cuBLAS | cuBLAS / Y |
| :--- | :--- | ---: | ---: | ---: |
| `bench_ab.py`, events, min of 15 interleaved rounds (`bench/base_*.json`) | small integers | 1562.3 / 1565.2 us | 1542.5 / 1546.8 us | 0.987 / 0.988 |
| `sbench.py`, CUDA graphs, min of 21 rounds (`bench/sb_*`) | small integers | 1561.6 / 1561.6 us | 1536.3 / 1541.4 us | 0.984 / 0.987 |
| the same | N(0, 1) | 1596.4 / 1579.3 us | 1547.7 / 1548.0 us | 0.970 / 0.980 |
| `prof/power.py`, 3 s back to back, mean | small integers | 1665.4 / 1674.2 us | 1639.8 / 1654.1 us | 0.985 / 0.988 |
| the same | N(0, 1) | 1701.2 / 1708.7 us | 1717.7 / 1720.1 us | **1.010 / 1.007** |

`prof/power.py` also sampled the SM clock and board power every 50 ms. On
small integers both kernels ran at the same clock (Y 2750-2767 MHz, cuBLAS
2745-2760) and Y drew less power (182-223 W against 229-243). On N(0, 1)
both drew 274-282 W, and cuBLAS throttled further: 2658-2660 MHz against Y's
2712-2716. So the gap has two parts:

- **Per clock cycle Y is 1.2-1.6% slower**, on any data.
- **Under sustained load on random data cuBLAS throttles more**, and Y is
  0.7-1.0% faster. The interleaved minimum, which ranks each kernel at its
  coolest moment, does not see this.

## Where the cycles go (`prof/`, `ncu`)

| | Y | cuBLAS |
| :--- | ---: | ---: |
| kernel | 128x128x32, 2x2 warps, 3 stages, `cp.async` | `ampere_s1688gemm_fp16_128x128_ldg8_stages_32x1_nn` |
| tensor pipe active | 48.78% | 49.53% |
| registers / thread, CTAs per SM | 231, 2 | 234, 2 |
| warp instructions | 81.8 M | 98.6 M |
| `ldmatrix` per 16x8x16 of work | 0.251 | 0.250 |
| integer instructions per 16x8x16 (IMAD, LEA, SHF, IADD3) | 0.86 | 0.23 |
| L2 hit rate, DRAM read | 93.5%, 163 MB | 95.1%, 128 MB |

The tensor pipe's 1.5% difference is the per-cycle gap. Y's stall samples
(`prof/y_src.csv`, by address): 97.9% in the K loop, 85% of them
`math pipe throttle` (a warp ready with an HMMA the pipe cannot yet take) and
9.4% `wait`, spread evenly over the HMMAs (about 500 samples each, the
pipe's issue cadence rather than any one dependency); 1.6% in the epilogue
and 0.6% in the prologue.

The loop's extra integer work (`y_opd.txt`: 148 instructions per K-tile, 64
of them HMMA) recomputes `k % 3` three times and every shared address from
it. It is not on the critical path: each scheduler issues about 7% as many
instructions as its tensor pipe has busy cycles, and the wait stalls do not
gather after it.

## m16n8k8 against m16n8k16 (`k8/`)

cuBLAS issues `m16n8k8`, Y `m16n8k16`. `k8/mma_k8_vs_k16.py`: one m16n8k16
against two m16n8k8 on the same fragments, f32 accumulate.

- **The same bits**, on mixed-sign and same-sign fragments at scales 1, 64
  and 1/64, chained 64 times: the tensor core runs k16 as two k8 passes.
  That is why Y's chain gives cuBLAS's bits at 4096^3 although cuBLAS uses
  the other instruction.
- **The same throughput**: 66 x 8 warps of 8 independent chains, 1.217 ms
  for each in one run and 1.292 / 1.291 ms in the other (`k8/out.txt`).

## Not pursued

What is left is structural: the prologue and epilogue (2.2% of the samples),
during which one CTA's warps feed the tensor pipe alone, and the CTA order's
L2 traffic. Neither has a quick lever, and under sustained load on random
data Y is already the faster kernel.
