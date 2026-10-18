# 512^3: what cuBLAS runs, and the tiles around Y's

At 512^3 Y takes 1.14-1.17x cuBLAS's time (`../README.md`), the largest gap
in that table above 256.

**What cuBLAS runs.** `ncu512.sh` profiles `prof512.py` (Y's
`gemm_f16_512`, the committed PTX, and `torch.mm` with a float32 output, three
calls each, both checked exact) under Nsight Compute with no cache flush, on a
GPU at 0.0% (`queue.txt`). `ncu_rows.txt` holds the two kernels' rows; the
report itself (1.3 MB, mostly PyTorch's input kernels) is not kept, and its
durations are of replayed passes.

| | Y `gemm_f16_512` | cuBLAS `ampere_s16816gemm_fp16_64x64_ldg8_stages_64x5_nn` |
| :--- | ---: | ---: |
| grid | 64 | 64 |
| threads | 256 (4x2 warps) | 128 |
| registers | 72 | 98 |
| shared memory | 38,912 B dynamic | 32,768 dynamic + 49,152 static |
| warp instructions | 444,416 | 343,296 |
| tensor pipe active | 28.1-28.2% | 30.8-31.6% |

Both run one 64x64 tile per CTA on 64 CTAs, one CTA an SM. cuBLAS's name
gives its K step and depth, 64 and 5 stages; Y's tuned tile is 64x64x32 with
4 stages. Y executes 1.29x the instructions.

**The tiles around Y's.** `sweep.sh` compiled `g512.ysu` with
`Y_CTA_OVERRIDE` (`ptx/`, named cta_m_cta_n_cta_k_warps_m_warps_n_stages;
`64_64_32_4_2_4.ptx` is byte for byte the committed `tests/gemm_f16_512.ptx`)
and timed the nine with `../bench_ab.py` in two processes, every column exact
first (`sweep1.json`, `sweep2.json`, logs beside them). Minimum microseconds,
run 1 / run 2:

| tile | us |
| :--- | ---: |
| cuBLAS | 5.82 / 5.82 |
| 64x64x32, 4x2 warps, 4 stages (the tuner's) | 6.69 / 6.68 |
| 64x64x32, 4x2, 6 | 6.73 / 6.73 |
| 64x64x64, 2x2, 3 | 6.89 / 6.89 |
| 64x64x32, 2x2, 5 | 6.96 / 6.96 |
| 64x64x64, 2x2, 4 | 6.98 / 6.96 |
| 64x64x64, 4x2, 5 | 7.19 / 7.20 |
| 64x64x64, 2x2, 6 (5 after the shared-memory clamp) | 7.32 / 7.31 |
| 64x64x64, 2x2, 5 (cuBLAS's shape) | 7.33 / 7.32 |
| 64x64x32, 2x2, 8 | 7.34 / 7.34 |
| Triton | 9.46 / 9.44 |

The tuner's tile is the fastest of the nine in both runs, and cuBLAS's shape
is among the slowest as Y emits it: changing the tile does not close the gap.
