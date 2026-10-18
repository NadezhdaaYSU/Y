# Decode-shaped F16 GEMMs with B from DRAM, 2026-10-18

The decode shapes, M x 4096 x 4096 with M of 1, 4, 8 and 32 (f16 A and B,
f32 C), had been timed only in loops that call the kernel again on one B.
That B is 33.5 MB, inside the 48 MB L2, so those loops read it from the L2
(`../f16_xbar_2026-10-18/decode/`). A decode step whose weights do not fit
in the L2 reads them from DRAM.

`bench_cold.py` times each kernel two ways. "From DRAM": each call takes
the next of four B copies (134 MB). "From L2": every call uses one copy.
Every column is first checked exact against float64 on all four copies
(small integers). Then CUDA events time back-to-back calls in 17
interleaved rounds, the first two dropped; minimum of 15. Y's kernels are
the committed `tests/gemm_f16_decode_m<M>.ptx` of `ed504e3`, renamed
(`ptx/`); cuBLAS is `torch.mm` with a float32 output. RTX 4070 Ti SUPER,
driver 615.71.09, PyTorch 2.13.0+cu130. Two runs, each started by
`gpu_quiet_then.sh` at a 10-sample mean of 5.0% and 2.0% with no game
running (`queue.txt`, `env_<r>.txt`); the SM clock was 2460-2775 MHz.
Microseconds, run 1 / run 2 (`table.py`):

| M | Y from DRAM | cuBLAS from DRAM | cuBLAS / Y | Y from L2 | cuBLAS from L2 | cuBLAS / Y |
|---|---|---|---|---|---|---|
| 1 | 55.513 / 56.179 | 55.788 / 55.840 | 1.005 / 0.994 | 19.949 / 20.275 | 26.790 / 26.720 | 1.343 / 1.318 |
| 4 | 55.616 / 55.917 | 56.026 / 56.407 | 1.007 / 1.009 | 23.943 / 23.821 | 53.830 / 54.699 | 2.248 / 2.296 |
| 8 | 56.103 / 56.076 | 58.710 / 57.440 | 1.046 / 1.024 | 23.129 / 23.468 | 50.310 / 50.598 | 2.175 / 2.156 |
| 32 | 56.160 / 56.316 | 55.597 / 55.936 | 0.990 / 0.993 | 30.810 / 31.219 | 23.089 / 23.649 | 0.749 / 0.758 |

From DRAM the two tie at every M (0.99-1.05x). Both move 575-618 GB/s of A,
B and C, against the card's 672 GB/s. From the L2, Y is ahead at M = 1, 4
and 8 and behind at M = 32.

`m32/` holds eight tiles at M = 32 (`Y_CTA_OVERRIDE`, the compiler of
`ed504e3`; `t_32_64_64_2_2_3` is the committed one), timed the same way:

| M = 32 tile | from DRAM | cuBLAS / tile | from L2 | cuBLAS / tile |
|---|---|---|---|---|
| 32, 128, 64, 2, 4, 3 | 56.273 / 56.441 | 0.998 / 0.996 | 29.446 / 29.587 | 0.804 / 0.791 |
| 32, 64, 64, 2, 2, 3 | 56.244 / 56.422 | 0.998 / 0.996 | 30.957 / 30.714 | 0.765 / 0.762 |
| 32, 64, 128, 2, 2, 2 | 56.671 / 56.690 | 0.991 / 0.992 | 33.420 / 32.832 | 0.708 / 0.713 |
| 32, 64, 32, 2, 2, 4 | 55.654 / 55.790 | 1.009 / 1.008 | 33.522 / 33.094 | 0.706 / 0.708 |
| 32, 128, 32, 2, 2, 4 | 56.141 / 56.166 | 1.000 / 1.001 | 33.167 / 33.286 | 0.714 / 0.704 |
| 32, 128, 64, 2, 2, 3 | 56.415 / 56.499 | 0.995 / 0.995 | 34.560 / 34.683 | 0.685 / 0.675 |
| 32, 32, 64, 2, 1, 4 | 57.016 / 57.190 | 0.985 / 0.983 | 40.781 / 40.262 | 0.581 / 0.582 |
| 32, 256, 32, 2, 4, 3 | 75.019 / 75.642 | 0.748 / 0.743 | 75.983 / 74.963 | 0.312 / 0.312 |
| cuBLAS | 56.141 / 56.218 | 1.000 / 1.000 | 23.674 / 23.418 | 1.000 / 1.000 |

From DRAM every tile but 32x256 is within 1.7% of cuBLAS. From the L2 the
best, 32x128x64 on 2x4 warps, takes 1.24-1.26x cuBLAS's time. The committed
tile takes 1.31x in this sweep; no tile closes that case.
