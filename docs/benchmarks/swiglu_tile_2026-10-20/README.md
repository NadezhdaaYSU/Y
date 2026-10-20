# The fused SwiGLU GEMM's tile, 2026-10-20

RTX 4070 Ti SUPER, driver 615.71.09, `ptxas` from CUDA 13.4, PyTorch
2.13.0+cu130. No game running; the 10-sample mean utilisation before the two
runs was 28.2% and 22.1%, read right after the previous job (`queue.txt`); the
two runs agree within 0.5% at every size.

`gemm_f16_swiglu_<N>` (Out = SiLU(X @ Wg) * (X @ Wu), f16 in, f32 out) had a
fixed 128x128x32 tile on 4x4 warps. `queue.sh` compiled it with each tile in
`Y_SWIGLU_TILE=cta_m,cta_n,cta_k,warps_m,warps_n` (`<tile>/`, swizzled shared
memory, 4 stages for all; `ptxas_<N>.txt`: no spills, 103-160 registers) and
timed them with `bench_swiglu.py` (as in `../quiet_2026-10-20/swiglu/`; every
fused tile gives the same bits, and every column is within 1e-4 of float64)
against Y's unfused path (two Y GEMMs, `F.silu(g) * u`) and PyTorch's (two
`torch.mm` with a float32 output). Minimum us, run 1 / 2:

| tile | 512 | 1024 | 2048 | 4096 |
| :--- | ---: | ---: | ---: | ---: |
| 128x128x32, 4x4 (old) | 32.30 / 32.31 | 59.46 / 59.56 | 444.40 / 445.25 | 3450.29 / 3473.09 |
| 64x64x32, 4x2 (new) | 11.06 / 11.06 | 59.31 / 59.35 | 431.11 / 431.81 | 3338.72 / 3355.14 |
| 64x64x32, 2x2 | 12.04 / 12.06 | 60.55 / 60.58 | 432.85 / 433.78 | 3325.95 / 3329.54 |
| 64x64x64, 2x2 | 11.58 / 11.59 | 73.42 / 73.60 | 505.39 / 506.52 | 3745.28 / 3749.38 |
| 128x64x32, 4x2 | 18.45 / 18.46 | 65.26 / 65.05 | 478.11 / 480.54 | 3682.82 / 3690.83 |
| 64x128x32, 2x4 | 19.17 / 19.18 | 67.06 / 67.17 | 488.35 / 489.47 | 3736.06 / 3772.93 |
| Y unfused | 15.14 / 15.14 | 65.05 / 65.17 | 493.65 / 493.94 | 3687.95 / 3734.85 |
| PyTorch | 15.98 / 16.02 | 67.82 / 68.07 | 462.85 / 463.22 | 3640.83 / 3638.93 |

64x64 on 4x2 warps is the fastest at 512 and 2048, level with the old tile at
1024 (1.002-1.004x), and 1.03x it at 4096, where 64x64 on 2x2 warps is 0.4-0.8%
faster still. It is the new default for every size: 2.92x the old tile at 512,
where the old one made 16 CTAs for 66 SMs; 1.10-1.37x Y's unfused path and
1.07-1.45x PyTorch's from 512 to 4096.
