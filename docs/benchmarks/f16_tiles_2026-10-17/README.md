# F16 GEMM tiles re-measured, 2026-10-17

The tiles in `.ysu_hw_profile`'s `AUTOTUNE_F16_*` lines were measured on the
kernel before `8d605e9`, which rewrote its K loop
(`../f16_gemm_loop_2026-10-17/`). They were measured again with that commit's
compiler: `Y g<shape>.ysu --emit-ptx --autotune-force` for each shape of that
record, on a copy of the main checkout's profile. Each shape ran when the
GPU's utilisation, averaged over 10 samples, was at most 5% and no game was
running.

`tune.txt` holds the tuner's ranking for each shape: the fastest round of
interleaved rounds, with the median and the slowest. The tuner keeps 1 to 8
copies of B, so the working set is either L2-resident (512 and 1024) or
larger than L2 (2048 and up, and every decode shape); its header line says
which. Three tiles changed; the others stayed:

| shape | tile before | tile after | tuner, before -> after (us) |
| :--- | :--- | :--- | :--- |
| 512^3 | 64x64x64, 4x2 warps, 3 stages | 64x64x32, 4x2 warps, 4 stages | 6.78 -> 6.73 |
| 1024^3 | 64x128x32, 2x2 warps, 2 stages | the same, 3 stages | 30.17 -> 29.79 |
| 8 x 4096 x 4096 | 32x64x64, 2x2 warps, 3 stages | the same, 4 stages | a tie: its eight fastest within its +/-0.29 us noise |

`bench_ab.py` (`../f16_gemm_loop_2026-10-17/`) then timed the two tiles of
each changed shape against each other, both compiled by `8d605e9`. It ran
four times, each in its own process (`ab_1.json` to `ab_4.json`). Minimum
microseconds:

| shape | before | after | after / before, each run | median |
| :--- | :--- | :--- | :--- | ---: |
| 512^3 | 6.85, 6.75, 6.71, 6.73 | 6.83, 6.74, 6.66, 6.61 | 0.997, 0.999, 0.992, 0.982 | 0.995 |
| 1024^3 | 29.73, 29.99, 29.96, 30.15 | 29.61, 29.72, 29.73, 29.77 | 0.996, 0.991, 0.992, 0.988 | 0.992 |
| 8 x 4096 x 4096 | 24.43, 24.40, 24.09, 24.05 | 23.32, 23.18, 22.95, 22.88 | 0.955, 0.950, 0.953, 0.951 | 0.952 |

At 1024, Triton's GEMM took 30.17-30.73 us in the same runs. `bench_ab.py`
uses one copy of each operand, so at 8 x 4096 x 4096 its 32 MB B stays in
the 50 MB L2 between calls. That shape gains 4.8% there and ties with B
streamed from DRAM.

## Shapes that borrow a tile

Two committed artifacts are at shapes with no measured tile:
`tensor_core_gemm_f16_rect` (512 x 1024 x 256) and `gemm_f16_uneven_grid`
(1280 x 1024 x 1024). They borrow the tile of the nearest measured shape
(`load_persisted_nearest`): 512^3's and 1024^3's. With the new tiles,
borrowed, against the old ones (`borrowed_1.json` to `borrowed_4.json`):

| shape | old tile | new tile, borrowed | new / old, median of 4 | cuBLAS |
| :--- | ---: | ---: | ---: | ---: |
| 512 x 1024 x 256 | 8.16-8.48 | 7.18-7.49 | 0.883 | 7.86-8.18 |
| 1280 x 1024 x 1024 | 45.79-46.00 | 48.74-48.87 | 1.064 | 43.02-45.15 |

1024^3's 3-stage tile was 6.4% slower at 1280 x 1024 x 1024 than its 2-stage
tile. By shared memory, the 3-stage tile (41,472 bytes) fits 2 CTAs per SM,
so the 160 CTAs of that shape take two waves; the 2-stage tile (27,648
bytes) fits 3, one wave. At 1024^3 the grid is 128 CTAs, and either fits
in one wave.

Both shapes were then measured for themselves (`tune.txt`), and the tuner
chose 64x128x32 on 2x4 warps with 3 stages for the first, and 128x64x32 on
2x2 warps with 2 stages for the second. Against the old tiles
(`measured_1.json` to `measured_4.json`):

| shape | old tile | measured tile | new / old, each run | cuBLAS |
| :--- | ---: | ---: | :--- | ---: |
| 512 x 1024 x 256 | 8.14-8.52 | 6.51-6.82 | 0.800 in all four | 7.86-8.19 |
| 1280 x 1024 x 1024 | 45.81-45.96 | 44.92-45.06 | 0.980-0.982 | 42.86-43.07 |

The main checkout's profile now holds the five new lines (`profile.diff`).
The seven committed F16 artifacts at these shapes are regenerated from it:
`gemm_f16_512`, `gemm_f16_1024`, `gemm_f16_bias_relu_512` and `_1024`,
`gemm_f16_decode_m8`, `gemm_f16_uneven_grid` and `tensor_core_gemm_f16_rect`.
The plain 512 and 1024 GEMMs, `rect` and `uneven_grid` are, byte for byte,
the PTX timed here, and the decode kernel differs from it only in its name.
The two bias+ReLU GEMMs take the same tiles and were not timed.
