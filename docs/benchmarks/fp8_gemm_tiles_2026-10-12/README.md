# FP8 GEMM tile choice, 2026-10-12

Y's `@tile(N, N, N)` FP8 GEMM from the compiler at `07902d0` ("before"; its
FP8 kernel is the one recorded in `../fp8_gemm_2026-10-12/`) and with this
change ("after"), against the Triton kernel of
`python/examples/run_head_to_head_comparison.py`. All three take float32 A
and B, quantize them to e4m3 inside the kernel, multiply on the e4m3 tensor
cores and write float32 C. Every column was checked equal to `A @ B` bit for
bit (small integers times 2^-3 are exact in e4m3) before it was timed.
`results.json` holds every sample, the PTX hashes, Triton's chosen
configuration per size and the toolchain; `bench.py` produced it.

Minimum over 15 interleaved rounds (9 at 8192), microseconds, RTX 4070 Ti
SUPER (66 SMs), driver 615.71.09, CUDA 13.4, PyTorch 2.13.0, Triton 3.7.1:

| N | before | after | Triton | after vs before | after vs Triton |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 7.2 | 7.1 | 10.4 | 1.01x faster | 0.69x |
| 512 | 11.8 | 10.2 | 8.8 | 1.16x faster | 1.17x |
| 768 | 30.9 | 24.2 | 24.9 | 1.28x faster | 0.97x |
| 1024 | 42.1 | 41.3 | 32.7 | 1.02x faster | 1.26x |
| 1536 | 151.9 | 140.3 | 133.4 | 1.08x faster | 1.05x |
| 2048 | 275.6 | 207.7 | 200.2 | 1.33x faster | 1.04x |
| 2560 | 543.6 | 474.1 | 444.2 | 1.15x faster | 1.07x |
| 3072 | 841.2 | 711.6 | 722.9 | 1.18x faster | 0.98x |
| 4096 | 1999.4 | 1537.0 | 1568.1 | 1.30x faster | 0.98x |
| 8192 | 16583.0 | 12377.6 | 16091.0 | 1.34x faster | 0.77x |

**Contended**: a game used 23-54% of the GPU from 512 on
(`gpu_busy_before` per size in `results.json`; 3% at 256). The columns were
interleaved and the minimum taken; Triton's minimum at 4096 here (1568 us)
is within 1% of the clean run in `../fp8_gemm_2026-10-12/` (1577 us).
Triton's time at a size also depends on the configuration its autotuner
picks in that process: at 1024 it ranged from 33 to 52 us across these runs.

## What changed

Profiled at 4096^3 with Nsight Compute, the previous kernel and Triton's both
moved 4.4 GB through L2 at 80% of its throughput; Y executed half Triton's
instructions but issued in 9.6% of cycles to Triton's 23.7%, its 8 warps
waiting together on the global-load queue, global loads, the tensor pipe,
the shared-memory queue and the barrier in turn.

- **Fragments come from `ldmatrix`.** An m16n8k32 e4m3 fragment register
  holds 4 consecutive k bytes of one row, two of `ldmatrix.m8n8`'s b16
  elements, so one `ldmatrix.x4` loads an A tile's four registers or two B
  tiles'. 48 `ld.shared.b32` per warp per K-tile became 12; 3-6% on its own
  at 1024-4096.
- **A tile table.** The padded path takes its tile from `fp8_padded_tile`:
  the largest tile that still gives about two waves of CTAs. A 256x128 tile
  of eight 64x64 warp tiles reads 25% less A and B per output than 128x128
  and does twice the tensor-core work per fragment load. Below that, 128x64,
  128x128 (the previous tile), and 64x64 with a 32-wide K step; when M or N
  is at most 512, 64x64 as before, on 8 warps while the grid is one wave.
- The staging divides each tile's A quads and B blocks exactly among its
  threads and refuses a tile where they do not divide: the first version of
  this change divided 256 B blocks among 512 threads, gave each none, and
  left B unstaged (4085 of 4096 elements wrong, caught by the exactness check).

## The sweep

Each shape timed against Triton in the same run, after an exactness check;
the cells are the shape's minimum divided by Triton's (below 1: faster), the
best of the runs it appeared in. `sweep/*.json` holds the runs. The kernel's
tile was chosen with `Y_FP8_TILE=cta_m,cta_n,warps_m,warps_n,cta_k`, an
override the committed compiler does not have; `sweep/tile_override.patch`
adds it back. The two 16-warp 256x128 rows also needed the partial-warp B
staging the committed compiler refuses (see above).

| tile | 256 | 384 | 512 | 768 | 1024 | 1536 | 2048 | 2560 | 3072 | 4096 | 8192 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128x256, 8 warps (2x4), K 32 |  |  |  |  | 1.38 |  | 1.18 |  |  | 1.15 | 0.88 |
| 128x256, 16 warps (4x4), K 32 |  |  |  |  | 1.33 |  | 1.15 |  |  | 1.10 | 0.87 |
| **256x128, 8 warps (4x2), K 32** |  |  |  |  | 0.98 | 1.09 | 1.02 | 1.05 | 1.00 | 0.99 | 0.77 |
| 256x128, 16 warps (4x4), K 32 |  |  |  |  | 1.41 |  | 1.18 |  |  | 1.14 | 0.89 |
| 256x128, 16 warps (8x2), K 32 |  |  |  |  | 0.97 | 1.11 | 1.02 | 1.11 | 1.06 | 1.04 | 0.79 |
| 128x128, 4 warps (2x2), K 32 | 1.19 |  | 2.01 | 1.40 | 0.90 | 1.06 | 1.10 | 1.05 | 1.01 | 0.99 | 0.77 |
| 128x128, 8 warps (2x4), K 32 |  |  |  |  | 0.98 |  | 1.35 |  |  | 1.26 | 1.08 |
| 128x128, 8 warps (4x2), K 32 |  |  |  |  | 0.99 |  | 1.35 |  |  | 1.27 | 1.08 |
| **128x128, 8 warps (4x2), K 64** (before) | 0.97 |  | 1.78 | 1.23 | 0.98 | 1.13 | 1.32 | 1.19 | 1.16 | 1.21 | 0.99 |
| **128x64, 4 warps (2x2), K 32** | 0.75 |  | 1.34 | 0.99 | 0.81 | 1.05 |  | 1.17 | 1.25 |  |  |
| 64x128, 4 warps (2x2), K 32 | 0.83 |  | 1.48 | 1.15 |  |  |  |  |  |  |  |
| **64x64, 4 warps (2x2), K 32** | 0.55 |  | 1.14 | 0.98 | 0.87 | 1.31 |  | 1.51 | 1.61 |  |  |
| **64x64, 4 warps (2x2), K 64** (before) | 0.52 | 0.85 | 1.12 | 1.05 |  |  |  |  |  |  |  |
| 64x64, 8 warps (2x4), K 32 |  |  | 1.08 | 1.04 |  |  |  |  |  |  |  |
| **64x64, 8 warps (2x4), K 64** | 0.80 | 0.76 | 0.98 | 1.43 |  |  |  |  |  |  |  |
| 64x64, 8 warps (4x2), K 32 |  |  | 1.15 | 1.05 |  |  |  |  |  |  |  |
| 64x64, 8 warps (4x2), K 64 |  |  | 1.08 | 1.54 |  |  |  |  |  |  |  |
| 32x64, 2 warps (1x2), K 32 | 0.52 |  | 1.16 | 1.33 |  |  |  |  |  |  |  |
| 32x64, 4 warps (2x2), K 64 |  |  | 1.13 | 1.45 |  |  |  |  |  |  |  |
| 64x32, 4 warps (2x2), K 64 |  |  | 1.07 | 1.37 |  |  |  |  |  |  |  |

Bold: the tiles `fp8_padded_tile` uses. Not timed, because `ptxas` spilled:
128x128 on 4 warps with a 64-wide K step, 256x128 on 8 warps with a 64-wide
K step (which also needs 60 KB of shared memory), 64x64 on one warp, and
128x64 and 64x128 on two.

## What remains

At 512 and 1024 the grid is about one wave (64 CTAs on 66 SMs), and
Triton's three-stage pipeline (`num_stages=3`) hides the global-load
latency that this kernel's one-tile-ahead register prefetch does not. 1536
and 2560 are 5-7% behind.

## Reproduce

    python3 bench.py emit <checkout-before> before/
    python3 bench.py emit <checkout-after> after/
    PYTHONPATH=<checkout>/python python3 bench.py time before/ after/ results.json
