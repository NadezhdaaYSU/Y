# FP32 GEMM on CUDA cores against cuBLAS (2026-11-05)

`@tile(M, N, K) kernel g(A, B, C: GlobalMemory<F32>)`, emitted by
`src/ptx_emitter/f32_simt_gemm.rs`. The goal was to **match** cuBLAS, so that a
Y program does not give up speed for Y's checks; beating it was not the aim.

- NVIDIA GeForce RTX 4070 Ti SUPER (sm_89, 66 SMs), driver 615.78.08
- ptxas 13.4 (driver JIT of the emitted PTX); cuBLAS from torch 2.13.0+cu130
  (CUDA 13.0), `allow_tf32 = False`, `float32_matmul_precision("highest")`;
  cupy 14.1.1 launches Y's kernel
- `bench.py` (this directory) is the harness; `final.txt` the last run on the
  committed code, `run_before_final.txt` an earlier run of the same code

## Result

Every shape is **bit-identical** to an independent kernel that folds `fmaf`
over k = 0..K-1, on the header's grid and on a 3x1 grid (`final.txt`).

| shape | tile | cuBLAS TF/s | Y / cuBLAS speed (two runs) |
| :--- | :--- | ---: | ---: |
| 8192^3 | large | 27.6 | 0.98, 0.975 |
| 4096^3 | large | 30.8 | 0.96, 1.00 |
| 3072^3 | large | 29.4 | 0.93, 0.96 |
| 2048^3 | large | 30.0 | 1.03, 1.04 |
| 1536^3 | mid | 22.4 | 0.98, 1.01 |
| 1024^3 | mid | 22.2 | 1.20, 1.20 |
| 768^3 | mid | 18.6 | 0.81, 0.85 |
| 512^3 | small | 14.3 | 1.19, 1.20 |
| 4096 x 4096 x 1024 | large | 30.5 | 0.96, 1.00 |
| 1024 x 4096 x 4096 | large | 30.4 | 1.02, 1.02 |
| 1024 x 1024 x 4096 | mid | 28.9 | 0.95, 0.94 |
| 256 x 4096 x 4096 | mid | 27.3 | 0.95, 0.94 |
| 128 x 4096 x 4096 | mid | 22.4 | 0.99, 1.02 |
| 4096 x 128 x 4096 | mid | 20.1 | 1.04, 1.01 |
| 64 x 8192 x 2048 | mid | 17.7 | 1.03, 1.03 |

Runs differ by about 3%: under this load the card reaches its ~285 W cap and
its clock moves between 2.3 and 2.6 GHz. Both sides are measured in the same
interleaved rounds.

At 256^3 the event timing measures cupy's launch (about 0.56x for every tile).
By kernel time (ncu, `kernel_time_256.txt`), cuBLAS runs two kernels,
5.7-6.2 us of split-K plus a 2.4-2.5 us reduction; Y runs one of 9.1 us.

### Where cuBLAS is ahead, and why Y does not follow

At 768^3, and at the 1024 x 1024 x 4096 and 256 x 4096 x 4096 shapes, cuBLAS
splits K (`cublas_kernels.txt`: a `Memset` or `splitKreduce` beside the GEMM).
Y does not, by design. Each element of Y's C is one `fma.rn.f32` chain in k
order, so its bits do not depend on M, the tile or the grid. A split would
make the answer depend on the shape.

The split also makes cuBLAS more accurate on those shapes. Measured as
max |C - C64| / (|A| |B|) on randn inputs:

| shape | cuBLAS | Y |
| :--- | ---: | ---: |
| 1024^3 | 1.0e-7 | 3.6e-7 |
| 2048^3 | 1.5e-7 | 3.8e-7 |
| 2048 x 2048 x 8192 | 1.5e-7 | 4.0e-7 |
| 4096^3 | 3.8e-7 | 3.8e-7 (no split) |

So on the split shapes the comparison is not between equal outputs.

## What it took

Each step was measured before it was kept.

1. **Ceiling.** A pure-FFMA kernel runs at 42.7 TF/s at 2640 MHz. cuBLAS runs
   `cutlass_80_simt_sgemm_256x128_8x4_nn_align1` from 2048 up. Under the
   ~285 W cap that is about 85% of the capped ceiling, so there is little
   headroom at large sizes.
2. **First plain kernel** (128x128x8, register-staged): 0.84-0.93x from 2048 up.
   Smem padding, a fragment double buffer and BK = 16 each pushed it past 128
   registers and halved occupancy (0.64-0.85x).
3. **CUTLASS's structure:** a 3-4 stage `cp.async` pipeline, A transposed into
   shared memory by 4-byte copies, 128x256 tiles. This gave 0.92-0.95x.
4. **Compile-time shapes** gave the largest gain. ncu showed the same
   instruction counts as cuBLAS and no bank conflicts, but each K tile issued
   about 150 integer instructions in a row to recompute its copy addresses.
   With M, N and K constant, each copy is one `cp.async` at an immediate
   offset from a pointer advanced once per tile. With BK = 16 this reached
   0.96-1.08x.
5. **Grouped raster** of 8 tile rows kept B's columns in L2: +3-5% at
   4096-8192.
6. **In Y's emitter**, the same design first ran at 0.87x where the CUDA
   version ran at 1.04x. The bits were identical, so only scheduling
   differed:
   - Reusing two fragment buffers made ptxas cluster the shared loads (MIO
     throttle 0.19 against 0.03). Fresh registers for each k step: 0.92x.
   - Putting every load and copy of the tile before its FMAs, the order NVVM
     uses: 0.96x.
   - Predicated 64-bit pointer adds were the last 6%. Branching around the
     copies and the advance, as NVVM does, gave 1.01-1.02x, equal to the
     CUDA version. Predicated copies with unconditional adds reached only
     0.97x.
   - NVVM's FMA order (each output 2, 4, 8 or 16 k steps at a time), the
     `.maxntid` directive, and one tile per CTA instead of the grid-strided
     loop did not help.
7. **Tile choice.** Each tile was forced on each gap shape (`tile_sweep.txt`).
   The rule's choice was the fastest or within noise everywhere. A fourth,
   8-warp 4x4-per-thread tile was never faster, so it was dropped.

## Checks

`tests/f32_simt_gemm.rs`:
- the tile rule, `.maxnreg`, and ptxas with no spills;
- refusals: off-tile shapes, the host backend, a near-miss signature;
- bit-exact against `f32::mul_add` folded in order, on six shapes covering
  all three tiles and a single K tile, each on three grids including more
  CTAs than tiles;
- the block-size trap, run in a child process;
- compute-sanitizer memcheck and racecheck clean.

14 mutants of the emitter were all killed. Three of them (a missing barrier
between tiles, a load past K, a prologue load past K) give correct bits and
were killed only by the sanitizer test.
