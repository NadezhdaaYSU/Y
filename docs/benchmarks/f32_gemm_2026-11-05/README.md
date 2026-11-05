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

## Any size (same day)

Since the same day the kernel takes any M, N and K below 2^22.

- **Edges.** Rows past M and k past K are `cp.async` copies of size 0,
  which read nothing (memcheck, racecheck and initcheck are clean on shapes
  from 1x1x1 to 2047 x 4099 x 36). Columns past N are clamped on read.
  Nothing past the edges is stored.
- **K tail.** The last, partial K tile runs only its valid steps, so a
  ragged K keeps the in-order chain, -0.0 included.
- **N % 4 != 0.** A B row is not 16-byte aligned, so B is copied and C
  stored 4 bytes at a time.
- **Aligned shapes are unchanged.** They emit SASS identical to before
  (checked at five shapes).

`ragged_final.txt` (`bench.py ragged`), every shape bit-identical:

| shape | tile | Y / cuBLAS |
| :--- | :--- | ---: |
| 1000^3 | mid | 1.25 |
| 1500^3 | mid | 0.95 |
| 3000^3 | large | 0.94 |
| 1024 x 1028 x 1024 | mid | 0.80 |
| 1023 x 1025 x 1027 (N odd) | mid | 0.59 |
| 2047 x 2049 x 2051 (N odd) | mid | 0.80 |
| 4095 x 4097 x 512 (N odd) | large | 0.76 |
| 4096 x 4097 x 4096 (N odd) | large | 0.86 |
| 1024 x 50257 x 768 (GPT-2 vocabulary, N odd) | large | 0.88 |
| 4096 x 50257 x 768 | large | 0.90 |
| 1 x 4096 x 4096 | small | 0.93 |
| 1 x 50257 x 768 | mid | 0.97 |

The two losses (`ragged_tile_sweep.txt`):

- **Odd N.** Copying B 4 bytes at a time doubles the copy instructions.
  ncu at 1024 x 1025 x 1024 on the mid tile: MIO throttle 0.32 against
  0.10 at N = 1028, and issue 62% against 74%. Only the 128x256 tile has
  enough FMAs per copied element to hide it, as cuBLAS's align1 kernels do.
  Mid-sized shapes have too few of those tiles to fill the GPU.
- **A nearly empty last tile column.** At 1024 x 1028 x 1024 it adds a
  wave, and every tile measured 0.74-0.78x.

The tile rule changed with this: the large tile now needs its waves at
least 80% full and at least 128 rows.
- 2047 x 2049 x 2051 went from the large tile (0.65x) to mid (0.80x).
- 1 x 50257 x 768 went from large (0.68x) to mid (0.97x).

Below about 512^3 the event timing here measures cupy's launch (1x1x1:
0.25x), not the kernel.

## Remainder tiles (same day)

Measuring Y alone at equal tile counts separated the two ragged-edge
losses.

- **A nearly empty last tile row or column costs a whole wave.**
  - 1024 x 1028 x 1024 took 151 us, the same as 1024 x 1152 x 1024 (147 us,
    also 144 tiles), against 85.5 us for 1024^3 (128 tiles).
  - 1025 x 1024 x 1024 took 154 us against 88.
  - 4096 x 4097 x 4096 needed 8.2 waves of large tiles, rounded up to 9.
- **4-byte B copies (N odd) cost the rest.** At equal tile counts they
  added 37% on the mid tile (207 against 151 us) and 14% on the large one.

Leftover rows and columns of at most half a tile now go to narrow tiles in
the same launch:
- rows: 32-row strips (`rows32x256`, `rows32x128`, `rows32x64`);
- columns: 64 or 32 columns wide (`rem128x64`, `rem64x32`).

The header states which sets a kernel uses. A narrow tile costs about a
third of a whole one, which sets two limits:
- **At most half a tile.** Splitting off more is slower: 1000^3 left 40 of
  64 rows and 104 of 128 columns, took 117 us split and 78 us whole, so it
  stays whole.
- **At most 50 tiles on that side.** With more, the ragged line is under 2%
  of the work: at 1024 x 50257 x 768 (197 tile columns) splitting was 3%
  slower.

Each output is still one in-order chain, so the bits do not change.

`remainder_final.txt`, against cuBLAS (the earlier column is the same
shape before remainder tiles):

| shape | before | now |
| :--- | ---: | ---: |
| 1025 x 1024 x 1024 | (154 us) | 1.09 |
| 1024 x 1028 x 1024 | 0.80 | 0.91 |
| 1023 x 1025 x 1027 (N odd) | 0.59 | 0.80 |
| 2047 x 2049 x 2051 (N odd) | 0.80 | 0.90 |
| 3000^3 | 0.94 | 0.97 |
| 4095 x 4097 x 512 (N odd) | 0.76 | 0.83 |
| 4096 x 4097 x 4096 (N odd) | 0.86 | 0.97 |
| 1 x 4096 x 4096 | 0.93 | 0.99 |
| 1000^3, 1500^3, 1 x 50257 x 768 | 1.25, 0.95, 0.97 | 1.26, 0.94, 0.98 (not split) |

Aligned shapes are as before (0.96-1.24x in the same run).

- **GPT-2's projection, not split.** 1024 x 50257 x 768 measured 0.79-0.85x
  in this session's runs, against 0.87-0.88x earlier. Its old and new
  kernels give identical bits. Timed together in one process, the new one
  was 0.8% and 3.9% slower in two runs. That may be a small real cost of
  computing the copy roles per tile, but most of the spread is the card's
  clock, which sits at its power cap.
- **Tried and dropped:** a 128x128 tile (`remainder_experiments.txt`) was
  within 3% of the current tiles on two K-heavy shapes and slower on the
  rest.

## Checks

`tests/f32_simt_gemm.rs`:
- the tile rule, `.maxnreg`, and ptxas with no spills;
- refusals: N or K of 2^22, the host backend, a near-miss signature;
- bit-exact against `f32::mul_add` folded in order, on 24 shapes, each on
  three grids including more CTAs than tiles. They cover all three tiles
  and all five remainder tiles, a single K tile, 1x1x1, N odd and
  N = 2 mod 4, a column edge alone, leftover rows that exactly fill their
  strips, and ragged M, N and K on every tile;
- the remainder sets each shape gets, from its header;
- a ragged K keeping a -0.0 sum;
- the block-size trap, run in a child process;
- compute-sanitizer memcheck and racecheck clean on 14 shapes, 10 of them
  ragged.

The emitter was mutated 41 times, and 40 of the mutants were killed
(the last 11 target the remainder sets). Most
mutants of the edge handling (size-0 copies, the column clamp, a missing
barrier between tiles) still give correct bits, so only the sanitizer test
kills them. One survivor is equivalent: the tail's `wait_group 0` against
`1`, where with 3 stages the newest group is always empty.
