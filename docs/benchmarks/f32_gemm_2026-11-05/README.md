# FP32 GEMM on CUDA cores against cuBLAS (2026-11-05)

`@tile(M, N, K) kernel g(A, B, C: GlobalMemory<F32>)`, emitted by
`src/ptx_emitter/f32_simt_gemm.rs`, and since 2026-11-06 the same with a
workspace `W: GlobalMemory<U64>`, which splits K in order
([below](#k-split-in-order-2026-11-06)). The goal was to **match** cuBLAS, so that a
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

## B rows at their own width (2026-11-06)

With N % 4 != 0, B was copied 4 bytes at a time, and at 1024 x 50257 x 768
that saturated the global-load queue. ncu: `lg_throttle` 0.57 against 0.04
at N = 50176, issue 66% against 81%; the aligned shape ran at 1.07x cuBLAS,
the odd one at 0.87x.

The fix uses compile-time alignment. Every K tile starts at a multiple of
16 rows, so whether row `kk` of a tile is 16-, 8- or 4-byte aligned depends
only on `kk * N mod 4`, which is known when compiling. Each row is now
copied at its widest width:

| N | 16-byte rows | 8-byte rows | 4-byte rows | B copies per thread (large tile) |
| :--- | :--- | :--- | :--- | :--- |
| odd | a quarter | a quarter | half | 16 -> 11 |
| 2 mod 4 | half | half | none | 16 -> 6 |

The shared-memory layout is unchanged and the copies stay asynchronous.
Four restrictions, each measured:
- **N odd only on the 256-column tiles.** On the 128-column mid tile the
  mixed widths were slower (1023 x 1025 x 1027: 1.15x the time;
  `row_widths_ab.txt`).
- **Not with a ragged column edge.** A chunk wider than 4 bytes would
  straddle N. Copying it in part (the rest zero-filled) reads, by memcheck,
  16 bytes at the end of B's last row, 8 past the allocation. Such a set
  keeps one width that divides N: 8 bytes for N even, else 4.
- **Splitting the edge off.** Because of the rule above, splitting the
  edge into remainder tiles is what lets the whole tiles take the widths.
  So the 50-tile limit on splitting is waived there:
  - 1024 x 30522 x 768: 0.86x unsplit, 0.97x split;
  - 1024 x 50257 x 768: 0.83x, 0.85x.
- **Not with a K tail.** The K-tail guards on every copy, beside the
  per-width registers, spilled: about 400 bytes on the 64-row tiles, and
  616-920 bytes even under the large tile's 255. Raising the mid tile to
  255 registers removed its spills but ran odd N 10-15% slower (ptxas used
  210 registers), so the 170 budget stays.

The old and new kernels, timed together (`row_widths_ab.txt`), give the
same bits on every shape:

| shape | before | now |
| :--- | ---: | ---: |
| 1024 x 30522 x 768 (BERT's vocabulary) | 0.85 | 1.00 |
| 4096 x 30522 x 768 | 0.88 | 0.99 |
| 2048 x 2050 x 2048 | 0.94 | 1.02 |
| 2047 x 2050 x 2051 (K tail: one 8-byte width) | 0.90 | 0.96 |
| 1024 x 1026 x 1024 | 0.75 | 0.79 |
| 1024 x 50257 x 768 (GPT-2's vocabulary) | 0.81 | 0.85 |
| 4096 x 50257 x 768 | 0.87 | 0.87 |
| 4096 x 4097 x 4096 | 0.99 | 0.98 |
| 1023 x 1025 x 1027, 2047 x 2049 x 2051 (unchanged code) | 0.78, 0.90 | 0.77, 0.91 |
| 1024^3, 4096^3 (unchanged code) | 1.15, 1.02 | 1.15, 0.98 |

The last two rows' SASS is identical before and after. Their differences
are the card's run-to-run spread (it sits at its power cap).

## K split in order (2026-11-06)

The losses at 768^3 and similar shapes were idle SMs, not slow tiles. The
three-parameter kernel alone, mid tiles at K = 768 (`streamk_why.txt`, and
an earlier run of the same in brackets):

| mid tiles | per SM | Y time |
| :--- | :--- | ---: |
| 66 | one | 51.1 us (38.1) |
| 72 (768^3) | one, six SMs two | 88.7 us (67.7) |
| 132 | two | 90.8 us (67.2) |
| 198 | three only fit two at a time | 145.4 us (119.4) |

So at 768^3, 60 SMs finished and waited while six ran a second tile. A
second mid CTA on an SM adds only 13-14% throughput. cuBLAS splits K there,
which changes the summation order.

**This splits K in order.** A fourth parameter, the workspace
`W: GlobalMemory<U64>` (one word per tile), lets the kernel run P CTAs,
the number that fit at once. Each CTA:
1. runs whole tiles grid-strided, so each wave is P consecutive tiles, as
   in the three-parameter kernel;
2. takes an equal share of the (tile, K tile) units of the last tiles, one
   to two waves' worth.

A tile cut between two CTAs is continued, not summed. The first stores its
exact f32 accumulators in C and writes in `W[tile]` how far it got. The
next waits for that, loads them, and carries the same `fma` chain on. So
every element is still the one chain over k = 0..K-1, the same bits as the
three-parameter form.

- **Order of a CTA's work.** The head of its last tile first (it waits on
  nothing, and the next CTA needs it), then whole tiles, then the
  continuation of its first tile. A CTA waits only on lower-numbered CTAs,
  each of which has published before it waits.
- **`W` needs no clearing.** Each word is tagged with the launch's
  `%gridid`, so a word left by an earlier launch, or garbage, never matches.
  The tests fill it with garbage and never clear it.
- **A wait over 30 s traps** instead of hanging (a grid larger than the
  CTAs that fit at once whose CTAs did not start in index order, or two
  launches sharing one `W`).

Before building it, a CUDA C++ probe of the same kernel showed the gain
(`streamk_probe.py`, `streamk_probe.txt`): 768^3 0.815x cuBLAS one tile
per CTA, 1.214x split on 66 CTAs, bit for bit. Split on 132 CTAs it was
0.762x: with less than a tile per CTA, a tile is cut in three and the
pieces wait on each other.

**The first kernel split every tile's K evenly and lost at 5120^3**: 0.896x
cuBLAS, against 1.017x for one tile per CTA (`streamk_even_split.txt`). The
tiles in flight were 12 apart, so they shared no rows of A or columns of B
in L2. Running whole tiles in waves first and splitting only the last ones
fixed it: 1.035x in the same harness.

**The plan** (`f32_gemm_streamk_plan`) is a model with this card's SM
rates (a second mid CTA +14%, a second or third small CTA +8%): one tile
per CTA in waves, the remainder tiles at their measured cost, against
stream-K on P = SMs x j CTAs. It takes stream-K where the model says at
least 4% faster, and only where each CTA's share is at least a tile. Only
for N % 4 == 0: with N % 4 != 0 the plain path's remainder tiles are what
let B's rows take their own widths. Elsewhere `W` is unused, and the kernel
is the three-parameter one: identical SASS (a test checks it), and 0.01%
apart in time (`streamk_order_ab.txt`).

### Result

`streamk_final.txt` (`streamk_bench.py`), two runs. Run 1 timed the
kernels in a fixed order, which at the card's power cap handicaps the last
one, here the workspace form: with W unused and identical SASS it measured
2.6-4.8% slower at 4096^3, and 0.01% apart in a shuffled order
(`streamk_order_ab.txt`). Run 2 shuffles the order every round. Every
output is bit-exact against the in-order oracle on four grids (the plan's,
half of it, 1 and 3 CTAs), with `W` full of garbage, and again after the
timing's hundreds of launches on one `W`.

| shape | tile, grid | without W (run 1, 2) | with W (run 1, 2) | time with / without (run 2) |
| :--- | :--- | ---: | ---: | ---: |
| 768^3 | mid, 66 | 0.850, 0.857 | 1.239, 1.247 | 0.687 |
| 896^3 | mid, 66 | 0.994, 0.993 | 1.089, 1.089 | 0.912 |
| 1152^3 | mid, 132 | 1.009, 0.993 | 1.211, 1.194 | 0.832 |
| 1280^3 | mid, 132 | 0.930, 0.944 | 1.310, 1.274 | 0.741 |
| 1536^3 | mid, 132 | 1.051, 1.028 | 1.207, 1.203 | 0.855 |
| 1500^3 (ragged, K tail) | mid, 132 | 0.956, 0.964 | 1.098, 1.119 | 0.862 |
| 2304^3 | large, 66 | 0.924, 0.919 | 1.073, 1.072 | 0.858 |
| 2560^3 | mid, 132 | 0.974, 0.980 | 1.039, 1.049 | 0.934 |
| 3072^3 | large, 66 | 0.958, 0.967 | 1.061, 1.068 | 0.905 |
| 3000^3 (ragged, K tail) | large, 66 | 0.956, 0.966 | 0.992, 1.000 | 0.966 |
| 5120^3 | large, 66 | 0.998, 1.070 | 1.036, 1.144 | 0.935 |
| 640^3 | small, 66 | 0.945, 0.946 | 1.038, 1.045 | 0.906 |
| 512 x 768 x 768 | small, 66 | 0.806, 0.805 | 0.940, 0.920 | 0.875 |
| 1024 x 1028 x 1024 | mid, 132 | 0.987, 0.959 | 1.110, 1.113 | 0.862 |
| 2048 x 768 x 3072 | mid, 132 | 0.719, 0.725 | 0.973, 0.976 | 0.743 |
| 512 x 3072 x 768 | mid, 132 | 1.050, 1.072 | 1.114, 1.116 | 0.961 |
| 1024^3 | W unused | 1.258, 1.242 | 1.264, 1.243 | 0.999 |
| 4096^3 | W unused | 0.987, 1.075 | 0.942, 1.077 | 0.998 |
| 1024 x 1024 x 4096 | W unused | 0.961, 0.978 | 0.963, 0.982 | 0.996 |
| 256 x 4096 x 4096 | W unused | 0.941, 0.948 | 0.941, 0.951 | 0.997 |

cuBLAS itself moved 6% between the runs at 4096^3 and 5120^3 (the power
cap), so the last column, from one shuffled run, is the steadier measure.

Still behind cuBLAS with `W`: 512 x 768 x 768 (0.92-0.94x), 2048 x 768 x
3072 (0.97-0.98x, from 0.72x), and the K-heavy shapes whose tiles already
fill the SMs (0.94-0.98x), where cuBLAS's split adds parallelism that an
in-order chain cannot. N % 4 != 0 is unchanged.

**Grids larger than the CTAs that fit at once** finish with the same bits
on this card (`streamk_oversubscribed.txt`: 2P + 5, 4P and 1000 CTAs; 1000
cuts each of 768^3's tiles into about 14 pieces). That rests on CTAs
starting in index order, which CUDA does not promise; the 30 s trap is the
fail-safe if they do not.

## Checks

`tests/f32_simt_gemm.rs`:
- the tile rule, `.maxnreg`, and ptxas with no spills;
- refusals: N or K of 2^22, the host backend, a near-miss signature;
- bit-exact against `f32::mul_add` folded in order, on 31 shapes, each on
  three grids including more CTAs than tiles. They cover:
  - all three tiles and all five remainder tiles;
  - a single K tile, and 1x1x1;
  - N odd and N = 2 mod 4, a column edge alone;
  - leftover rows that exactly fill their strips;
  - ragged M, N and K on every tile;
  - B at each row's width with K past the prologue, and the one-width
    fallbacks;
- the remainder sets and B copy widths each shape gets, from its header;
- a ragged K keeping a -0.0 sum;
- the block-size trap, run in a child process;
- compute-sanitizer memcheck and racecheck clean on 17 shapes, 13 of them
  ragged;
- the workspace form (2026-11-06):
  - the plan's choice on 11 shapes, and ptxas with no spills;
  - where `W` is unused, SASS identical to the three-parameter kernel;
  - bit-exact on 16 shapes (each tile; ragged M and N; a K tail; one, two
    and many K tiles), each on 7 grids: the plan's twice (the second launch
    finds the first's words in `W`), 1, 2 and 3 CTAs, the plan's + 5, and
    more CTAs than tiles. `W` holds garbage and is never cleared;
  - memcheck and racecheck clean on 8 of them, `W` exactly as large as the
    header says;
  - a grid of 8.6e9 CTAs (65536 x 65535 x 2, legal in CUDA) traps at once;
    with the check removed it ran for a minute and returned;
  - the host backend refuses it, and a `W` of U32 is refused.

The emitter was mutated 52 times (11 against the remainder sets, 11
against the per-row B widths), and 50 of the mutants were killed. Most
mutants of the edge handling (size-0 copies, the column clamp, a missing
barrier between tiles) still give correct bits, so only the sanitizer test
kills them. Two survivors are equivalent:
- the tail's `wait_group 0` against `1`, where with 3 stages the newest
  group is always empty;
- a K guard reading the first width's row for every width, where a K tail
  always means a single width.

Each test gap a survivor exposed was closed by adding a shape:
- leftover rows that exactly fill their strips;
- several B widths with K past the prologue;
- an edge with no K tail.

The stream-K code was mutated 13 times (`streamk_mutate.py`,
`streamk_mutants.txt`), and all 13 were killed:
- wrong bits: continuation loads from K tile 0, a continuation starting
  from zero, the K tail run as a full tile, no launch tag, no barrier after
  the wait, whole tiles not grid-strided;
- the 30 s trap, each in a launch that could never finish: a wait no
  producer satisfies, split tiles not offset past the whole waves,
  publishing where a segment started instead of where it stopped;
- the plan test: a threshold of 0.5 instead of 1.04;
- the sanitizer: loads past a segment's end in the loop and in the
  prologue, and a loop K limit that forgot where its segment started. That
  last one survived the first run (every sanitized shape had at most 3 K
  tiles, so the prologue loaded the partial tile) and was killed once two
  shapes with a K tail and many K tiles were added.
