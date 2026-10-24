# F16 GEMM at LLM layer shapes against cuBLAS, 2026-10-24

RTX 4070 Ti SUPER (66 SMs, 48 MB L2), driver 615.71.09, `ptxas` from CUDA
13.4, PyTorch 2.13.0+cu130. The desktop was in use, no game (`env_*.txt`):
contended by the 5% rule. Compiler: 53cbd17.

The records so far timed squares, decode (M <= 32, N = K = 4096) and
512 x 512 x K. These are prefill and MLP shapes: M from 64 to 2048 against
N x K = 4096 x 4096, 11008 x 4096 and 4096 x 11008 (Llama-2-7B's MLP),
14336 x 4096 and 4096 x 14336 (Llama-3-8B's) - 30 shapes, f16 in, f32 out,
chain accumulation (M > 32). The committed profile has no entry for any of
them, so a compile without `--autotune` borrows a measured tile from a shape
within 2x per dimension, or takes the analytic model's.

## Out of the box and tuned (`ab_1`, `at_1`, `at_2`)

`../f16_square_2026-10-18/bench_ab.py`: every column exact on small integers
first, then timed (one B, reused: 32-118 MB, partly L2-resident), minimum of
15 interleaved rounds. `auto` is the compile without `--autotune`
(`gen.sh`), `tuned` with it (`gen_tuned.sh`, into a copy of the profile;
the tuner times B streamed from DRAM). cuBLAS / Y, run 1 / 2 (above 1, Y is
faster):

| shape | auto tile | auto | tuned tile | tuned |
| :--- | :--- | ---: | :--- | ---: |
| 64 x 4096 x 4096 | 32x64 2x2 (M = 32's) | **0.821 / 0.836** | 64x64x64 2x2 s3 | 1.002 / 1.005 |
| 128 x 4096 x 4096 | 64x64 2x2 | 1.062 / 1.067 | 64x64x64 2x2 s3 | 1.030 / 1.034 |
| 256 x 4096 x 4096 | 64x64 2x2 | 1.005 / 1.011 | 64x64x32 2x2 s4 | 1.005 / 1.001 |
| 512 x 4096 x 4096 | 64x128 2x4 | 1.024 / 1.024 | 64x64x64 2x2 s3 | 1.001 / 1.001 |
| 1024 x 4096 x 4096 | 128x64 2x2 | 1.016 / 1.016 | 64x128x32 2x2 s3 | 1.021 / 1.021 |
| 2048 x 4096 x 4096 | 128x64 2x2 | 1.018 / 1.019 | 128x128x32 2x2 s2 | 0.991 / 0.993 |
| 64 x 11008 x 4096 | 64x64 2x2 | 1.031 / 1.032 | 64x64x64 2x2 s2 | 1.032 / 1.024 |
| 128 x 11008 x 4096 | 64x64 2x2 | **0.975 / 0.975** | 64x64x64 2x2 s3 | 0.965 / 0.966 |
| 256 x 11008 x 4096 | 64x64 2x2 | **0.980 / 0.980** | 64x64x64 2x2 s3 | 0.986 / 0.981 |
| 512 x 11008 x 4096 | 64x128 2x4 | 1.082 / 1.085 | 64x64x64 2x2 s3 | 1.131 / 1.132 |
| 1024 x 11008 x 4096 | 128x128 2x2 | **0.954 / 0.954** | 128x64x32 2x2 s3 | 0.976 / 0.974 |
| 2048 x 11008 x 4096 | 128x128 2x2 | 1.041 / 1.049 | 64x128x32 2x2 s3 | 1.054 / 1.063 |
| 64 x 4096 x 11008 | 64x64 2x2 | 1.046 / 1.064 | 64x64x64 2x2 s3 | 1.078 / 1.083 |
| 128 x 4096 x 11008 | 64x64 2x2 | 1.072 / 1.072 | 64x64x64 2x2 s4 | 1.098 / 1.090 |
| 256 x 4096 x 11008 | 64x64 2x2 | 0.982 / 0.979 | 64x64x64 2x2 s3 | 0.968 / 0.971 |
| 512 x 4096 x 11008 | 64x128 2x4 | 0.999 / 0.999 | 64x64x32 2x2 s4 | 0.991 / 0.990 |
| 1024 x 4096 x 11008 | 128x128 2x2 | 0.987 / 0.987 | 128x64x32 2x2 s4 | 1.003 / 1.003 |
| 2048 x 4096 x 11008 | 128x128 2x2 | 0.988 / 0.988 | 64x128x32 2x2 s3 | 1.007 / 1.007 |
| 64 x 14336 x 4096 | 64x64 2x2 | 1.023 / 1.021 | 64x64x64 2x2 s3 | 1.000 / 1.001 |
| 128 x 14336 x 4096 | 64x64 2x2 | 0.982 / 0.935 | 64x64x32 2x2 s3 | 1.077 / 1.036 |
| 256 x 14336 x 4096 | 64x64 2x2 | 1.136 / 1.127 | 64x64x64 2x2 s3 | 1.139 / 1.131 |
| 512 x 14336 x 4096 | 64x128 2x4 | 1.133 / 1.132 | 64x64x64 2x2 s3 | 1.142 / 1.141 |
| 1024 x 14336 x 4096 | 128x128 2x2 | 1.000 / 0.999 | 128x64x32 2x2 s2 | 0.987 / 0.986 |
| 2048 x 14336 x 4096 | 128x128 2x2 | 0.998 / 0.997 | 64x128x32 2x2 s4 | 1.029 / 0.998 |
| 64 x 4096 x 14336 | 64x64 2x2 | 1.030 / 1.068 | 32x64x64 2x2 s2 | 1.026 / 1.032 |
| 128 x 4096 x 14336 | 64x64 2x2 | 1.078 / 1.081 | 64x64x64 2x2 s3 | 1.068 / 1.075 |
| 256 x 4096 x 14336 | 64x64 2x2 | 0.977 / 0.974 | 64x64x64 2x2 s3 | 0.971 / 0.968 |
| 512 x 4096 x 14336 | 64x128 2x4 | 0.997 / 0.997 | 64x64x64 2x2 s3 | 0.986 / 0.987 |
| 1024 x 4096 x 14336 | 128x128 2x2 | 0.986 / 0.986 | 128x64x32 2x2 s4 | 1.001 / 1.002 |
| 2048 x 4096 x 14336 | 128x128 2x2 | 0.986 / 0.987 | 128x128x32 2x2 s2 | 0.979 / 0.977 |

Y is level with cuBLAS or ahead at most shapes, up to 1.14x (M = 256-512,
N = 14336). Three findings:

**1. The worst loss is a borrowed decode tile.** 64 x 4096 x 4096 is within
2x of the profile's M = 32 decode entry, which was tuned for decode
(float32 accumulation, B from DRAM) and takes 1.22x cuBLAS's time at M = 64.
`m64/`: without that entry the analytic model picks 64x64x32 2x2, 0.97-0.98x
cuBLAS; the tuned tile is 1.01x (`noborrow` in `m64_*.json`: 31.0 / 30.4 us,
borrowed 36.2 / 36.6, tuned 29.7 / 29.7, cuBLAS 30.0 / 29.9). Since this
record, `load_persisted_nearest` never borrows across M = 32
(`tests/gemm_tile_borrow.rs`), so that shape gets the analytic tile.

**2. The tuner's pick was slower than the tile it replaced at 13 of 30
shapes, in both runs, by 0.4-3.5%** (faster at 15, by up to 22%, the M = 64
case). It chose 2x2 warps at every shape. `tuner_logs/`: at 512 x 4096 x 4096 its
fastest candidate by minimum was 64x128x64 2x4 s3 (213.72 us), and it took
64x64x64 2x2 s3 (214.84 us) "on warp count then K depth", all nine inside
its band of +-2.30 us; here that pick is 2.3% slower than the 2x4-warp tile
the compile without `--autotune` borrowed. At 1024 x 11008 x 4096 its
regime disagrees with this harness's: streaming B, 128x64x32 s3 led
(1074.27 us) and 64x128x32 s3 came fifth (1090.67); with one reused B the
order flips (below).

**3. M = 128-256 against N or K of 11008 is 2-3.5% behind with every tile
tried.** `tiles/` (`sweep_tiles.sh`, eight tiles, two runs):

| shape | best tile | cuBLAS / Y | runner-up |
| :--- | :--- | ---: | :--- |
| 128 x 11008 x 4096 | 64x64x32 2x2 s4 | 0.974 / 0.974 | 128x64x32 4x2 s3, 0.932-0.935 |
| 256 x 11008 x 4096 | 64x64x32 2x2 s4 | 0.979 / 0.978 | 64x128x32 2x2 s3, 0.913-0.914 |
| 256 x 4096 x 11008 | 64x128x32 2x2 s3 | 1.004 / 1.005 | 128x128x32 4x2 s3, 1.000-1.001 |
| 1024 x 11008 x 4096 | 64x128x32 2x2 s3 | 1.011 / 1.015 | 64x64x32 2x2 s4, 0.990-0.994 |

So two of the four have a tile that beats cuBLAS, which neither compile
found; two do not. Splitting K does not help there (`split/`, `sbench.py`,
CUDA graphs, two runs on integers and N(0, 1)): split 2 and 4, float32 or
chained, 64x64 or 64x128 tiles, all 0.73-0.92x cuBLAS, against the unsplit
64x64x32 s4's 0.954-0.981 - the float32 partials (11 MB per split at
M = 128) cost more than the waves they balance. cuBLAS runs
`ampere_s1688gemm_fp16_128x128_ldg8_stages_32x1_nn` there (`prof/`): one
128x128 tile row, 86 CTAs, loads staged through registers, one shared
stage; Y's own 128x128 tile is 0.83x at that shape.

## Why cuBLAS wins at 128 x 11008 x 4096 (`prof/cyc_*.txt`)

`ncu`, one launch each, N(0, 1) inputs, in SM cycles (the clock is not
comparable across `ncu` runs):

| | Y 128x128 2x2 s3 | Y 64x64 2x2 s4 | cuBLAS |
| :--- | ---: | ---: | ---: |
| grid | (86, 1, 1) | (172, 2, 1) | **(86, 1, 3)** |
| elapsed, slowest SM | 557,620 | 424,504 | 432,572 |
| active cycles, busiest / least busy SM | 553,711 / 336,560 | 422,035 / 349,290 | 427,954 / 342,427 |
| tensor-pipe cycles, busiest SM (sum over SMs 45,088,768 for all three) | 1,048,576 | 786,432 | 704,512 |

cuBLAS (`ampere_s1688gemm_fp16_128x128_ldg8_stages_32x1_nn`) splits K in
three inside the one kernel - 258 CTAs, no reduction kernel - so its
busiest SM carries 1.34 tiles' worth of tensor work. Y's 128x128 grid gives
whole tiles to SMs, two to some and one to the rest: exactly 2 x 524,288
tensor cycles on the busiest. Y's 64x64 tile balances as well as cuBLAS
(and takes fewer cycles here) but loads about twice the tile bytes from the
L2: 344 CTAs of 64 x 4096 + 4096 x 64 halves, 344 MB, against 258 of
128 x 1365 + 1365 x 128, 180 MB (computed from the tiles, not measured) -
which may be why it trails at the boost clock of `bench_ab.py`. Y's split K, with its float32 partials and second
launch, costs more than it balances (`split/`). An in-kernel split whose
slices add into C in turn would match cuBLAS's structure; it needs a
per-tile flag and CTAs that wait on one another, which CUDA does not
guarantee to make progress.
