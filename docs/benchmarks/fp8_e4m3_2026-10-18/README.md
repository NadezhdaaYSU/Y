# FP8 GEMM over e4m3 bytes, and float32 accumulation, 2026-10-18

The head-to-head of 2026-10-13 (`../head_to_head_2026-10-13/`) left one row
where Y lost: its FP8 GEMM at 8192^3 took 1.12x the time of PyTorch's
`torch._scaled_mm` pipeline. Both compute float32 C from float32 A [M][K] and
B [K][N] through e4m3 tensor cores. Y's kernel quantises inside its K loop,
so every CTA converts every element of its A rows and B columns, 64 times
over at 8192. PyTorch quantises each operand once, in separate kernels
(about 4.1 ms at 8192), then runs cuBLASLt's GEMM (6.49 ms).

This change adds:

- `E4M3`, FP8 e4m3 as a `GlobalMemory` element type (manual §20.5): a store
  rounds an F32 to nearest-even and saturates at +-448, a load widens
  exactly. Quantising a matrix is then ordinary Y code (`kernels/quant.ysu`).
- An `@tile` GEMM over e4m3 bytes (manual §9.12): A [M][K] and B [N][K] as
  `_scaled_mm` takes them, copied with `cp.async` into XOR-swizzled shared
  tiles.
- Float32 accumulation in both FP8 GEMMs: each `mma` starts from zero and is
  added into the accumulator with `add.rn.f32` (below).

RTX 4070 Ti SUPER (66 SMs), driver 615.71.09, PyTorch 2.13.0+cu130, Triton
3.7.1, `ptxas` 13.4. Every timed run began with the GPU's 10-sample mean utilisation at or
below 5% and no game running (`QUIET` line of each log), and read the SM
clock after every sample (in each `.json`). Every column was checked before it
was timed: equal to the float64 product bit for bit, from small integers
times 1/8, which every implementation quantises exactly.

## The result: float32 in, float32 out

`bench_pipeline.py`, minimum microseconds over 15 interleaved rounds, run 1;
run 2 (`pipeline_1`, `pipeline_2`). "Y e4m3" is three Y kernels: A quantised,
B quantised and transposed to [N][K], the GEMM over the bytes. "PyTorch" is
`(A / s).to(float8_e4m3fn)`, the same for B with `.t().contiguous().t()`,
and `_scaled_mm`. "Y fused" and "Triton" quantise inside their GEMM.

| N | Y e4m3 | Y fused | PyTorch | Triton | Y e4m3 / PyTorch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 25.8; 25.7 | 26.9; 26.9 | 47.9; 47.6 | 52.9; 41.2 | 0.54; 0.54 |
| 2048 | 171.9; 171.8 | 192.1; 192.4 | 276.4; 281.5 | 201.7; 201.5 | 0.62; 0.61 |
| 4096 | 1088.8; 1085.2 | 1477.4; 1474.6 | 1802.5; 1791.2 | 1586.4; 1578.7 | 0.60; 0.61 |
| 8192 | 7417.6; 7386.6 | 11502.0; 11346.4 | 10501.8; 10443.8 | 15424.5; 15369.7 | 0.71; 0.71 |

Triton's 1024 time differs between the runs: its autotuner chose in each
process (`triton_config` in the `.json`).

The parts, run 1:

| N | Y quantise A | Y quantise and transpose B | Y GEMM | torch quantise A | torch quantise and transpose B | `_scaled_mm` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 4.2 | 5.2 | 16.4 | 7.9 | 23.1 | 16.8 |
| 2048 | 12.8 | 16.3 | 104.5 | 21.7 | 96.7 | 105.2 |
| 4096 | 130.6 | 136.7 | 798.9 | 334.6 | 648.7 | 817.1 |
| 8192 | 534.5 | 549.3 | 6320.3 | 1412.7 | 2755.1 | 6354.9 |

The GEMM over e4m3 bytes takes 0.97-0.99x `_scaled_mm`'s time up to 4096 and
ties it at 8192 (0.995x in run 1, 1.005x in run 2). The rest of the gap is
the quantisation: 1.08 ms for Y at 8192, 4.17 ms for torch.

Y's quantised bytes equal torch's, every byte, at every size
(`Y quant A == torch` in the logs). `pipeline_0` is the same benchmark with
the GEMM as first written: a 128x128x64 tile on 8 warps, two CTAs per SM,
chained accumulation. It measured 25.5 / 170.5 / 1105 / 7433 us.

## The GEMM over e4m3 bytes

`bench_e4m3.py` times PTX variants against `_scaled_mm` on the same bytes;
`Y_FP8_E4M3_TILE` pins a tile. Minimum, as a ratio to `_scaled_mm`'s time in
the same run.

**First, chaining the mmas** (`sweep_1`; PTX in `ptx/sweep_1/`): 128x128x64
on 4x2 warps with 3 stages, two CTAs per SM - cuBLASLt's own tile - took
0.978x at 1024, 0.985x at 2048, 0.998x at 4096 and 1.019x at 8192. Launch
groups of 16 M tiles tied with 8. From 2048 up the other four were 3-13%
slower than it: 4 stages at one CTA per SM, 128x256, 256x128 and a 32-wide
K step. At 1024 the first was 0.6% faster, and 128x256 and 256x128, 32 CTAs
there, were 73% slower.

**What cuBLASLt does that this did not.** Nsight Compute at 8192
(`profile/`, `prof_e4m3.py`), taken while a game also used the GPU: the
instruction counts are the kernels' own, the durations may include the game.
Y's kernel took 6.61 ms and cuBLASLt's 6.48,
both with the tensor pipe 23-24% busy and mostly stalled on it. Y's K loop is
76 SASS instructions - 32 QMMA, 12 LDSM, 4 LDGSTS; cuBLASLt's is 175, of which
64 are FADD (`profile/loopmix_8192.txt`). cuBLASLt adds its mma results into
separate float32 registers.

**Why** (`accum_precision.py`): each GEMM against the float64 product of the
same e4m3 bytes. A product of two e4m3 values is exact in float32, so every
difference is in the summation. Median relative error:

| inputs, K | Y, chained | `_scaled_mm` | Y, added in float32 | the exact sum rounded to float32 |
| :--- | ---: | ---: | ---: | ---: |
| `randn * 8`, 1024 | 3.97e-4 | 1.30e-4 | 5.52e-5 | 0 |
| `randn * 8`, 4096 | 9.73e-4 | 1.37e-4 | 5.60e-5 | 1.45e-8 |
| `rand * 16`, 1024 | 1.36e-2 | 5.71e-4 | 6.47e-5 | 2.03e-8 |
| `rand * 16`, 4096 | 6.62e-2 | 5.72e-4 | 6.50e-5 | 1.99e-8 |

With positive inputs the chained error grows with K: 24x `_scaled_mm`'s at
1024 and 116x at 4096, in proportion to the number of mmas chained (K / 32:
32 and 128). Each step loses about 2^-11 of the sum: the tensor core adds an
mma's products into its accumulator at about 11 bits, truncating.
`emit_fp8_mma_into` starts each mma from zero and adds its four results with
`add.rn.f32`. The error left, 6.5e-5 (about 2^-14) whatever K, is the mma's
own sum of 32 products: the 128 float32 additions at K = 4096 add at most
128 x 2^-24 = 7.6e-6. `_scaled_mm`'s 64 FADD per 32 QMMA are enough to add
every second mma's results, and its error sits between. The fused FP8 GEMM had the same flaw,
6.62e-2, and so does the head-to-head's Triton kernel (`tl.dot` on e4m3):
`fused_precision.py`, `precision_fused_before.log` and `_after.log`.

**Its cost** (`acc_1`, `ptx/accum/`), the 128x128 tile added in float32
against chained: 1.089x / 1.050x / 1.000x / 1.012x `_scaled_mm` against 0.981x
/ 0.986x / 0.997x / 1.018x at 1024 / 2048 / 4096 / 8192. One CTA per SM
without a register cap (214 registers) was 1.09-1.20x.

**Smaller tiles** (`sweep_3` at 1024-4096, `sweep_4` at 4096-8192; PTX in
`ptx/sweep_3/`, `ptx/sweep_4/`), every one adding in float32. 64x64x64 on 2x2
warps with 3 stages, four CTAs per SM, was the fastest at every size: 0.987x
at 1024 and 2048, 0.963x and 0.955x at 4096, 0.969x at 8192, and 0.955x at
8192 launched in groups of 16 M tiles. Against it, 128x64 tied at 1024 and
was 5% and 2% slower at 2048 and 4096; 64x128 on 4 and on 8 warps, 64x64x128
and 128x128x32 were 1-14% slower. It is the default, in groups of 16; the
final runs above use it.

**The fused GEMM** (`bench_fused_ab.py`, `fused_ab_1`; PTX in `ptx/fused/`):
the compiler at `f5ae72b` ("before") against this one ("after"), which adds
in float32. 7.8 / 7.8, 27.5 / 26.9, 192.6 / 192.2, 1509.2 / 1505.3 and
11572 / 11437 us at 512 / 1024 / 2048 / 4096 / 8192: no cost, since the
kernel's time goes to its conversion. Triton took 9.2, 32.8, 200.7, 1596.2
and 15433.

## Quantising

`profile/quantize_8192.txt`, at 8192 x 8192 (`profile/prof_quant.py`). A one
byte store per element is at the memory bound for A (568 us, 95% of DRAM
throughput). Transposed, `Q[c * rows + r] = X[idx] * inv_s` writes one 32-byte
L2 sector per byte: 67.2 million write sectors, 2.81 ms at 93% of L2
throughput. `kernels/quant.ysu`'s transposing kernel has a lane read four rows
of one column and store their four bytes with one `block_ptr2d_store_v4`, a
warp covering 8 columns and 16 rows: 4.2 million write sectors, 557 us at 93%
of DRAM throughput. torch's quantisation took 1.41 ms for A and 2.76 ms for B
(`pipeline_1`).

## The conversions

`sweep_e4m3.py` (`conversions.log`): every float32 bit pattern through a Y
store equals `x.to(torch.float8_e4m3fn)` (NaN for NaN), and torch equals a
round-half-even reference on a sample of every 64th; every byte through a Y
load widens to torch's value; and an F64 beside every e4m3 tie rounds once
(rounding through float32 first gets 189 of those 512 values wrong).

## Files

- `bench_pipeline.py`, `pipeline_*.json`/`.log`: the result table.
- `bench_e4m3.py`, `sweep_1`, `acc_1`, `sweep_3`, `sweep_4`: the tile sweeps.
- `bench_fused_ab.py`, `fused_ab_1`: the fused GEMM before and after.
- `accum_precision.py`, `fused_precision.py`, `precision_*.log`: the errors.
- `sweep_e4m3.py`, `conversions.log`: the element type's conversions.
- `kernels/`: the GEMM sources, the quantising kernels, and the one-byte-store
  ones (`quant_naive.ysu`).
- `ptx/`: every variant timed, by sweep; `ptx/final/` the default tile.
- `profile/`: the 8192 profile summaries and the scripts that made them. The
  reports themselves (12 MB) are not kept.
