# The FP32 GEMM below sm_80 without copy guards in the K loop (2026-11-15)

Below sm_80 the `@tile` FP32 GEMM (`src/ptx_emitter/f32_simt_gemm.rs`) copies
with plain loads, so a guarded copy is a predicated load and a zero fill.
Until `5885bf6` the K loop guarded every copy whenever K was not a multiple
of 16 or M left a ragged last row of tiles, though at most one iteration
loads the partial K tile. Those guards spilled. On a Tesla T4,
1023 x 1025 x 1027 ran at 0.22x cuBLAS
(`docs/benchmarks/f32_gemm_sm75_2026-11-12/`, which put it down to N = 1 mod
4; it was not).

Now, below sm_80, the short K tile comes first. It runs steps 0..K%16-1 of a
whole tile read from k = 0, and the whole tiles after it end at K. A ragged
last row of tiles slides back to end at row M-1 and stores only its own rows.
Every element of C is still one `fma.rn.f32` chain over k = 0..K-1 in order.

## Registers (`ptxas -v`, no GPU needed)

| shape | sm_75 before | sm_75 after | sm_70 after | sm_61 after |
|---|---|---|---|---|
| 1024 x 1024 x 1024 | 128, no spill | the same kernel | | |
| 1024 x 1025 x 1024 (N odd alone) | 125, no spill | the same kernel | | |
| 1024 x 1024 x 1027 (K tail alone) | 255, 424 B spilled | 130, none | 141, none | 141, none |
| 1023 x 1024 x 1024 (M ragged alone) | 242, none | 128, none | 143, none | 139, none |
| 1023 x 1025 x 1027 | 255, 688 B spilled | 128, none | 141, none | 127, none |

The sm_80 and sm_89 kernels, the aligned sm_75 ones and the stream-K ones
are byte for byte what they were (76 sm_80/sm_89 kernels compared). At two or
three K tiles a remainder set still spills 4-52 bytes (`docs/known_issues.md`).

## Bits

- `python/tests/test_f32_gemm_below_sm80.py`: the sm_75 kernels run in Y's
  CPU PTX interpreter (which refuses any access outside A, B or C) against an
  in-order fma32 oracle, and their K loops hold no guarded load.
- Before the commit, the same interpreter check over 25 shapes (1023 x 1025 x
  1027 included, old and new builds, two grids each) was bit-exact. Five
  planted errors in the change were each caught.
- On the RTX 4070 Ti SUPER, whose driver compiles sm_75 PTX for it: tests/
  f32_simt_gemm.rs (`built_for_sm_75_each_element_is_the_same_fma_chain`, 33
  shapes) passes, and 8 edge shapes on two grids each are bit-exact against
  an `fmaf` kernel under compute-sanitizer memcheck (0 errors) and racecheck
  (0 hazards).

## Speed: the sm_75 build on an RTX 4070 Ti SUPER

`f32_vs_cublas.py` with `Y_BENCH_TARGET=sm_75`, `Y_LIB_PATH` set to each
`liby.so` (the wheel before `5885bf6` and the wheel at it). Driver 615.78.08,
torch 2.13.0+cu130, cuBLAS TF32 off, the card otherwise idle. 11 interleaved
rounds against cuBLAS, the minimum of each side. Two rounds of (before,
after), the order flipped in the second. Raw output in `rtx4070tis/`. This
is the sm_75 code on an sm_89 card, not a T4.

| shape | before (round 1, 2) | after (round 1, 2) |
|---|---|---|
| 512^3 | 1.044x, 1.044x | 1.059x, 1.025x |
| 1024^3 | 1.205x, 1.194x | 1.170x, 1.182x |
| 2048^3 | 0.879x, 0.934x | 0.882x, 0.863x |
| 4096^3 | 0.837x, 0.797x | 0.848x, 0.850x |
| 1024 x 4096 x 4096 | 0.825x, 0.843x | 0.863x, 0.852x |
| 4096 x 4096 x 1024 | 0.866x, 0.867x | 0.848x, 0.822x |
| 1024 x 1025 x 1024 | 0.855x, 0.782x | 0.816x, 0.848x |
| **1023 x 1025 x 1027** | 0.293x, 0.278x | 0.946x, 0.840x |
| **1024 x 1024 x 1027** | 0.345x, 0.347x | 1.198x, 1.193x |
| **1023 x 1024 x 1024** | 0.726x, 0.701x | 1.166x, 1.174x |
| **2047 x 2049 x 2051** | 0.249x, 0.247x | 1.007x, 1.049x |
| **4096 x 4096 x 4100** | 0.211x, 0.213x | 0.891x, 0.870x |
| **1025^3** | 0.253x, 0.252x | 0.916x, 0.877x |
| **400 x 2049 x 1025** | 0.313x, 0.311x | 1.209x, 1.178x |
| **100 x 1024 x 1027** | 0.171x, 0.168x | 0.539x, 0.518x |

The first seven are the same kernel before and after; they moved by
0.94-1.02x, which is the noise here. Every row was bit-exact against the
`fmaf` chain in every run.

## Then: the next tile's loads first, their stores after the FMAs

A plain load's shared store waits for the load. With the copies spread over
the k steps, ptxas (after if-converting the copy branches) left 30-80
instructions between each `LDG` and its `STS`, where a global load takes
hundreds of cycles. Now, below sm_80, every load of the next tile issues at
the top of the iteration (with the pointer advance) and the 12 stores come
after the FMAs. In the sm_75 SASS for 1024^3 the loop starts with 12 `LDG`
and ends with `STS x12`. This costs 18-40 registers (1024^3: 128 -> 146; no
spill on sm_75, sm_70 or sm_61). The large tile keeps the old order: on
sm_70 it is at the register cap without the prefetch, and 2048 x 4097 x 40
spilled 176 bytes with it. The 4-52-byte spills left at two or three K
tiles are gone.

The same procedure as above, the libraries at `5885bf6` (fix) and with the
prefetch, raw output in `rtx4070tis_prefetch/`:

| shape | fix (round 1, 2) | prefetch (round 1, 2) |
|---|---|---|
| 512^3 | 1.036x, 1.068x | 1.089x, 1.082x |
| 1024^3 | 1.222x, 1.183x | 1.236x, 1.219x |
| 2048^3 | 0.891x, 0.904x | 0.895x, 0.947x |
| 4096^3 | 0.844x, 0.846x | 0.838x, 0.865x |
| 1024 x 4096 x 4096 | 0.854x, 0.873x | 0.872x, 0.896x |
| 4096 x 4096 x 1024 | 0.849x, 0.825x | 0.835x, 0.902x |
| **128 x 4096 x 4096** | 0.633x, 0.612x | 1.048x, 0.982x |
| **4096 x 128 x 4096** | 0.588x, 0.573x | 0.875x, 0.925x |
| **64 x 8192 x 2048** | 0.634x, 0.633x | 1.036x, 1.028x |
| 1023 x 1025 x 1027 | 0.943x, 0.959x | 0.959x, 0.882x |
| 1024 x 1024 x 1027 | 1.172x, 1.168x | 1.235x, 1.161x |
| 2047 x 2049 x 2051 | 1.079x, 1.066x | 1.021x, 1.054x |
| 4096 x 4096 x 4100 | 0.898x, 0.908x | 0.971x, 0.956x |
| 400 x 2049 x 1025 | 1.193x, 1.215x | 1.310x, 1.323x |
| 100 x 1024 x 1027 | 0.536x, 0.536x | 0.559x, 0.558x |

It helps most where few tiles share the card, so few warps share an SM to
hide the latency (the three in bold). A T4 runs the mid tile at one CTA (4
warps) per SM at every shape, so that is the case there throughout; it is
not measured there yet. Every row bit-exact in every run; memcheck and
racecheck clean on 10 shapes (sm_75 build, this card).

`100 x 1024 x 1027` (32 tiles for 66 SMs) stays at 0.54-0.56x: it is short
of tiles, not latency.

## Speed on a Tesla T4

Kaggle job `y-f32-t4` version 3 (`t4_v3/`): Tesla T4, driver 580.178.04,
torch 2.13.0+cu126. sm_75 PTX built here with a 40-SM profile: `before` (the
compiler before `5885bf6`), `head` (`834f1b4`: no guards, loads first) and
`stages2` (an experiment: `head` with 2 stages instead of 3, so two mid-tile
CTAs fit an SM). Each was loaded with cupy, every element checked against an
`fmaf` chain on the card (all 39 bit-exact), and timed against cuBLAS (TF32
off), 11 rounds in shuffled order, the minimum per side.

| shape | cuBLAS TF | before | head |
|---|---|---|---|
| 1023 x 1025 x 1027 | 3.72 | 0.94 TF, 0.25x | 3.21 TF, 0.86x |
| 1024 x 1024 x 1027 | 4.76 | 1.06 TF, 0.22x | 3.99 TF, 0.84x |
| 2047 x 2049 x 2051 | 4.83 | 0.98 TF, 0.20x | 3.39 TF, 0.70x |
| 4096 x 4096 x 4100 | 3.46 | 1.18 TF, 0.34x | 3.63 TF, 1.05x |

That is 3.4-3.8 times the speed at the shapes the guards hit, far outside
the noise. The rest of that run is not: the T4 is a 70 W card that clocks
down under load, and the minimum per side took whichever sample caught it at
boost. cuBLAS read 3.5-6.0 TF across the shapes, and one `stages2` sample
7.25 TF (89% of the card's peak). So it does not settle `head` against
`before` at aligned shapes, or `stages2` against `head`; the raw figures are
in `t4_v3/`.

Version 4 (`t4_v4/`) timed `head` and `stages2` again: 3 sweeps of 15 rounds
in shuffled order, each sample about 0.1 s of launches, the median of the 45
per-round ratios (cuBLAS time / Y time) with their quartiles, and the SM
clock and power logged (the card ran at 1365-1545 MHz, 36-66 W). All 26
bit-exact. Medians run lower than version 3's minima for both sides
(cuBLAS 2.4-3.8 TF against 3.5-6.0).

| shape | cuBLAS TF | head | stages2 | stages2 tile |
|---|---|---|---|---|
| 64 x 8192 x 2048 | 2.64 | 1.009x [0.952-1.065] | 1.018x [0.959-1.066] | mid |
| 128 x 4096 x 4096 | 2.97 | 0.966x [0.939-1.045] | 1.041x [0.957-1.088] | mid |
| 512^3 | 3.82 | 0.680x [0.666-0.694] | 0.687x [0.675-0.696] | small |
| 1023 x 1025 x 1027 | 2.54 | 0.913x [0.883-0.942] | 1.036x [0.946-1.069] | mid |
| 1024^3 | 2.95 | 0.986x [0.939-1.048] | 1.018x [0.954-1.069] | mid |
| 1024 x 1024 x 1027 | 2.88 | 1.028x [0.954-1.052] | 1.002x [0.961-1.074] | mid |
| 1024 x 4096 x 4096 | 3.39 | 1.030x [0.942-1.086] | 1.016x [0.991-1.050] | large |
| 2047 x 2049 x 2051 | 2.86 | 0.964x [0.897-1.005] | 0.923x [0.886-0.986] | large |
| 2048^3 | 3.26 | 1.060x [0.959-1.091] | 1.016x [0.980-1.076] | large |
| 4096 x 128 x 4096 | 2.38 | 1.023x [1.000-1.083] | 1.085x [0.996-1.094] | mid |
| 4096 x 4096 x 1024 | 3.10 | 1.109x [0.990-1.119] | 1.232x [1.136-1.296] | large |
| 4096^3 | 3.33 | 1.031x [0.945-1.053] | 1.188x [1.125-1.226] | large |
| 4096 x 4096 x 4100 | 3.13 | 1.076x [0.976-1.091] | 1.234x [1.134-1.276] | large |

`head` (one 3-stage mid CTA an SM) is at 0.91-1.11x cuBLAS on every shape
but 512^3 (the small tile, 0.68x). `stages2` set every tile to 2 stages,
which did two things: the mid tile fit two CTAs an SM, and the large tile
(128 x 256, 74 KB at 3 stages) shrank to 49.7 KB and fit a T4 at all, and
the tile rule then chose it at seven shapes. So the column mixes them:

- the mid tile at 2 stages: 1023 x 1025 x 1027 0.91x -> 1.04x (quartiles
  apart), 4096 x 128 x 4096 1.02x -> 1.09x, 128 x 4096 x 4096 0.97x ->
  1.04x, 1024 x 1024 x 1027 1.03x -> 1.00x, the others within each other's
  quartiles. That is now the default on Turing, where three stages leave
  one CTA an SM and two leave two (`f32_gemm_turing_stages`);
- the large tile at 2 stages: 1.19-1.23x at the shapes with about 13 waves
  of its tiles (4096^3, 4096 x 4096 x 1024, 4096 x 4096 x 4100), no better
  or worse at about 3 waves (2048^3 1.06x -> 1.02x, 2047 x 2049 x 2051
  0.96x -> 0.92x, where it spilled 76 bytes at 255 registers). Not taken:
  where it pays needs measuring, and it has no prefetch (`prefetch`
  excludes it for Volta's register cap).
