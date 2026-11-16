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
  0.96x -> 0.92x, where it spilled 76 bytes at 255 registers).

Version 5 (`t4_v5/`, the same method) swept the large tile's waves (its
tiles over 40 SMs) against the mid tile, both at 2 stages, with and without
the prefetch on the large tile (224 registers with it, 221 without). All 24
bit-exact.

| shape | waves | mid | large | large + prefetch |
|---|---|---|---|---|
| 2048^3 | 3.2 | 1.025x [0.972-1.076] | 0.975x [0.952-1.022] | 1.014x [0.964-1.044] |
| 2560^3 | 5.0 | 1.061x [0.970-1.084] | 1.204x [1.130-1.256] | 1.144x [1.100-1.202] |
| 4096 x 2048 x 2048 | 6.4 | 0.965x [0.910-1.006] | 1.043x [0.998-1.102] | 1.016x [0.989-1.049] |
| 3072^3 | 7.2 | 1.048x [1.008-1.083] | 1.094x [1.052-1.165] | 1.070x [1.031-1.136] |
| 4096 x 4096 x 1024 | 12.8 | 1.058x [0.980-1.088] | 1.200x [1.154-1.272] | 1.193x [1.167-1.248] |
| 4096^3 | 12.8 | 1.021x [1.006-1.064] | 1.197x [1.166-1.232] | 1.168x [1.128-1.226] |
| 6144 x 4096 x 1024 | 19.2 | 1.115x [1.072-1.143] | 1.163x [1.113-1.250] | 1.167x [1.085-1.248] |
| 8192 x 8192 x 512 | 51.2 | 1.000x [0.954-1.028] | 1.059x [1.016-1.087] | 1.026x [0.996-1.057] |

The large tile wins wherever its waves are more than 80% full and loses at
2048^3, exactly 80% (3.2 waves: the last a fifth full). The prefetch does
not help it. So on Turing the 2-stage large tile is now taken where the
large-tile rule holds with more than 80% (not at least 80%), without the
prefetch, and only where measured: N a multiple of 4 and M of 128. With a
ragged M and an odd N it spilled 76 bytes at 255 registers. Its bits:
the interpreter check on 4 shapes (one with a column remainder set at 2
stages), and memcheck and racecheck on 3 (sm_75 build, RTX 4070 Ti SUPER).

## 512^3 and the other shapes on the small tile

512^3 is 64 small tiles (64 x 64) for 40 SMs: 24 SMs run two, 16 one, so
at best 80% of the card's CTA slots work. Version 6 (`t4_small/`) tried the
small tile at 2 stages and the 32 x 64 and 64 x 32 tiles (128 tiles, 4 x 4
outputs a thread) as the main tile, at 3 and 4 stages: the smaller tiles
lost on every shape (512^3: 0.68x -> 0.54-0.57x), 2 stages changed nothing.
A thread computing 4 x 4 instead of 4 x 8 outputs loads more from shared
memory for each FMA, and that cost more than the balance gained.

What balances it without losing the in-order chain is splitting K in order:
the workspace form (`W: GlobalMemory<U64>`), whose stream-K schedule runs
40 CTAs and lets a CTA continue a tile from the exact accumulators another
left in C. Its segments now load the next K tile first and store it after
the FMAs too (`prefetch`; before, they spread the copies). Job
`y-f32-t4-w` version 2 (`t4_w/`), medians of 45 paired rounds against
cuBLAS, all bit-exact:

| shape | three-parameter | `W`, before | `W`, with the prefetch |
|---|---|---|---|
| 512^3 | 0.689x [0.68-0.70] | 0.753x [0.75-0.76] | 0.779x [0.76-0.78] |
| 384 x 512 x 1024 | 0.591x [0.56-0.64] | 0.662x [0.64-0.70] | 0.823x [0.77-0.85] |
| 256 x 1024 x 1024 | 0.661x [0.65-0.69] | 0.644x [0.61-0.68] | 0.757x [0.70-0.77] |
| 512 x 512 x 2048 | 0.684x [0.66-0.70] | 0.673x [0.65-0.68] | 0.769x [0.72-0.78] |
| 512 x 256 x 512 | 0.839x | 0.839x (`W` unused) | 0.839x |
| 256 x 512 x 4096 | 0.674x | 0.675x (`W` unused) | 0.673x |

The three-parameter form cannot do this (it has nowhere to say how far a
tile got), so at 512^3 it stays at 0.69x on a T4.

## The mid tile where the small one doubles up on an SM

That was the wrong comparison. Version 7 (`t4_mid/`) set the mid tile (64 x
128, two stages on Turing) against the small one at the same shapes, and
two 2-warp 8 x 8 small tiles (64 x 64, 2 and 3 stages). All 24 bit-exact.

| shape | mid tiles / 40 SMs | small | mid | small, 2 warps (2 / 3 stages) |
|---|---|---|---|---|
| 512^3 | 32 | 0.678x [0.61-0.75] | 0.930x [0.85-1.01] | 0.484x / 0.479x |
| 512 x 512 x 2048 | 32 | 0.728x [0.70-0.75] | 0.955x [0.92-0.98] | 0.572x / 0.575x |
| 256 x 1024 x 1024 | 32 | 0.645x [0.60-0.69] | 0.871x [0.82-0.96] | 0.489x / 0.505x |
| 384 x 512 x 1024 | 24 | 0.609x [0.53-0.68] | 0.706x [0.62-0.74] | 0.391x / 0.392x |
| 512 x 256 x 512 | 16 | 0.827x [0.80-0.87] | 0.609x [0.60-0.63] | 0.518x / 0.520x |
| 256 x 512 x 4096 | 16 | 0.706x [0.69-0.72] | 0.600x [0.57-0.62] | 0.487x / 0.488x |

So 32 mid tiles on 40 SMs beat 64 small ones, which put two CTAs on 24
SMs. If a second small CTA adds on a T4 what it adds on the RTX 4070 Ti
SUPER (8%; not measured here), 512^3 says a mid tile takes about 1.35
times a small tile's time there, for twice the outputs; on the 4070 Ti it
takes 1.65 times, and the tile rule, read off that card, took the mid tile
from 90% of a wave of it (36 tiles on 40 SMs). The 2-warp tiles leave
half of an SM's schedulers idle.

Version 8 (`t4_rule/`) placed the bound: the small tile against the mid,
built with a scratch switch that forced the mid tile (the committed build
of each shape is byte for byte one of the two). All 26 bit-exact.

| shape | mid tiles | small tiles | small | mid |
|---|---|---|---|---|
| 512^3 | 32 | 64 | 0.683x [0.68-0.70] | 0.924x [0.89-0.94] |
| 500^3 | 32 | 64 | 0.666x [0.66-0.68] | 0.911x [0.88-0.92] |
| 1024 x 256 x 1024 | 32 | 64 | 0.681x [0.65-0.73] | 0.943x [0.82-0.97] |
| 1024 x 256 x 256 | 32 | 64 | 0.721x [0.72-0.73] | 0.952x [0.94-0.96] |
| 2048 x 128 x 512 | 32 | 64 | 0.735x [0.70-0.75] | 0.968x [0.91-1.04] |
| 512 x 480 x 1000 | 32 | 64 | 0.659x [0.63-0.71] | 0.842x [0.75-0.89] |
| 256 x 1024 x 1024 | 32 | 64 | 0.659x [0.63-0.68] | 0.790x [0.76-0.91] |
| 448 x 576 x 512 | 35 | 63 | 0.756x [0.74-0.79] | 0.794x [0.77-0.82] |
| 384 x 640 x 512 | 30 | 60 | 0.781x [0.77-0.78] | 0.918x [0.91-0.92] |
| 448 x 512 x 512 | 28 | 56 | 0.858x [0.85-0.86] | 0.999x [0.99-1.01] |
| 384 x 512 x 1024 | 24 | 48 | 0.609x [0.55-0.63] | 0.697x [0.62-0.74] |
| 512 x 256 x 512 | 16 | 32 | 0.834x [0.82-0.85] | 0.605x [0.60-0.61] |
| 2048 x 64 x 512 | 32 | 32 | 1.108x [1.08-1.14] | 0.897x [0.89-0.91] |

The mid tile wins wherever the small tile would put two CTAs on some SM,
down to 24 mid tiles (60% of a wave; at 384 x 512 x 1024 the quartiles
overlap, in both versions, with the medians 14-16% apart). Where 40 SMs
take every small tile at once it loses (512 x 256 x 512, and 2048 x 64 x
512, where its 128 columns are half empty). So on Turing the mid tile is
now taken from 60% of a wave where the small tile would exceed one; 24 to
35 mid tiles was the small tile before. Between 50% and 60% (21-23 mid
tiles, 41-47 small) it was not measured and stays the small tile. The
rule is in `f32_gemm_tile`; the tile choice changes no bit of C.

With `W` the comparison is a third way: the small tile split in order over
the 40 SMs, which the workspace form did before (`t4_w/`: 384 x 512 x 1024
at 0.82x), against the mid tile in one wave, `W` unused. Version 3 of
`y-f32-t4-w` (`t4_wband/`) timed the two at the shapes the rule above gives
the mid tile, from 48 small tiles to 64. All 18 bit-exact.

| shape | small tiles | mid tiles (CTAs) | small, split | mid |
|---|---|---|---|---|
| 384 x 512 x 1024 | 48 | 24 | 0.738x [0.68-0.79] | 0.673x [0.66-0.69] |
| 448 x 448 x 768 | 49 | 28 (35) | 0.938x [0.81-1.01] | 0.856x [0.77-0.90] |
| 320 x 640 x 1024 | 50 | 25 | 0.796x [0.74-0.86] | 0.727x [0.71-0.75] |
| 832 x 256 x 1024 | 52 | 26 | 0.769x [0.71-0.85] | 0.769x [0.76-0.80] |
| 576 x 384 x 1024 | 54 | 27 | 0.789x [0.74-0.82] | 0.780x [0.76-0.81] |
| 448 x 512 x 512 | 56 | 28 | 0.943x [0.93-0.95] | 1.000x [0.99-1.01] |
| 384 x 640 x 512 | 60 | 30 | 0.841x [0.84-0.85] | 0.917x [0.91-0.92] |
| 448 x 576 x 512 | 63 | 35 (42) | 0.879x [0.84-0.91] | 0.800x [0.79-0.83] |
| 512^3 | 64 | 32 | 0.790x [0.79-0.79] | 0.918x [0.90-0.93] |

The split wins to about 1.3 small tiles an SM (52 on 40), the two tie at
52-54, and the mid tile wins above. 448 x 576 x 512 is the exception: its
last 64 columns go to 14 remainder tiles beside 28 mid ones, 42 CTAs, so
the mid launch is no longer one wave. (Its whole-tile count, 35, is what
the rule above reads; at 448 x 448 x 768 the remainder makes 35 CTAs, still
one wave.) So with `W`, where the rule above takes the mid tile below 90%
of a wave, the small tile is split instead up to 1.3 small tiles an SM, or
where the mid launch with its remainder tiles exceeds one CTA an SM
(`f32_gemm_turing_split_small`). That picks the faster of the two at all
nine shapes but the ties.

Where that leaves the two forms on a T4 (the medians above):

| shape | before (`600c1a5`), three parameters / `W` | now, three parameters / `W` |
|---|---|---|
| 512^3 | 0.68x / 0.78x | 0.92x / 0.92x |
| 448 x 512 x 512 | 0.86x / 0.94x | 1.00x / 1.00x |
| 384 x 640 x 512 | 0.78x / 0.84x | 0.92x / 0.92x |
| 384 x 512 x 1024 | 0.61x / 0.74-0.82x | 0.70x / 0.74-0.82x |
| 448 x 576 x 512 | 0.76x / 0.88x | 0.79x / 0.88x |

## Not taken: K tiles of 32 on the small tile

Profiled on the RTX 4070 Ti SUPER (`ncu`; the T4 on Kaggle refuses access
to its performance counters), the sm_75 small tile at 512 x 256 x 512 spent
a third of its warp samples with its shared stores waiting on the global
loads issued at the top of the same K tile (long scoreboard, 34%). A K tile
of 32 instead of 16 doubles the FMAs between the two; there it ran 6-9%
faster (one `ncu` duration each, 4 shapes), bit-exact in the CPU
interpreter. On the T4 (job `y-f32-t4-b`, `t4_bk32/`, all 30 bit-exact) it
was slower at every shape, at two stages and at three:

| shape | small tiles | BK 16 | BK 32, 2 stages | BK 32, 3 stages |
|---|---|---|---|---|
| 512 x 256 x 512 | 32 | 0.835x [0.83-0.84] | 0.750x [0.74-0.76] | 0.753x |
| 448 x 512 x 512 | 56 | 0.866x [0.83-0.90] | 0.679x [0.66-0.71] | 0.694x |
| 384 x 640 x 512 | 60 | 0.778x [0.75-0.80] | 0.613x [0.60-0.63] | 0.612x |
| 256 x 512 x 4096 | 32 | 0.678x [0.63-0.68] | 0.643x [0.61-0.66] | 0.649x |
| 2048 x 64 x 512 | 32 | 1.116x [1.07-1.19] | 0.974x [0.96-1.02] | 0.962x |
| 1024 x 128 x 2048 | 32 | 0.967x [0.95-0.99] | 0.875x [0.84-0.95] | 0.903x |
| 128 x 512 x 2048 | 16 | 0.450x [0.37-0.46] | 0.361x [0.33-0.38] | 0.362x |

The 4070 Ti issues an FMA a clock per scheduler, the T4 one every two, so
the same wait is twice as large a share of the time there; and BK 32 puts
each warp's transposed stores of A four to a bank instead of two. Which of
these decides it on the T4 is not measured. The 16-tile shapes (128 x 512 x
2048 at 0.45x, 256 x 256 x 8192 at 0.54x) leave 24 SMs idle, and an
in-order chain cannot split K to fill them without `W`, whose plan needs a
tile an SM.

## Few tiles: the 32-row tiles

With 16 small tiles or fewer and a long K (128 x 512 x 2048 at 0.44x), 24
of the T4's 40 SMs idle, and an in-order chain cannot split K to fill them.
More, smaller tiles can: 32 x 64 (4 warps, 4 x 4 outputs a thread, the
small tile's row remainder) and 32 x 32 (2 warps). A thread then loads
more from shared memory per FMA, which is why they lost where the small
tile already filled the SMs (version 6), but where most of a wave is
empty the CTAs win. Version 9 (`t4_thin/`, all 33 bit-exact) forced each
as the whole tile:

| shape | small tiles | small | 32 x 64 | 32 x 32 |
|---|---|---|---|---|
| 128 x 256 x 4096 | 8 | 0.326x [0.30-0.35] | 0.435x [0.40-0.49] | 0.503x [0.45-0.56] |
| 64 x 512 x 8192 | 8 | 0.286x [0.26-0.30] | 0.367x [0.34-0.39] | 0.405x [0.37-0.43] |
| 128 x 512 x 512 | 16 | 0.616x [0.61-0.62] | 0.749x [0.74-0.75] | 0.602x [0.60-0.61] |
| 128 x 512 x 2048 | 16 | 0.437x [0.40-0.48] | 0.539x [0.46-0.57] | 0.436x [0.39-0.48] |
| 64 x 1024 x 4096 | 16 | 0.522x [0.47-0.55] | 0.570x [0.52-0.58] | 0.472x [0.43-0.49] |
| 256 x 256 x 8192 | 16 | 0.540x [0.53-0.56] | 0.585x [0.58-0.59] | 0.466x [0.45-0.47] |
| 192 x 384 x 4096 | 18 | 0.640x [0.63-0.66] | 0.698x [0.68-0.73] | 0.572x [0.56-0.59] |
| 128 x 640 x 2048 | 20 | 0.525x [0.51-0.59] | 0.637x [0.58-0.66] | 0.510x [0.47-0.56] |
| 320 x 256 x 2048 | 20 | 0.534x [0.52-0.59] | 0.592x [0.56-0.65] | 0.495x [0.46-0.55] |
| 384 x 256 x 2048 | 24 | 0.568x [0.55-0.59] | 0.534x [0.51-0.56] | 0.518x [0.47-0.54] |
| 128 x 1024 x 1024 | 32 | 0.707x [0.62-0.74] | 0.586x [0.56-0.61] | 0.557x [0.50-0.58] |

At every shape the fastest is the 32-row tile with the most tiles that
still fit one wave, if that is more than the tile it replaces; past a
wave the small tile is back ahead. That is the rule now on Turing
(`f32_gemm_turing_thin`): 32 x 32 where its tiles fit a wave and outnumber
32 x 64's, else 32 x 64 where its tiles fit and outnumber the small
tile's. The 32 x 32 tile's 64 threads cannot share a launch with the
128-thread remainder tiles, so it needs M and N multiples of 32; the thin
tiles take no remainder sets, their ragged edges are guarded inside the
tile (M at least 33, where the last row slides back, so the K loop stays
unguarded: no spill at 10 ragged shapes, bit-exact in the CPU interpreter
at 9). Version 10 (`t4_thin_rule/`, all 24 bit-exact) checked the rule's
choice at eight shapes it had not seen; it picked the fastest at all eight:

| shape | the rule's tile | small | 32 x 64 | 32 x 32 |
|---|---|---|---|---|
| 64 x 512 x 1024 | 32 x 32 (32 tiles) | 0.325x [0.32-0.33] | 0.537x [0.52-0.54] | 0.560x [0.55-0.57] |
| 160 x 256 x 4096 | 32 x 32 (40) | 0.347x [0.30-0.41] | 0.463x [0.39-0.59] | 0.534x [0.45-0.61] |
| 64 x 640 x 4096 | 32 x 32 (40) | 0.315x [0.28-0.37] | 0.409x [0.37-0.44] | 0.446x [0.38-0.52] |
| 96 x 512 x 2048 | 32 x 64 (24) | 0.445x [0.38-0.47] | 0.550x [0.49-0.59] | 0.441x [0.37-0.49] |
| 192 x 256 x 2048 | 32 x 64 (24) | 0.365x [0.34-0.39] | 0.451x [0.40-0.49] | 0.375x [0.34-0.41] |
| 128 x 384 x 2048 | 32 x 64 (24) | 0.406x [0.39-0.43] | 0.477x [0.42-0.50] | 0.413x [0.39-0.44] |
| 32 x 1024 x 2048 | 32 x 32 (32) | 0.626x [0.60-0.66] | 0.537x [0.52-0.56] | 0.639x [0.59-0.69] |
| 32 x 512 x 4096 | 32 x 32 (16) | 0.340x [0.29-0.36] | 0.294x [0.28-0.30] | 0.350x [0.31-0.36] |

(At 32 rows 32 x 64 has no more tiles than the small tile and is not
taken; 32 x 32 is level with the small tile there.) These shapes are still
behind cuBLAS, which splits K; the gain is 17-72% where M is above 32.

## Where the T4 stands: a sweep, and `W` at a K tail

Job `y-f32-t4-b` version 2 (`t4_sweep/`) timed the pushed build (`1d148f8`)
at 16 common shapes, three-parameter form and `W`, all 32 bit-exact:

| shape | three parameters | `W` |
|---|---|---|
| 4096^3 | 1.202x [1.10-1.22] | 1.191x |
| 1024 x 30522 x 768 (BERT's vocabulary) | 1.128x [1.03-1.16] | 1.132x |
| 1024 x 50257 x 768 (GPT-2's head) | 1.058x [0.98-1.08] | 1.062x |
| 3000^3 | 1.033x [0.99-1.06] | 1.044x |
| 512 x 4864 x 896 (Qwen2.5-0.5B's MLP up) | 1.069x [1.05-1.11] | 1.016x |
| 2048^3 | 0.996x [0.98-1.03] | 1.030x (`W` unused) |
| 2047 x 2049 x 2051 | 0.993x [0.95-1.01] | 0.993x (`W` unused) |
| 1536^3 | 0.978x [0.92-0.99] | 1.000x [0.95-1.01] |
| 1000^3 | 1.019x [0.93-1.05] | 0.839x [0.81-0.89] |
| 768^3 | 0.911x [0.91-0.93] | 0.907x [0.90-0.91] |
| 512 x 896 x 896 (Qwen2.5-0.5B's attention projections) | 0.886x [0.88-0.91] | 1.041x [1.01-1.08] |
| 512 x 896 x 4864 (its MLP down) | 0.829x [0.81-0.88] | 0.924x [0.83-1.00] |
| 32 x 4096 x 4096 | 0.884x [0.85-0.90] | 0.860x |
| 128 x 896 x 896 | 0.673x [0.64-0.73] | 0.682x |
| 8 x 4096 x 4096 | 0.773x [0.74-0.82] | 0.765x |
| 1 x 50257 x 768 | 0.552x [0.52-0.63] | 0.549x |

With `W`, 1000^3 ran 18% slower than without. Below sm_80 the short first
K tile and the sliding last tile row (`short_first`, `slide_m`) do not
apply to a stream-K segment, which can start anywhere in a tile's K, so a
segment's K loop guards its copies at a K tail or a ragged M: all 12 loads
of 1000^3's loop are predicated, against none at 768^3. So below sm_80 `W`
now splits only where M and K are whole tiles, and elsewhere is the
three-parameter kernel (1000^3 1.02x); its header says why. The sm_80 and
sm_89 kernels, where a guarded copy is a `cp.async` of size 0, are
unchanged.

The one-row and few-row shapes read all of B for one to 32 rows of A, so
they are bound by memory, not by the FMAs. Y streams B at 137-142 GB/s
there (the T4's DRAM is rated 320); cuBLAS at 253 GB/s for one row, 171
for 8 and 159 for 32. Y runs them on 32-row tiles of the mid tile's width,
so a one-row GEMM computes 31 rows for nothing and its loads are those of
a 32-row tile; the one row is the next gap to close.

## One to 16 rows: tiles no taller than M

Those rows went to the 32-row strips of the mid tile's width, so a one-row
GEMM computed 31 rows for nothing; and they stream B, at 137-142 GB/s
against cuBLAS's 159-253. Two shorter tiles, 8 x 256 (4 warps of 8 x 64)
and 16 x 128 (4 warps of 16 x 32), both 4 x 4 outputs a thread, built with
a scratch switch (version 11, `t4_skinny/`, all 33 bit-exact):

| shape | 32-row tile | 16 x 128 | 8 x 256 |
|---|---|---|---|
| 1 x 4096 x 4096 | 0.634x [0.63-0.64] | 0.782x [0.77-0.79] | 0.802x [0.80-0.82] |
| 1 x 30520 x 768 | 0.654x [0.65-0.67] | 0.837x [0.83-0.86] | 1.066x [1.05-1.07] |
| 1 x 50256 x 768 | 0.584x [0.58-0.59] | 0.735x [0.73-0.75] | 0.948x [0.94-0.96] |
| 4 x 4096 x 4096 | 0.540x [0.51-0.55] | 0.666x [0.62-0.67] | 0.694x [0.69-0.70] |
| 8 x 4096 x 4096 | 0.929x [0.85-0.95] | 1.155x [1.07-1.17] | 1.482x [1.33-1.51] |
| 8 x 4864 x 896 | 0.640x [0.55-0.67] | 0.823x [0.79-0.84] | 1.018x [0.87-1.04] |
| 8 x 11008 x 4096 | 0.864x [0.85-0.88] | 1.100x [1.08-1.12] | 1.304x [1.29-1.32] |
| 16 x 896 x 4864 | 0.303x [0.26-0.31] | 0.346x [0.29-0.35] | 0.308x [0.27-0.31] |
| 16 x 4096 x 4096 | 0.760x [0.75-0.78] | 1.123x [1.07-1.15] | 1.028x [1.01-1.06] |
| 24 x 4096 x 4096 | 0.837x [0.83-0.85] | 0.730x [0.71-0.75] | 0.684x [0.67-0.69] |
| 32 x 4096 x 4096 | 0.923x [0.90-0.93] | 0.734x [0.71-0.75] | 0.634x [0.62-0.64] |

So on Turing up to 8 rows take 8 x 256 and up to 16 rows 16 x 128. The 8 x
256 tile's 4-byte B copies (N odd) do not divide over its 128 threads, so
at N odd it is 16 x 128 too. Neither takes remainder sets, and `W` does
not split them (not measured). Version 12 (`t4_skinny_rule/`, all 24
bit-exact) set that rule against the commit before it at shapes it was not
read off:

| shape | short-tile CTAs | before | short tile |
|---|---|---|---|
| 1 x 30522 x 768 (N = 2 mod 4: 8-byte copies) | 120 | 0.597x [0.58-0.65] | 1.007x [0.94-1.08] |
| 1 x 50257 x 768 (N odd: 16 x 128) | 393 | 0.462x [0.46-0.48] | 0.590x [0.58-0.63] |
| 16 x 50257 x 768 | 393 | 0.698x [0.69-0.71] | 0.933x [0.93-0.94] |
| 12 x 3072 x 768 | 24 | 0.559x [0.52-0.58] | 0.703x [0.65-0.73] |
| 2 x 4096 x 11008 | 16 | 0.542x [0.52-0.55] | 0.704x [0.67-0.71] |
| 1 x 4864 x 896 | 19 | 0.620x [0.59-0.65] | 0.726x [0.71-0.76] |
| 7 x 4098 x 1027 | 17 | 0.623x [0.53-0.65] | 0.707x [0.59-0.76] |
| 11 x 777 x 4097 | 7 | 0.266x [0.22-0.27] | 0.242x [0.22-0.25] |
| 8 x 896 x 896 | 4 | 0.368x [0.37-0.37] | 0.348x [0.34-0.35] |
| 4 x 896 x 4864 | 4 | 0.291x [0.25-0.30] | 0.216x [0.20-0.22] |
| 1 x 1024 x 1024 | 4 | 0.224x [0.22-0.24] | 0.163x [0.15-0.17] |
| 1 x 896 x 4864 | 4 | 0.274x [0.26-0.28] | 0.204x [0.20-0.21] |

With 4-7 CTAs the short tiles lose to the 14-16 small tiles they replace;
with 16 or more they won everywhere. So they are taken where they make at
least 40% of the SMs' count in CTAs (16 on a T4), and the rest keep the
tile they had. Those few-column GEMMs (896 to 1024 outputs a row, K up to
4864) stay far behind: 896 in-order chains are all the parallelism there
is, and cuBLAS splits K.

The four tiles added on 2026-11-16 (32 x 32, 32 x 64, 16 x 128, 8 x 256),
built for sm_75 and run on the RTX 4070 Ti SUPER: compute-sanitizer
memcheck 0 errors and racecheck 0 hazards on 7 shapes (64 x 64 x 16, 33 x
65 x 17, 100 x 500 x 3000, 8 x 8192 x 40, 5 x 8194 x 35, 1 x 8193 x 40, 13
x 4097 x 33; ragged M and N, odd N, K tails), each bit-exact on two grids.
`built_for_sm_75_each_element_is_the_same_fma_chain` now asserts that its
35 shapes run every tile Turing can choose (two shapes were added for the
8 x 256 tile, which none reached).

## One to 4 rows: an in-order GEMV

At a few hundred to a few thousand columns even the short tiles left SMs
idle (1 x 1024 x 1024 at 0.22x), and for one row an 8-row tile still
computes 7 rows for nothing. A GEMV-shaped kernel does neither: each thread
owns one column j of C, grid-strided, and runs C[i][j] = fma(A[i][k],
B[k][j], C[i][j]) for k = 0..K-1 in order, the same bits as every tile. It
loads U rows of its column of B before their FMAs (a warp's loads of one
row are one 128-byte line) and reads A, the same address across a warp,
through the read-only cache.

It was first written as CUDA C and compiled on the T4 with NVRTC (job
`y-f32-t4-gemv`, `t4_gemv_proto/`, the bench script beside it), in five
block sizes and unrolls, all 72 bit-exact. Blocks of 32 threads with 32
rows ahead won where 128-thread blocks would leave most SMs without one,
128 threads with 16 rows elsewhere; up to 4 rows it beat the tiles (1 x
1024 x 1024 0.22x -> 0.51x, 1 x 4096 x 4096 0.73x -> 1.18x), at 8 rows it
did not (8 x 4096 x 4096 1.45x on the 8 x 256 tile, 0.95x). The compiler
now emits it on Turing for one to 4 rows (`emit_f32_gemv_kernel`; 128
threads where those give 75% of the SMs a CTA). Version 13 (`t4_gemv/`,
all 28 bit-exact), the emitted kernel against the commit before:

| shape | before | GEMV |
|---|---|---|
| 1 x 896 x 4864 | 0.272x [0.26-0.28] | 0.562x [0.54-0.57] |
| 1 x 1024 x 1024 | 0.209x [0.20-0.22] | 0.481x [0.46-0.50] |
| 1 x 2048 x 2048 | 0.575x [0.56-0.63] | 1.062x [1.06-1.07] |
| 1 x 3840 x 1024 | 0.575x [0.55-0.64] | 1.078x [1.07-1.09] |
| 1 x 4096 x 4096 | 0.804x [0.80-0.86] | 1.172x [1.16-1.18] |
| 1 x 4864 x 896 | 0.769x [0.75-0.87] | 1.054x [1.05-1.06] |
| 1 x 50257 x 768 | 0.619x [0.60-0.78] | 0.790x [0.78-0.79] |
| 1 x 30522 x 768 | 1.026x [0.97-1.09] | 0.921x [0.89-0.93] |
| 2 x 768 x 3072 | 0.297x [0.29-0.31] | 0.547x [0.53-0.56] |
| 2 x 4096 x 11008 | 0.700x [0.70-0.74] | 0.966x [0.96-1.00] |
| 3 x 1001 x 37 | 1.986x [1.93-2.04] | 2.000x [1.94-2.06] |
| 4 x 896 x 4864 | 0.297x [0.27-0.30] | 0.379x [0.34-0.38] |
| 4 x 4096 x 4096 | 0.702x [0.67-0.72] | 0.799x [0.74-0.81] |
| 4 x 11008 x 4096 | 0.830x [0.82-0.84] | 0.996x [0.96-1.03] |

It lost one: 1 x 30522 x 768, where the 8 x 256 tile has 120 CTAs. Version
14 (`t4_gemv_band/`, all 20 bit-exact; the T4's clock fell to 585-945 MHz
for part of the run, so its quartiles are wide) placed that edge at even N:

| shape | 8 x 256 CTAs | GEMV | 8 x 256 |
|---|---|---|---|
| 1 x 12288 x 1024 | 48 | 1.456x [1.44-1.55] | 1.035x [0.99-1.08] |
| 1 x 16384 x 1024 | 64 | 1.263x [1.21-1.33] | 1.089x [1.03-1.10] |
| 1 x 20480 x 1024 | 80 | 1.138x [1.12-1.29] | 1.059x [0.99-1.09] |
| 1 x 24576 x 768 | 96 | 1.066x [0.91-1.16] | 0.885x [0.84-0.94] |
| 1 x 30522 x 768 | 120 | 1.178x [0.97-1.31] | 1.298x [1.10-1.32] |
| 1 x 32000 x 4096 | 125 | 0.974x [0.87-1.01] | 1.288x [1.20-1.35] |
| 1 x 65536 x 512 | 256 | 0.890x [0.84-1.40] | 1.158x [0.94-1.33] |
| 2 x 30522 x 768 | 120 | 0.791x [0.72-0.90] | 1.122x [1.01-1.15] |
| 4 x 30522 x 768 | 120 | 0.720x [0.66-0.79] | 1.599x [1.49-1.65] |
| 4 x 32000 x 4096 | 125 | 1.355x [1.30-1.38] | 1.911x [1.88-1.97] |

With three waves of its CTAs the 8 x 256 tile streams B faster; so at N
even with at least three waves the tile stays, elsewhere one to 4 rows
take the GEMV (at N odd its rival is the 16 x 128 tile, which lost at 1 x
50257 x 768). Every shape here, built by the committed compiler, is byte
for byte the faster variant timed. Bits: the CPU interpreter at 4 ragged
shapes on two grids each, the RTX 4070 Ti SUPER at 9 (memcheck and
racecheck clean at 6), and `built_for_sm_75_each_element_is_the_same_fma_chain`,
which now runs both GEMV blocks. A grid of 2^25 CTAs or more would have
wrapped a 32-bit column stride (at 2^32, to 0: a loop without end), so the
column index is 64-bit. Reversing the FMAs of one block changes C by an ulp
and the exactness test fails on it.
