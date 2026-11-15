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

## Speed on a Tesla T4

Pending: the Kaggle job `y-f32-t4` (version 2) times both libraries in one
session on these shapes and three more.
