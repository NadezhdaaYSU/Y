# The FP32 GEMM below sm_80, on a Tesla T4 (2026-11-12)

Until `9454e4f` every Y GEMM needed sm_80. The FP32 one (`@tile(M, N, K)`
on `GlobalMemory<F32>`, `src/ptx_emitter/f32_simt_gemm.rs`) stages through
`cp.async`, and for an older target the compiler refused it by name: on a
Tesla T4 (Kaggle's and Colab's free GPU), a V100 or a GTX 1050 Ti there was
none. Below sm_80 each copy is now a load into a register and a shared
store, the 74 KB large tile is left out where it does not fit a block, and
registers are budgeted for the CTAs the target's shared memory admits. The
arithmetic, one in-order `fma.rn.f32` chain per element, is unchanged.

## Bits

- `sm75_on_sm89_exact.py`: 10 shapes (ragged, one row, one column, N odd)
  built for sm_75 and for sm_89 and run on the RTX 4070 Ti SUPER, whose
  driver compiles sm_75 PTX for it; on the header's grid and on a 3x1 grid
  every element equals the in-order `fmaf` oracle, and the two builds equal
  each other. `tests/f32_simt_gemm.rs` does the same for its 33 exactness
  shapes (`built_for_sm_75_each_element_is_the_same_fma_chain`).
- `t4.txt`: on the T4 itself, all 10 shapes of `f32_vs_cublas.py` bit-exact.

## Speed

`f32_vs_cublas.py` (`t4.txt`): Tesla T4, driver 580.178.04, torch 2.13.0
(CUDA 12.6), on Kaggle; Y compiled through the wheel's `liby.so` for the
card, cuBLAS through `torch.mm` with TF32 off; 11 alternating rounds after a
3 s warm-up, the minimum of each. The T4 is a 70 W card that clocks down
under load, so absolute TFLOPS move; the ratio is the number to read.

| shape | tile | cuBLAS TFLOPS | Y TFLOPS | Y / cuBLAS |
|---|---|---|---|---|
| 512^3 | small | 5.23 | 3.64 | 0.70x |
| 1024^3 | mid | 4.60 | 3.78 | 0.82x |
| 2048^3 | mid | 4.73 | 3.69 | 0.78x |
| 4096^3 | mid | 4.00 | 3.83 | 0.96x |
| 1023 x 1025 x 1027 | mid | 4.45 | 0.97 | 0.22x |
| 1024 x 4096 x 4096 | mid | 4.81 | 3.66 | 0.76x |
| 128 x 4096 x 4096 | mid | 4.59 | 3.54 | 0.77x |
| 4096 x 128 x 4096 | mid | 3.93 | 3.36 | 0.85x |
| 64 x 8192 x 2048 | mid | 4.18 | 3.49 | 0.83x |
| 4096 x 4096 x 1024 | mid | 4.91 | 3.64 | 0.74x |

0.70-0.96x where N is a multiple of 4. At N = 1 mod 4 the many narrow B
copies' loads spill (ptxas hoists them ahead of their stores): 0.22x, and
`docs/known_issues.md` records it.

**Correction (2026-11-15):** the cause was not N. 1023 x 1025 x 1027 also has
a K tail and a ragged M, and for either one the K loop guarded every copy,
which below sm_80 is a predicated load and a zero fill. That is what spilled.
N odd alone (1024 x 1025 x 1024) is 125 registers and no spill; K alone (1024
x 1024 x 1027) spilled 424 bytes. The loop's copies take no guards now
(`docs/known_issues.md`); the table above is the code before that change.

The plain loads also hide less latency
than `cp.async`; the usual remedy on these cards (CUTLASS's sm_70 SIMT
kernels) is to load the next K tile into registers during the current one's
FMAs, which this version does not do.

For reference, the same sm_75 build on the RTX 4070 Ti SUPER
(`sm75_on_sm89_exact.py`'s shapes through `f32_vs_cublas.py` with the target
forced): 1024^3 1.21x cuBLAS (the cp.async build 1.25x), 4096^3 0.83x (the mid
tile; cp.async uses the large one), 1023 x 1025 x 1027 0.29x.
