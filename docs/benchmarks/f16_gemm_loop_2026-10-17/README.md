# F16 GEMM K loop, 2026-10-13 to 2026-10-17

Y's `@tile(N, N, N)` F16 GEMM (f16 A and B, f32 C) before this change
(`ff3526f`) and after it. The other columns are Triton, the GEMM in
`bench_ab.py` with its autotuner's choice of configuration at each size, and
cuBLAS (`torch.mm` with an f32 output). Every column was checked equal to
float64 `A @ B`, bit for bit, before it was timed. The inputs are small
integers, so every answer is exact.

Y's tiles come from the main checkout's `.ysu_hw_profile`, whose measured
`AUTOTUNE` lines pick:

- 64x64x64 on 4x2 warps, 3 stages, at 512;
- 64x128x32 on 2x2 warps, 2 stages, at 1024;
- 128x128x32 on 2x2 warps, 2 stages, from 2048.

Triton's autotuner picked:

- 64x64x64 on 4 warps, 3 stages, at 512;
- 64x128x32 on 4 warps, 4 stages, at 1024;
- 128x128x32 on 4 warps, 4 stages, at 2048 and 4096;
- 128x256x32 on 8 warps, 3 stages, at 8192.

The kernels timed are, byte for byte, `tests/gemm_f16_<N>.ptx` at
`ff3526f` ("before") and at `8d605e9` ("after"). The tiles were then
measured again with the new loop, and three of them changed
(`../f16_tiles_2026-10-17/`). Each run started when the
GPU's utilisation, averaged over 10 samples, was at most 5% and no game was
running. Run 1 started at 0.9% and run 2 at 0.6%.

Minimum over 15 interleaved rounds in microseconds, run 1; run 2 (`run1.json`,
`run2.json`). RTX 4070 Ti SUPER, driver 615.71.09, CUDA 13.4 `ptxas`, PyTorch
2.13.0+cu130, Triton 3.7.1:

| M=N=K | before | after | Triton | cuBLAS | after / Triton |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 7.1; 7.0 | 6.7; 6.7 | 8.4; 8.3 | 5.9; 5.8 | 0.802; 0.808 |
| 1024 | 30.8; 30.6 | 30.3; 30.2 | 30.2; 30.2 | 28.8; 28.8 | 1.003; 0.999 |
| 2048 | 215.0; 215.0 | 206.9; 206.9 | 207.2; 207.3 | 204.1; 204.0 | 0.998; 0.998 |
| 4096 | 1646.1; 1642.8 | 1583.1; 1579.4 | 1596.9; 1588.2 | 1546.2; 1543.5 | 0.991; 0.994 |
| 8192 | 12713.0; 12713.5 | 12294.1; 12293.6 | 13730.3; 13708.8 | 12403.7; 12398.1 | 0.895; 0.897 |

The SM clock, read after every sample, stayed between 2460 and 2775 MHz.
From 1024 to 8192 every median is within 2.4% of its minimum. At 512 the
samples are bimodal. In run 2, the first seven or eight rounds of Y and cuBLAS
were slower (this change 7.4 against 6.7 us). In both runs, Triton's samples
sat at 9.4-9.6 us with one or two at 8.3-8.5. So the minima at 512 carry the
least weight.

M x 4096 x 4096 with M of 1 to 32 (`decode.json`), the tile measured for
those shapes (32x64 on 2x2 warps), on a GPU 1.2% busy:

| M | before | after | cuBLAS |
| ---: | ---: | ---: | ---: |
| 1 | 20.8 | 20.7 | 26.6 |
| 4 | 21.8 | 21.8 | 53.4 |
| 8 | 23.9 | 23.9 | 50.3 |
| 32 | 31.1 | 31.0 | 23.7 |

At M = 1, 4 and 8 the tile does not divide M, so these shapes keep the
per-chunk copies. At 32 the addresses are hoisted, and the time does not
move. `kernels/` holds these kernels' sources and PTX. The PTX is
`tests/gemm_f16_decode_m<M>.ptx` at `ff3526f` and at `8d605e9`, with the
kernel renamed. (This paragraph said they were other kernels, with other K
and N, until the tiles were measured again.)

## Where the time went

`profile/` holds both kernels at 1024^3 (`before_1024.ncu-repz` and
`after_1024.ncu-repz`, made with `prof_f16.py`). `analysis_1024.txt` and
`analysis_4096.txt` hold the region split (`regions.py`) and the loop mix
with its stall samples (`loopmix.py`) for before, after and Triton. The
4096 reports are not kept; they are 3.2-3.3 MB each. In every profile
Triton's configuration is pinned to what its autotuner picked.

Split at the K loop, the epilogue was not where Y lost time, as it had been
for the FP8 GEMM. At 4096, 1.7% of Y's stall samples fell after the loop and
1.1% of Triton's, and both wrote C with the same 131,072 `STG.E.128`. Both
spent 98% of their samples in the loop:

| per K iteration, 4096^3 | Y before | Triton | Y after |
| :--- | ---: | ---: | ---: |
| SASS instructions | 181 | 132 | 149 |
| HMMA | 64 | 64 | 64 |
| IMAD | 44 | 17 | 24 |
| ISETP | 25 | 4 | 6 |
| LDSM | 24 | 16 | 16 |
| LDGSTS | 8 | 8 | 8 |
| long-scoreboard stall samples in the loop | 10,560 | 446 | 963 |
| MIO-throttle stall samples in the loop | 13,424 | 4,258 | 9,531 |

Two things caused most of the difference:

- Every `cp.async` chunk recomputed its whole address and both bound checks
  at every K-tile. The long-scoreboard stalls sat on those IMADs: each one
  overwrote a register that the previous chunk's LDGSTS was still reading.
- B's fragments were loaded with two `ldmatrix.x2.trans` per 16 columns, and
  the loop's MIO-throttle stalls sat on them.

12 of the 149 instructions after the change are `@!PT LDS RZ, [RZ]`, which
are never executed: ptxas pads copy groups with them, and they carry part
of the remaining MIO-throttle samples.

## What changed

All of it is in `emit_tensor_core_gemm_kernel`'s pipelined path:

1. B's fragments are loaded with one `ldmatrix.x4.trans` per 16 columns. The
   registers are the same, in the same order (`emit_gemm_compute_block`,
   which the SwiGLU GEMM shares).
2. Where M, N and K are whole multiples of the tile, each thread's copy
   addresses are computed once, before the loop, and each chunk is
   `[base + immediate]` (`emit_gemm_tile_load_async_setup`). The bound checks
   become one CTA-wide predicate.
3. Where that leaves at most two copies after each MMA group of the first
   16-deep K step, the copies are issued between the MMAs. The K-tile wait
   and the barrier then move to the top of the loop.

`steps/` times each step, in one process a run, on a quiet GPU. `steps/ptx/`
holds the PTX of every step. Minimum microseconds:

| run | step | 512 | 1024 | 2048 | 4096 | 8192 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `steps_1_2.json` | before | 7.0 | 30.7 | 215.4 | 1645.6 | 12738.2 |
| | 1 and 2 (`new`) | 6.7 | 30.2 | 212.0 | 1610.8 | 12453.2 |
| | 3, copies between MMAs at every size (`new2`) | 6.8 | 31.3 | 209.6 | 1601.5 | 12367.4 |
| | Triton | 9.4 | 30.2 | 207.2 | 1596.6 | 13971.4 |
| `steps_2b_3.json` | before | 7.0 | 30.6 | 214.9 | 1642.5 | 12716.0 |
| | copies between MMAs only where at most two follow each group (`final`) | 6.7 | 30.0 | 209.1 | 1599.0 | 12344.3 |
| | and the wait at the loop's top, every tile (`rot`) | 6.7 | 31.5 | 206.8 | 1580.0 | 12288.5 |
| | Triton | 8.3 | 30.8 | 207.2 | 1595.4 | 13939.8 |

Steps 1 and 2 alone left the copies back to back, and they filled the
load/store queue: at 4096 the loop had 7,510 `lg_throttle` samples, 7,325 of
them on B's second to fourth copies. Spread out, the copies met a wait (`DEPBAR`) that ptxas had hoisted
ahead of the iteration's last 38 MMAs, with 24,200 long-scoreboard samples on
the MMA after it (`profile/steps_4096.txt`). At the top of the loop, that wait cannot move above the
previous iteration's MMAs.

The 64x128 tile at 1024 has two MMA groups for six copies. There,
interleaving measured 3.6% slower and the moved wait 5%. So that tile issues
its copies first and waits at the end of the iteration, as before. The
change is `rot` where the copies spread and `final` where they do not. Moving
the wait without interleaving the copies (`rotni`) was slower at every size
(`steps/decide.json`, 2026-10-17):

| step (`decide.json`) | 1024 | 2048 | 4096 |
| :--- | ---: | ---: | ---: |
| before | 30.5 | 215.4 | 1645.6 |
| the change, with `mad.wide` (`r2`) | 29.9 | 207.3 | 1582.1 |
| with `mul.wide` (`r3`) | 30.0 | 207.5 | 1594.4 |
| the wait at the top without interleaving (`rotni`) | 31.6 | 223.3 | 1708.0 |
| Triton | 30.2 | 207.1 | 1596.8 |

## The K-tile's offset: `mad.wide`, `mul.wide` or a shift

Each K-tile adds `k0 * step` bytes to a thread's hoisted base address. The
step is 2 bytes for A and 2 * N for B. Written as one `mad.wide.u32`, ptxas
made the offset a LEA and LEA.HI.X pair, but the translation validator does
not model `mad.wide`. Written as `mul.wide.u32` and `add.s64`, it became one
IMAD.WIDE.U32, two SASS instructions fewer in the loop, and was slower. In
six runs, each in its own process (`steps/r2r3_*.json`), the slowdown at 4096
was 0.69-0.81%, median 0.77%. At 8192 the median was 0.89%, and at 2048 0.13%.
As a shift (`cvt.u64.u32`, `shl.b64`, `add.s64`), where the step is a power of
two, ptxas makes the LEA pair again. That form is what this change emits; it
uses only instructions the validator models. Four runs against `mad.wide`
(`steps/r2r4_*.json`) gave median ratios of 1.0004, 0.9998 and 0.9996 at 2048,
4096 and 8192. A step that is not a power of two is a `mul.wide`.

## Still open

At 1024 this change ties Triton. Its loop is 94 instructions against
Triton's 91, but it has 653 long-scoreboard stall samples against Triton's 66
(`profile/analysis_1024.txt`). That tile keeps the copies back to back, at
the start of the iteration. Its two stages make the wait at the end of the
iteration a wait for every copy, the ones just issued included, and ptxas
moved that wait up to after the fourth of the iteration's 32 MMAs. Measured
again with the new loop, the tile at 1024 takes three stages
(`../f16_tiles_2026-10-17/`): 29.6-29.8 us against Triton's 30.2-30.7 in the
same runs.

## Reproduce

    # PTX from each compiler, in a directory holding the profile:
    Y g4096.ysu --emit-ptx -o after/g4096.ptx      # and before/, each size
    PYTHONPATH=<checkout>/python python3 bench_ab.py before=before,after=after out.json [sizes]
    cd profile && ncu --profile-from-start off --set full --import-source yes \
        -o out python3 prof_f16.py 1024 && python3 regions.py out.ncu-repz
