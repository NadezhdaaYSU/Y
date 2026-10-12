# FP8 GEMM with `cp.async` stages, 2026-10-12

Y's `@tile(N, N, N)` FP8 GEMM from the compiler at `6a7e307` ("before"; the
tiles of `../fp8_gemm_tiles_2026-10-12/`) and with this change ("after"),
against the Triton kernel of `python/examples/run_head_to_head_comparison.py`.
All three take float32 A and B, quantize them to e4m3 inside the kernel,
multiply on the e4m3 tensor cores and write float32 C; every column was
checked equal to `A @ B` bit for bit before it was timed. `bench.py` (the
tiles record's, unchanged) produced `results.json`.

Minimum over 15 interleaved rounds (9 at 8192), microseconds, RTX 4070 Ti
SUPER, driver 615.71.09, CUDA 13.4, PyTorch 2.13.0, Triton 3.7.1:

| N | before | after | Triton | after vs before | after vs Triton |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 5.0 | 4.7 | 11.3 | 1.06x faster | 0.42x |
| 512 | 10.2 | 9.5 | 9.4 | 1.07x faster | 1.00x |
| 768 | 24.4 | 24.5 | 24.8 | same tile | 0.99x |
| 1024 | 41.0 | 34.6 | 32.6 | 1.18x faster | 1.06x |
| 1536 | 139.1 | 121.5 | 132.7 | 1.14x faster | 0.92x |
| 2048 | 207.4 | 214.0 | 200.3 | 1.03x slower | 1.07x |
| 2560 | 472.3 | 427.8 | 443.6 | 1.10x faster | 0.96x |
| 3072 | 706.4 | 695.3 | 726.0 | 1.02x faster | 0.96x |
| 4096 | 1540.0 | 1510.4 | 1571.3 | 1.02x faster | 0.96x |
| 8192 | 11794.1 | 11576.9 | 15304.9 | 1.02x faster | 0.76x |

The GPU was idle when the run started (`gpu_busy_before`, 0%); the readings
per size, 20-50%, are taken right after that size's warm-up and Triton's
autotuning, which they include. A browser was open.

## What changed

At 1024^3 the previous kernel took 1.26x Triton's time. Nsight Compute's
per-instruction stall samples put a quarter of them on one `FMUL`, the scale
multiply of a value loaded a whole K-tile earlier: its warps waited 11.7
cycles per instruction on global loads, Triton's 1.0. A kernel that
prefetches into registers can keep only as many K-tiles in flight as its
registers hold; Triton's kernel issues no `ld.global` at all.

`Fp8Tile::stages` 2 (`emit_fp8_k_loop_async` in `src/ptx_emitter.rs`):

- raw f32 K-tiles are copied into shared memory with 16-byte `cp.async`s,
  each warp copying consecutive chunks of a row, rows and columns past M and
  N zero-filled;
- each iteration copies tile k+2, multiplies tile k from its e4m3 stage,
  waits for tile k+1 and converts it into the other e4m3 stage, between two
  barriers, so two K-tiles are in flight during each multiply and none waits
  in a register (117 registers at 128x128, from 143);
- B's raw rows are stored XOR-swizzled by 16-byte chunk, `(row >> 2) & 7`,
  so that the conversion reads them in the transposed stores' lane order
  (eight lanes on rows four apart) without bank conflicts;
- the stages need dynamic shared memory, 45 KB at 64x64 and 90 KB at
  128x128, which the PTX states for the launcher.

`fp8_padded_tile` uses it at 64x64 when M or N is at most 512 and the grid is
one wave, and at 128x128 when there are at least 64 such tiles; 768^3 keeps
the register path (64x64, K step 32), as do small shapes with more than one
wave of CTAs.

## The sweep

`sweep/*.json`: each tile against Triton in the same run, exact first, with
`Y_FP8_TILE=cta_m,cta_n,warps_m,warps_n,cta_k,stages`, an override the
committed compiler does not have (`sweep/tile_override.patch` adds it back).
The tiles timed, `stages` 2 against the register path (0):

- 512-2048 (`sizes_512_2048.json`): 128x128 with `cp.async` is fastest from
  1024 to 1536 (34.8 us against 41.4 at 1024); at 512, 64x64 with a 32-wide
  K step and `cp.async` (9.4 against 10.1); at 768 the register 64x64 tile
  (24.3 against 26.7).
- 256-512 (`small_256_512.json`): 64x64 `cp.async` fastest at each (4.7,
  7.0, 9.5 us).
- 2048-8192 (`sizes_2048_8192.json`, `sizes_2560_4096.json`): 128x128
  `cp.async` against the previous 256x128 register tile: 427.4 against 471.0
  at 2560, 692.2 against 713.7 at 3072, 1510.9 against 1536.6 at 4096,
  12221 against 12444 at 8192; 214.3 against 205.8 at 2048.
- Around 2048 (`around_2048.json`): 147.6 against 176.4 at 1792, 297.0
  against 328.0 at 2304. The 256x128 tile wins at 2048 alone, so the rule
  does not special-case it.
- Rectangular shapes with 512 CTAs of 64x64 (printed, not saved): 512 x
  4096 x 1024 100.7 us with `cp.async` against 97.3 on the register path,
  4096 x 512 x 1024 98.7 against 96.8, 256 x 8192 x 512 55.3 against 55.4.
  Hence the one-wave condition.

## Tests

`python/y_lang/tests/test_fp8_gemm.py` runs every tile `fp8_padded_tile`
picks, `cp.async` and register, with ragged edges, exactly, and runs four
ragged shapes under `compute-sanitizer --tool memcheck` with PyTorch's
caching allocator off: a copy past A's last row feeds only outputs the
epilogue drops, which the exactness check cannot see (1,541 invalid reads
when the zero-fill is removed). Each of 11 mutants of the `cp.async` path
fails a test; `test_fp8_gemm_kernel_async_copies_raw_tiles` checks the
loop's structure.

## Reproduce

    python3 bench.py emit <checkout-before> before/
    python3 bench.py emit <checkout-after> after/
    PYTHONPATH=<checkout>/python python3 bench.py time before/ after/ results.json
