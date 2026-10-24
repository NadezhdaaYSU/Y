# Compensated accumulation as the decode default, 2026-10-24

RTX 4070 Ti SUPER (66 SMs, 48 MB L2), driver 615.71.09, `ptxas` from CUDA
13.4, PyTorch 2.13.0+cu130 (`timing/gpu.txt`).

## What changed

`PtxEmitter::f16_accumulation` with `Y_F16_ACCUMULATE` unset:

| GEMM | before | now |
| :--- | :--- | :--- |
| at most 32 rows, unsplit | `F32` (each mma's result added in float32) | **`Kahan`** (those adds compensated once per K-tile) |
| any split of a split-K GEMM | `F32` | `F32` |
| more than 32 rows | `Chain` | `Chain` |

`Y_F16_ACCUMULATE=f32|kahan|mma` still selects one mode for every shape.
The mode and its accuracy and speed at N = K = 4096 are in
`../f16_kahan_2026-10-20/`: lower rel_F and max |E|/(|A||B|) than cuBLAS
and SGEMM in all 63 cases measured; from DRAM within 0.3% of `F32`; with B
in the L2 1.19-1.20x `F32`'s time at M = 1-8 and 1.06x at 32, still
0.39-0.92x cuBLAS's. Split K keeps `F32` because compensation took
1.26-1.40x its time at 512 x 512.

The fused SwiGLU kernel follows the same rule, but its fixed 64-row tile
must divide M, so with no `Y_SWIGLU_TILE` it never has 32 rows or fewer
(the compiler stops at an assertion for M = 8). It was not timed.

## Decode at N or K = 11008 and 14336 (`timing/`)

Llama-2-7B's and Llama-3-8B's MLP widths, M = 1, 8 and 32. B is 90-117 MB,
more than the L2, so it streams from DRAM even though `bench_ab.py` reuses
one B. `run.sh` (calling `gen.sh` and
`../f16_square_2026-10-18/bench_ab.py`): every column exact on small
integers, then the minimum of 15 interleaved rounds, two runs. No game was
running. The Steam client was open and utilisation read 8-21% before each
run, so by the 5% rule both runs are contended. `table.py` gives
(`timing/table.md`, run 1 / run 2):

| M x N x K | tile | f32 | kahan | mma | cuBLAS | kahan / f32 | cuBLAS / kahan |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1x11008x4096 | 16x64 1x2 | 151.6 / 150.1 | 149.8 / 149.8 | 148.4 / 149.7 | 154.9 / 154.6 | 0.988 / 0.998 | 1.034 / 1.032 |
| 8x11008x4096 | 16x64 1x2 | 149.8 / 151.4 | 153.2 / 150.4 | 150.7 / 149.5 | 152.8 / 153.8 | 1.023 / 0.994 | 0.997 / 1.023 |
| 32x11008x4096 | 32x64 2x2 | 153.9 / 153.4 | 155.4 / 153.2 | 155.4 / 153.1 | 178.6 / 176.2 | 1.010 / 0.999 | 1.149 / 1.150 |
| 1x4096x11008 | 16x64 1x2 | 150.9 / 149.4 | 148.7 / 148.7 | 149.2 / 148.0 | 156.8 / 155.4 | 0.986 / 0.995 | 1.054 / 1.045 |
| 8x4096x11008 | 16x64 1x2 | 149.8 / 149.6 | 151.3 / 150.6 | 149.8 / 149.4 | 154.2 / 153.4 | 1.010 / 1.007 | 1.019 / 1.019 |
| 32x4096x11008 | 32x64 2x2 | 155.1 / 155.5 | 159.1 / 158.8 | 152.2 / 153.3 | 161.2 / 160.1 | 1.026 / 1.022 | 1.013 / 1.008 |
| 1x14336x4096 | 16x64 1x2 | 206.0 / 205.0 | 204.8 / 207.1 | 202.5 / 201.2 | 199.4 / 199.7 | 0.994 / 1.010 | 0.974 / 0.964 |
| 8x14336x4096 | 16x64 1x2 | 206.9 / 207.9 | 207.8 / 210.2 | 203.9 / 203.2 | 200.7 / 201.2 | 1.004 / 1.011 | 0.966 / 0.957 |
| 32x14336x4096 | 32x64 2x2 | 199.7 / 197.5 | 199.7 / 198.5 | 199.9 / 197.3 | 200.0 / 198.1 | 1.000 / 1.005 | 1.001 / 0.998 |
| 1x4096x14336 | 16x64 1x2 | 195.1 / 192.2 | 195.5 / 192.4 | 194.1 / 191.9 | 201.9 / 199.8 | 1.002 / 1.001 | 1.033 / 1.038 |
| 8x4096x14336 | 16x64 1x2 | 195.6 / 192.5 | 195.4 / 192.8 | 195.1 / 192.3 | 198.8 / 196.0 | 0.999 / 1.001 | 1.017 / 1.017 |
| 32x4096x14336 | 32x64 2x2 | 201.9 / 200.0 | 207.1 / 203.8 | 199.0 / 196.6 | 204.6 / 202.6 | 1.026 / 1.019 | 0.988 / 0.994 |

- **At M = 1 and 8 compensation is within noise of `F32`**: 0.99-1.02x its
  time, the run-to-run spread being up to 3% (8 x 11008 x 4096).
- **At M = 32 with K = 4096 it is 1.00-1.01x, and with K = 11008 or 14336
  1.02-1.03x in both runs.** These shapes take the analytic model's
  32x64x32 tile (no profile entry is within 2x), whose 32-deep K-tile is
  folded every 2 mmas; the measured decode tile at N = K = 4096 is 64 deep
  and folds every 4.
- **Against cuBLAS** the default is ahead or level at 9 of the 12 shapes,
  up to 1.15x (32 x 11008 x 4096). It trails at 1 and 8 x 14336 x 4096,
  0.96-0.97x, as `F32` (0.97x) and the chain (0.98-0.99x) do on that 16x64
  tile, and at 32 x 4096 x 14336, 0.99x, where `F32` is 1.01x.

`timing_game_running/` is a first attempt that ran while Roblox (Sober) had
the GPU: its waiter checked for a game before queueing for the GPU lock, and
the game had restarted by the time the lock came. It is kept and not used;
`run.sh` now checks once it holds the lock and before and after each run.

## Committed PTX (`check/`)

Regenerated with the re-tuned profile, exactly the four decode kernels
change, `gemm_f16_decode_m{1,4,8,32}.ptx` (`regen.txt`; `test_drift`
differs under every compiler). Registers (`registers.txt`): 118 -> 139 at
M = 1, 4 and 8, 104 -> 124 at 32, no spills. Each has 16 compensated
accumulators: 48 `sub.rn.f32` (the fold's twice, and the finite check's)
and 16 `setp.eq.b32`.

## Tests and mutants (`mutants/`)

`tests/ptx_f16_accumulate.rs` checks the default at M = 1, 8, 32 (compensated,
with the tile sums of `F16Accumulation::Kahan`), 33 and 256 (chained), the
`f32` knob at M = 8 (plain float32), and the fused SwiGLU kernel at 32 rows
under `Y_SWIGLU_TILE=32,64,32,2,2` (compensated) and at 64 (chained).
`tests/ptx_f16_split_k.rs` checks that a split of 8 rows stays plain float32,
with `Y_F16_ACCUMULATE=kahan` as the control. Six mutants of the default
(`mutants.py`): the old `F32`, no split arm, the decode arm first, `< 32`,
`<= 33`, and compensation everywhere. All six fail, each at the assertion
`failures.txt` names.

## Validator gates

`build_corpus`, `regress`, `fpgate`, `docgate`, `gap` and `unroll`, on HEAD
and on this tree (`gates.txt`).
