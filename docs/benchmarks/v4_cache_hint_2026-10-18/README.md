# The v4 built-ins without an evict-first hint, 2026-10-18

The head-to-head's elementwise and RMSNorm rows (`../head_to_head_2026-10-18/`)
are 100,000 elements and 128 x 1024: about a microsecond, launch latency. At
sizes where the data decides (`find.log`, `probe.py`, the compiler of
`bc0bbc3`), Y's `vec_add_v4`, `swiglu_v4` and `rmsnorm_v4` kernels tied Triton
once the data was larger than the L2 and took 2.3-2.9x Triton's time when it
fitted: 67.93 against 23.07 us for a vector add of 2^22 floats, 36.48 against
13.63 for RMSNorm on 4096 x 1024.

Every load and store of those built-ins, and the 3-D block-pointer v4 store,
was `ld.global.cs` / `st.global.cs`: evict-first in L1 and L2
(`LDG.E.EF.128` / `STG.E.EF.128` in SASS, `sass_forms.txt`). A benchmark that
calls a kernel again on the same buffers, like a model whose next kernel reads
this one's output, found its lines gone from L2 and read them from DRAM: on
data that fitted Y reached 690-790 GB/s, little more than the 610-625 GB/s
it reached from DRAM.

`probe_cs.py` times three versions of each kernel's PTX against Triton's:
`cs`, what the compiler emitted; `plain`, the same PTX with the hint removed;
`ca`, `.ca` loads and plain stores (`ld.global.ca`, the cache operator an
element read takes, is `LDG.E.128.STRONG.SM`). `ptx/` holds all nine; the
script read them under the names `y_<kernel>.ptx` (`cs`) and
`y_<kernel>_<variant>.ptx`. The compiler with this change emits, byte for
byte, the `plain` PTX of all three kernels.

Each column is checked first (vector add exact, the others within 1e-5 of the
largest output against float64), then timed by CUDA events over back-to-back
calls, 17 interleaved rounds of which the first two are dropped; minimum of
15. Two runs, each started by `gpu_quiet_then.sh` at a 10-sample mean of at
most 5% with no game running (`queue.txt`: 2.4% and 0.6%). RTX 4070 Ti SUPER
(48 MB L2), driver 615.71.09, PyTorch 2.13.0+cu130, Triton 3.7.1; the SM clock
was 2475-2790 MHz. Microseconds, run 1 / run 2 (`tables.py`):

| shape | footprint | Y cs | Y plain | Y ca | Triton | cs / plain | plain vs Triton |
|---|---|---|---|---|---|---|---|
| vec_add 1048576 | 13 MB | 5.05 / 5.52 | 4.97 / 5.00 | 4.97 / 4.98 | 5.50 / 5.51 | 1.02x / 1.10x | 1.11x / 1.10x |
| swiglu 1048576 | 13 MB | 5.04 / 5.20 | 4.98 / 5.01 | 4.99 / 5.03 | 5.54 / 5.58 | 1.01x / 1.04x | 1.11x / 1.11x |
| vec_add 4194304 | 50 MB | 72.15 / 72.92 | 26.81 / 27.25 | 27.48 / 27.39 | 25.80 / 25.85 | 2.69x / 2.68x | 0.96x / 0.95x |
| swiglu 4194304 | 50 MB | 71.95 / 71.75 | 24.80 / 25.01 | 27.31 / 27.18 | 26.03 / 35.49 | 2.90x / 2.87x | 1.05x / 1.42x |
| vec_add 16777216 | 201 MB | 322.29 / 323.24 | 324.10 / 325.52 | 323.58 / 324.79 | 323.36 / 326.36 | 0.99x / 0.99x | 1.00x / 1.00x |
| swiglu 16777216 | 201 MB | 321.62 / 322.32 | 325.06 / 325.06 | 323.36 / 324.61 | 324.72 / 325.86 | 0.99x / 0.99x | 1.00x / 1.00x |
| RMSNorm 4096 x 1024 | 34 MB | 42.71 / 43.94 | 13.23 / 13.31 | 13.25 / 13.29 | 13.55 / 13.58 | 3.23x / 3.30x | 1.02x / 1.02x |
| RMSNorm 16384 x 1024 | 134 MB | 218.74 / 219.86 | 218.35 / 219.25 | 218.03 / 219.13 | 218.34 / 219.59 | 1.00x / 1.00x | 1.00x / 1.00x |
| RMSNorm 4096 x 4096 | 134 MB | 217.39 / 217.88 | 217.88 / 219.39 | 217.95 / 218.96 | 218.27 / 219.37 | 1.00x / 0.99x | 1.00x / 1.00x |
| RMSNorm 16384 x 4096 | 537 MB | 866.65 / 871.08 | 872.09 / 874.99 | 872.11 / 875.18 | 873.81 / 879.07 | 0.99x / 1.00x | 1.00x / 1.00x |

"cs / plain" above 1 is the hint's cost; "plain vs Triton" above 1 is Y
faster. The footprint counts every byte the kernel reads or writes once.

- Without the hint the 50 MB and 34 MB shapes take 2.68-3.30x less time,
  and Y is 0.95-0.96x Triton on the vector add, 1.05x and 1.42x on SwiGLU
  (Triton's own minimum moved from 26.03 to 35.49 us between the runs) and
  1.02x on RMSNorm. At 13 MB Y without the hint is 1.10-1.11x Triton, and
  the hint costs 1-10%.
- From 134 MB up the hint saved at most 1.1% (run 1's SwiGLU at 2^24, 321.62
  against 325.06 us) and cost 0.2-0.3% on RMSNorm at 16384 x 1024; Y without
  it is 1.00x Triton in every one of those rows in both runs.
- `.ca` loads are within 0.5% of plain ones except at 2^22: 2.5% and 0.5%
  slower on the vector add, 10.1% and 8.7% on SwiGLU. 2^22 floats is the
  L2's own size (below), and element reads, which take `.ca`, tie plain ones
  at every footprint either side of it ("Element reads").

## At the L2's size

The 50 MB shapes above sit on the L2's own size, and there the order of the
columns does not hold from one run to the next. `edge/probe_edge.py` (the
compiler of `957899d`, two runs: `edge/edge1.log`, `edge/edge2.log`) times
`vec_add_v4`, a one-element-a-thread vector add (`C[i] = A[i] + B[i]`) and
Triton's at ten footprints, each checked exact first. Triton's time over
Y's, run 1 / run 2:

| footprint | `vec_add_v4` | one element a thread |
|---|---|---|
| 16-44 MB | 1.080-1.124 / 1.080-1.136 | 1.078-1.164 / 1.090-1.160 |
| 48 MB | 1.005 / 0.907 | 0.806 / 0.693 |
| 52 MB | 0.928 / 0.987 | 1.187 / 1.214 |
| 56 MB | 0.985 / 1.061 | 1.037 / 0.969 |
| 64 and 96 MB | 1.001, 0.992 / 1.010, 1.002 | 1.015, 0.985 / 1.031, 1.004 |

Below 48 MB Y's vector adds are 1.08-1.16x Triton's in both runs, and from
64 MB they tie; between, either kernel leads by up to 1.44x, in an order that
changes between runs. The 50 MB rows of the first table are of that kind.

A read that should stream can still say so: `@cache_policy(L2_STREAM)` on an
element read emits `ld.global.cs` (`tests/ptx_cache_policy.rs`).

## Element reads

An element read (`A[i]`) emits `ld.global.ca`, which ptxas makes
`LDG.E.STRONG.SM`; without the operator it is `LDG.E`
(`element_reads/sass_forms.txt`). `element_reads/probe_ca2.py` (the compiler
of `957899d`) compiles a one-element-a-thread vector add (`C[i] = A[i] +
B[i]`) and multiply-add (`D[i] = A[i] * B[i] + C[i]`) from Y source, times
them as emitted and with `.ca` removed (`element_reads/*.ptx`) against
Triton's, each column checked first, at footprints away from the L2's size.
Two runs, each started by `gpu_quiet_then.sh` at 4.1% and 2.7% with no game
running (`element_reads/queue.txt`). Microseconds, run 1 / run 2
(`element_reads/table.py`):

| kernel | footprint | Y plain | Y `.ca` | Triton | `.ca` / plain | Triton / plain |
|---|---|---|---|---|---|---|
| add1 | 24 MB | 7.91 / 7.89 | 7.91 / 7.89 | 9.13 / 9.07 | 1.000 / 1.000 | 1.154 / 1.150 |
| fma1 | 24 MB | 7.35 / 7.33 | 7.35 / 7.32 | 8.52 / 8.55 | 1.000 / 0.999 | 1.159 / 1.166 |
| add1 | 32 MB | 10.17 / 10.11 | 10.15 / 10.10 | 11.36 / 11.29 | 0.998 / 0.999 | 1.117 / 1.117 |
| fma1 | 32 MB | 9.38 / 9.34 | 9.38 / 9.33 | 10.78 / 10.74 | 1.000 / 0.999 | 1.149 / 1.150 |
| add1 | 40 MB | 12.45 / 12.40 | 12.41 / 12.37 | 13.73 / 13.65 | 0.997 / 0.998 | 1.103 / 1.101 |
| fma1 | 40 MB | 11.29 / 11.35 | 11.28 / 11.36 | 12.64 / 12.61 | 0.999 / 1.001 | 1.120 / 1.111 |
| add1 | 96 MB | 154.15 / 154.03 | 154.35 / 154.30 | 154.41 / 153.94 | 1.001 / 1.002 | 1.002 / 0.999 |
| fma1 | 96 MB | 155.20 / 155.29 | 155.26 / 155.32 | 155.60 / 155.59 | 1.000 / 1.000 | 1.003 / 1.002 |
| add1 | 192 MB | 308.00 / 308.31 | 308.45 / 308.00 | 310.39 / 309.36 | 1.001 / 0.999 | 1.008 / 1.003 |
| fma1 | 192 MB | 308.22 / 308.41 | 308.64 / 308.57 | 309.48 / 309.02 | 1.001 / 1.001 | 1.004 / 1.002 |

`.ca` takes 0.997-1.002x plain's time in every row, so element reads keep
it. The timing script imports `bench_ab.py` from
`../f16_square_2026-10-18/` for its PTX loader.
