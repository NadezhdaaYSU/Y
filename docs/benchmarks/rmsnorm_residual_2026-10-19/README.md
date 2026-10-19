# `rmsnorm_residual_<d>` with h in registers, 2026-10-19

The fused residual-add RMSNorm (`h = X + Residual`, `NewResidual = f16(h)`,
`Out = f16(h * rsqrt(mean(h^2) + 1e-5) * Weight)`) had been compared only
with FlashInfer, at 128, 1024 and 8192 rows of 4096. RTX 4070 Ti SUPER
(48 MB L2), driver 615.71.09, PyTorch 2.13.0+cu130, Triton 3.7.1. Every run
here was started by `gpu_quiet_then.sh` (`../paged_attention_2026-10-18/`)
at a 10-sample mean of at most 5% with no game running (each directory's
`queue.txt` or the log's first line). Every column is checked against
float64 first: NewResidual exactly, Out within 2^-10 of the largest output.
CUDA events over back-to-back calls, 17 interleaved rounds, the first two
dropped; minimum of 15.

## What it was

`discovery/probe_nr.py` timed the committed `tests/rmsnorm_residual_4096.ptx`
(and `rope_128`) against Triton kernels doing the same arithmetic, two runs
(`discovery/table.py`). Inside the L2, at 512 and 1024 rows (17 and 34 MB),
Y took 1.08-1.10x Triton's time; from DRAM they tied; at 128 rows Y was
ahead. RoPE took 1.02-1.06x Triton's time inside the L2 and tied from DRAM.

The kernel ran both passes as loops over the thread index, which `ptxas`
keeps rolled: a thread had one group's two 16-byte loads in flight at a
time, staged h through `hidden * 4` bytes of static shared memory, and
loaded Weight only after the reduction. Its warp count was the most of
{8, 4, 2, 1} dividing the groups evenly, so hidden 896 and 12032 ran on one
warp a row and 3072 and 5120 on four. Two more defects showed on the way:
`ptxas` refuses the static shared memory from hidden 12288 up (more than
48 KB), so those kernels could not be built, and a hidden size that is not a
multiple of 8 compiled and left the end of every row unwritten.

## The change

- **h stays in registers.** Each thread's groups are unrolled; every load
  (X, Residual and Weight for every group) is issued before any arithmetic,
  and NewResidual is stored as soon as h exists.
- **A predicated last group.** Where the groups do not divide evenly over
  the threads, the last one's loads are zero-filled and its stores
  predicated off.
- **Measured warp counts.** One group a thread on up to 16 warps; past
  that, up to two a thread on up to 32.
- **A fallback for very large rows.** Past 8 groups a thread (4 at 1024
  threads), from hidden 65536, the passes stay loops, and pass 2 reads X and
  Residual again instead of staging h.
- **A trap on a wrong launch.** The block size is fixed at compile time,
  and the PTX header states it; any other block size traps.
- **A refusal.** A hidden size that is not a multiple of 8 is refused.

`hand/mk_reg.py` wrote the register version of the 4096 kernel by hand
first, with the old warp count; `hand/ab_1.log` has it and the emitted
version bit for bit equal to the committed kernel, and the emitted one
1.11x faster at 17 and 34 MB. `warps/` then timed 1 to 32 warps a row at hidden 896-16384
(`probe_warps.py`, `warps_1-3.log`, `ptx/`): more warps were faster inside
the L2 at every size from 3072 up, and the rule above is within 1.1% of
the fastest count measured at each size inside the L2 and 1.4% from DRAM. (`warps/dims_1.log` is an earlier run of
the unrolled kernel with the old warp counts; its PTX is not kept.)

## Measured

`final/probe_final.py`, two runs, both started at 0.0%: the parent
compiler's kernel ("before", `final/ptx/r<d>_before.ptx`; from 12288 up
`ptxas` refuses it), this change's ("after", `r<d>_after.ptx`; `r4096_after.ptx`
is `tests/rmsnorm_residual_4096.ptx`), Triton, and FlashInfer's
`fused_add_rmsnorm` (in place; its buffers are restored before each
sample, outside the timed region). Each column's time over "after"'s, run 1
/ run 2 (`final/table.py`); above 1, "after" is faster:

| hidden | footprint | after (us) | before / after | Triton / after | FlashInfer / after |
|---|---|---|---|---|---|
| 896 | 4 MB | 2.91 / 2.88 | 1.227 / 1.247 | 1.979 / 2.021 | 1.478 / 1.497 |
| 896 | 17 MB | 7.47 / 7.50 | 1.226 / 1.215 | 0.984 / 0.980 | 1.100 / 1.107 |
| 896 | 34 MB | 13.26 / 13.35 | 1.171 / 1.157 | 0.988 / 0.983 | 1.081 / 1.075 |
| 896 | 134 MB | 217.75 / 218.25 | 1.004 / 1.005 | 1.006 / 1.000 | 0.997 / 0.995 |
| 2048 | 4 MB | 2.90 / 2.91 | 1.097 / 1.086 | 2.003 / 2.010 | 1.476 / 1.488 |
| 2048 | 17 MB | 7.59 / 7.60 | 1.054 / 1.058 | 0.993 / 0.995 | 1.163 / 1.167 |
| 2048 | 34 MB | 13.37 / 13.40 | 1.034 / 1.035 | 0.998 / 0.999 | 1.107 / 1.108 |
| 2048 | 134 MB | 217.25 / 217.72 | 1.001 / 0.999 | 1.004 / 1.004 | 1.005 / 1.004 |
| 3072 | 4 MB | 3.04 / 3.04 | 1.141 / 1.148 | 1.908 / 1.928 | 1.408 / 1.418 |
| 3072 | 17 MB | 7.67 / 7.67 | 1.220 / 1.229 | 1.037 / 1.044 | 1.145 / 1.143 |
| 3072 | 34 MB | 13.36 / 13.46 | 1.211 / 1.205 | 1.029 / 1.025 | 1.139 / 1.131 |
| 3072 | 134 MB | 217.80 / 218.27 | 1.005 / 1.002 | 1.003 / 1.000 | 1.002 / 1.000 |
| 4096 | 4 MB | 3.02 / 3.03 | 1.136 / 1.129 | 1.934 / 1.921 | 1.421 / 1.426 |
| 4096 | 17 MB | 7.69 / 7.68 | 1.161 / 1.168 | 1.057 / 1.065 | 1.161 / 1.180 |
| 4096 | 34 MB | 13.45 / 13.57 | 1.164 / 1.157 | 1.043 / 1.043 | 1.145 / 1.146 |
| 4096 | 134 MB | 217.64 / 218.06 | 1.002 / 1.001 | 1.005 / 0.998 | 1.002 / 1.001 |
| 5120 | 4 MB | 2.99 / 2.98 | 1.268 / 1.322 | 1.880 / 1.963 | 1.431 / 1.450 |
| 5120 | 17 MB | 7.62 / 7.61 | 1.280 / 1.277 | 1.205 / 1.191 | 1.172 / 1.181 |
| 5120 | 34 MB | 13.41 / 13.51 | 1.257 / 1.254 | 1.098 / 1.094 | 1.148 / 1.150 |
| 5120 | 134 MB | 218.14 / 217.17 | 1.004 / 1.005 | 1.003 / 1.005 | 0.998 / 1.001 |
| 8192 | 4 MB | 3.15 / 3.03 | 1.184 / 1.231 | 1.857 / 1.931 | 1.359 / 1.429 |
| 8192 | 17 MB | 8.23 / 8.06 | 1.265 / 1.303 | 1.061 / 1.096 | 1.132 / 1.155 |
| 8192 | 34 MB | 13.95 / 13.89 | 1.211 / 1.203 | 1.060 / 1.070 | 1.173 / 1.158 |
| 8192 | 134 MB | 217.72 / 217.95 | 1.005 / 1.000 | 1.005 / 0.997 | 1.004 / 1.002 |
| 12032 | 4 MB | 3.34 / 3.36 | 4.913 / 4.887 | 1.751 / 1.649 | 1.284 / 1.289 |
| 12032 | 17 MB | 8.82 / 8.81 | 3.567 / 3.736 | 1.012 / 1.026 | 1.160 / 1.201 |
| 12032 | 34 MB | 14.67 / 14.62 | 3.263 / 3.291 | 1.087 / 1.092 | 1.173 / 1.186 |
| 12032 | 134 MB | 217.64 / 217.56 | 1.423 / 1.425 | 1.003 / 1.001 | 1.004 / 1.009 |
| 16384 | 4 MB | 3.36 / 3.35 | - | 1.741 / 1.570 | 1.283 / 1.299 |
| 16384 | 17 MB | 9.37 / 9.39 | - | 0.956 / 0.958 | 0.991 / 1.004 |
| 16384 | 34 MB | 15.84 / 16.16 | - | 1.013 / 0.998 | 1.048 / 1.036 |
| 16384 | 134 MB | 218.47 / 218.03 | - | 1.000 / 0.999 | 1.000 / 1.006 |

- **Against the old kernel:** inside the L2 it is 1.03-1.32x faster at
  every size, and 3.3-4.9x at 12032, which ran on one warp. From DRAM it is
  level, except at 12032 (1.42x).
- **Against FlashInfer:** 1.07-1.20x faster at 17 and 34 MB at every
  hidden size but 16384 (0.99-1.05x), and 1.28-1.50x at 4 MB.
- **Against Triton:** 1.01-1.21x faster from 3072 to 12032 inside the
  L2, level at 2048, 1-2% behind at 896, and at 16384 4% behind at 17 MB
  and level at 34 MB. It is 1.57-2.02x faster at 4 MB.
- **From DRAM:** the new kernel, Triton and FlashInfer tie at every size.

Out's bits are the old kernel's only where the warp count did not change
(2048). Elsewhere a thread's share of the sum of squares changed, and with
it the rounding; every column is within 2^-10 of float64.

## Tests and mutants

`python/y_lang/tests/test_rmsnorm_residual.py` covers 10 hidden sizes (1 to
32 warps, a predicated last group, and the loop path at 65536), each on 1,
5 and 67 rows. It checks:
- NewResidual exactly, Out against float64;
- three launches bit for bit alike;
- the warp counts above;
- that the loads come first, unrolled;
- that a launch with half the block size traps (in a child process);
- that hidden 100 is refused.

Against the parent's compiler all four tests fail. `mutants/run.sh` applies
nine mutants of the new emitter code, rebuilding each and running the file;
each fails it (`mutants/summary.txt`; the control row passes):
- no predicate on the last group;
- its loads not zeroed;
- Weight offset by the row;
- NewResidual from X;
- the guard comparing the wrong size;
- no guard;
- the old warp rule;
- the loop path storing no NewResidual;
- no refusal.
