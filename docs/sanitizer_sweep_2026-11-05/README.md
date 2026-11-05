# compute-sanitizer over the GPU test suite (2026-11-04/05)

Every test binary that launches a Y kernel on the GPU (`binaries.txt`: the
`tests/*.rs` files that call `CudaContext::new`, `cuLaunchKernel` or
`KernelModule`, less three that time kernels and only slow down) was run
under NVIDIA compute-sanitizer, one binary at a time, at commit `10d9d44`.

- compute-sanitizer 2026.3.0.0, CUDA 13.4, driver 615.78.08
- NVIDIA GeForce RTX 4070 Ti SUPER (sm_89), the desktop GPU
- `--print-limit` uncapped for the counts below, `--show-backtrace no`

## Result

| tool | binaries that ran kernels | finding |
| :--- | :--- | :--- |
| memcheck | 38 (the four `zk_gpu_*` with `--features zk`) | **no invalid or misaligned access** in any kernel any test launches |
| racecheck | 18 with shared-memory kernels: GEMMs (F16 split K, int8, scaled int8, MMA layout), attention, RMSNorm, swizzle, collectives, exact PV, the grid-sum drift reduction, NTT, MSM, Groth16, field | **no shared-memory hazard** |
| synccheck | 27 of the 40 before the session ended | no barrier error in those 27; the other 13 were not run |
| initcheck | not run | - |

`adaptive_jit_gpu` and `adaptive_jit_ffi_gpu` ran nothing (every test is
`#[ignore]`d), and the `zk_gpu_*` binaries run nothing without `--features
zk`; they were rerun with it (`memcheck_uncapped_and_zk.txt`).

## The errors memcheck did report are Y's own traps

Eight binaries reported errors - 37,328 in `zero_drift_scatter` alone - and
every one is a `@bounds`, `@ZeroDrift` or launch-contract check stopping a
kernel on purpose, which is what those tests test: each trapping thread is a
`Trace/breakpoint trap` record, and the context it poisons adds `Program hit
CUDA_ERROR_LAUNCH_FAILED` and `CUDA API Error: Sticky error detected` records
for the calls that follow. Counted by type, uncapped, they add up to the
sanitizer's own ERROR SUMMARY in every binary
(`memcheck_recount_and_racecheck.txt`):

| binary | traps | launch failures | sticky errors | = ERROR SUMMARY |
| :--- | ---: | ---: | ---: | ---: |
| attention_kfix_is_checked_at_launch | 3072 | 66 | 60 | 3198 |
| ptx_rmsnorm_v4 | 1664 | 15 | 12 | 1691 |
| zero_drift_scatter | 35820 | 808 | 700 | 37328 |
| zero_drift_grid_sum | 45 | 192 | 160 | 397 |
| zero_drift_f32_conversion | 13 | 52 | 39 | 104 |
| zero_drift_stop_point | 10 | 46 | 36 | 92 |
| zero_drift_runtime_checks | 5 | 20 | 15 | 40 |
| bounds_runtime_check | 3 | 12 | 9 | 24 |

(The number of trapping threads varies a little from run to run: the first
pass counted 37,104 in `zero_drift_scatter`.)

## Two counting traps

- **`--print-limit` hides records.** The first pass printed 30 per binary;
  that the printed ones were traps said nothing about the 37,074 it did not
  print. The counts above are from uncapped reruns.
- **A record on the harness's line.** Every count came out exactly one short
  of the summary: the test harness prints `test name ... ` with no newline, so
  the first sanitizer record of that test begins on the same line and a
  `^=========` match misses it. Matching anywhere in the line closes it. A
  count that does not reconcile is not a clean result.

`tools/ydb/ycheck.py` (`ydb --gpu LAUNCH --check TOOL`) applies both lessons
to one launch: it reconciles with the summary and prints the sanitizer's whole
output when it cannot.

## What this does not show

- Only the kernels, shapes and inputs the tests launch are checked; a
  sanitizer checks executions, not all inputs.
- `#[ignore]`d tests (benchmarks, long sweeps) and the three timing tests were
  not run.
- synccheck covered 27 of 40 binaries and initcheck none.

## Rerun

```bash
cargo test --release --no-run              # and --features zk for zk_gpu_*
docs/sanitizer_sweep_2026-11-05/sweep.sh memcheck target/sanitizer_memcheck
```

`memcheck_summary.txt` is the first pass (30 records printed per binary),
`memcheck_uncapped_and_zk.txt` and `memcheck_recount_and_racecheck.txt` the
uncapped recount and the racecheck pass, `synccheck_partial.txt` the
synccheck pass as far as it went. Per-binary logs were not kept: the
trapping binaries' logs run to megabytes and hold nothing but the records
counted above.
