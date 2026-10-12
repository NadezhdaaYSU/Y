# FP8 GEMM, before and after 2026-10-12

Y's `@tile(N, N, N)` FP8 GEMM from the compiler at `10001f3` ("before") and
with this change ("after"), against the Triton kernel of
`python/examples/run_head_to_head_comparison.py`. All three take float32 A
and B, quantize them to e4m3 inside the kernel, multiply on the e4m3 tensor
cores and write float32 C. Every column was checked equal to `A @ B` bit for
bit (small integers times 2^-3 are exact in e4m3) before it was timed.
`results.json` holds every sample, the PTX hashes, Triton's chosen
configuration and the toolchain; `bench.py` produced it.

Minimum over 15 interleaved rounds (9 at 8192), microseconds, RTX 4070 Ti
SUPER, driver 615.71.09, CUDA 13.4, PyTorch 2.13.0, Triton 3.7.1:

| N | before | after | Triton | after vs before | after vs Triton |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 16.0 | 7.1 | 9.2 | 2.25x faster | 0.77x |
| 512 | 29.6 | 11.8 | 8.7 | 2.51x faster | 1.36x |
| 768 | 63.2 | 31.1 | 24.8 | 2.03x faster | 1.25x |
| 1024 | 82.5 | 42.2 | 32.6 | 1.95x faster | 1.29x |
| 2048 | 416.8 | 275.7 | 200.0 | 1.51x faster | 1.38x |
| 4096 | 3734.7 | 1991.2 | 1577.0 | 1.88x faster | 1.26x |
| 8192 | 36026.0 | 17014.8 | 16678.7 | 2.12x faster | 1.02x |

**Contended**: a Chromium window used 6-55% of the GPU during the run
(`gpu_busy_before` per size in `results.json`). The columns were
interleaved and the minimum taken, and three earlier exploratory runs gave
the same ratios within about 5%, but no run of this comparison was made on
an idle GPU. Triton's times also depend on the configuration its autotuner
picks in that process: its 1024 time varied from 33 to 53 us between runs.

What changed (see `emit_fp8_gemm_kernel`'s session-6 notes in
`src/ptx_emitter.rs`), each found with Nsight Compute rather than assumed:

- B was staged in global B's N-contiguous layout, so each B fragment
  register was gathered with four `ld.shared.u8`; at 4096^3, 226M of 302M
  shared-load wavefronts were bank-conflict replays. B is now stored
  transposed and both tiles have 80-byte rows: one conflict-free
  `ld.shared.b32` per register.
- The prefetch of the next K-tile loaded, converted and stored before the
  tensor-core work, so warps waited on global memory every iteration. The
  loads now stay in registers across the math.
- At 8192^3 both kernels are DRAM-bound and Y read 17.2 GB to Triton's 9.2:
  in raster order the resident CTAs shared one B column and swept all of A.
  CTAs now launch in groups of 8 M-tiles.

At 4096^3 the after kernel executes fewer instructions than Triton's (164M
to 265M) and reads less DRAM (352 MB to 623 MB), but each instruction waits
about twice as long: the global-load queue, the shared-memory queue, the
tensor pipe and the barrier stall in turn, because each K-step's loads,
fragment loads, mma and stores run as separate bursts. Two attempts at that
were measured and rejected: forcing two CTAs per SM (spills; slower), and
interleaving those phases in the emitted PTX (6-11% slower).

Reproduce:

    python3 bench.py emit <checkout-before> before/
    python3 bench.py emit <checkout-after> after/
    PYTHONPATH=<checkout>/python python3 bench.py time before/ after/ results.json
