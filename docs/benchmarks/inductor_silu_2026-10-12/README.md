# `silu` and `sigmoid` in `y_inductor`, 2026-10-12

SwiGLU's activation, `silu(g) * u`, through four paths:

- `y_inductor` before this change (from `fc3221d`): `silu` ran eagerly and
  the multiply was a Y kernel;
- `y_inductor` after it: one Y kernel;
- `torch.compile`, whose Inductor backend generates a Triton kernel;
- eager PyTorch.

`bench_swiglu.py` produced `run1.json` and `run2.json`; each holds every
sample, the GPU's state before and after, and the versions.

## Method

- Before each column is timed, its output is compared with eager's, bit for
  bit. Both Y columns equal eager in every element. `torch.compile`'s output
  differs in about a quarter of them: 261,713 of 2^20, 1,044,124 of 2^22 and
  4,171,540 of 2^24, by at most 8.6e-8 relative to the largest output.
- Timing follows the head-to-head (`../head_to_head_2026-10-11/`): CUDA
  events around back-to-back calls from Python (*launch*) and around a
  CUDA-graph replay of the same calls (*graph*), 15 interleaved rounds after 3
  warm-up rounds, minimum per call.
- RTX 4070 Ti SUPER, driver 615.71.09, PyTorch 2.13.0+cu130, Triton 3.7.1,
  Python 3.14.7. **A game was running.** Mean GPU utilisation was 32% before
  run 1 and 6% before run 2 (70% and 19% after), so these are contended
  figures. Each column suffered the same contention, interleaved with the
  others.

## Results

Minimum microseconds per call, launch / graph, run 1 and run 2:

| n | Y before | Y after | torch.compile | eager |
| ---: | ---: | ---: | ---: | ---: |
| 2^20 | 9.5 / 8.4; 9.5 / 8.4 | 6.0 / 4.5; 6.4 / 4.5 | 15.9 / 5.0; 16.3 / 5.0 | 10.2 / 9.1; 10.2 / 9.1 |
| 2^22 | 90.8 / 90.6; 92.3 / 91.0 | 80.4 / 80.4; 80.5 / 80.4 | 80.4 / 79.9; 79.1 / 77.2 | 89.9 / 89.8; 90.3 / 89.2 |
| 2^24 | 542.3 / 541.5; 542.7 / 540.7 | 322.4 / 322.8; 323.2 / 323.0 | 323.6 / 323.0; 329.1 / 323.2 | 541.1 / 541.3; 541.1 / 541.7 |

- **Against before:** one kernel instead of two is 1.13x faster at 2^22 and
  1.68x at 2^24 (graph). At 2^22 the eager `silu`'s 16 MB output stays in the
  48 MB L2 for the multiply, so the unfused path costs less there. At 2^20 it
  is 1.87x faster.
- **Against `torch.compile`:** at 2^22 and 2^24, where both kernels stream
  from DRAM at about 625 GB/s, Y takes 0.98-1.04x its time across the two
  runs and both regimes. Each is ahead in some runs. At 2^20, where the
  tensors sit in L2, Y takes 0.9x Triton's time replayed from a graph (4.5
  against 5.0 us). Launched from Python it takes 0.4x (6.0-6.4 against
  15.9-16.3 us), the difference between the two launch paths.

## The load order

`loads_after_silu.json` is the same benchmark, run once (utilisation 15%),
before the kernel loaded its inputs first. Its `before_commit` field reads
`HEAD`, which was `fc3221d`. `fuse_by_partitions` puts an input's
placeholder where it is first used, so `u` was loaded after the `silu`. ptxas does not move a load above the IEEE division's slow-path call:
that held for Y's `ld.global.ca` and also when the load was rewritten as a
plain `ld.global`. So each thread waited for two memory latencies in a row.
That cost nothing measurable at 2^22 and 2^24, where enough warps hide the
latency, but at 2^20 the fused kernel took 11.7 us from a graph, against
4.5 us with the loads first and 8.4 us for the two unfused kernels.

## Reproduce

    python3 bench_swiglu.py <checkout> <before-commit> <out.json>

with `Y_LIB_PATH` naming the checkout's `liby.so` if the package cannot
find it.
