# `rmsnorm_v4` without the scale broadcast, 2026-10-13

The head-to-head's RMSNorm row, 128 rows of 1024 float32, one row per block
of 256 threads. Three kernels:

- `rmsnorm_v4` as the compiler at `b664885` emits it ("old");
- with this change ("new");
- the head-to-head's Triton kernel.

`rms_old_new.py` produced `run1.json` and `run2.json`.

## Method

- Both Y kernels are compiled from the same source by the two compilers'
  `Y` binaries, and run on four shapes (128x1024, 64x4096, 256x512, 300x128).
  The new kernel's output equals the old one's bit for bit on every shape;
  the script stops otherwise.
- Timing: each kernel's 500 launches captured in a CUDA graph and replayed,
  21 interleaved rounds after 3 warm-up rounds, minimum and median per
  launch. The clocks are brought up first with 30 4096x4096 matmuls. Without
  that, one run measured every kernel 4.6x slower at an idle clock. The SM
  clock read 2775 MHz in every round of both runs.
- RTX 4070 Ti SUPER, driver 615.71.09, PyTorch 2.13.0+cu130, Triton 3.7.1,
  Python 3.14.7. No game was running. The GPU read 38% busy before run 1,
  just after the previous run ended, and 7% before run 2.

## Results

Microseconds per launch, minimum (median):

| | old | new | Triton |
| :--- | ---: | ---: | ---: |
| run 1 | 1.362 (1.386) | 1.219 (1.243) | 1.311 (1.319) |
| run 2 | 1.370 (1.399) | 1.221 (1.265) | 1.311 (1.319) |

The new kernel takes 0.89x the old one's time and 0.93x Triton's. The old
kernel took 1.04x Triton's, as in the head-to-head (1.40 against 1.33 us,
`../head_to_head_2026-10-11/`).

## What changed

The old lowering:

1. Each warp reduced its squares and stored the partial.
2. A barrier.
3. Only warp 0 read the partials and reduced them; the other warps reduced
   zeros.
4. Thread 0 computed the scale and stored it in shared memory.
5. A second barrier.
6. Every thread loaded the scale.

Now every warp reads all the partials, lane `l` reading warp `l`'s, and
reduces them with the same butterfly. Every lane gets the sum thread 0 had,
bit for bit, and computes the scale itself. Steps 4-6 go: one barrier and
one shared store per call instead of two of each.

Inside a `for` or `while` the next row's partials are stored into the same
shared words, so a second barrier follows the loads there. The old lowering
had that hazard unguarded: a warp a row ahead could store its partial over
the scale that the others had not yet read.

## Reproduce

    Y_HW_PROFILE=<profile> python3 rms_old_new.py <old Y> <new Y> <checkout>/python [out.json]

with `Y_LIB_PATH` naming a `liby.so` if the package cannot find one.
