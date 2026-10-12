# Launching from Python, 2026-10-12

What one `TorchKernel.launch` costs, before and after this change, beside
Triton's launch and PyTorch's own add, for a 100,000-element f32 vector add
(each column checked equal to `a + b` first). `launch_bench.py` produced the
four `*.json` files; each holds every sample, the GPU, PyTorch, Triton and
Python versions, and the GPU's utilisation when it started. Before and after
ran in separate processes, alternated twice.

Median microseconds per call over 15 interleaved rounds, RTX 4070 Ti SUPER,
PyTorch 2.13.0, Triton 3.7.1, Python 3.14.7:

| | before (runs 1, 2) | after (runs 1, 2) | Triton | PyTorch |
| :--- | ---: | ---: | ---: | ---: |
| CPU time per call | 6.67, 6.58 | 2.94, 2.93 | 5.67-5.73 | 1.97-2.04 |
| per call, CUDA events | 6.70, 6.57 | 2.96, 2.95 | 5.71-5.79 | 2.03-2.58 |

*CPU time* is the calls timed on the host clock without waiting for the GPU:
the launch path's own cost. *CUDA events* times back-to-back calls as the
head-to-head's *launch* mode does (`../head_to_head_2026-10-11/`, where Y
measured 6.62 us); it cannot go below the GPU's time per kernel, and a game
used 20-28% of the GPU during these runs, which shows in that row's spread,
not in the CPU row.

## Where the time went

A cProfile of 20,000 launches, before:

- `torch.cuda.current_stream()` built a `Stream` object (and resolved the
  device) on every call: 40% of the launch. The current stream's raw handle
  is now read with `torch._C._cuda_getCurrentRawStream(_cuda_getDevice())`,
  one C call each, resolved once; Triton's launcher reads it the same way.
- `torch.cuda.is_available()` ran twice per launch and read environment
  variables; the module is already loaded through the driver by then, so it
  is not asked.
- The interpreter switch (`Y_INTERPRETER`/`Y_DEBUG`) is still read on every
  launch, so setting it between launches still takes effect; it now looks the
  names up in the dict `os.environ` keeps (about 0.1 us) and falls back to the
  public lookup where that dict is absent or `os.environ` was replaced.
- Imports and per-argument `hasattr`/`callable` calls moved out of the path.

What remains is mostly `cuLaunchKernel` itself through `ctypes`, 1.95 us.

## Fixed on the way

- The launch passed no dynamic shared memory, so a kernel staging through
  `.extern .shared` died with an illegal address: Y's F16 tile GEMMs could not
  run from Python at all. The launch now passes what the kernel's PTX states
  ("Dynamic shared memory required: N bytes"), and above 48 KB raises the
  function's limit with `cuFuncSetAttribute` first.
- A kernel name the module lacked ran an entry named `main` instead, when the
  module had one. It is refused.

`python/y_lang/tests/test_launch.py` checks both, that a launch runs on the
caller's current stream and on an explicit one (on a non-blocking stream,
where a launch on the wrong one races), and the interpreter switch; each of
seven mutants of the launch path fails it.

## Reproduce

    PYTHONPATH=<checkout>/python python3 launch_bench.py <label> <results.json>
