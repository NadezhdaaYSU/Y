# Launching from Python, 2026-10-17

This record measures what one `TorchKernel.launch` costs through the native
launch path (`python/y_lang/_launch.c`) and through ctypes, beside Triton's
launch and PyTorch's own add. The kernel is a 100,000-element f32 vector add,
and every column was checked equal to `a + b` first.

`launch_bench.py` produced `run1.json` and `run2.json`, each in its own
process. Each file holds every sample, the SM clock read after each one, the
GPU, CPU, PyTorch, Triton and Python versions, and the module the native
column loaded. Both Y columns time the same kernel, loaded once with
`Y_LAUNCHER=native` and once with `Y_LAUNCHER=ctypes`. Each run started when
the GPU's utilisation, averaged over 10 samples, was at most 5% (1.5% and
1.8%) and no game was running.

Median microseconds per call over 15 interleaved rounds, run 1; run 2. RTX
4070 Ti SUPER at 2490 and 2520 MHz, AMD Ryzen 9 9950X, PyTorch 2.13.0,
Triton 3.7.1, Python 3.14.7:

| | Y, native | Y, ctypes | Triton | PyTorch |
| :--- | ---: | ---: | ---: | ---: |
| CPU time per call | 1.72; 1.72 | 2.93; 2.93 | 5.66; 5.58 | 2.03; 2.02 |
| per call, CUDA events | 1.73; 1.72 | 2.94; 2.94 | 5.68; 5.64 | 2.06; 2.04 |

*CPU time* is the calls timed on the host clock without waiting for the
GPU: the launch path's own cost. *CUDA events* times back-to-back calls, as
the head-to-head's *launch* mode does. It cannot go below the GPU's time per
kernel.

The ctypes column reproduces the 2.9 us of `../launch_2026-10-12/`. Two
earlier runs (`unwarmed/`) gave the same CPU times: 1.70 and 1.63 us for the
native path, and 2.02 and 1.96 for PyTorch. Those runs did not bring the clocks
up first or log them. Their CUDA-events medians rose to 6.4-7.2 us in every
column of run 2 and in the ctypes column of run 1, as a kernel run at a
lowered clock would. The script now runs 30 4096^2 matmuls before its rounds
and reads the SM clock after every sample.

## Where the time went

Before this change, a launch spent about 0.9 us in ctypes converting
`cuLaunchKernel`'s eleven arguments. An eleven-argument ctypes call to an
empty C function takes 0.87 us, against 0.13 for one argument
(`ctypes_arity.py`, `noop.c`, `ctypes_arity.txt`). `../launch_2026-10-12/`
measured the ctypes call, driver included, at 1.95 of the 2.9 us. The other
1 us was Python around the call. It filled the four parameter slots one
ctypes attribute at a time, after a `getattr` and a `callable` each, read
the current stream, and checked the interpreter switch.

The driver's own `cuLaunchKernel` on a non-blocking stream, called from C,
takes 1.01-1.04 us (`floor.c` on `vec_add.ptx`, `floor.txt`). That is 59-60%
of the native path's 1.72 us. A cProfile of 20,000 native launches
(`profile.txt`) puts 74% of their time inside the one C call (the driver, and
the three `data_ptr` calls it makes). Reading the current stream takes 10%.

## Reproduce

    PYTHONPATH=<checkout>/python python3 launch_bench.py <label> <results.json>
    PYTHONPATH=<checkout>/python python3 launch_bench.py profile
    gcc -O2 -shared -fPIC noop.c -o libnoop.so && python3 ctypes_arity.py
    gcc -O2 -I<cuda>/include floor.c -o floor -lcuda && ./floor vec_add.ptx
