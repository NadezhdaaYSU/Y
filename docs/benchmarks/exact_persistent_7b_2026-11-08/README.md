# Persistent reductions at 7B: faster alone, slower in the model (2026-11-08)

A negative result, kept with its data. Not adopted.

## The idea

In a 7B decode step the two residual-add + `ExactRMSNorm` + quantise kernels
per layer take 18 us each, 1.0 ms of a 17 ms step
(`../exact_greedy_loop_7b_2026-11-08/profile_greedy.log`). Inductor compiles
an inner reduction persistently (the row kept in registers, one pass) only up
to 1024 elements; the hidden size is 3584, so these are looped, and a looped
reduction recomputes its fused inputs in each pass, here including a float64
divide that this GeForce part runs at 1/64 rate.

`norm_bench.py` compiles that boundary alone at the decode shape (32 x 3584)
and times its GPU kernels with the profiler, after holding the GPU busy for
2 s (from idle clocks one kernel measured anywhere from 40 to 217 us), with
Inductor's caches off. Each variant's outputs were checked bit for bit
against eager:

| variant | GPU us/call |
|---|---|
| looped (Inductor's rule) | 31.5, 31.7 |
| persistent up to 4096 (custom `InductorChoices`) | 21.4, 21.5 |
| cooperative reductions | 31.5, 31.7 |
| cooperative, forced | 38.6, 39.0 |
| float32 norm (not exact; reference only) | 4.4, 4.2 |
| float64 with a reciprocal multiply (not exact; reference only) | 17.3 looped, 10.7 persistent |

So float64 is most of the kernel, and its divide is half of the persistent
one.

## In the model

`y_lang.exact.model.compile_forward` installed the persistent rule as
`inductor_choices_class` (in the compile cache's key) for every inner
reduction of at most 4096 elements, behind `Y_EXACT_PERSISTENT_ROWS`
(`persist_check.py` checked that it made the boundary one `triton_per` kernel
with eager's bits, and that `=0` restored the looped one). 7B exact, Y GEMM,
int8 V, q/k/v merged, `--loop greedy`, one configuration per process
(`run.sh`, `run.out`):

| process | tok/s/seq |
|---|---|
| vLLM FP8, CUDA graphs | 58.19 |
| Y, Inductor's rule | 54.4 |
| Y, persistent to 4096 | 53.2 |
| Y, persistent to 4096 | 53.3 |
| Y, Inductor's rule | 54.4 |
| vLLM FP8, CUDA graphs | 59.17 |

Persistent is 2.2% slower in the model. Its record
(`gpu_sm89_y_vint8_persistent_compiled.json`) has the CPU record's tokens
(digest `7700e3e28f1d21d1`, 0 of 5 cases differ), so it was only a speed
question. The rule reaches every inner reduction up to 4096 elements in
prefill and decode, not only the boundary measured alone; which kernels got
slower was not profiled. The code was removed (`persist_check.py` no longer
runs against the package; `norm_bench.py` carries its own copy of the rule).

The divide remains the target: a correctly rounded float64 quotient from a
multiply and two FMAs, applied to that kernel only.
