# The exact pipeline on Y's own int8 GEMM, 2026-11-07

`tools/exact_model.py` computes every `ExactLinear` product on the GPU with one
of three kernels, chosen by `Y_EXACT_GEMM`:
- `triton` (the default): `tools/w8a8_gemv.py`;
- `torch`: `torch._int_mm`, i.e. cuBLASLt;
- `y`: Y's own tensor-core GEMM, the one `proofs/Int8GemmSchedule.v` and
  `proofs/Int8GemmExact.v` prove, called through `y_lang.int8_gemm.int8_mm`.

All three sums are exact, and the epilogue is the same torch code, so the mode
should change no bit of the output. This folder records that it does not.

## Correctness: CPU against GPU, 0.5B

`tools/exact_cross_device.py`, Qwen2.5-0.5B-Instruct, 5 cases x 160 tokens,
greedy, workload hash `7aba481ae45aa49e`:

| record | device | GEMM | exact digest | stock digest | against the CPU |
|---|---|---|---|---|---|
| `cpu_zen5_0p5b.json` | Zen 5, eager | CPU `_int_mm` | `30eec169d2f38da4` | `096231ba30bfa89d` | - |
| `gpu_sm89_y_eager.json` | RTX 4070 Ti SUPER, eager | Y | `30eec169d2f38da4` | `1c6a3e4a85967a22` | exact 0/5, stock 5/5 |
| `gpu_sm89_y_compiled.json` | same, `torch.compile` | Y | `30eec169d2f38da4` | `4c8b56c3817664dd` | exact 0/5, stock 5/5 |
| `gpu_sm89_triton_eager.json` | same, eager | Triton | `30eec169d2f38da4` | `1c6a3e4a85967a22` | exact 0/5, stock 5/5 |

The comparisons are in `compare_cpu_vs_*.txt`, and `records.sh` reproduces the
three GPU records from the CPU one. Each record passed the harness's gate
(canary and degeneracy) and reproduced itself on a second run. Stock bf16 is
the control: it changed all 5 cases each time, so the comparison can fail.

The CPU record ran at the lowest priority on 8 threads, with the GPU in use
by a game. The GPU records ran while that game drew frames on the same card.
A bit-identity check needs no idle card; a busy one gives the same answer
later. The Triton record is a regression check on the default path: this
change edits the code around it.

`selftest_y.log` and `selftest_default.log` are `tools/exact_selftest.py` in
each mode. In Y mode `check_y_gemm_agrees` compares:
- Y's kernel with a float64 product over the whole int8 range, at row counts
  either side of the op's padding buckets;
- `ExactLinear` on Y's kernel with the `_int_mm` path, in 12 layer calls:
  plain and two-digit activations, an up and a down projection, eager and
  compiled. The down projection takes the split-K path.

Its control is a changed weight, which must change the output. The other
checks already build their layers in the active mode, so in Y mode the
compiled-GPU-against-CPU check runs Y's kernel too.

## Speed

Idle GPU, two runs each (`split_sweep_run*.log`, `layer_run*.log`), the runs
within about 1% of each other.

**The split factor** (`split_sweep.py`, Y's kernel alone, decode shapes):
`split_factor` picks the fastest factor, or one within 7% of it, at every
shape but one. That shape is the 7B down projection (56 tiles, K = 18944):
the rule leaves it unsplit, where 6 ways measured 13-15% faster. Its 4- and
8-way splits gain only 3-7%, so no rule was taken from it. Splitting gives
1.6-1.8x on the 0.5B down projection (4 ways) and on the 7B key and value
projections (7 ways).

**One ExactLinear, eager** (`layer_bench.py`), Y against Triton:
- faster on most prefill shapes, 1.25-1.76x at M = 256-2048;
- faster on the 0.5B down projection at decode (1.00-1.24x) and the 7B key
  and value projections (1.04-1.21x);
- slower on the 7B gate/up and down layers at decode (0.90-0.91x: 144 us
  against 131 us). The kernel alone is 115 us there. The rest is the eager
  epilogue, four torch passes where Triton fuses its scaling;
- 0.53x on the 0.5B up projection at M = 2048, the same epilogue over a
  40 MB tensor;
- 0.74-0.97x on the small 0.5B layers at decode, where every mode spends
  about 50 us on eager launches around a GEMM of about 5 us.

Against cuBLASLt (`torch._int_mm` and the same epilogue), Y is 0.88-2.5x,
mostly above 0.95x.

**Decode throughput, compiled, is not measured yet.** Under `torch.compile`
the epilogue fuses into one kernel, so the eager penalties above should
shrink. `e2e.sh` measures it, but each process took 13 minutes on this card,
and the run was stopped after one.

## How the op launches the kernel

`int8_mm(a, b)` takes `a [M][K]` and `b [N][K]`, int8, contiguous, 16-byte
aligned, on one CUDA device. It compiles the `@tile(M', N, K)` kernel for the
device's target, where M' is M rounded up to a multiple of 16 up to 128 rows,
or of 128 above. It zero-pads the rows, launches as the PTX header says, and
drops the padding.

Where the output tiles fill at most half the SMs, K is split over
`gridDim.z` into a zeroed C. The factor is the smallest within 10% of the
least `ceil(tiles * z / SMs) * ceil(k_tiles / z)`, with at least 8 K tiles a
split. On this card that is 4 for the 0.5B down projection at decode (14
tiles) and 7 for the 7B key and value projections (8 tiles). Under
`torch.compile` the op is `torch.ops.y_lang.int8_mm`, a custom op, so dynamo
keeps it in the graph.

Host CPU time of one call at 32 x 896 x 896 (the minimum of 5 batches of 300
calls, measured with the card in use, which host time barely depends on):

| path | CPU per call |
|---|---:|
| `int8_mm`, eager | 4.8 us |
| `torch._int_mm` | 3.2 us |
| Triton `w8a8_matmul` | 9.9 us |
| `int8_mm` through the custom-op dispatcher | 9.4 us |
