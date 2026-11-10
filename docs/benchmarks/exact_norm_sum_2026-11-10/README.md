# The norm's sum of squares, the same bits in any order (2026-11-10)

`ExactRMSNorm` summed its float64 squares in each device's order: a loop on
the CPU, a tree in Inductor's kernel, another in `norm_kernel`. The squares
are exact, the additions round, so CPU == GPU held only where the sums' last
bits did not move a float32 result across a rounding midpoint. Every record
agreed; nothing guaranteed it.

`y_lang.exact.model.sum_of_squares` now puts each square on a fixed grid,
2^-32 of the row's largest square's unit, adds the grid values as integers
(two 40-bit limbs) and rounds once. Integer addition is exact, so every order
and every device gives the same bits. `norm_kernel` computes the same
integers. It is not `math.fsum`'s correctly rounded sum: the grid drops less
than 2^-66 of the row's largest square at H = 3584, so the two differ only
where the exact sum lies that close to a float64 rounding boundary
(`test_not_fsum_where_the_grid_drops_a_square` builds one such row).

## Same tokens

Qwen2.5-0.5B-Instruct, `tools/exact_cross_device.py`, 5 cases x 160 tokens,
exact arm (`records.sh`):

| record | digest | cases differing |
|---|---|---|
| `cpu_0p5b.json`: CPU (Zen 5), eager, new norm | `30eec169d2f38da4` | 0 of 5 against the old CPU record (`../exact_y_gemm_2026-11-07/cpu_zen5_0p5b.json`) |
| `gpu_default_compiled.json`: RTX 4070 Ti SUPER, compiled, Y GEMM + int8 V, torch norm | `30eec169d2f38da4` | 0 of 5 against `cpu_0p5b.json` |
| `gpu_all_compiled.json`: as above + Triton norm, Y attention, in-place cache | `30eec169d2f38da4` | 0 of 5 against `cpu_0p5b.json` |

Qwen2.5-7B-Instruct, CPU (Zen 5), eager, exact arm, `--low-memory`
(`cpu_zen5_eager_7b.json`, `cpu_7b.log`, 8 minutes): digest
`7700e3e28f1d21d1`, weight fingerprint `fa21999abf0abea9`, 0 of 5 cases
differ from the CPU record before the change
(`../exact_cross_device_7b_2026-11-04/cpu_zen5_eager.json`). It is now the
7B reference: `../exact_y_kernels_7b/run.sh` compares its GPU record with it.

The weight fingerprints are equal in each comparison.

## Cost

32 x 3584 (the 7B decode shape), CUDA graphs, 15 shuffled rounds after a
clock warm-up, the minimum (median); rough: desktop applications on the GPU
and the 7B CPU record running on the host. Two runs each, outputs identical
in both:

| | before | after |
|---|---|---|
| `norm_kernel` (`norm_bench.py`, `norm_bench.run*.txt`) | 6.66 (6.98), 6.75 (6.83) us | 6.64 (6.73), 6.70 (7.19) us |
| default path compiled: residual add + norm + int8 row quantiser (`norm_bench_torch.py`, `norm_bench_torch.run*.txt`) | 29.86 (31.51), 29.80 (30.97) us | 23.16 (24.10), 22.96 (24.47) us |

The kernel is unchanged within the noise; the compiled default path is
about a fifth faster.

The kernel kept its speed by doing its reductions in two passes, not five:
its exponent maximum and the row's kind (normal, subnormal, inf/NaN) in
one, both limbs in the other. Before that the integer sum cost it 1.3 us
(6.01 -> 7.31 us, in a run whose output was not kept). The compiled default path gets faster because its float64
adds ran at 1/64 rate on a GeForce part, and the integer adds do not.

## Tests

`python/y_lang/tests/test_exact_sum_of_squares.py`: the torch path on the
CPU, on CUDA and compiled, and the Triton kernel's own sum (`WRITE_SUM`),
against a Python-integer oracle, on rows as given and permuted; a float64
sum of the same rows changes bits under permutation (the control). 19
mutants fail it; the 4 that pass change nothing.
