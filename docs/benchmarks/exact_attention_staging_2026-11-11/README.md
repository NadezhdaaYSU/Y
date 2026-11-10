# The attention kernel waits on memory once a chunk (2026-11-11)

`tests/exact_decode_attention.ysu` (`Y_EXACT_ATTN=y`) took 284 us a 0.5B
decode step reading the in-place cache (`Y_EXACT_KV_PACK=inplace`) against
235 us after the cat (rough, `docs/known_issues.md`). This is why, and the
change that came of it.

RTX 4070 Ti SUPER (sm_89, 66 SMs, 48 MB L2), driver 615.78.08, torch
2.13.0+cu130, ptxas 13.4. Kernel alone on synthetic operands, no model
loaded: CUDA graphs of 16 launches or more, 15 shuffled rounds after a 2 s
clock warm-up, the minimum (median in the raw files). Desktop applications
were open; the GPU showed no other load.

## The cache's level, not its layout

`attn_l2.py` (`l2_hot_cold.txt`): the same rows read L2-resident (one copy,
read again and again) or DRAM-resident (a ring of copies cycled past 4x the
L2), contiguous (the cat cache) or as views of buffers of 1024 rows a head
(`AppendLayer`'s layout). Microseconds a launch, old kernel:

| shape | T | hot contiguous | hot view | cold contiguous | cold view |
|---|---|---|---|---|---|
| 0.5B (B 32, 2 KV heads x 7, D 64) | 32 | 6.51 | 6.52 | 7.94 | 8.01 |
| | 100 | 16.08 | 16.10 | 20.43 | 20.42 |
| | 256 | 29.99 | 30.11 | 39.79 | 40.43 |
| 7B (B 32, 4 KV heads x 7, D 128) | 32 | 8.81 | 8.78 | 10.64 | 10.68 |
| | 100 | 19.86 | 19.93 | 25.04 | 25.07 |
| | 256 | 36.73 | 36.74 | 47.21 | 47.19 |

The layout costs nothing; DRAM costs 1.22-1.33x. The cat writes the whole
cache every step and leaves it in L2 for the attention that follows; in
place, the rows were last touched a step ago and every other layer's weights
have been through L2 since. Even hot, the kernel was far from bandwidth
(7B, T = 100: 19.9 us for 5.3 MB, 7.9 us at DRAM's 672 GB/s): it waited on
latency. Each warp staged its share of a 32-key chunk of K one row at a
time, a load and then the shared-memory store that needs it, about five
round trips in a row (only D/8 lanes of a warp loading); each key's K scale
was one more load after the barrier; and with 7 warps, warp 0 staged two
groups of V rows one after the other.

## The change

- K: a chunk's 32 rows lie one after another in the cache, and a packed row
  is exactly `D/8 + 1` sixteen-byte units, the shared-memory row the kernel
  already used (its extra unit staggered the banks). So the chunk is copied
  as it lies, every thread of the block taking three units with their loads
  out before their stores (544 units at D = 128 under 3 x 224 threads: one
  pass with 7 warps).
- The K scale is word 0 of that extra unit, now staged with the row, and read
  from shared memory.
- V: two groups of four rows a warp, eight loads out before the stores.

Every value is what it was: the same words reach the same shared-memory
places, and nothing after them changed.

## Two blocks an SM need 128 registers

The first build of this was 0.63-0.87x the old kernel's time at 0.5B and
1.08-1.31x at 7B (`ab_first_try.txt`, four K loads in flight). One change at a
time (`ab_variants.txt`): the K staging alone took 7B to 1.24-1.38x; the
scale read changed nothing (within 2%); the V staging gained up to 5%. The
driver said why (`occ.py`, `occupancy.txt`): the old kernel took 124
registers and 2 blocks an SM; the new one 137-142 and 1; the V change alone,
at 128, still 2. That is 7-warp blocks needing 128 or fewer for two to fit:
an SM's four sub-partitions hold 16384 registers each, and 14 warps put 4 on
some partition, 16384 / (4 x 32) = 128. 0.5B launches 64 blocks on 66 SMs,
one an SM either way; 7B launches 128, which at one an SM run in two waves.

The backend writes `.maxnreg 255` for this kernel: a bucket of its own
virtual-register count, sized for an assumed 256-thread block and one block
an SM. The same PTX with `.maxnreg 128` fits in 122-128 registers, no
spills, two blocks an SM, and is faster at every shape
(`ab_variants_cap128.txt`; the `:128` arguments of `attn_ab.py` rewrite the
directive): three loads in flight matched four and needed fewer registers.

So the language got `@launch_bounds(max_threads[, min_blocks])` (manual
§9.18; `tests/ptx_launch_bounds.rs`): the PTX `.maxntid` and `.minnctapersm`,
from which `ptxas` derives the register limit itself, in place of the
guessed `.maxnreg`. `ptxas` ignores bounds an SM cannot meet with only a
warning, so the compiler refuses them from a per-target table, which the
test holds to `ptxas` at both edges on all 13 targets. The kernel states
`@launch_bounds(256, 2)` and takes 124 registers, no spills, two blocks an
SM.

## Result

`attn_ab.py` (`ab_final.txt`): the kernel at `e7bd856` against this one, in
one process, output bits compared before timing (equal at every shape, with
and without a mask):

```sh
git show e7bd856:tests/exact_decode_attention.ysu > /tmp/before.ysu
PYTHONPATH=python python3 docs/benchmarks/exact_attention_staging_2026-11-11/attn_ab.py \
  /tmp/before.ysu tests/exact_decode_attention.ysu
```

Microseconds a launch, new / old:

| shape | T | hot | hot, mask | cold | cold, mask |
|---|---|---|---|---|---|
| 0.5B | 32 | 6.53 -> 5.72 (0.876) | 0.880 | 8.20 -> 6.96 (0.849) | 0.840 |
| | 100 | 16.03 -> 12.91 (0.806) | 0.812 | 20.22 -> 15.57 (0.770) | 0.777 |
| | 256 | 29.93 -> 23.09 (0.772) | 0.777 | 39.77 -> 28.62 (0.720) | 0.724 |
| | 300 | 57.95 -> 38.09 (0.657) | 0.675 | 68.79 -> 44.05 (0.640) | 0.644 |
| 7B | 32 | 8.76 -> 8.24 (0.940) | 0.937 | 10.66 -> 9.89 (0.928) | 0.927 |
| | 100 | 19.83 -> 17.43 (0.879) | 0.882 | 25.06 -> 21.68 (0.865) | 0.865 |
| | 256 | 36.64 -> 31.23 (0.852) | 0.852 | 47.05 -> 40.16 (0.854) | 0.854 |
| | 300 | 76.19 -> 54.95 (0.721) | 0.739 | 89.12 -> 65.70 (0.737) | 0.734 |

Past 256 keys the kernel stages K in every phase (it keeps no logits in
registers there), so it gains most. Cold still costs 1.16-1.29x hot: the
latency is fewer round trips now, not hidden.

What this is worth in a decode step is not measured here. At 7B, the 5-9%
that `0673bdd` took off this kernel was about 0.3% of a step, so attention
was about 4% of one, and 10-15% of it is about 0.5%.
`../exact_y_kernels_7b/run.sh` times the switches at 7B.

## Same tokens

`records.sh`: Qwen2.5-0.5B-Instruct, `tools/exact_cross_device.py`, 5 cases
x 160 tokens, exact arm, compiled, against the CPU record of the integer
norm sum (`../exact_norm_sum_2026-11-10/cpu_0p5b.json`, digest
`30eec169d2f38da4`); `records.out`:

| record | switches | digest | cases differing |
|---|---|---|---|
| `gpu_all_compiled.json` | Y GEMM, int8 V, Triton norm, Y attention, in-place cache | `30eec169d2f38da4` | 0 of 5 |
| `gpu_attn_cat_compiled.json` | Y GEMM, int8 V, Y attention, cat cache | `30eec169d2f38da4` | 0 of 5 |

The weight fingerprints are equal; the recorder's self-check passed in both.

## Tests

- `python/y_lang/tests/test_exact_decode_attention.py`: bits equal to the
  torch ops, now also at 1, 2, 4 and 8 warps a block (the staging loop's
  passes and V's second group depend on it).
- `tests/ptx_launch_bounds.rs`: the directives, the refusals, the table
  against `ptxas`, a launch past the bound refused on the device, and this
  kernel under its bounds (128 registers or fewer) and without them (more).
- `python/y_lang/tests/test_interpreter.py`: the CPU interpreter refuses a
  block past `.maxntid`, as the driver does.
- 25 mutants (`mutate.py`, `mutants.txt`): 24 fail a test, one of them
  (the third K unit off by one) only once the 1-to-8-warp cases were added;
  the 25th (two units an iteration where three are stored) only stores some
  units twice.
