# Warp shuffles and reductions in their value's type (2026-10-26)

## The bug

`shfl_sync_bfly`, `shfl_sync_bfly_b32`, `warp_reduce_sum` and `warp_reduce_max` wrote an f32
register whatever they were given (`probe/README.txt`):

| Value | Before | Effect |
|---|---|---|
| `I32`, `U32` shuffle | `shfl` into `%f`, then `cvt.rzi.s32.f32` | assembles; `shfl_sync_bfly(t + 1000, 1)` stored 0 on every lane on the GPU |
| `I32`, `U32` sum and max | `add.f32` / `max.f32` on the integers' bits, then `cvt.rzi` | assembles; wrong results |
| `F64`, `I64`, `U64` | a 64-bit register in a `.b32` shuffle | ptxas rejects the PTX |

## The fix (`src/ptx_emitter.rs`)

- **`emit_bfly_typed`** shuffles into a register of the value's own type. A 64-bit value moves as its two halves: `mov.b64 {lo, hi}`, two `shfl.sync.bfly.b32`, `mov.b64` back. `emit_warp_butterfly_shuffle`, which the built-ins' f32 reductions use, goes through it.
- **`emit_warp_reduce`** uses `add` or `max` of the type (`.f32`, `.f64`, `.s32`, `.u32`, `.s64`, `.u64`). An integer sum wraps, as `+` does.
- **The dispatch** takes any integer lane offset as the u32 lane index. It refuses a float offset, which would have reached `shfl` as its bits, and refuses `shfl_sync_bfly_b32` of a 64-bit value.

The F32 path emits the same text: all 60 committed `tests/*.ptx` regenerate identically
under this compiler and 7d6037f's (`check/regen.txt`). No `tests/*.ptx` changes, so the
validator's gates read unchanged PTX.

## Tests (`tests/ptx_typed_collectives.rs`)

- **`collectives_use_the_values_own_type`:** each of the six types compiles to its own
  instructions, with no f32 instruction for the non-F32 types and two halves for the
  64-bit ones, and assembles for sm_89.
- **`a_shuffle_refuses_what_it_cannot_move_and_takes_any_integer_offset`:**
  - `shfl_sync_bfly_b32` of F64, I64 and U64 is refused; I32 compiles as the control.
  - A float offset is refused under both spellings.
  - An I64 offset compiles and assembles.
- **`every_type_shuffles_and_reduces_exactly_on_the_gpu`:**
  - two warps of every type through the xor-1 and xor-5 shuffles, the sum and the max;
  - the values are ones a float or 32-bit reading gets wrong: negatives, above 2^24, the top bit set, both 64-bit halves in use;
  - each result is compared bit for bit against a CPU emulation of the same butterfly, on poisoned outputs.

All three tests fail on 7d6037f's compiler and pass on this one.

## Mutants (`mutants/`)

13 mutants of the new code, run against this file, `ptx_intrinsics_assemble`,
`ptx_divergent_collectives` and `ptx_rmsnorm_v4` with the GPU tests included: **13 killed**
(`mutants.txt`; the tests each failed are in `killed_by.txt`). The restored source passes
and hashes as before (`sources_before.sha256`).

## Why it came up

A WTCS (connectivity score) kernel needs warp-wide integer prefix counts and float64
extremes. The first probe of `shfl_sync_bfly` on an `I32` returned 0.
