# `@ZeroDrift`'s f32 term conversion, before and after (2026-10-27)

**What changed.** Each f32 term added to a fixed-point `@ZeroDrift` accumulator was widened to f64, scaled by 2^32, offset by ±0.5 and truncated. That is five f64 instructions per term, on a card that runs f64 at 1/64 of the f32 rate.

The term is now converted with integer arithmetic on its bits. It gives the same value: round to nearest, ties away from zero.

With narrow bounds (below 2^61 raw units, i.e. below 2^29 in Q32.32) two more things changed:
- **The overflow test is gone.** A term now traps at 2^62 instead of 2^63, which makes the test provably dead.
- **The range is one unsigned compare** instead of two.

## Measured

**Setup.** `bench.py` compiles `rowsum.ysu` with the compiler before the change (b54b1b1) and after it, checks the results, and times them.

- **Machine:** RTX 4070 Ti SUPER, PyTorch 2.13.0, Y's `.ysu_hw_profile` for this card.
- **Data:** N(0, 3) values clipped to ±10, 978 columns, seeded.
- **Timing:** minimum over 25 interleaved rounds of 20 launches, after a 2 s clock ramp.
- **Conditions:** no game running before or after; the GPU was at 2% before the run.

**The same bits.**
- The new kernel equals the old one in **462,260 of 462,260** sums (46,226 rows × 10 summation orders).
- It also equals the documented semantics, emulated in NumPy, in all 462,260.
- Neither `@ZeroDrift` kernel depends on the order. The plain f32 kernel does, in all 46,226 rows.

**Time, relative to the plain f32 row sum:**

| | 46,226 rows (bandwidth-bound) | 4,096 rows (16 blocks; latency-bound) |
|---|---|---|
| plain f32 | 1.00 (304 µs) | 1.00 (52 µs) |
| `@ZeroDrift`, before | 3.72 | 7.27 |
| `@ZeroDrift`, after | **1.30** | **3.71** |

**SASS of `rowsum_zd`:** 176 instructions with 31 f64 before; 80 instructions with 3 f64 after. All 3 are in the read-back after the loop.

**What is left.** The loop now runs the conversion's integer instructions and one trap test for each term, plus one range test for each store, and no f64. In the bandwidth-bound case that costs 1.30x. Where few threads hide the latency of that chain it still costs 3.71x, which remains open (`docs/known_issues.md`).

**Later (2026-10-28).** Most of the 3.71x was the trap test, not the conversion: its branch kept ptxas from unrolling the loop, so each term's loads waited for the previous term. With the stop moved to the loop's end it takes 1.64x (and 0.98x where bandwidth limits it): `../zero_drift_stop_point_2026-10-28/`.

## The probe

The compiler's accumulate probe (`probe.log`, three fresh runs) now times a compensated chain for its Kahan row: 6721–6856 ps per accumulate. It used to time a plain `add.f32` chain under that name (about 1440 ps). The exact representations are unchanged:
- Q16.16: 1811–1816 ps;
- Q32.32: 2138–2153 ps;
- I64: 2145–2155 ps.

F64 is 17537–17843 ps.

## Mutation test

`mutants.txt`: 15 mutants of the new code. 14 fail a test.

The survivor clamps the rounding shift at 32 instead of 31. PTX clamps shift amounts the same way, and any shift above 24 gives 0 for an f32 significand either way, so it is an equivalent mutant.

## Files

| File | What it is |
|---|---|
| `bench.py` | the benchmark |
| `bench.log` | its output |
| `results.json` | its output, with every timing |
| `rowsum.ysu` | the kernels |
| `rowsum_old.ptx`, `rowsum_new.ptx` | both compilers' PTX |
| `probe.log` | the probe's three fresh runs |
| `mutants.txt` | the mutation test |
