# The exact decode step: what is proved, what is assumed

`proofs/ExactDecodeStep.v` holds one machine-checked theorem about one decode
step of Y's exact inference path (`python/y_lang/exact/`) on a Qwen2 model.
This document says what that theorem states and what it rests on, and where
a paper may and may not use the word "proved". The proof inventory and the
gaps that came before it are in
[`docs/exact_decode_proof_inventory.md`](exact_decode_proof_inventory.md).

## The claim in one paragraph

Model one decode step as a function of a **machine**. A machine says how
every reduction in the step is computed: which association and order each
sum uses, in which carrier (int32, int64, float32 or float64), and how each
maximum and the argmax are found, separately at every layer, head, row and
column. A tile shape, a K split, a thread count, an equal-length batch of
any size, a device and a kernel choice each give a machine. **Every two
valid machines produce the same next token and the same new cache rows.**
A machine is valid when each sum it computes is exact up to that site's
bound, which every carrier the code uses meets for every shape the code
accepts. The floating-point operations that are not reductions are taken as
given functions, the same on every device; that is the trust boundary below.

**A ragged batch changes nothing either.** A row left-padded to the length
of a longer prompt carries masked padding keys in its cache; every valid
execution of that row gives the same next token and cache rows as the same
prompt decoded alone (`left_padding_changes_nothing`). This needs three
conditions on the run, stated below, which hold whenever the run's logits
and quantiser scales are finite.

## The theorems

All are in `proofs/ExactDecodeStep.v` unless named otherwise. Every one ends
in `Print Assumptions ... Closed under the global context`: no axiom, no
`Admitted`. `tests/proofs_are_checked.rs` checks this for all 30 files.

| Theorem | What it says |
|---|---|
| `every_valid_execution_takes_the_same_decode_step` | Two valid machines, the same weights, cache and token: the same next token and new cache rows. |
| `every_valid_execution_takes_the_reference_step` | ... and that is the step of the reference machine, whose sums are integer sums. |
| `left_padding_changes_nothing` | A valid execution of a row with P masked padding keys before its real ones gives the reference step of the same row with no padding, given `run_pad_ok`. |
| `attend_padding` | The same for one attention call: masked keys ahead of the real ones change no output, given `attend_pad_ok`. |
| `code_executions_agree` | Any two executions that put each sum in a carrier the code uses there, with any schedules, agree. |
| `the_code's_carriers_meet_every_need` | For every shape the code accepts, each carrier the code uses at a site holds that site's bound. |
| `qwen25_shapes_are_accepted` | Qwen2.5 0.5B, 1.5B, 3B and 7B, at activation widths 127 and 511, for every T from 1 to 32768. |
| `the_premises_are_satisfiable_at_qwen25_0_5b` | The reference machine is valid there; the premises are not vacuous. |
| `ExactReduce.any_schedule_gives_the_sum` | Any association and order of an integer sum in a carrier gives the sum while the absolute values fit the cap. |
| `ExactDecodeWidths.the_derived_activation_width_fits_the_accumulator`, `the_derived_key_width_fits_float32`, `the_digit_width_fits_float32` | The widths the code derives from the shapes satisfy the bounds, for every shape. |
| `ExactDecodeWidths.the_width_functions_return_what_the_python_returns` | The Coq width functions and integer `exp2` return what the Python returns, on 674 generated samples at and around every ceiling. |
| `the_digits_recombine` | The base-2^d digit split of `p.V` recombines to the weight exactly. |

The bounds per site (`need_st`) are these, with K the layer's contraction
length, a its activation width, D the head dimension, kl the K width and T
the number of keys:

| Site | Bound | Carriers the code uses |
|---|---|---|
| linear layer, one digit | K · 127 · 127 | int32; float32 when K · 127² < 2^24; float64 |
| linear layer, two digits (a > 127) | K · 128 · 127, K · 127 · 127, combine K · (a + 128) · 127 | int32, float64 |
| attention score | D · 127 · kl | float32 (`matmul`), int32 (Y's `dp2a` kernel) |
| norm limb sums | H · 2^40 | int64 |
| softmax denominator | T · 2^28 | int64 |
| `p.V` in one sum | T · 2^28 · 127 | int64 (Y's `exact_pv`), float64 |
| `p.V` digit sums | T · (2^d − 1) · 127, d = `digit_width(T)` | float32 |
| `p.V` recombination | T · 2^28 · 127 | float64 |

## The trust boundary

The theorem is parameterised over these functions. It holds whatever they
are, **provided each is the same function on every device and in every
compilation**. That proviso is not proved. Each item is what the code does
and what equality across devices relies on.

| Operation (Section variable) | In the code | Relies on |
|---|---|---|
| Quantiser scale (`fscale`) | `amax(|x|) / divisor(lv)`, 1 where 0 | One IEEE float32 divide, correctly rounded. The divisor is a tensor, so torch does not use a reciprocal multiply, and `eager_numerics.division_rounding` keeps Inductor from using one. The max is taken by integer key: on non-negative finite floats the IEEE order is the order of the bit patterns. |
| Quantiser rounding (`fround`) | `round(x / s)` | An IEEE divide, then round half to even. |
| Linear epilogue (`fepi`) | `float32(acc) * sx * sw + bias` | int32 → float32 conversion with round to nearest, then two multiplies and an add in that order, with no fused multiply-add (separate eager kernels; `emulate_precision_casts` turns off Triton's FP fusion in Inductor). |
| Norm finish (`fnorm`) | `float64(big)·2^40 + float64(small)`, `· 2^(e−332)`, `/ H`, `+ eps`, `sqrt`, `x / r`, to float32, `· weight` | One correctly rounded add of two exact operands, an exact power-of-two scale, IEEE divide and square root (not `rsqrt`), round to float32. |
| RoPE (`frope`) | table lookup, `q·cos + rotate_half(q)·sin` | Host-built tables, each entry correctly rounded (mpmath near midpoints); on the device, multiplies and an add with no FMA. |
| Logit and exp argument (`fqa`, `flogit`, `targ`) | `s · (sq · scale · log2 e · 65536) · sk`, `round(m − logit)` | float64 multiplies in that order; a subtract and a round. `targ_ival`: it reads the max only through its IEEE value, so a max of −0 and one of +0 agree (both round to 0). |
| V-scale fold (`fw`, `fold_round`, `fold_scale`) | `p · sv`, `round(w · (2^28 / safe))`, `safe / 2^28` | float64 operations; `2^28 / safe` is a reciprocal times a power of two, which equals the divide. `fold_ival`, as for `targ`. |
| Attention output (`fout`) | `(acc · wscale) / max(l, 1)`, to float32 | float64 multiply and divide. |
| Residual, SiLU product (`fadd`, `fsilu_mul`) | float32 add; `ExactSiLU(g) · u` | ExactSiLU is integer interpolation in a host table followed by one float32 multiply. |
| Embedding (`embed`) | a lookup | Nothing beyond reading memory. |

Hardware and compiler assumptions behind every row:

- IEEE 754 binary32 and binary64 arithmetic with round to nearest even and
  **no flush to zero of subnormals** in these operations. The Triton
  kernels use `tl.div_rn` because libdevice's `div_rn` flushes subnormals
  (`w8a8_gemv.py`); torch's own kernels do not flush.
- No contraction of a multiply into a following add (FMA) in the
  elementwise operations, as above.
- For kernels that are not Y's (cuBLASLt's int8 GEMM, oneDNN's int8 GEMM on
  a CPU, Triton's `tl.dot`, MKL or cuBLAS float32 matmuls for the scores,
  `torch.sum` on int64): each output element is **some association of that
  element's own terms in the stated carrier**. The CPU's int8 GEMM is
  measured against float64 at import (`cpu_int_mm_is_exact`) because
  `vpmaddubsw` saturates on CPUs without VNNI. TF32 would not break the
  scores: their operands are integers below 2^11.
- The argmax returns the first maximal index (torch documents this).

## What the theorem does not cover

- **Rows holding an infinity or a NaN.** The norm then takes a float path,
  and the padding conditions below can fail.
- **Ragged batches beyond `attend_pad_ok`.** The padding theorem needs, at
  every attention call of the unpadded run: (1) the mask constant
  `finfo(float64).min` is at most the largest real logit; (2) a padding
  key's folded weight `0 · sv` is at most the largest real one; (3) the fold
  rounds it to 0. All three hold when the logits and scales are finite: the
  mask constant is the most negative finite float64, a real key at the max
  has weight `2^28 · sv > 0`, and `round(0 · x) = 0` for finite `x`. They are
  premises of the theorem, not proved from the code.
- **The prefill**, except as one decode step per prompt token (in a ragged
  prefill, each row's keys after its own position are masked the same way).
- **Triton's two-digit kernel** (`w8a8_matmul_hilo`), which accumulates
  `128 · hi` and `lo` in one interleaved int32 sum. The separate form
  (`torch._int_mm`, Y's GEMM, float64) is modelled. The per-term bound
  that kernel needs is proved (`the_split_terms_fit_the_accumulator`).
- **Y's decode-attention kernel** (`Y_EXACT_ATTN=y`) is a machine of the
  model, but the executable check below does not run it (it needs the packed
  cache).
- **That the model is the code.** The model is a transcription. What ties
  it to the code is the executable check and the generated constants, next.

## How the model is tied to the code

1. **Constants.** `proofs/ExactDecodeConstants.v` is generated from
   `python/y_lang/exact/` by `tools/exact_decode_constants.py`;
   `python/tests/test_exact_decode_constants.py` fails on any byte of
   difference. The width proofs check their definitions against the samples
   in that file, so a width function changed in the Python breaks
   `ExactDecodeWidths.v`.
2. **Executable check.** `tools/exact_decode_check.py` evaluates the model
   with the reference machine (exact integer sums, the transcribed widths,
   `exp2`, norm limbs and digit split; the trusted operations are the code's
   torch operations on the CPU) and compares it with the real path, bit for
   bit, at each chosen layer's output and new K and V rows, the final norm,
   the logits and the token. A bound-stress stage then feeds each linear
   layer the input that saturates one column (every activation at ±a, signs
   matching the row with the largest Σ|w|), because real activations keep
   most sums far below their bounds.

Results, Qwen2.5-0.5B-Instruct at revision `7ae5576`, prompt "The capital of
France is", all 24 layers, 4 decode steps:

| Where | Width | Compared | Bit-identical |
|---|---|---|---|
| AMD Ryzen 9 9950X (CPU path), with the ragged stage | 127 | 575 | 575 |
| AMD Ryzen 9 9950X (CPU path), with the ragged stage | 511 | 575 | 575 |
| Tesla T4, driver 580.178.04, torch 2.13.0+cu126 (GPU path vs the model on the Kaggle CPU), with the ragged stage | 127 | 817 | 817 |
| Tesla T4, same | 511 | 817 | 817 |
| Kaggle Intel Xeon @ 2.00GHz (CPU path), Python 3.13, with the ragged stage | 127 | 575 | 575 |
| Tesla T4, `Y_EXACT_V_INT8=1` (`p.V` on Y's `exact_pv`, one int64 sum), with the ragged stage | 127 | 817 | 817 |

On the T4 the 817 add 242 checks that the int8 weights and scales built on
the GPU equal those built on the CPU. The first `V_INT8` attempt did not
run: the job had no Z3, so Y refused to compile `exact_pv` ("could not verify
invariant ... the SMT solver could not be run"), the compiler failing closed.
With `z3-solver` installed, a second run of all four configurations gave the
same counts as the first.

The ragged stage (`--ragged`) decodes the batch ["The capital of France
is", a 14-token prompt], so the first row carries 9 padding keys. It
compares that row with the prompt decoded alone: the logits and token at
every step, each layer's decode output, and the K and V cache rows, 154
tensors over 24 layers and 4 steps, all bit-identical and finite, on every
row of the table. With `Y_EXACT_KV_PACK=1` the cache holds packed int8 rows;
the check reads them back with the code's unpacker and compares each new
row's packed integers and scales with its own quantisation (CPU: 70 of 70).

Each run decodes " Paris. It is the". Of the 421 non-ragged checks, 121 are the stress stage
(five linear layers in each of 24 layers, and the LM head). Its peak column
sums reach 1.17 × 2^24 at width 127 (`down_proj`) and 4.71 × 2^24 at 511.
The raw reports are in
[`docs/benchmarks/exact_decode_check_2026-11-15/`](benchmarks/exact_decode_check_2026-11-15/).
`python/tests/test_exact_decode_check.py` runs a short form of the check in
the Python suite, skipping it where the pinned model is not cached.

## Negative controls

`tools/exact_decode_mutations.py` applies each mutation to a temporary copy
(never the working tree), runs an unmutated control first, and runs the gate
the mutation must break: the Coq chain after regenerating the constants
(`coq`), or the executable check on the CPU (`exec`).

| Mutation | Plants | Gate | Result |
|---|---|---|---|
| control | nothing | coq, exec | both hold |
| float32 denominator | a float sum in a reduction (`p.sum` in float32) | exec | caught: 2 mismatches |
| float32 denominator in the proof | the same, allowed by the proof's carrier table | coq | caught: `ExactDecodeStep.v` fails |
| float32 linear accumulation | the int8 GEMM run as a float32 matmul | exec | caught: 3 mismatches, all in the stress stage |
| wrong index | head h reads the K scales of head h mod n_kv, not h // rep | exec | caught: 6 mismatches, the tokens change |
| digit ceiling widened | `DIGIT_MAX` 16383 → 32767 | coq | caught: `ExactDecodeWidths.v` fails; exec holds |
| score budget widened | float32 budget 2^26 with K at 4095 levels | coq, exec | caught by both (exec by the code's own assert) |
| accumulator margin dropped | the `− 128` in `act_levels_for` | coq | caught: the samples no longer match |
| masked key let back in | a masked key's weight no longer forced to 0 (`tq` and `p` unmasked) | exec | caught: 14 mismatches, the tokens change (the same mask also makes the prefill causal) |
| masked key let back in, in the proof | the model's masked weight no longer 0 | coq | caught: `ExactDecodeStep.v` fails |

Two of these are lessons rather than confirmations:

- **Before the stress stage, the float32 linear accumulation passed every
  check on real activations.** Real sums stay below 2^24 except in the
  worst case, so a check on real inputs alone could not see it.
- **A wrong index cannot break the agreement theorem.** Both machines run
  the same model, so the theorem is about reductions, not dataflow. Only the
  executable check pins the model's dataflow to the code. The same holds
  for the widened digit ceiling in the other direction: no input at these
  shapes reaches it, so only the proof sees it.

## Where a paper may say "proved"

May say, with the theorem named:

- "We prove in Coq that, in a model of one Qwen2 decode step, every two
  executions whose reductions are exact up to the derived bounds produce
  the same next token and cache rows, whatever association, order, tiling,
  K split, thread count, batch size or device they use."
- "We prove that the carriers the implementation uses meet those bounds for
  every shape it accepts, including Qwen2.5 0.5B to 7B at activation widths
  127 and 511 and contexts up to 32768 tokens."
- "We prove that the width functions return the bounds the reductions need
  for every input, and check them against the implementation on 674
  generated samples."
- "We prove that a row of a ragged batch decodes to the same token and cache
  rows as the prompt alone, under three conditions on the run that hold
  whenever its logits and scales are finite."

Must qualify, or say "checked" or "measured":

- That the implementation computes the model: **checked**, bit for bit, on
  real activations and bound-stressing inputs, on the CPUs and GPU named
  above. It is not proved.
- Cross-device bit identity: **proved conditional** on the trust boundary
  (same IEEE elementwise operations, no FTZ, no FMA contraction, kernels
  that compute each output from its own terms), and **measured** on the
  hardware listed.
- Ragged batches: **proved conditional** on `attend_pad_ok` at every
  attention call, and **checked** bit for bit on a real ragged batch.
- The prefill as one pass, rows with inf or NaN, Triton's interleaved
  two-digit kernel: **not proved**.

Must not say: that the implementation is verified; that Y's GPU kernels are
proved in general (only those with their own proofs, listed in the
inventory); that any other system lacks such a proof (not checked); that Y
is the first or only system with this property.
