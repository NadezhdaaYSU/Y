# Exact decode: what the Coq proofs cover, and where the gaps are

Inventory of `proofs/*.v` against one decode step of the exact inference path
in `python/y_lang/exact/`. Written 2026-11-14 at commit `a7043ad`, before any
proof of a whole decode step was started. It lists what each file proves,
which stage of a decode step that covers, and what nothing covers. The files
added since are at the end, with what is still open.

The theorem statements quoted here are copied from the proof files.
`python/tests/test_proof_inventory.py` fails if a quoted statement is no longer
in its file, or if a proof file is missing from this inventory.

## Summary

- 26 proof files, 360 theorems and corollaries, no `Admitted`, every
  `Print Assumptions` closed (`tests/proofs_are_checked.rs`).
- 5 files are about kernels the decode path can run: `Int8GemmExact.v`,
  `Int8GemmSchedule.v`, `ExactPvExact.v`, `ExactPvLaunchContract.v`,
  `NormGridSum.v`. The first four apply only when an environment variable
  selects Y's own kernel (`Y_EXACT_GEMM=y`, `Y_EXACT_V_INT8=1`). In the default
  configuration only `NormGridSum.v` describes code that runs.
- 12 files are about Y's CPU `@ZeroDrift` GEMM, which `python/y_lang/exact/`
  never calls.
- 2 files are about the PTX attention kernels in `src/exact_attention.rs`,
  which the Python decode path does not launch either.
- 1 file (`SoftmaxErrorBound.v`) is an accuracy bound, not a determinism claim.
- 2 files are shared libraries; 4 are about other subsystems.
- No theorem mentions a layer, a decode step or a model. Every result is about
  one kernel or one reduction.

## One decode step, stage by stage

The path is `Qwen2` with `ExactLinear`, `ExactRMSNorm`, `ExactSiLU`,
`ExactRotary` and `exact_attention` installed (`model.convert`,
`attention.build_exact_model`). A reduction is an operation that combines more
than two values, so its result could depend on the order. Everything else is
elementwise.

| # | Stage | Code | Reductions | Carried in | Proof today |
|---|---|---|---|---|---|
| 1 | RMSNorm | `model.sum_of_squares`, `norm_kernel` | max of the exponents; sum of high limbs; sum of low limbs | int64 | `NormGridSum.v`: limbs, order, bounds. Not the max. |
| 2 | Linear layers: merged q/k/v, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `lm_head` | `model.ExactLinear` | max of `abs(x)` over the row; `sum_k x8[k] * w8[n][k]`; with wide activations two such sums and `128 * hi + lo` | int32 (Triton `tl.dot`, `torch._int_mm`, Y's `mma`), or float32 below 2^24, or float64 below 2^53 | `Int8GemmExact.v` and `Int8GemmSchedule.v`, for Y's GPU kernel only. Nothing for the other four carriers, the digit split, or the max. |
| 3 | RoPE | `model.ExactRotary`, transformers' `apply_rotary_pos_emb` | none | float32 | none needed for order; the float operations are trusted |
| 4 | Quantise Q, and the new K and V rows | `attention.q8_per_tensor`, `kv.quantize_rows` | max of `abs(x)` over each row | float32 | none |
| 5 | Scores | `attention.exact_attention_core` (`matmul`), or `exact_decode_attention.ysu` (`dp2a`) | `sum_d q[d] * k[t][d]` | float32 below 2^24, or int32 | none for either. `k_levels_for` and the score budget are Python asserts. |
| 6 | Logit, its max, the exp argument | same | max over the unmasked keys | float64 | none |
| 7 | Integer `exp2` | `attention.exp2_neg_q16_16` | none | int64 | `SoftmaxErrorBound.v` assumes an abstract `expf`. The recipe itself is not in Coq, and its range `0..2^28` is not proved. |
| 8 | Denominator | `p.sum` | `sum_t p[t]` | int64 | none for this path |
| 9 | Fold the V scale | `kv.fold_scale_into_weights` | max over keys of `p * sv` | float64 | none |
| 10 | `p.V` | `attention.exact_pv`: Y's kernel, or base-2^dbits digits | `sum_t W[t] * v[t][d]`; per digit a float32 matmul, recombined in float64 | int64, or float32 below 2^24 then float64 below 2^53 | `ExactPvExact.v` and `ExactPvLaunchContract.v`, for Y's kernel only. Nothing for the digit split, which is the default. |
| 11 | Output divide | same | none | float64 | trusted |
| 12 | Residual adds, SiLU, the MLP product | transformers, `model.ExactSiLU` | none | float32, int32 | trusted; the SiLU table interpolation is integer but elementwise |
| 13 | Next token | `greedy.greedy` | argmax over the vocabulary | float32 | none |

## Gaps

1. **Nothing composes.** The theorems are per kernel. No statement says that a
   layer, or a decode step, gives the same result under two different
   executions.
2. **The default kernels are not Y's.** With no environment variable set, the
   linear layers run Triton's `tl.dot` (GPU, sm_80 and later),
   `torch._int_mm` (GPU below sm_80, and CPUs that pass
   `cpu_int_mm_is_exact`) or a float matmul; the scores run a float32
   `matmul`; `p.V` runs the digit split. None has a kernel-level proof, and
   none can: their source is not Y's. What can be proved is one level up: any
   order of exact additions gives the same sum when the bound holds. That
   theorem does not exist yet in a form that covers all carriers.
3. **The widths are asserted in Python, not proved.** `act_levels_for`,
   `k_levels_for`, `digit_width`, `DIGIT_MAX`, the `(a_levels + 128)`
   accumulator bound, the 2^24 score budget, the 2^53 bounds. Each is an
   `assert` or a refusal in the Python. No theorem derives the bound from the
   function that picks the width, for every input.
4. **Max reductions are not covered.** There are five per layer (stages 1, 2,
   4, 6, 9) and the final argmax. Their order varies between devices like any
   sum's.
5. **The integer `exp2` is not in Coq.** Its output range is what bounds
   stages 8 and 10.
6. **No premise is instantiated at a real model.** Kernel theorems are stated
   for abstract `K`, `T`, `D`. Nothing shows that Qwen2.5-0.5B's dimensions
   (or 1.5B, 3B, 7B) satisfy them, at activation levels 127 and 511.
7. **The trust boundary is not written in one place.** The float operations
   that are not reductions (the quantisers' divides, the epilogue multiplies,
   the norm's divide and square root, RoPE, the residual adds, the output
   divide) are argued in comments and in `docs/bit_identical_decode.md`. No
   list states what each relies on.
8. **`exact_decode_attention.ysu` has no Coq proof.** It carries `@bounds`
   and `@invariant` annotations that Z3 checks, and differential tests
   against the torch path.
9. **No drift test for the Python constants.** `ExactGemmSchedule.v` and
   `AttentionSchedule.v` are generated from Rust and compared byte for byte.
   Nothing ties a proof to `ACC_BUDGET`, `DIGIT_MAX`, `P_BITS`, `SUMSQ_F` or
   the width functions.
10. **No executable link between a proof and the Python path.** The kernel
    proofs are tied to emitted PTX or generated schedules. Nothing runs a
    proof's definitions against `python/y_lang/exact/` on real activations.

## The files

### `proofs/Int8GemmExact.v`

The value of Y's int8 tensor-core GEMM: each output element is the source dot product, in int32, at any tile shape, split-K factor and landing order.

- On the decode path: yes, with `Y_EXACT_GEMM=y` on sm_80 and later.
- Covers: Linear layers (stage 2), on that kernel only.
- Theorems and corollaries: 23. Quoted below: 3.

```coq
Theorem the_emitted_int8_gemm_holds_the_source_dot_products :
  forall (A B : nat -> nat -> Z) m K S T KT nz order wms wns ni nj tm tn wm wn mi nj' g t j,
    0 <= m ->
    (forall r k, Z.abs (A r k) <= m) ->
    (forall c k, Z.abs (B c k) <= m) ->
    Z.of_nat K * (m * m) <= MC.I32MAX ->
    (K = 32 * S)%nat -> (0 < T)%nat -> (S <= KT * T)%nat -> (0 < nz)%nat ->
    Permutation order (seq 0 nz) ->
    let brow := (tm * (wms * (ni * 16)) + wm * (ni * 16))%nat in
    let bcol := (tn * (wns * (nj * 8)) + wn * (nj * 8))%nat in
    let row := SCH.out_row wms ni tm wm mi g (j / 2) in
    let col := SCH.out_col wns nj tn wn nj' t (j mod 2) in
    let A' := fun r k => if (k <? K)%nat then A r k else 0 in
    let B' := fun c k => if (k <? K)%nat then B c k else 0 in
    D.combine Z.add (SCH.k_tile_owner T nz) (step A' B' brow bcol mi nj' (d_row g j) (d_col t j)) (KT * T) nz
      = sum_k (prod A B row col) K
    /\ wcombine 0 (map (fun w => wpart (SCH.k_tile_owner (32 * T) nz) (prod A B row col) w K) order)
      = sum_k (prod A B row col) K
    /\ wpart (SCH.k_tile_owner (32 * T) 1) (prod A B row col) 0 K = sum_k (prod A B row col) K.
```

```coq
Corollary the_split_k_combine_is_exact_in_int32 :
  forall A B m K row col nz order,
    0 <= m ->
    (forall r k, Z.abs (A r k) <= m) ->
    (forall c k, Z.abs (B c k) <= m) ->
    Z.of_nat K * (m * m) <= MC.I32MAX ->
    (0 < nz)%nat ->
    Permutation order (seq 0 nz) ->
    wcombine 0 (map (fun w => wclass (prod A B row col) w nz K) order)
    = sum_k (prod A B row col) K.
```

```coq
Theorem bounded_products_accumulate_exactly : forall f n c,
  0 <= c -> (forall i, Z.abs (f i) <= c) -> Z.of_nat n * c <= MC.I32MAX ->
  MC.wsum f 0 n = sum_k f n.
```

Also proved: `the_lanes_cover_the_a_fragment`, `the_lanes_cover_the_b_fragment`, `the_stage_slots_are_disjoint`, `the_copy_lands_at_its_swizzled_place`, `the_ldmatrix_a_fragment_is_the_source_element`, `the_ldmatrix_b_fragment_is_the_source_element`, `the_k_loop_is_the_contraction`, `every_split_factor_gives_the_contraction`, `the_bound_is_one_k_step_wide`, `the_licence_admits_131040_and_refuses_131072`, `the_measured_overflow_is_two_s_complement`, `the_signed_minimum_exposes_the_old_licence`, `the_split_k_accumulation_is_exact_in_int32`, `each_launch_is_licensed_and_three_of_them_wrap`, `the_third_launch_wraps_to_the_measured_value`, `the_combine_needs_a_zeroed_destination`, `from_zero_the_same_partial_is_exact`, `the_owned_split_is_exact_in_int32`, `the_staged_a_register_is_the_mma_fragment`, `the_staged_b_register_is_the_mma_fragment`.

### `proofs/Int8GemmSchedule.v`

The same kernel's schedule: one owner per element of C, every output tile computed once on any grid, the K split partitions the contraction, the atomic reduction is order-independent.

- On the decode path: yes, with `Y_EXACT_GEMM=y`.
- Covers: Linear layers (stage 2), on that kernel only.
- Theorems and corollaries: 37. Quoted below: 2.

```coq
Theorem c_element_has_exactly_one_owner :
  forall wms wns ni nj r c, 0 < wms -> 0 < wns -> 0 < ni -> 0 < nj ->
    (exists tm wm i g h tn wn j t e,
       wm < wms /\ i < ni /\ g < 8 /\ h < 2 /\ wn < wns /\ j < nj /\ t < 4 /\ e < 2
       /\ out_row wms ni tm wm i g h = r /\ out_col wns nj tn wn j t e = c)
    /\ (forall tm1 wm1 i1 g1 h1 tm2 wm2 i2 g2 h2,
          wm1 < wms -> wm2 < wms -> i1 < ni -> i2 < ni ->
          g1 < 8 -> g2 < 8 -> h1 < 2 -> h2 < 2 ->
          out_row wms ni tm1 wm1 i1 g1 h1 = r -> out_row wms ni tm2 wm2 i2 g2 h2 = r ->
          tm1 = tm2 /\ wm1 = wm2 /\ i1 = i2 /\ g1 = g2 /\ h1 = h2)
    /\ (forall tn1 wn1 j1 t1 e1 tn2 wn2 j2 t2 e2,
          wn1 < wns -> wn2 < wns -> j1 < nj -> j2 < nj ->
          t1 < 4 -> t2 < 4 -> e1 < 2 -> e2 < 2 ->
          out_col wns nj tn1 wn1 j1 t1 e1 = c -> out_col wns nj tn2 wn2 j2 t2 e2 = c ->
          tn1 = tn2 /\ wn1 = wn2 /\ j1 = j2 /\ t1 = t2 /\ e1 = e2).
```

```coq
Theorem the_atomic_reduction_is_order_independent :
  forall T nz f S order, 0 < nz -> Permutation order (seq 0 nz) ->
    fold_right Z.add 0%Z (map (fun w => D.part Z.add (k_tile_owner T nz) f w S) order)
    = D.acc_range Z.add f 0 S.
```

Also proved: `lane_decomposition_is_injective`, `a_warps_lanes_fill_the_tile`, `out_row_injective`, `out_col_injective`, `out_row_onto`, `out_col_onto`, `a_thread_is_its_warp_and_lane`, `the_trap_is_what_confines_a_warp_to_its_own_tile`, `without_the_trap_a_fifth_warp_lands_in_the_next_tile`, `the_copies_cover_the_stage_once`, `without_the_trap_a_short_block_leaves_rows_uncopied`, `cta_pid_is_unique`, `without_the_grid_trap_the_stride_wraps`, `the_tile_increment_detects_its_wrap`, `every_output_tile_has_exactly_one_owner`, `grouped_is_a_bijection`, `grouped_one_tile_row`, `grouped_one_tile_column`, `every_output_tile_is_computed_exactly_once`, `every_k_step_has_exactly_one_owner`, `the_split_k_sums_to_the_whole_contraction`, `any_split_factor_gives_the_same_answer`, `a_rounding_accumulate_would_break_the_split`, `a_rounding_accumulate_would_break_the_landing_order`, `the_stripe_stays_below_the_k_tiles`, `the_stripe_reaches_every_tile_of_its_class`, `the_zero_filled_tail_adds_nothing`, `a_storing_cta_computes_the_whole_contraction`, `every_storing_cta_writes_the_same_value`, `the_unsplit_launch_stores_the_whole_contraction`, `a_split_cta_holds_only_part_of_the_contraction`, `two_split_ctas_would_store_different_values`, `the_reducing_epilogue_still_splits`, `whole_blocks_are_in_or_out`, `an_uncovered_shape_is_refused`.

### `proofs/ExactPvExact.v`

The `exact_pv` kernel: its int64 accumulation over the emitted 32-bit indices is the source dot product, with the index and accumulator licences and their measured failures.

- On the decode path: yes, with `Y_EXACT_V_INT8=1`.
- Covers: p.V (stage 10), on that kernel only.
- Theorems and corollaries: 17. Quoted below: 2.

```coq
Theorem the_emitted_exact_pv_holds_the_source_dot_product :
  forall Pf Vf b q Qx Tx Dx d NP NV (n : nat),
    0 <= b -> 0 <= q -> 0 <= d -> 1 <= Dx -> 1 <= Tx -> 0 <= Qx ->
    Z.of_nat n <= Tx ->
    (forall z, p_domain (Pf z)) -> (forall z, v_domain (Vf z)) ->
    (forall t, (t < n)%nat -> (b * Qx + q) * Tx + Z.of_nat t <= MC.I32MAX) ->
    (forall t, (t < n)%nat -> (b * Tx + Z.of_nat t) * Dx + d <= MC.I32MAX) ->
    (forall t, (t < n)%nat -> (b * Qx + q) * Tx + Z.of_nat t < NP) ->
    (forall t, (t < n)%nat -> (b * Tx + Z.of_nat t) * Dx + d < NV) ->
    Z.of_nat n * (PMAX * VABS) <= I64MAX ->
    wacc (emitted_term Pf Vf b q Qx Tx Dx d NP NV) n
      = sum_upto (src_term Pf Vf b q Qx Tx Dx d) n.
```

```coq
Theorem bounded_terms_accumulate_exactly : forall f n c,
  0 <= c -> (forall k, Z.abs (f k) <= c) -> Z.of_nat n * c <= I64MAX ->
  wacc f n = sum_upto f n.
```

Also proved: `acc_exact_in_int64`, `the_product_never_wraps`, `the_accumulator_bound_is_one_iteration_wide`, `the_measured_overflow_is_two_s_complement`, `the_semantic_domain_raises_the_accumulator_ceiling`, `the_v_index_is_the_mathematical_one`, `the_p_index_is_the_mathematical_one`, `the_index_ceiling_binds_first`, `the_measured_index_wrap_reads_v0_twice`, `the_measured_double_count_is_what_the_device_returned`, `the_output_index_is_the_mathematical_one`, `every_output_element_is_written_by_one_thread`, `without_the_index_licence_the_answer_is_wrong`, `the_accumulator_licence_is_load_bearing`, `the_loop_lengths_exist`.

### `proofs/ExactPvLaunchContract.v`

The shape the Rust launcher checks implies every licence `ExactPvExact.v` needs.

- On the decode path: yes, with `Y_EXACT_V_INT8=1`.
- Covers: p.V (stage 10), on that kernel only.
- Theorems and corollaries: 6. Quoted below: 1.

```coq
Theorem checked_shape_instantiates_the_source_dot_product :
  forall Pf Vf B Q T D b q d (n : nat),
    checked_shape B Q T D ->
    0 <= b < B -> 0 <= q < Q -> 0 <= d < D ->
    Z.of_nat n = T ->
    (forall z, EP.p_domain (Pf z)) ->
    (forall z, EP.v_domain (Vf z)) ->
    EP.wacc (EP.emitted_term Pf Vf b q Q T D d (B * Q * T) (B * T * D)) n
      = EP.sum_upto (EP.src_term Pf Vf b q Q T D d) n.
```

Also proved: `checked_shape_licenses_every_live_index`, `checked_shape_licenses_the_output_index`, `geometry_covers_the_output_rectangle`, `geometry_output_coordinates_are_unique`, `disjoint_live_byte_spans_never_alias`.

### `proofs/NormGridSum.v`

ExactRMSNorm's sum of squares: the two 40-bit limbs are the floor grid value, their sums do not depend on the order, and they fit below 2^53 for rows of at most 8192 values.

- On the decode path: yes, always.
- Covers: RMSNorm (stage 1): the integer part.
- Theorems and corollaries: 7. Quoted below: 4.

```coq
Theorem the_limbs_are_the_grid_value : forall m2 s,
  0 <= m2 < 2 ^ 48 -> 0 <= s ->
  hi m2 s * 2 ^ 40 + lo m2 s = grid m2 s.
```

```coq
Theorem the_sum_does_not_depend_on_the_order : forall l l',
  Permutation l l' -> big l = big l' /\ small l = small l'.
```

```coq
Theorem the_limb_sums_fit : forall l, Forall ok l -> Z.of_nat (length l) <= 8192 ->
  0 <= zsum fhi l < 2 ^ 53 /\ 0 <= zsum flo l < 2 ^ 53 /\
  0 <= small l < 2 ^ 40 /\ 0 <= big l < 2 ^ 53.
```

```coq
Theorem the_carried_limbs_are_the_row_sum : forall l, Forall ok l ->
  big l * 2 ^ 40 + small l = zsum fgrid l.
```

Also proved: `the_torch_mask_is_the_kernel_mask`, `the_grid_value_is_the_floor`, `past_8192_values_big_can_reach_2_53`.

### `proofs/SoftmaxErrorBound.v`

An error bound for the integer softmax against the ideal one, for an abstract `expf` meeting a stated interface. It bounds how far the output is from the real softmax; it says nothing about bit equality.

- On the decode path: accuracy, not determinism.
- Covers: Attention weights (stages 7 to 10), as an accuracy bound for the PTX attention kernels.
- Theorems and corollaries: 22. Quoted below: 2.

```coq
Corollary the_output_quotient_is_within_the_bound : forall n,
  0 < Lq n -> 0 < Wtot n ->
  Qabs (Oq n / Lq n - Wnum n / Wtot n) <= (2#1) * VMAXQ * Err n / Lq n.
```

```coq
Corollary the_bound_holds_at_every_launch_geometry : forall n k,
  (0 < k)%nat ->
  inject_Z (GS.combine Z.add p k n k) == Lq n
  /\ inject_Z (GS.combine Z.add (fun i => p i * v i)%Z k n k) == Oq n.
```

Also proved: `the_rounding_is_within_a_half_ulp`, `an_unsaturated_argument_is_within_a_half_ulp`, `saturation_means_the_exact_exponent_is_astronomically_large`, `the_per_unit_factor_is_what_one_unit_of_log2_costs`, `the_weight_past_thirty_halvings_is_below_an_ulp`, `the_swept_domain_covers_the_admitted_one`, `the_weight_is_within_a_relative_ulp`, `the_attention_output_is_within_the_bound`, `the_max_subtraction_puts_a_floor_under_the_total_weight`, `a_total_weight_below_one_ulp_admits_a_zero_denominator`, `the_bound_at_a_long_context`, `the_quantized_temperature_shifts_the_exponent_by_at_most_half_a_delta`, `the_quantized_temperature_moves_the_ideal_weight_by_at_most_a_beta_power`, `at_head_dim_128_the_temperature_moves_a_weight_by_under_a_two_thousandth`, `a_zero_multiplier_gives_every_key_the_same_weight`, `the_two_readings_of_the_multiplier_disagree_above_two_to_the_thirty_one`, `the_signed_reading_zeroes_a_weight_the_unsigned_one_keeps`, `no_rational_squares_to_two`, `exact_homogeneity_is_unsatisfiable_over_Q`, `the_interface_is_satisfiable`.

### `proofs/GridStrideSplit.v`

A grid-stride reduction (worker `w` of `n` takes the indices congruent to `w`) equals the plain sum at any worker count and in any landing order. This is the first proof here that spends commutativity.

- On the decode path: no: `src/exact_attention.rs` kernels.
- Covers: The older PTX attention kernels. The decode path in `python/y_lang/exact/` does not launch them.
- Theorems and corollaries: 11. Quoted below: 2.

```coq
Theorem grid_stride_exact : forall f n S,
  (0 < n)%nat ->
  combine Z.add f n S n = sum_upto Z.add f S.
```

```coq
Theorem atomics_may_land_in_any_order : forall f n S order,
  (0 < n)%nat ->
  Permutation order (seq 0 n) ->
  fold_right Z.add 0 (map (fun w => class_sum Z.add f w n S) order)
  = sum_upto Z.add f S.
```

Also proved: `stride_classes_partition`, `any_worker_count_agrees`, `rounding_breaks_the_stride_split`, `rounding_is_order_dependent`, `exact_is_order_independent`, `the_two_accumulates_differ_on_this_input`, `the_bound_is_one_unit_wide`, `the_emitted_launch_geometry_visits_every_key_exactly_once`, `every_folded_worker_is_a_real_thread`.

### `proofs/AttentionSchedule.v`

Generated from the Rust emitter: the PTX attention kernels' worker index is a mixed-radix number of the launch coordinates, so distinct threads are distinct workers.

- On the decode path: no: `src/exact_attention.rs` kernels.
- Covers: The older PTX attention kernels.
- Theorems and corollaries: 9. Quoted below: 2.

```coq
Theorem distinct_threads_are_distinct_workers :
  forall cz1 cx1 tx1 cz2 cx2 tx2 ncx ntx,
    0 < ncx -> 0 < ntx ->
    cx1 < ncx -> cx2 < ncx -> tx1 < ntx -> tx2 < ntx ->
    worker_accum cz1 ncx cx1 ntx tx1 = worker_accum cz2 ncx cx2 ntx tx2 ->
    cz1 = cz2 /\ cx1 = cx2 /\ tx1 = tx2.
```

```coq
Theorem every_worker_is_some_thread :
  forall w ncz ncx ntx,
    0 < ncx -> 0 < ntx ->
    w < nworkers_accum ncz ncx ntx ->
    exists cz cx tx,
      cz < ncz /\ cx < ncx /\ tx < ntx /\
      worker_accum cz ncx cx ntx tx = w.
```

Also proved: `the_worker_index_is_a_mixed_radix_number`, `the_worker_count_is_the_product_of_the_extents`, `the_scores_worker_index_is_a_mixed_radix_number`, `the_scores_worker_count_is_the_product_of_the_extents`, `the_worker_is_below_the_worker_count`, `dropping_the_z_extent_overflows_the_worker_count`, `the_scores_schedule_is_the_accumulate_schedule_at_one_z`.

### `proofs/Decomposition.v`

The partition obligation proved once: contiguous parts and owner-mapped parts both combine to the plain fold under `Z.add`, with rounding counterexamples.

- On the decode path: library.
- Covers: Used by the GEMM and attention schedule proofs.
- Theorems and corollaries: 12. Quoted below: 2.

```coq
Theorem contiguous_exact : forall edge f n N,
  edge 0%nat = 0%nat ->
  edge n = N ->
  (forall u, (edge u <= edge (S u))%nat) ->
  acc_parts Z.add edge f n = acc_range Z.add f 0 N.
```

```coq
Theorem decomposition_exact : forall owner f N n,
  (forall i, (i < N)%nat -> (owner i < n)%nat) ->
  combine Z.add owner f N n = acc_range Z.add f 0 N.
```

Also proved: `contiguous_count_agrees`, `widths_cover_the_extent`, `parts_tile`, `decomposition_count_agrees`, `rounding_breaks_a_contiguous_split`, `exact_survives_the_same_split`, `rounding_breaks_an_interleaved_split`, `exact_survives_the_interleaved_split`, `the_two_accumulates_differ_on_this_input`, `the_interleaved_split_is_not_contiguous`.

### `proofs/MixedRadix.v`

Positional indices: pack/unpack round trips, range, onto, and two-digit uniqueness.

- On the decode path: library.
- Covers: Used by the index arguments of the other files.
- Theorems and corollaries: 8. Quoted below: 2.

```coq
Theorem pack_unpack : forall B q r,
  (0 < B)%nat -> (r < B)%nat -> hi B (pack B q r) = q /\ lo B (pack B q r) = r.
```

```coq
Theorem two_digit_unique : forall B0 B1 q1 m1 r1 q2 m2 r2,
  (0 < B0)%nat -> (0 < B1)%nat ->
  (m1 < B1)%nat -> (m2 < B1)%nat -> (r1 < B0)%nat -> (r2 < B0)%nat ->
  (q1 * (B1 * B0) + m1 * B0 + r1 = q2 * (B1 * B0) + m2 * B0 + r2)%nat ->
  q1 = q2 /\ m1 = m2 /\ r1 = r2.
```

Also proved: `unpack_pack`, `pack_in_range`, `pack_onto`, `the_low_digit_bound_cannot_be_dropped`, `without_the_digit_bound_it_is_not_injective`, `the_high_digit_needs_no_bound`.

### `proofs/ExactGemmSchedule.v`

Generated from `src/cpu_gemm.rs`: the CPU exact GEMM's constants and index maps, and that the emitted loop bounds are the models the other files use.

- On the decode path: no: CPU `@ZeroDrift` GEMM.
- Covers: Y's CPU exact GEMM (`vpdpwssd`). `python/y_lang/exact/` never calls it: on a CPU the linear layers run `torch._int_mm` or a float64 matmul.
- Theorems and corollaries: 12. Quoted below: 1.

```coq
Theorem the_schedule_is_the_shipped_one :
  MR = 6 /\ NRV = 4 /\ LANES = 16 /\ NR = 64
  /\ VEC_ELEMS = 32 /\ FLUSH_K_PAIRS = 64
  /\ KSPLIT_MIN_BAND = 128 /\ KSPLIT_MAX_THREADS = 64.
```

Also proved: `jobs_bytes_is_a_record_per_thread`, `the_emitted_width_is_the_tiling_model_at_the_loop_variable`, `the_emitted_panel_index_is_the_tile_index`, `the_emitted_tile_count_is_the_tiling_model`, `the_emitted_chunk_end_is_the_flush_model`, `the_emitted_band_length_is_the_ksplit_model`, `the_emitted_a_index_is_the_pair_element`, `the_tile_geometry_is_consistent`, `slot_b_is_the_plain_interleave`, `the_tile_fits_the_register_file`, `ksplit_threads_is_never_zero`.

### `proofs/ExactGemmKsplit.v`

The K split into contiguous bands sums to the whole contraction at any thread count.

- On the decode path: no: CPU `@ZeroDrift` GEMM.
- Covers: CPU exact GEMM.
- Theorems and corollaries: 7. Quoted below: 2.

```coq
Theorem ksplit_exact : forall f K nthr,
  (0 < nthr)%nat ->
  acc_bands Z.add f K nthr nthr = acc_range Z.add f 0 K.
```

```coq
Corollary any_thread_count_agrees : forall f K n1 n2,
  (0 < n1)%nat -> (0 < n2)%nat ->
  acc_bands Z.add f K n1 n1 = acc_bands Z.add f K n2 n2.
```

Also proved: `bands_tile`, `dropping_the_remainder_loses_terms`, `rounding_breaks_the_split`, `exact_survives_the_same_split`, `the_two_accumulates_differ_on_this_input`.

### `proofs/ExactGemmMSplit.v`

The M split is a partition of the rows: every row has one owner and gets its naive value, with no algebra needed.

- On the decode path: no: CPU `@ZeroDrift` GEMM.
- Covers: CPU exact GEMM.
- Theorems and corollaries: 12. Quoted below: 1.

```coq
Theorem every_row_gets_its_naive_value : forall f M mthr K r,
    (0 < mthr)%nat -> (r < M)%nat ->
    exists t, (t < mthr)%nat /\ owns M mthr t r /\
      msplit_row f M mthr K t (r - roff M mthr t) = row_naive f K r.
```

Also proved: `rows_tile`, `owner_unique`, `owner_exists`, `msplit_exact`, `any_thread_count_agrees`, `msplit_needs_no_algebra`, `an_msplit_survives_the_accumulate_that_breaks_a_ksplit`, `the_accumulate_really_does_round`, `the_rounding_accumulate_is_not_associative`, `the_bands_are_not_all_one`, `a_ragged_extent_still_tiles`.

### `proofs/ExactGemmTiling.v`

The output tiling writes every element of C exactly once, with the clamped ragged tail.

- On the decode path: no: CPU `@ZeroDrift` GEMM.
- Covers: CPU exact GEMM.
- Theorems and corollaries: 19. Quoted below: 1.

```coq
Theorem c_written_exactly_once : forall M N MR NR ldc ti1 fi1 tj1 fj1 ti2 fi2 tj2 fj2,
  (0 < MR)%nat -> (0 < NR)%nat -> (N <= ldc)%nat ->
  (fi1 < tw M MR ti1)%nat -> (fj1 < tw N NR tj1)%nat ->
  (fi2 < tw M MR ti2)%nat -> (fj2 < tw N NR tj2)%nat ->
  addr ldc (toff MR ti1) fi1 (toff NR tj1) fj1
  = addr ldc (toff MR ti2) fi2 (toff NR tj2) fj2 ->
  ti1 = ti2 /\ fi1 = fi2 /\ tj1 = tj2 /\ fj1 = fj2.
```

Also proved: `tiles_cover`, `tile_index_in_range`, `tile_index_injective`, `tile_index_surjective`, `every_live_element_is_written`, `unclamped_tail_writes_out_of_bounds`, `the_clamped_tail_stays_in_bounds`, `row_stride_below_n_aliases`, `the_emitted_row_loop_enumerates_the_tiles`, `the_emitted_row_loop_runs_once_per_tile`, `the_emitted_column_loop_enumerates_the_tiles`, `the_emitted_column_loop_runs_once_per_tile`, `the_emitted_fold_back_visits_each_offset_once`, `the_emitted_fold_back_runs_the_tile_width`, `the_emitted_fold_back_column_runs_the_tile_width`, `the_fold_back_stays_inside_the_live_rectangle`, `the_fold_back_trip_count_is_what_keeps_it_in_bounds`, `the_two_panel_loops_are_different`.

### `proofs/ExactGemmPacking.v`

The operand packing: slot maps are bijections into the panels, and a full padded tile yields the live dot product.

- On the decode path: no: CPU `@ZeroDrift` GEMM.
- Covers: CPU exact GEMM.
- Theorems and corollaries: 14. Quoted below: 1.

```coq
Theorem padded_product_is_the_live_dot_product :
  forall (arow bcol : nat -> Z) (i j mrows ncols kc : nat),
    (i < mrows)%nat -> (j < ncols)%nat ->
    sum_pairs
      (fun p h =>
         packed arow mrows kc i (2 * p + h)%nat
         * packed bcol ncols kc j (2 * p + h)%nat)
      (kpairs kc)
    = sum_k (fun k => arow k * bcol k) kc.
```

Also proved: `pack_a_slot_bijective`, `pack_a_slot_in_panel`, `pack_a_slot_onto`, `pack_b_slot_bijective`, `slot_b_is_the_plain_interleave`, `pack_b_slot_in_panel`, `a_dead_row_contributes_nothing`, `garbage_in_the_pad_changes_the_answer`, `the_masked_version_agrees_on_the_same_input`, `panel_decodes_its_own_write`, `panel_is_the_only_solution`, `the_group_bound_is_load_bearing`, `inside_the_group_the_same_panel_agrees`.

### `proofs/ExactGemmMicro.v`

The int32 flush: the chunked accumulation is the plain sum, and the operand bound keeps every int32 chunk from wrapping.

- On the decode path: no: CPU `@ZeroDrift` GEMM (its `wrap32` is reused by the int8 GEMM proof).
- Covers: CPU exact GEMM.
- Theorems and corollaries: 10. Quoted below: 2.

```coq
Theorem flush_exact_in_int32 : forall f lo len,
  (forall i, (i <= len)%nat -> in_i32 (sum_from f lo i)) ->
  wsum f lo len = sum_from f lo len.
```

```coq
Theorem operand_bound_gives_no_overflow : forall f lo len Fl m,
  0 <= m ->
  (len <= Fl)%nat ->
  2 * Z.of_nat Fl * m * m <= I32MAX ->
  (forall p, Z.abs (f p) <= 2 * m * m) ->
  forall i, (i <= len)%nat -> in_i32 (sum_from f lo i).
```

Also proved: `flush_exact`, `stopping_at_the_last_whole_interval_loses_the_tail`, `overflow_breaks_the_flush`, `the_licence_makes_the_chunk_exact`, `the_default_interval_licenses_4095_and_not_4096`, `the_4096_case_exceeds_by_exactly_one`, `the_packed_column_is_the_stored_column`, `a_wrong_lane_stride_permutes_the_columns`.

### `proofs/ExactGemmRegisterTile.v`

The register tile's routing: one `vpdpwssd` step is two k terms of the right row and column.

- On the decode path: no: CPU `@ZeroDrift` GEMM.
- Covers: CPU exact GEMM.
- Theorems and corollaries: 11. Quoted below: 1.

```coq
Theorem one_step_is_two_k_terms : forall acc Ap Bp p i v l,
  vpdpwssd acc
           (broadcast (i32_lo Ap (p * MR + i)) (i32_hi Ap (p * MR + i)))
           (bvec (fun e => Bp (p * (NR * 2) + e)%nat) v) l
  = acc l
    + Ap (p * (MR * 2) + slot_a i 0)%nat * Bp (p * (NR * 2) + slot_b (col_of v l) 0)%nat
    + Ap (p * (MR * 2) + slot_a i 1)%nat * Bp (p * (NR * 2) + slot_b (col_of v l) 1)%nat.
```

Also proved: `the_lane_consumes_its_own_column`, `the_i32_load_is_the_packed_pair`, `swapping_the_pair_halves_computes_a_different_function`, `a_symmetric_operand_hides_the_swap`, `tile_position_injective`, `tile_position_in_range`, `tile_position_surjective`, `only_the_live_rectangle_is_stored`, `a_padding_column_never_reaches_c`, `a_padding_row_never_reaches_c`.

### `proofs/ExactGemmComposition.v`

The five self-contained GEMM proofs agree on their shared slot maps, and a packed lane accumulates the right source elements for one step.

- On the decode path: no: CPU `@ZeroDrift` GEMM.
- Covers: CPU exact GEMM.
- Theorems and corollaries: 12. Quoted below: 1.

```coq
Theorem the_packed_panels_route_to_the_right_source_elements :
  forall (A B : nat -> nat -> Z) (mrows ncols kc p : nat) acc i v l,
    (i < ExactGemmRegisterTile.MR)%nat ->
    (v < ExactGemmRegisterTile.NRV)%nat -> (l < 16)%nat ->
    ExactGemmRegisterTile.vpdpwssd
      acc
      (ExactGemmRegisterTile.broadcast
         (ExactGemmPacking.panel A mrows kc ExactGemmPacking.MR
            (p * (2 * ExactGemmRegisterTile.MR)
             + ExactGemmRegisterTile.slot_a i 0)%nat)
         (ExactGemmPacking.panel A mrows kc ExactGemmPacking.MR
            (p * (2 * ExactGemmRegisterTile.MR)
             + ExactGemmRegisterTile.slot_a i 1)%nat))
      (ExactGemmRegisterTile.bvec
         (fun e => ExactGemmPacking.panel (fun j k => B k j) ncols kc
                     ExactGemmPacking.NR
                     (p * (2 * ExactGemmRegisterTile.NR) + e)%nat) v)
      l
    = acc l
      + ExactGemmPacking.packed (A i) mrows kc i (2 * p)%nat
        * ExactGemmPacking.packed
            (fun k => B k (ExactGemmRegisterTile.col_of v l)) ncols kc
            (ExactGemmRegisterTile.col_of v l) (2 * p)%nat
      + ExactGemmPacking.packed (A i) mrows kc i (2 * p + 1)%nat
        * ExactGemmPacking.packed
            (fun k => B k (ExactGemmRegisterTile.col_of v l)) ncols kc
            (ExactGemmRegisterTile.col_of v l) (2 * p + 1)%nat.
```

Also proved: `packing_and_register_tile_agree_on_the_b_slot`, `micro_and_register_tile_agree_on_the_b_slot`, `packing_and_micro_agree_on_the_b_slot`, `packing_and_register_tile_agree_on_the_a_slot`, `micro_and_register_tile_agree_on_the_column`, `the_tile_shape_is_the_same_everywhere`, `the_agreement_is_not_vacuous`, `the_lane_accumulates_the_source_elements`, `a_dead_row_contributes_nothing`, `the_flush_interval_and_the_output_tile_are_the_same_family`, `the_agreement_covers_the_ragged_case`.

### `proofs/ExactGemmChain.v`

One accumulator lane end to end, then one whole MR x NR tile, hold the source dot products.

- On the decode path: no: CPU `@ZeroDrift` GEMM.
- Covers: CPU exact GEMM.
- Theorems and corollaries: 15. Quoted below: 1.

```coq
Theorem the_tile_holds_the_source_dot_products :
  forall A B C0 mrows ncols kc Fl m i j,
    (0 < Fl)%nat -> 0 <= m ->
    2 * Z.of_nat Fl * m * m <= MC.I32MAX ->
    (forall idx k, Z.abs (A idx k) <= m) ->
    (forall k idx, Z.abs (B k idx) <= m) ->
    (i < RT.MR)%nat -> (j < RT.NR)%nat ->
    tile_after C0
      (PK.panel A mrows kc PK.MR)
      (PK.panel (fun c k => B k c) ncols kc PK.NR)
      Fl mrows ncols (PK.kpairs kc) i j
    = if RT.stored mrows ncols i j
      then C0 i j + PK.sum_k (fun k => A i k * B k j) kc
      else C0 i j.
```

Also proved: `the_k_pair_loop_computes_the_source_dot_product`, `a_dead_row_accumulates_nothing`, `a_dead_column_accumulates_nothing`, `the_chain_is_not_vacuous`, `the_flush_schedule_computes_the_source_dot_product`, `the_whole_chain_is_not_vacuous`, `the_emitted_lane_computes_the_source_dot_product`, `violating_the_licence_breaks_the_chain`, `at_the_licensed_magnitude_the_chain_holds`, `the_lane_map_is_a_two_sided_inverse`, `distinct_columns_use_distinct_lanes`, `a_dead_position_leaves_c_untouched`, `the_tile_lift_is_not_vacuous`, `the_store_predicate_is_redundant`.

### `proofs/ExactGemmWhole.v`

Every element of C from the threaded CPU GEMM is the source dot product, at any thread count.

- On the decode path: no: CPU `@ZeroDrift` GEMM.
- Covers: CPU exact GEMM.
- Theorems and corollaries: 13. Quoted below: 1.

```coq
Theorem the_threaded_gemm_holds_the_source_dot_products :
  forall A B M N K Fl m nthr r c,
    (0 < Fl)%nat -> 0 <= m ->
    2 * Z.of_nat Fl * m * m <= ExactGemmMicro.I32MAX ->
    (forall idx k, Z.abs (A idx k) <= m) ->
    (forall k idx, Z.abs (B k idx) <= m) ->
    (0 < nthr)%nat -> (r < M)%nat -> (c < N)%nat ->
    thread_sum A B M N K Fl nthr r c nthr
    = PK.sum_k (fun k => A r k * B k c) K.
```

Also proved: `the_whole_output_holds_the_source_dot_products`, `the_zeroing_loop_visits_each_slot_once`, `the_scratch_is_zeroed_wherever_the_fold_back_reads`, `the_zeroing_loop_covers_exactly_the_tile`, `one_trip_short_leaves_the_corner_stale`, `the_position_decomposition_is_the_tilings`, `the_whole_chain_is_not_vacuous`, `both_bands_contribute`, `the_packing_buffers_hold_every_slot_written`, `the_workers_records_cannot_overlap`, `every_row_of_c_is_written_by_exactly_one_band`, `the_row_bands_leave_no_row_unwritten`.

### `proofs/ExactGemmAllocation.v`

Every packing-buffer and scratch write is inside its allocation, and each allocation is exactly the write set.

- On the decode path: no: CPU `@ZeroDrift` GEMM.
- Covers: CPU exact GEMM (memory safety).
- Theorems and corollaries: 13. Quoted below: 1.

```coq
Theorem the_a_allocation_is_exactly_the_write_set : forall mtiles kp,
  (0 < mtiles)%nat -> (0 < kp)%nat ->
  S (a_write (mtiles - 1) (kp - 1) (SCH.MR - 1) 1 kp)
  = SCH.panel_a_elems mtiles kp.
```

Also proved: `the_stride_is_the_packing_proofs_panel_extent`, `the_a_buffer_is_one_panel_per_row_tile`, `pack_a_write_is_inside_the_allocation`, `pack_b_write_is_inside_the_allocation`, `the_scratch_store_is_inside_the_tile`, `the_b_allocation_is_exactly_the_write_set`, `the_scratch_tile_is_exactly_the_store_set`, `one_element_short_overflows_the_a_panel`, `one_element_short_overflows_the_b_panel`, `one_element_short_overflows_the_scratch_tile`, `the_shipped_sizes_are_not_degenerate`, `the_two_panel_extents_are_different`.

### `proofs/ExactGemmThreading.v`

The threading layer's records and private buffers: slots are disjoint, and the reduction over the private tiles is the K-split sum.

- On the decode path: no: CPU `@ZeroDrift` GEMM.
- Covers: CPU exact GEMM.
- Theorems and corollaries: 23. Quoted below: 1.

```coq
Theorem the_reduction_is_the_ksplit_sum : forall f K nthr,
  (0 < nthr)%nat ->
  reduce_into 0
    (fun t => KS.acc_range Z.add f (KS.boff K nthr t) (KS.blen K nthr t)) nthr
  = KS.acc_range Z.add f 0 K.
```

Also proved: `job_slot_in_range`, `job_slot_injective`, `the_job_array_is_exactly_the_record_set`, `one_slot_short_overflows_the_job_array`, `a_thirteenth_field_is_the_next_threads_first`, `tid_slot_in_range`, `the_tid_array_is_exactly_the_thread_set`, `live_slot_in_range`, `the_live_array_is_exactly_the_thread_set`, `the_flag_joins_exactly_the_threads_that_started`, `the_sentinel_agrees_only_if_no_live_thread_has_id_zero`, `a_live_thread_with_id_zero_is_not_joined`, `private_slot_in_range`, `the_private_c_is_read_back_exactly_once`, `every_element_of_the_private_buffer_is_reached`, `the_private_c_allocation_is_exactly_the_reduced_set`, `one_element_short_overflows_the_private_c`, `reading_the_private_buffer_at_the_callers_stride_leaves_it`, `a_row_stride_below_cols_aliases`, `the_destination_is_injective_when_the_stride_admits_the_row`, `a_destination_that_is_not_zeroed_is_wrong`, `a_zeroed_destination_is_exactly_the_partials`.

### `proofs/GemmBandSplit.v`

The f32 kernel's band edges tile their extents; the exactness theorems hold for `Z.add` and are shown to fail under rounding, which is what the f32 kernel does.

- On the decode path: no: f32 CPU GEMM.
- Covers: Not on the exact path (a float kernel).
- Theorems and corollaries: 13. Quoted below: 1.

```coq
Theorem prop_ksplit_exact : forall f n ext,
  (0 < n)%nat ->
  pacc_bands Z.add f n ext n = KS.acc_range Z.add f 0 ext.
```

Also proved: `pedge_zero`, `pedge_last`, `pedge_monotone`, `prop_bands_tile`, `gedge_zero`, `gedge_last`, `gedge_monotone`, `every_edge_snaps_to_a_granule_or_the_extent`, `granule_split_exact`, `the_two_splits_are_different`, `rounding_breaks_the_proportional_split_too`, `exact_survives_the_proportional_split`.

### `proofs/GpuWarpTiling.v`

A CTA tile's rows are written exactly once when the warp geometry divides, with the measured gap when it does not.

- On the decode path: no: float tensor-core GEMMs.
- Covers: Not on the exact path.
- Theorems and corollaries: 9. Quoted below: 1.

```coq
Theorem cta_rows_written_exactly_once :
    forall x, x < cta ->
    exists! t, let '(w, i, r) := t in
      w < warps /\ i < num /\ r < frag /\ row stride frag w i r = x.
```

Also proved: `the_stride_is_the_written_extent`, `rows_in_range`, `rows_injective`, `rows_onto`, `the_measured_gap`, `the_measured_hole_count`, `the_legal_tile_has_no_gap`, `the_emitted_count_is_the_model`.

### `proofs/CountingSort.v`

The MSM bucket scatter writes every slot exactly once.

- On the decode path: no: MSM.
- Covers: Not on the exact path (zero-knowledge backend).
- Theorems and corollaries: 23. Quoted below: 1.

```coq
Theorem the_msm_scatter_writes_every_slot_exactly_once :
  forall counts grp nchunk nb ng s,
  (nchunk <= ng * grp)%nat ->
  (s < edge (bucket_total counts nchunk) nb)%nat ->
  exists b gi r,
    (b < nb)%nat /\ (gi < ng)%nat /\ (r < group_width counts grp nchunk b gi)%nat /\
    (cursor counts grp nchunk b gi + r)%nat = s /\
    (forall b' gi' r',
       (b' < nb)%nat -> (gi' < ng)%nat ->
       (r' < group_width counts grp nchunk b' gi')%nat ->
       (cursor counts grp nchunk b' gi' + r')%nat = s ->
       b' = b /\ gi' = gi /\ r' = r).
```

Also proved: `the_bucket_edges_are_a_decomposition_edge_function`, `widths_of_an_edge_recover_it`, `slot_in_range`, `slot_injective`, `slot_onto`, `every_slot_is_written_exactly_once`, `the_reduction_theorem_still_applies`, `chained_starts_are_prefix_edges`, `a_chained_run_stays_in_the_bucket`, `the_chained_runs_do_not_overlap`, `the_chained_runs_tile_the_bucket`, `without_the_chaining_a_slot_can_go_unwritten`, `without_the_chaining_two_groups_can_write_one_slot`, `an_empty_group_still_chains`, `dest_in_range`, `dest_injective`, `dest_onto`, `every_idx_slot_is_written_exactly_once`, `no_uniform_radix_describes_this`, `the_group_widths_exhaust_every_bucket`, `the_cursor_is_the_destination`, `the_msm_instance_is_not_vacuous`.

### `proofs/ZkControlFlow.v`

The R1CS control-flow lowering agrees with the operational semantics.

- On the decode path: no: R1CS.
- Covers: Not on the exact path (zero-knowledge backend).
- Theorems and corollaries: 2. Quoted below: 1.

```coq
Theorem low_correct : forall s e, bools e s -> agrees e s.
```

Also proved: `output_is_the_returned_value`.

## Added after this inventory (2026-11-15)

Four files. `ExactReduce.v`, `ExactDecodeConstants.v` and `ExactDecodeWidths.v`
close gaps 3, 5 and 9 and the sum half of gaps 2 and 4; `ExactDecodeStep.v`
closes gap 1 (the decode step) and gap 6 (Qwen2.5 0.5B to 7B at widths 127
and 511). What is still open is listed at the end.

### `proofs/ExactReduce.v`

Any order and any association of a sum of integers, in int32, int64, float32
or float64, gives the mathematical sum while the absolute values of the terms
sum to at most the carrier's cap. This is the statement that covers kernels
whose schedule is not Y's (Triton, cuBLASLt, oneDNN, `torch.sum`). Max and
argmax are modelled by their specifications.

- On the decode path: yes, every sum (stages 1, 2, 5, 8, 10), as a statement
  about carriers, not about any kernel's code.
- Theorems and corollaries: 15. Quoted below: 4.

```coq
Theorem any_schedule_gives_the_sum : forall B add terms t,
  exact_within B add ->
  covers t (length terms) ->
  abs_sum terms <= B ->
  teval add terms t = zsum terms.
```

```coq
Theorem every_carrier_is_exact_within_its_cap : forall c,
  exact_within (cap c) (cadd c).
```

```coq
Theorem past_the_cap_two_schedules_disagree :
  covers left_first 3 /\ covers right_first 3
  /\ abs_sum spike = cap F32 + 2
  /\ teval (cadd F32) spike left_first = 2 ^ 24
  /\ teval (cadd F32) spike right_first = 2 ^ 24 + 2
  /\ zsum spike = 2 ^ 24 + 2.
```

```coq
Theorem first_argmax_unique : forall d l i1 i2,
    is_first_argmax d l i1 -> is_first_argmax d l i2 -> i1 = i2.
```

### `proofs/ExactDecodeConstants.v`

Generated from `python/y_lang/exact/` by `tools/exact_decode_constants.py`:
the constants each bound reads, and samples of what the Python width
functions and the integer `exp2` return.
`python/tests/test_exact_decode_constants.py` compares it with the code byte
for byte.

- On the decode path: it is the path's constants.
- Theorems and corollaries: 2. Quoted below: 1.

```coq
Theorem the_exp_table_has_64_entries_from_one : length EXP_TABLE = 64%nat
  /\ nth 0 EXP_TABLE 0 = 2 ^ 30.
```

### `proofs/ExactDecodeWidths.v`

`act_levels_for`, `k_levels_for`, `digit_width`, `split_digits`, the integer
`exp2` and the norm's bit fields, transcribed, with the bound each reduction
needs proved for every input, and checked against the generated samples.

- On the decode path: yes, the widths of stages 1, 2, 5, 7 and 10.
- Theorems and corollaries: 14. Quoted below: 4.

```coq
Theorem the_derived_activation_width_fits_the_accumulator : forall k requested,
  0 < k -> 0 <= requested ->
  k * 127 * 127 < C.ACC_BUDGET ->
  let a := act_levels_for k C.W_LEVELS requested in
  127 <= a <= Z.max 127 (Z.min C.DIGIT_MAX requested)
  /\ (a + 1) mod 128 = 0
  /\ (127 < a -> k * (a + 128) * 127 < C.ACC_BUDGET).
```

```coq
Theorem the_derived_key_width_fits_float32 : forall d q,
  0 < d -> 0 < q ->
  d * q * 127 <= C.SCORE_BUDGET ->
  let kl := k_levels_for d q in
  127 <= kl <= Z.max 127 C.K_LEVELS /\ d * q * kl <= C.SCORE_BUDGET.
```

```coq
Theorem the_integer_exp_is_a_q28_weight : forall t,
  0 <= exp2_neg_q16_16 t <= 2 ^ 28.
```

```coq
Theorem the_width_functions_return_what_the_python_returns :
  forallb (triple_ok (fun k r => act_levels_for k C.W_LEVELS r)) C.ACT_LEVELS_SAMPLES = true
  /\ forallb (triple_ok k_levels_for) C.K_LEVELS_SAMPLES = true
  /\ forallb (triple_ok digit_width) C.DIGIT_WIDTH_SAMPLES = true
  /\ forallb (fun s => let '(q, h, l) := s in (split_hi q =? h) && (split_lo q =? l))
             C.SPLIT_SAMPLES = true
  /\ forallb (fun s => exp2_neg_q16_16 (fst s) =? snd s) C.EXP2_SAMPLES = true.
```

### `proofs/ExactDecodeStep.v`

The capstone. A model of one decode step of the exact path (both norms, q/k/v,
RoPE, attention over the cache, o_proj, the MLP, the residuals, the final
norm, the LM head, the argmax), parameterised by a MACHINE: how every sum,
max and argmax is computed, site by site. Any two valid machines give the same
next token and cache rows. The trusted float operations are Section variables.

- On the decode path: all of it, at the level of the dataflow and the
  reductions; see `docs/exact_decode_theorem.md` for what is assumed.
- Theorems and corollaries: 19. Quoted below: 5.

```coq
Theorem every_valid_execution_takes_the_same_decode_step :
  forall dm T M1 M2 mw caches bl tok pos,
    step_premises dm T mw caches tok ->
    valid dm T M1 -> valid dm T M2 ->
    step M1 dm mw caches bl tok pos = step M2 dm mw caches bl tok pos.
```

```coq
Theorem the_code's_carriers_meet_every_need : forall dm T d st c,
  code_shape_ok dm T -> code_uses dm T d st c -> need_st dm T d st <= ER.cap c.
```

```coq
Theorem qwen25_shapes_are_accepted : forall mk req T,
  In mk [qwen25_0_5b; qwen25_1_5b; qwen25_3b; qwen25_7b] -> In req [127; 511] -> 1 <= T <= 32768 ->
  code_shape_ok (mk req) T.
```

```coq
Theorem a_float32_denominator_is_not_exact :
  ~ exact_sum (2 * 2 ^ 28) (sched_sum ER.F32 ltree).
```

```coq
Theorem left_padding_changes_nothing : forall dm T M mw ls P R0 tok pos,
  step_premises dm T mw (map snd (padded_ls ls)) tok -> valid dm T M ->
  mlayers mw = map (fun t => fst (fst t)) ls ->
  run_pad_ok dm pos P R0 0 ls (embed tok) ->
  step M dm mw (map snd (padded_ls ls)) (repeat true P ++ repeat false R0) tok pos
  = step Mref dm mw (map snd (real_ls ls)) (repeat false R0) tok pos.
```

## Added for exact training (2026-11-22)

### `proofs/ExactTrainArith.v`

The integer arithmetic of `python/y_lang/exact/train_lm.py` between the
Python and `ExactReduce.v`: the balanced base-128 digit split that carries
12-bit gradients through int8 GEMMs (enough digits, the fewest, and exactly
the values the code's check accepts), the stacked GEMM's shifted digit-pair
blocks being the dot product of the values in any order inside the bound the
code checks, the quantizers' rounding shift being round-half-even, and the
fixed-point sums being exact in any int64 order. Refutations: the digit count
used until 2026-11-21 leaves 1 over at 14 bits, round half up differs on a
tie, two terms of 2^62 wrap int64, a scale bound of 512 could leave float64.

- On the training path: every product (`Ops.matmul`), every fixed-point sum
  (`Ops.sum`, `scatter_rows`) and every quantizer's rounding, as integer
  arithmetic. Not modelled: that the floats' bit fields are what the
  rounding is given, the elementwise IEEE operations, the int8 GEMM itself
  (`Int8GemmExact.v`). The float32 formula of `scaled_round` is this
  rounding by `ExactScaledRound.v`.
- Theorems and corollaries: 24. Quoted below: 5.

```coq
Theorem nothing_left_iff_in_reach : forall n r,
  rest n r = 0 <-> - reach n 64 <= r <= reach n 63.
```

```coq
Theorem ndigits_hold : forall bits q,
  9 <= bits <= 64 -> Z.abs q <= top bits ->
  rest (ndigits bits) q = 0 /\ value (digits (ndigits bits) q) = q.
```

```coq
Theorem every_order_of_the_blocks_is_exact : forall c K na nb A B da db t,
  0 <= da -> 0 <= db ->
  (forall k i, (k < K)%nat -> (i < na)%nat -> Z.abs (A k i) <= da) ->
  (forall k j, (k < K)%nat -> (j < nb)%nat -> Z.abs (B k j) <= db) ->
  Z.of_nat K * reach na da * reach nb db <= ER.cap c ->
  ER.covers t (length (blocks K na nb A B)) ->
  ER.teval (ER.cadd c) (blocks K na nb A B) t = srange K (fun k => wval na (A k) * wval nb (B k)).
```

```coq
Theorem shift_round_is_nearest : forall m r, 0 <= m -> 1 <= r ->
  nearest_even m r (shift_round m r).
```

```coq
Theorem fixed_point_sums_are_exact : forall c terms t,
  0 <= c <= 61 -> Z.of_nat (length terms) <= 2 ^ c ->
  Forall (fun q => Z.abs q <= 2 ^ (61 - c)) terms ->
  ER.covers t (length terms) ->
  ER.teval (ER.cadd ER.I64) terms t = ER.zsum terms.
```

## Added for exact training's bytes (2026-11-27)

### `proofs/ExactTrainWidths.v`

The machine types of `python/y_lang/exact/train_lm.py`'s quantized values,
narrowed on 2026-11-27 (`q_dtype`: int8 to 8 bits, int16 to 15, int32
above). A clamped value fits its type; every value the digit split forms
stays in int16 up to 15 bits, joined to `q_dtype`'s cut; every digit fits
the int8 stack. Refutations: at 16 bits the split leaves int16, and int8
cannot hold 9 bits.

- On the training path: every `q8` and `digits` call and `exact_product`'s
  digit stacks. Not modelled: that torch's int8 and int16 arithmetic is two's
  complement with an arithmetic shift (tested on the CPU and the GPU).
- Theorems and corollaries: 11. Quoted below: 2.

```coq
Theorem q8_fits_its_type : forall bits q,
  1 <= bits <= 32 -> - TA.top bits <= q <= TA.top bits -> fits (q_width bits) q.
```

```coq
Theorem the_split_is_safe_wherever_q_dtype_gives_int16 : forall bits q n,
  9 <= bits -> q_width bits = 16 -> - TA.top bits <= q <= TA.top bits -> split_in_int16 n q.
```

### `proofs/ExactRowCopy.v`

A training piece of one row runs as two, the row and a copy no target reads
(`lora.SFT._accumulate`, 2026-11-26). Row-wise results, maxima over the rows
with a copy of a row, the values quantized on those maxima and integer sums
with a zero gradient row are the row alone's. Refutations: a scale that
depends on the row count (`exact_sum`'s) changes, and a row that is not a
copy can move the maximum.

- On the training path: the three kinds of operation a piece is made of.
  Not modelled: that no length-dependent sum runs over a piece's rows (a fact
  about the code, checked by the tiny-model test against the CPU and by the
  T4 run's step 5).
- Theorems and lemmas: 10. Quoted below: 2.

```coq
Theorem a_copy_keeps_the_quantized : forall (q : Z -> Z -> Z) xs c, In c xs ->
  firstn (length xs) (map (fun x => q x (maxl (xs ++ [c]))) (xs ++ [c]))
  = map (fun x => q x (maxl xs)) xs.
```

```coq
Theorem a_zero_row_keeps_the_sum : forall xs gs c, length xs = length gs ->
  dot (xs ++ [c]) (gs ++ [0]) = dot xs gs.
```

### `proofs/ExactMlpKernel.v`

The integer and bit-field arithmetic of `python/y_lang/exact/mlp_kernel.py`,
the MLP's gate and up projections and their glue in one kernel (2026-11-27).
The K loop of masked BK-wide blocks is the dot product for every BK; every
partial sum, in any order, fits int32 to K = 133,144 (`usable` refuses K past
`qmm.MAX_K`); the sigmoid table's index stays in bounds, the lower clamp
being the load-bearing one; the interpolation fits int32 for the table's
bounds (checked on the built table by python/tests/test_exact_mlp_kernel.py);
the scale's float32 field is normal exactly when the kernel's `ok` holds.
Refutations: one more term can overflow, the index without the lower clamp
is negative, one step outside `ok` the field is zero or all ones.

- On the training path: `_mlp_kernel`'s integer work. Not modelled: its
  float32 roundings (the separate path's operations, tested bit for bit).
- Theorems and lemmas: 19. Quoted below: 3.

```coq
Theorem the_blocked_loop_is_the_dot : forall K BK f, (0 < BK)%nat ->
  sumto ((K + BK - 1) / BK) (fun b => sumto BK (fun j => masked K f (b * BK + j)%nat)) = sumto K f.
```

```coq
Theorem every_partial_sum_fits_int32 : forall ts : list Z,
  Forall (fun t => -16129 <= t <= 16129) ts -> Z.of_nat (length ts) <= 133144 ->
  in_int32 (fold_right Z.add 0 ts).
```

```coq
Theorem the_table_index_is_in_bounds : forall r TN,
  - 2 ^ 24 <= r <= 2 ^ 24 -> 1 <= TN -> 0 <= table_index r TN <= TN - 1.
```

### `proofs/ExactQmmSplit.v`

`python/y_lang/exact/qmm.py`'s in-register split of a 12-bit operand,
`lo = ((a + 64) & 127) - 64`, `hi = (a - lo) >> 7`, is `ExactTrainArith.v`'s
digit split, and `hi * 128 + lo` gives the value back; for |a| <= 2047 both
digits are int8. With K <= `MAX_K` = 131,040 the int32 accumulators hold
their sums in both modes, and the int64 join is below 2^53, so the float64
branch converts it exactly. Refutation: one-digit sums of 133,145 terms of
127 * 127 leave int32.

- On the training path: every `qmm` product. Not modelled: the float32
  epilogue's rounding and Triton's `tl.dot` itself (tested against
  `exact_product` by python/tests/test_exact_qmm.py). The join's margin is
  wide: its largest value reaches 2^53 only between K = 2^35 and 2^36.
- Theorems and lemmas: 9. Quoted below: 3.

```coq
Theorem qmm_lo_is_the_digit : forall a, lo a = TA.digit a.
```

```coq
Theorem the_accumulators_fit_int32 : forall K s,
  0 <= K <= 131040 ->
  (- (K * (64 * 127)) <= s <= K * (64 * 127) -> in_int32 s) /\
  (- (K * (16 * 127)) <= s <= K * (16 * 127) -> in_int32 s) /\
  (- (K * (127 * 127)) <= s <= K * (127 * 127) -> in_int32 s).
```

```coq
Theorem the_join_fails_only_past_2_35 :
  2 ^ 35 * (2047 * 127) < 2 ^ 53 /\ ~ (2 ^ 36 * (2047 * 127) < 2 ^ 53).
```

### `proofs/ExactDrawSearch.v`

`python/y_lang/exact/vocab_draw.py`'s draw never stores p: it sums blocks of
1,024 entries, counts the blocks whose running sum is at most u = r mod l,
recomputes that block and counts its entries the same way. For every block
size, number of blocks and p >= 0 with 0 <= u < l, the token k * B + j is the
flat count `rl.draw_rows` defines, and k and j are in range (the second pass
reads a block that exists). Underneath: a prefix predicate holds below its
count exactly. Refutations: one negative weight makes the two differ;
without `mod l`, u = l counts every block.

- On the sampling path: every fused draw. Not modelled: that the kernel's p
  are `softmax_ints`' (tested bit for bit by
  python/tests/test_exact_vocab_draw.py), int64 overflow of the sums (p <=
  2^28).
- Theorems and lemmas: 23. Quoted below: 1.

```coq
  Theorem the_two_level_search_is_the_flat_one : (k * B + j)%nat = flat.
```

### `proofs/ExactMmPadding.v`

`lora.mm_y_any` and `ExactLinear` pad K, rows and columns with zeros and
slice the result back, and `mm_y_any` runs K past 131,040 in pieces whose
int32 results it adds in int64; the code's reasons were comments. Zero
entries past K add nothing; padded rows and columns change no output the
slice keeps; the pieces add to the whole sum for every K; a piece fits
int32 and the total int64 below 2^43 terms. Refutation: a piece one term
longer than 133,144 can overflow int32.

- On the training path: every product on Y's GEMM or `torch._int_mm`. Not
  modelled: the GEMM itself (`Int8GemmExact.v` for Y's).
- Theorems and lemmas: 8. Quoted below: 2.

```coq
Theorem zero_padding_k_changes_no_sum : forall K e a b a' b' m n,
  (forall k, (k < K)%nat -> a' m k = a m k /\ b' n k = b n k) ->
  (forall k, (K <= k)%nat -> a' m k = 0 \/ b' n k = 0) ->
  dot (K + e) a' b' m n = dot K a b m n.
```

```coq
Theorem the_pieces_of_k_are_the_whole_sum : forall K (f : nat -> Z) (Y : nat), (0 < Y)%nat ->
  MK.sumto ((K + Y - 1) / Y) (fun c => MK.sumto Y (fun x => MK.masked K f (c * Y + x)%nat)) = MK.sumto K f.
```

### `proofs/ExactFloatOrder.v`

The exact quantizers take maxima of |x| on its bits (`exponent`, `q8`, the
fused MLP kernel's row maxima). For every mantissa width (float32 and
float64), finite non-negative floats' bits compare as their values do, so
the maximum of the bits is the bits of the maximum; and the exponent read
from the field bounds the value above, subnormals included, tightly within
a factor of 2 for normal floats. Refutations: with the sign bit the order
breaks; the all-ones field is no finite value.

- On the training path: every quantizer's scale. Not modelled: that
  `.view(int)` gives IEEE's bits (tested on ties, zeros and subnormals).
- Theorems and lemmas: 9. Quoted below: 2.

```coq
Theorem the_bits_are_ordered_as_the_values : forall p E1 M1 E2 M2, 0 <= p ->
  finite p E1 M1 -> finite p E2 M2 ->
  (bits p E1 M1 <= bits p E2 M2 <-> value p E1 M1 <= value p E2 M2).
```

```coq
Theorem the_field_bounds_the_value : forall p E M, 0 <= p -> finite p E M ->
  value p E M < 2 ^ (p + E).
```

### `proofs/ExactCeilLog2.v`

`exact_sum`'s scale uses `_ceil_log2(n)`, computed eagerly as
`(n - 1).bit_length()` and, compiled, by a loop of comparisons. Both are the
least k with 2^k >= n for every n >= 1, so a compiled step and an eager one
get the same scale; the loop ends within n steps. Refutation: floor(log2 n)
differs at n = 3.

- On the training path: every `exact_sum`. Not modelled: Python's
  `bit_length` (its specification) and torch.compile's tracing of the loop
  (compiled-against-eager tests).
- Theorems and lemmas: 6. Quoted below: 1.

```coq
Theorem the_two_definitions_agree : forall n, 1 <= n -> loop (Z.to_nat n) n 0 = bit_length (n - 1).
```

### `proofs/ExactPieces.v`

`rl.pieces` cuts a batch's rows into pieces for `GRPO.step`. For every batch
and bound, the pieces joined are the rows 0 .. n - 1 in order (every row in
exactly one piece, each piece consecutive), none is empty, a piece of two or
more rows is within `chunk_tokens`, and with no bound there is one piece.
Refutation: a single row longer than the bound is a piece past it, which is
all the code promises.

- On the training path: every GRPO step's pieces. Not modelled: that the
  step runs each piece once in order.
- Theorems and lemmas: 9. Quoted below: 2.

```coq
  Theorem the_pieces_are_the_rows_in_order : forall n, concat (pieces n) = seq 0 n.
```

```coq
  Theorem every_piece_of_two_or_more_rows_is_within_the_bound : forall n,
    Forall within (pieces n).
```

### `proofs/ExactQ8Clamp.v`

`q8`'s scale s = e - (bits - 1), with every |x| below 2^e, rounds each
x * 2^-s to the nearest even integer and clamps it to 2^(bits-1) - 1. The
rounding reaches at most one past that top; after the clamp the value is in
range and its error stays below one step 2^s (below half a step where the
clamp does nothing). Refutation: 8 bits, x = 127.5 * 2^s rounds to 128, so
without the clamp top is passed.

- On the training path: every quantizer, as integer rounding. Not modelled:
  the float formula the GPU computes (tested against the integer rounding).
- Theorems and lemmas: 7. Quoted below: 1.

```coq
  Theorem the_clamped_error_is_below_one_step : Z.abs (m - q8_mag bits m r * 2 ^ r) < 2 ^ r.
```

### `proofs/ExactKvPack.v`

The packed KV cache (`python/y_lang/exact/kv.py`, `Y_EXACT_KV_PACK=inplace`)
stores K's int16 values as bytes and each row's float32 scale as four bytes,
made by `_signed_byte` and shifts and read back as int16 or int8. The byte
is the low byte, signed (the `^ 0x80` checked on all 256 bytes) and kept on
a byte; every int16 K value's two bytes are int8 and rebuild it; every
scale's 32 bits rebuild from its four bytes (V) and from two int16 pieces
of them (K). Refutations: an unsigned top piece rebuilds -1 as 2^32 - 1; a
value past int16 has a high byte that is no int8 (`pack_k` refuses such
K_LEVELS).

- On the cached path: every packed K and V row. Not modelled: the float32
  bits' view, the storage's byte order (little-endian on x86 and CUDA), and
  `quantize_rows`' clamp.
- Theorems and lemmas: 13. Quoted below: 2.

```coq
Theorem k_bytes_rebuild_the_value : forall r, -32768 <= r <= 32767 ->
  (-128 <= Z.shiftr r 8 <= 127) /\ (-128 <= signed_byte r <= 127) /\
  Z.shiftr r 8 * 256 + ubyte (signed_byte r) = r.
```

```coq
Theorem v_scale_bytes_rebuild_the_bits : forall b, -2 ^ 31 <= b <= 2 ^ 31 - 1 ->
  rebuild8 (signed_byte b) (signed_byte (Z.shiftr b 8)) (signed_byte (Z.shiftr b 16))
           (signed_byte (Z.shiftr b 24)) = b.
```

### `proofs/ExactNormKernel.v`

The RMS norm kernel (`python/y_lang/exact/norm_kernel.py`) divides once
per row, multiplies per element, rounds the product q to float32 by adding
2^28 to its bits, and sends an element to the true divide when its bits fail
a guard. The guard passes only 0 and q's in float32's normal binades whose
dropped bits are more than 3 from a tie; there the kernel's float32 is
strictly the nearest to every value within 4 ulp64(q) of q. With each
float64 operation within 2^-53 of its exact result, q is within 3.0001 ulps
of the divide, so both round to the kernel's float32. Also: a normal x
widened by its bits keeps its value, and the row's scale `(e + 691) << 52`
is a finite 2^(e - 332). Refutations: with the dropped bits 3 from a tie
the divide can sit on the tie and round the other way, so the guard's 3
cannot be 2; a subnormal x does not widen by its bits.

- On the decode and training paths: every CUDA `ExactRMSNorm` row. Not
  modelled: that each float64 operation rounds to nearest (IEEE, |d| <=
  2^-53 in float64's normal range), the sign bit's plumbing, and Triton's
  bitcasts (tested against the divide on inputs built to sit on midpoints).
- Theorems and lemmas: 25. Quoted below: 2.

```coq
Theorem the_multiply_rounds_as_the_divide : forall E N P (t d1 d2 d3 : Q),
  897 <= E <= 1149 -> 0 <= N < 2 ^ 52 -> 3 < dist (FO.bits 52 E N) ->
  (0 < t)%Q -> (- u64 <= d1 <= u64)%Q -> (- u64 <= d2 <= u64)%Q -> (- u64 <= d3 <= u64)%Q ->
  (inject_Z (v64 E N) == t * (1 + d1) * (1 + d2))%Q ->
  (inject_Z P == t * (1 + d3))%Q ->
  forall e' m', FO.finite 23 e' m' -> FO.bits 23 e' m' <> f32 (FO.bits 52 E N) ->
  Z.abs (v64 E N - G E N) < Z.abs (v64 E N - v32 e' m') /\
  Z.abs (P - G E N) < Z.abs (P - v32 e' m').
```

```coq
Theorem the_guard_admits_only_zero_and_the_normal_binades : forall b, 0 <= b < 2 ^ 63 ->
  unsafe b = false ->
  b = 0 \/ (897 <= b / 2 ^ 52 <= 1149 /\ 0 <= b mod 2 ^ 52 < 2 ^ 52 /\
            b = FO.bits 52 (b / 2 ^ 52) (b mod 2 ^ 52) /\ 3 < dist b).
```

### `proofs/ExactExp2Int64.v`

The integer exp2 (`attention.exp2_neg_q16_16`) runs on torch int64, which
wraps past 2^63, while `ExactDecodeWidths.v` models it on unbounded
integers. For every int64 argument each value the code forms (the clamps,
each product before its shift, the series' partial sums, the rounding
constant and its sum) is in 0 .. 2^63 - 1, so the int64 code computes the
model's integers; `base * corr` is at most 2^60 and the rounded sum below
2^62. `(y * RECIP6_Q32) >> 32` is y / 6 for every y below 2^30 (the Rust
copy's comment claims it below 2^12), so the series' third term is y3 / 6;
at y = 2147483651 it gives one more. Refutations: the product reaches 2^60 (t = 0) and the rounding
constant 2^61 (n >= 60), so the docstring's "below 2^60" and the comment's
"no intermediate reaches 2^61" were both wrong; every product is past
float64's exact range 2^53, and at t = 1025 it is no float64, so the
docstring's "both types represent it exactly" was wrong too (both texts
are corrected with this file).

- On the decode and training paths: every attention weight. Not
  modelled: torch's `>>` as an arithmetic shift and `clamp` as min and max,
  and the table's values (pinned by the whole domain's digest).
- Theorems and lemmas: 23. Quoted below: 2.

```coq
  Theorem every_intermediate_fits_int64 : Forall int64_ok (steps t0).
```

```coq
Theorem every_product_is_past_the_float64_exact_range : forall t0, -2 ^ 63 <= t0 < 2 ^ 63 ->
  2 ^ 53 < base t0 * corr t0.
```

### `proofs/ExactScaledRound.v`

Every float32 quantizer of exact training rounds x * 2^ns by
`scaled_round`'s float32 formula: a = clamp(ns, -126, 127), b = clamp(ns -
a, -126, 127), `round((x * 2^a) * 2^b)`, each product rounded to float32
(2^ns itself is no float32 past 2^127). For every finite float32 x >= 0,
-126 <= ns <= 254 and x * 2^ns below 2^62, the formula is the
round-half-even of x * 2^ns, for every rounding that keeps a float32
unchanged and is monotone (IEEE round to nearest is one): a + b = ns, the
products are exact when the first one is at least 2^-126 (any K * 2^j with
K below 2^24 is a float32), and otherwise both sides are 0. Refutations:
without b, at ns = 140 the product rounds to 0 where the value is 1024; a
rounding that keeps every float32 but is not monotone breaks it.

- On the training path: every float32 quantizer (`q8` and the scaled
  sums). Not modelled: torch's float32 `*` as IEEE round to nearest and
  `round` as half to even (documented; tested against `scaled_round_int`),
  the sign.
- Theorems and lemmas: 9. Quoted below: 2.

```coq
  Theorem the_float32_formula_is_the_exact_rounding : forall E M ns,
    0 <= E <= 254 -> 0 <= M < 2 ^ 23 -> -126 <= ns <= 254 ->
    FO.value 23 E M * 2 ^ (126 + ns) < 2 ^ 62 * 2 ^ 275 ->
    formula E M ns = TA.shift_round (FO.value 23 E M * 2 ^ (126 + ns)) 275.
```

```coq
Theorem one_factor_alone_loses_the_value :
  TA.shift_round (FO.value 23 0 (2 ^ 19) * 2 ^ (126 + 127)) 275 = 0 /\
  TA.shift_round (FO.value 23 0 (2 ^ 19) * 2 ^ (126 + 140)) 275 = 1024.
```

### `proofs/ExactSoftmaxKernel.v`

Exact training's softmax kernel (`python/y_lang/exact/softmax_kernel.py`)
rests on three float32 claims. `_rne_nonneg`'s `(x + 2^23) - 2^23` below
2^23, and x itself above, is the round-half-even of every finite float32
x >= 0 (for a rounding that keeps float32s and rounds to the nearest
integer, ties to even, in [2^23, 2^24)). A row's weights are each in
0 .. 2^28 and sum to 2^28 .. 2^38 (its maximum key gives 2^28, at most 1024
keys), so after any monotone conversion each divide operand is 0 or at
least 1, the divisor at least 2^28, and the rounded quotient 0 or in
2^-38 .. 1: `div_rn`'s subnormal flush has nothing to act on. The row
maximum of P is in 2^-10 .. 1, so q8's scale is s = -16 .. -6, the range
flag cannot fire and `scaled_round`'s factors are a = -s, b = 0.
Refutations: without the branch, x = 2^23 + 1 rounds to 2^23; a row of
2048 keys can sum to 2^39.

- On the training path: every attention softmax on a GPU with Triton.
  Not modelled: IEEE's rounding itself (float32 `+`, `-`, the conversions
  and `div_rn` round to nearest, ties to even), that the row's maximum key
  gets t = 0, and a negative x's sign (the callers clamp it to 0).
- Theorems and lemmas: 15. Quoted below: 2.

```coq
  Theorem the_magic_add_rounds_half_even : forall E M, 0 <= E <= 254 -> 0 <= M < 2 ^ 23 ->
    rne_nonneg (FO.value 23 E M) = TA.shift_round (FO.value 23 E M) 149 * 2 ^ 149.
```

```coq
Theorem the_row_sum_is_between_2_28_and_2_38 : forall ts,
  Forall (fun t => 0 <= t) ts -> In 0 ts -> (length ts <= 1024)%nat ->
  let ps := map DW.exp2_neg_q16_16 ts in
  Forall (fun p => 0 <= p <= 2 ^ 28) ps /\ 2 ^ 28 <= zsum ps <= 2 ^ 38.
```

### `proofs/ExactRopeTable.v`

The RoPE table (`model.cos_sin_f32`) keeps y = fn(arg) * scale, computed
in float64 with the host's cos or sin, rounded to float32, and sends to
mpmath every entry whose y has its low 29 fraction bits within
`ROPE_MARGIN` = 10 of 2^28, is below 2^-126 or is not finite. A libm within
`ROPE_LIBM_ULPS` = 4 ulps and the multiply's rounding leave y less than 11
of its ulps from the true cos(arg) * scale (just above 9), and an entry the
guard passes has its float32 strictly the nearest to every value that
close: each entry is the correctly rounded float32, with no tie. The
window theorem reuses `ExactNormKernel.v`'s neighbours and holds for every
fraction. Refutation: with the distance 4, a libm error within the bound
can put the true value on the midpoint, so a margin of the libm's error
alone would not do.

- On the decode and training paths: every RoPE table entry. Not
  modelled: that the host's libm meets its bound (libms claim 1 ulp), the
  float64 multiply and the rounding to float32 as IEEE's, mpmath's answers
  for the entries it decides, and scale below 2^126.
- Theorems and lemmas: 6. Quoted below: 1.

```coq
Theorem the_rope_entry_is_correctly_rounded : forall E N (c cs s d : Q),
  897 <= E <= 1149 -> 0 <= N < 2 ^ 52 -> rope_margin < NK.dist (FO.bits 52 E N) ->
  (0 < cs)%Q -> (0 < s)%Q -> (- u64 <= d <= u64)%Q ->
  (Qabs (c - cs) <= 4 * (2 * u64) * cs)%Q ->
  (inject_Z (NK.v64 E N) == c * s * (1 + d))%Q ->
  forall e' m', FO.finite 23 e' m' -> FO.bits 23 e' m' <> NK.f32 (FO.bits 52 E N) ->
  (Qabs (cs * s - inject_Z (NK.G E N)) < Qabs (cs * s - inject_Z (NK.v32 e' m')))%Q.
```

### `proofs/ExactRowClamp.v`

Decode's per-row quantizers (`attention.quantize_queries`,
`kv.quantize_rows`, `model._q_per_row`, the W8A8 kernel's `_quant_rows`)
compute s = amax / L and round(x / s), clamped to -L .. L, with IEEE
divides. For a row whose scale is normal and L <= 2^21, every rounded
quotient stays below L + 1/2, so every integer within 1/2 of it is in
-L .. L: the clamp never changes a value. Refutations: with amax = 190 *
2^-149 and L = 127 the scale rounds to the smallest subnormal and x / s is
190, so the clamp is load-bearing exactly for a subnormal scale (the case
the W8A8 kernel's comment records); at L = 2^23 the margin is gone.

- On the decode path: every quantized query, K and V row and
  activation row. Not modelled: IEEE's division (the standard model, in
  the normal range), `round` and `rint` as round to nearest, the zero
  scale's replacement by 1, and the max (`exponent`'s order is
  `ExactFloatOrder.v`'s; this max is a float max).
- Theorems and lemmas: 6. Quoted below: 2.

```coq
  Theorem the_clamp_does_nothing_when_the_scale_is_normal : forall k : Z,
    Qabs (q - inject_Z k) <= 1 # 2 -> (- L <= k <= L)%Z.
```

```coq
Theorem a_subnormal_scale_needs_the_clamp :
  (2 * Z.abs (190 - 127 * 1) < 127 /\ 1 < 2 ^ 23 /\ 190 / 1 > 127)%Z.
```

### `proofs/ExactVFold.v`

`kv.fold_scale_into_weights` turns a query's Q0.28 weights p and the
keys' float32 V scales sv into integer weights W = round(w * 2^28 / wmax),
w = p * sv in float64, clamped to 0 .. 2^28. Any K * 2^j with K below
2^(p + 1) is a float of mantissa width p, so p * sv (29 bits times 24) is
itself a float64: w and its maximum are exact. With the two float64
roundings, w * c stays below 2^28 + 1/2, so every rounding of it is in
0 .. 2^28 and the clamp never changes a value. Refutation: with float32's
u the same bound reaches 2^28 + 32, so the fold needs float64.

- On the decode path: every attention row's V weights. Not modelled:
  IEEE's float64 multiply and divide (the standard model, in the normal
  range), `round` as round to nearest.
- Theorems and lemmas: 5. Quoted below: 2.

```coq
Theorem the_weight_product_is_exact_in_float64 : forall P e m,
  0 <= P <= 2 ^ 28 -> 0 <= e <= 254 -> 0 <= m < 2 ^ 23 ->
  exists E N, 0 <= E <= 2046 /\ 0 <= N < 2 ^ 52 /\
    FO.value 52 E N = P * (FO.value 23 e m * 2 ^ 925).
```

```coq
  Theorem the_fold_stays_within_2_28 : forall W : Z,
    Qabs (y - inject_Z W) <= 1 # 2 -> (0 <= W <= 2 ^ 28)%Z.
```

## Still open

- Not modelled: rows with an infinity or a NaN, the prefill as one pass.
  Ragged batches are covered by `left_padding_changes_nothing`, under
  run-level conditions that hold for finite values. Triton's interleaved
  two-digit kernel is covered by `sched_dot2_exact` (since 2026-11-15).
- The executable check has run Triton's interleaved kernel on one sm_89
  GPU (it needs sm_80 or later), and Y's decode-attention kernel
  (`Y_EXACT_ATTN=y`) on a Tesla T4 and that GPU; on no other GPUs.

The trust boundary, the executable check's results and the negative
controls are in [`docs/exact_decode_theorem.md`](exact_decode_theorem.md).
