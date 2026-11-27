(** * The integer exp2 in int64 (`python/y_lang/exact/attention.py`).

    `exp2_neg_q16_16(t, table)` gives `2^-t` (t in Q16.16) as a Q0.28 weight
    by a table of 64 values, a three-term series and shifts, on torch int64
    tensors. `ExactDecodeWidths.v` proves its range on unbounded integers
    (`the_integer_exp_is_a_q28_weight`), and `ExactDecodeStep.v` uses that
    model. int64 wraps past 2^63, so the model is the code only if no value
    the code forms leaves int64. A comment there and the function's docstring
    both say so ("no intermediate reaches 2^61", "every intermediate here is
    an integer below 2^60"); this file proves the bound the code needs and
    refutes both numbers.

    [steps] lists every value the Python forms, in its order (the clamps,
    each product before its shift, the series' partial sums, the rounding
    constant `1 << (sh - 1)` and the sum it is added to).
    [every_intermediate_fits_int64]: for every int64 t, each of them is in
    0 .. 2^63 - 1, so torch's int64 operations give the integers the model
    computes: no product wraps, and `>>` and `&` on a non-negative value are
    floor division and the remainder.
    [the_largest_values]: the product `base * corr` is at most 2^60 and the
    rounded sum below 2^62. [the_steps_are_the_model]: the last step is
    `ExactDecodeWidths.exp2_neg_q16_16` wherever the code keeps it (n < 30).
    [the_reciprocal_multiply_is_division_by_six]: `(y * RECIP6_Q32) >> 32` is
    `y / 6` for every y below 2^30 (`src/fixed_exp.rs` says so for y3 below
    2^12), so [the_series_divides_by_six]: the series' third term is y3 / 6.

    ** Refutations.

    [the_product_reaches_2_60] (t = 0) and
    [the_rounding_constant_reaches_2_61] (n >= 60): neither "below 2^60" nor
    "no intermediate reaches 2^61" holds.
    [every_product_is_past_the_float64_exact_range]: for every t, `base * corr`
    is above 2^53, and [float64_would_round_the_product] at t = 1025: it is
    no float64 (a float64 in [2^59, 2^60) is a multiple of 2^7). The
    docstring's "both types represent it exactly" is false; a float64 replica
    would need its own argument. (Measured on the whole domain 0 .. 31 << 16:
    1,937,810 of the 2,031,616 products are not float64s, though no output
    differs.)
    [the_reciprocal_multiply_fails_past_2_31]: at y = 2147483651 the
    multiply gives one more than y / 6, so the bound on y is needed.

    ** Not proved here.

    That torch's int64 `>>` is an arithmetic shift and `clamp` a min or max
    (their documented meaning); the table's values (pinned with the whole
    domain's digest, `EXP2_DOMAIN_DIGEST`).

    Checked with Rocq 9.1:  coqc -Q . "" ExactExp2Int64.v  *)

From Stdlib Require Import ZArith Lia List.
Import ListNotations.
Require ExactDecodeConstants ExactDecodeWidths.
Open Scope Z_scope.

Module C := ExactDecodeConstants.
Module DW := ExactDecodeWidths.

Definition int64_ok (x : Z) : Prop := 0 <= x < 2 ^ 63.

(** Every value `exp2_neg_q16_16` forms, in its order. *)
Definition steps (t0 : Z) : list Z :=
  let t := Z.max t0 0 in
  let n := Z.shiftr t 16 in
  let f := Z.land t 65535 in
  let idx := Z.min (Z.max (Z.shiftr f 10) 0) 63 in
  let base := nth (Z.to_nat idx) C.EXP_TABLE 0 in
  let d := Z.land f 1023 in
  let dl := d * C.LN2_Q30 in
  let y := Z.shiftr dl 16 in
  let yy := y * y in
  let y2 := Z.shiftr yy 30 in
  let y2y := y2 * y in
  let y3 := Z.shiftr y2y 30 in
  let y3r := y3 * C.RECIP6_Q32 in
  let y3o6 := Z.shiftr y3r 32 in
  let c1 := 2 ^ 30 - y in
  let c2 := c1 + Z.shiftr y2 1 in
  let corr := c2 - y3o6 in
  let bc := base * corr in
  let g := Z.shiftr bc 30 in
  let sh := Z.min (2 + n) 62 in
  let half := Z.shiftl 1 (sh - 1) in
  let s := g + half in
  [t; n; f; Z.shiftr f 10; idx; base; d; dl; y; yy; y2; y2y; y3; y3r; y3o6;
   c1; Z.shiftr y2 1; c2; corr; bc; g; 2 + n; sh; sh - 1; half; s; Z.shiftr s sh].

Lemma pow_facts : 2 ^ 16 = 65536 /\ 2 ^ 24 = 16777216 /\ 2 ^ 30 = 1073741824 /\
  2 ^ 32 = 4294967296 /\ 2 ^ 47 = 140737488355328 /\ 2 ^ 60 = 1152921504606846976 /\
  2 ^ 61 = 2305843009213693952 /\ 2 ^ 62 = 4611686018427387904 /\ 2 ^ 63 = 9223372036854775808.
Proof. repeat split; reflexivity. Qed.

Lemma nonneg_shift : forall a s, 0 <= a -> 0 <= s -> 0 <= Z.shiftr a s <= a.
Proof. exact DW.shiftr_bounds. Qed.

Section Bounds.
  Variable t0 : Z.
  Hypothesis Ht0 : -2 ^ 63 <= t0 < 2 ^ 63.

  Definition t := Z.max t0 0.
  Definition n := Z.shiftr t 16.
  Definition f := Z.land t 65535.
  Definition idx := Z.min (Z.max (Z.shiftr f 10) 0) 63.
  Definition base := nth (Z.to_nat idx) C.EXP_TABLE 0.
  Definition d := Z.land f 1023.
  Definition y := Z.shiftr (d * C.LN2_Q30) 16.
  Definition y2 := Z.shiftr (y * y) 30.
  Definition y3 := Z.shiftr (y2 * y) 30.
  Definition y3o6 := Z.shiftr (y3 * C.RECIP6_Q32) 32.
  Definition corr := 2 ^ 30 - y + Z.shiftr y2 1 - y3o6.
  Definition g := Z.shiftr (base * corr) 30.
  Definition sh := Z.min (2 + n) 62.

  Lemma b_t : 0 <= t < 2 ^ 63.
  Proof. unfold t. lia. Qed.

  Lemma b_n : 0 <= n < 2 ^ 47.
  Proof.
    pose proof b_t. pose proof pow_facts. unfold n. rewrite Z.shiftr_div_pow2 by lia. split.
    - apply Z.div_pos; lia.
    - apply Z.div_lt_upper_bound; lia.
  Qed.

  Lemma b_f : 0 <= f < 2 ^ 16.
  Proof.
    pose proof b_t. unfold f. replace 65535 with (Z.ones 16) by reflexivity.
    rewrite Z.land_ones by lia. apply Z.mod_pos_bound. lia.
  Qed.

  Lemma b_d : 0 <= d <= 1023.
  Proof.
    pose proof b_f. unfold d. change 1023 with (Z.ones 10). rewrite Z.land_ones by lia.
    pose proof (Z.mod_pos_bound f (2 ^ 10) ltac:(lia)).
    change (2 ^ 10) with 1024 in *. change (Z.ones 10) with 1023. lia.
  Qed.

  Lemma b_y : 0 <= y <= 2 ^ 24.
  Proof.
    pose proof b_d. pose proof pow_facts. unfold y, C.LN2_Q30. rewrite Z.shiftr_div_pow2 by lia. split.
    - apply Z.div_pos; lia.
    - apply Z.div_le_upper_bound; lia.
  Qed.

  Lemma b_series : 0 <= y2 <= y /\ 0 <= y3 <= y2 /\ 0 <= y3o6 <= y3.
  Proof.
    pose proof b_y. pose proof pow_facts.
    assert (A : 0 <= y2 <= y) by (unfold y2; apply DW.shiftr_mul_le; lia).
    assert (B : 0 <= y3 <= y2) by (unfold y3; apply DW.shiftr_mul_le; lia).
    split; [exact A | split; [exact B|]].
    unfold y3o6, C.RECIP6_Q32. apply DW.shiftr_mul_le; lia.
  Qed.

  Lemma b_corr : 0 <= corr <= 2 ^ 30.
  Proof.
    pose proof b_y. pose proof b_series as [A [B C3]]. pose proof pow_facts.
    pose proof (nonneg_shift y2 1 ltac:(lia) ltac:(lia)).
    unfold corr. lia.
  Qed.

  Lemma b_base : 0 <= base <= 2 ^ 30.
  Proof. exact (DW.exp_table_nth _). Qed.

  Lemma b_g : 0 <= g <= 2 ^ 30.
  Proof.
    pose proof b_base. pose proof b_corr. pose proof pow_facts.
    unfold g. rewrite Z.shiftr_div_pow2 by lia. split.
    - apply Z.div_pos; nia.
    - apply Z.div_le_upper_bound; nia.
  Qed.

  Lemma b_sh : 2 <= sh <= 62.
  Proof. pose proof b_n. unfold sh. lia. Qed.

  Theorem every_intermediate_fits_int64 : Forall int64_ok (steps t0).
  Proof.
    pose proof b_t. pose proof b_n. pose proof b_f. pose proof b_d. pose proof b_y.
    pose proof b_series as [A [B C3]]. pose proof b_corr. pose proof b_base. pose proof b_g. pose proof b_sh.
    pose proof pow_facts as P. destruct P as [P16 [P24 [P30 [P32 [P47 [P60 [P61 [P62 P63]]]]]]]].
    assert (H10 : 0 <= Z.shiftr f 10 <= f) by (apply nonneg_shift; lia).
    assert (Hdl : 0 <= d * C.LN2_Q30 <= 1023 * 744261118) by (unfold C.LN2_Q30; nia).
    assert (Hyy : 0 <= y * y <= 2 ^ 48) by (change (2 ^ 48) with (2 ^ 24 * 2 ^ 24); nia).
    assert (Hy2y : 0 <= y2 * y <= 2 ^ 48) by (change (2 ^ 48) with (2 ^ 24 * 2 ^ 24); nia).
    assert (Hy3r : 0 <= y3 * C.RECIP6_Q32 <= 2 ^ 24 * 715827883) by (unfold C.RECIP6_Q32; nia).
    assert (Hh : 0 <= Z.shiftr y2 1 <= y2) by (apply nonneg_shift; lia).
    assert (Hbc : 0 <= base * corr <= 2 ^ 60) by (change (2 ^ 60) with (2 ^ 30 * 2 ^ 30); nia).
    assert (Hhalf : Z.shiftl 1 (sh - 1) = 2 ^ (sh - 1)) by (apply Z.shiftl_1_l).
    assert (Hp : 2 <= 2 ^ (sh - 1) <= 2 ^ 61).
    { split.
      - change 2 with (2 ^ 1) at 1. apply Z.pow_le_mono_r; lia.
      - apply Z.pow_le_mono_r; lia. }
    assert (Hs : 0 <= Z.shiftr (g + 2 ^ (sh - 1)) sh <= g + 2 ^ (sh - 1)) by (apply nonneg_shift; lia).
    unfold steps. fold t n f idx base d. fold y. fold y2. fold y3. fold y3o6.
    replace (2 ^ 30 - y + Z.shiftr y2 1 - y3o6) with corr by reflexivity. fold g sh. rewrite Hhalf.
    unfold int64_ok. assert (0 <= idx <= 63) by (unfold idx; lia).
    repeat constructor; lia.
  Qed.

  Theorem the_largest_values : 0 <= base * corr <= 2 ^ 60 /\ g + Z.shiftl 1 (sh - 1) < 2 ^ 62.
  Proof.
    pose proof b_corr. pose proof b_base. pose proof b_g. pose proof b_sh. pose proof pow_facts.
    rewrite Z.shiftl_1_l. assert (2 ^ (sh - 1) <= 2 ^ 61) by (apply Z.pow_le_mono_r; lia).
    split; [change (2 ^ 60) with (2 ^ 30 * 2 ^ 30); nia | lia].
  Qed.
End Bounds.

Theorem the_steps_are_the_model : forall t0,
  Z.shiftr (Z.max t0 0) 16 < 30 -> DW.exp2_neg_q16_16 t0 = last (steps t0) 0.
Proof.
  intros t0 H. unfold DW.exp2_neg_q16_16.
  destruct (30 <=? Z.shiftr (Z.max t0 0) 16) eqn:E; [apply Z.leb_le in E; lia|]. reflexivity.
Qed.

(** `RECIP6_Q32` is 2^32 / 6 rounded up (6 times it is 2^32 + 2), so the
    multiply and shift is division by 6 while the rounding up stays below one
    sixth: for every y below 2^30, far past `y3`'s 2^24. *)
Theorem the_reciprocal_multiply_is_division_by_six : forall y, 0 <= y < 2 ^ 30 ->
  Z.shiftr (y * C.RECIP6_Q32) 32 = y / 6.
Proof.
  intros y Hy. unfold C.RECIP6_Q32. rewrite Z.shiftr_div_pow2 by lia.
  pose proof (Z.div_mod y 6 ltac:(lia)). pose proof (Z.mod_pos_bound y 6 ltac:(lia)).
  set (q := y / 6) in *. set (r := y mod 6) in *.
  change (2 ^ 32) with 4294967296. symmetry. apply Z.div_unique with (2 * q + r * 715827883); lia.
Qed.

Corollary the_series_divides_by_six : forall t0, -2 ^ 63 <= t0 < 2 ^ 63 -> y3o6 t0 = y3 t0 / 6.
Proof.
  intros t0 Ht0. pose proof (b_y t0 Ht0). pose proof (b_series t0 Ht0) as [A [B _]]. pose proof pow_facts.
  unfold y3o6. apply the_reciprocal_multiply_is_division_by_six. lia.
Qed.

(** ** Refutations. *)

Theorem the_product_reaches_2_60 : nth 19 (steps 0) 0 = 2 ^ 60.
Proof. vm_compute. reflexivity. Qed.

Theorem the_rounding_constant_reaches_2_61 : nth 24 (steps (60 * 2 ^ 16)) 0 = 2 ^ 61.
Proof. vm_compute. reflexivity. Qed.

Lemma table_above_2_29 : forall i, (i < 64)%nat -> 2 ^ 29 < nth i C.EXP_TABLE 0.
Proof.
  intros i Hi.
  assert (H : forallb (fun j => 2 ^ 29 <? nth j C.EXP_TABLE 0) (seq 0 64) = true) by (vm_compute; reflexivity).
  rewrite forallb_forall in H. specialize (H i ltac:(apply in_seq; lia)). apply Z.ltb_lt in H. exact H.
Qed.

Theorem every_product_is_past_the_float64_exact_range : forall t0, -2 ^ 63 <= t0 < 2 ^ 63 ->
  2 ^ 53 < base t0 * corr t0.
Proof.
  intros t0 Ht0. pose proof (b_y t0 Ht0) as Hy. pose proof (b_series t0 Ht0) as [A [B C3]].
  pose proof pow_facts. pose proof (nonneg_shift (y2 t0) 1 ltac:(lia) ltac:(lia)).
  assert (Hc : 2 ^ 30 - 2 ^ 24 - 2 ^ 24 <= corr t0) by (unfold corr; lia).
  assert (Hb : 2 ^ 29 < base t0).
  { unfold base. apply table_above_2_29. unfold idx. lia. }
  change (2 ^ 53) with (2 ^ 29 * 2 ^ 24). nia.
Qed.

Theorem float64_would_round_the_product :
  let p := nth 19 (steps 1025) 0 in 2 ^ 59 <= p < 2 ^ 60 /\ p mod 2 ^ 7 <> 0.
Proof. cbv zeta. split; [split|]; vm_compute; try reflexivity; intro H; discriminate H. Qed.

Theorem the_reciprocal_multiply_fails_past_2_31 :
  Z.shiftr (2147483651 * C.RECIP6_Q32) 32 <> 2147483651 / 6.
Proof. vm_compute. discriminate. Qed.

Print Assumptions every_intermediate_fits_int64.
Print Assumptions the_reciprocal_multiply_is_division_by_six.
Print Assumptions the_series_divides_by_six.
Print Assumptions the_reciprocal_multiply_fails_past_2_31.
Print Assumptions the_largest_values.
Print Assumptions the_steps_are_the_model.
Print Assumptions the_product_reaches_2_60.
Print Assumptions the_rounding_constant_reaches_2_61.
Print Assumptions every_product_is_past_the_float64_exact_range.
Print Assumptions float64_would_round_the_product.
