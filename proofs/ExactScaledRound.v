(** * `scaled_round`'s float32 formula (`python/y_lang/exact/train_lm.py`).

    Every float32 quantizer of exact training (`q8`, the scaled sums) needs
    round-half-even(x * 2^ns) for a float32 x and an integer ns = -s. A power
    of two past 2^127 is no float32, so `scaled_round` multiplies by two:

        a = clamp(ns, -126, 127);  b = clamp(ns - a, -126, 127)
        y = round((x * 2^a) * 2^b)

    each product rounded to float32. Its docstring argues this is exact for
    every ns the callers make (ns >= -126): each factor is a normal power of
    two, the products are exact unless the first underflows, and then the
    value is below 2^-126, whose round is 0 either way. This file proves it,
    for every rounding [rn] that keeps a float32 unchanged ([rn_fix]) and is
    monotone ([rn_mono]): IEEE round to nearest is one, so are the directed
    roundings.

    Units: values are integers times 2^-275 (2^-149, float32's step, times
    2^-126, the smallest factor), so a float32 with fields (E, M) is
    `ExactFloatOrder.value 23 E M * 2^126`. Magnitudes: x >= 0 (the code's
    products and `round` treat the sign symmetrically).

    [the_float32_formula_is_the_exact_rounding]: for every finite float32
    x >= 0 and -126 <= ns <= 254 with x * 2^ns below 2^62, the formula is
    `ExactTrainArith.shift_round (x * 2^ns) 275`, the round-half-even of
    x * 2^ns (`ExactTrainArith.shift_round_is_nearest`).
    [the_two_factors_are_the_scale]: a + b = ns and 0 <= b, for every such ns.
    [an_integer_times_a_power_of_two_is_a_float32]: the representability fact
    under the exact cases, K * 2^j for K below 2^24.

    ** Refutations.

    [one_factor_alone_loses_the_value]: without b, at ns = 140 the product
    x * 2^127 rounds to 0 where x * 2^140 is 1024 (x = 2^-130).
    [a_rounding_that_is_not_monotone_breaks_it]: an [rn] that keeps every
    float32 but sends one product below 2^-126 to 1 makes the formula give 1
    where the exact rounding gives 0, so the monotone hypothesis is needed.

    ** Not proved here.

    That torch's float32 `*` rounds to nearest (IEEE) and `round` is half to
    even on a float32 (its documentation; the tests compare the formula with
    `scaled_round_int`'s integers on ties, zeros, subnormals and extreme
    scales); the sign.

    Checked with Rocq 9.1:  coqc -Q . "" ExactScaledRound.v  *)

From Stdlib Require Import ZArith Lia.
Require ExactFloatOrder ExactTrainArith.
Open Scope Z_scope.

Module FO := ExactFloatOrder.
Module TA := ExactTrainArith.

Definition clamp (lo hi v : Z) : Z := Z.min (Z.max v lo) hi.

Theorem the_two_factors_are_the_scale : forall ns, -126 <= ns <= 254 ->
  let a := clamp (-126) 127 ns in
  let b := clamp (-126) 127 (ns - a) in
  a + b = ns /\ 0 <= b /\ -126 <= a <= 127 /\ b <= 127 /\ (ns <= 127 -> a = ns /\ b = 0).
Proof. intros ns H. cbv zeta. unfold clamp. lia. Qed.

(** A finite float32 is K * 2^c with K below 2^24. *)
Lemma split_value : forall E M, 0 <= E -> 0 <= M < 2 ^ 23 ->
  FO.value 23 E M = (if E =? 0 then M else 2 ^ 23 + M) * 2 ^ (Z.max (E - 1) 0) /\
  0 <= (if E =? 0 then M else 2 ^ 23 + M) < 2 ^ 24.
Proof.
  intros E M HE HM. unfold FO.value. destruct (E =? 0) eqn:Z0.
  - apply Z.eqb_eq in Z0. subst E. split; [|lia]. rewrite Z.max_r by lia. ring.
  - apply Z.eqb_neq in Z0. split; [|lia]. rewrite Z.max_l by lia. reflexivity.
Qed.

Theorem an_integer_times_a_power_of_two_is_a_float32 : forall K j,
  0 <= K < 2 ^ 24 -> 0 <= j -> K * 2 ^ j < 2 ^ 211 ->
  exists E M, 0 <= E <= 254 /\ 0 <= M < 2 ^ 23 /\ FO.value 23 E M = K * 2 ^ j.
Proof.
  intros K j HK Hj Hb.
  destruct (Z_lt_le_dec (K * 2 ^ j) (2 ^ 23)) as [Ls|Ge].
  - exists 0, (K * 2 ^ j). assert (0 <= K * 2 ^ j) by (apply Z.mul_nonneg_nonneg; [lia | apply Z.pow_nonneg; lia]).
    repeat split; try lia; try reflexivity.
  - assert (HK0 : 0 < K) by (destruct (Z.eq_dec K 0) as [->|]; [cbn in Ge; lia | lia]).
    set (lk := Z.log2 K).
    destruct (Z.log2_spec K HK0) as [L1 L2]. fold lk in L1, L2.
    assert (Hlk : 0 <= lk <= 23).
    { split; [apply Z.log2_nonneg|]. destruct (Z_le_gt_dec lk 23) as [|G]; [lia|].
      assert (2 ^ 24 <= 2 ^ lk) by (apply Z.pow_le_mono_r; lia). lia. }
    assert (P1 : 2 ^ (lk + 1) = 2 * 2 ^ lk) by (rewrite Z.pow_add_r by lia; ring).
    assert (Hge : 23 <= lk + j).
    { destruct (Z_le_gt_dec 23 (lk + j)) as [|G]; [lia|].
      assert (K * 2 ^ j < 2 ^ (lk + 1) * 2 ^ j) by (apply Z.mul_lt_mono_pos_r; [apply Z.pow_pos_nonneg; lia | lia]).
      rewrite <- Z.pow_add_r in H by lia. assert (2 ^ (lk + 1 + j) <= 2 ^ 23) by (apply Z.pow_le_mono_r; lia). lia. }
    exists (lk + j - 22), (K * 2 ^ (23 - lk) - 2 ^ 23).
    assert (S1 : 2 ^ lk * 2 ^ (23 - lk) = 2 ^ 23) by (rewrite <- Z.pow_add_r by lia; f_equal; lia).
    assert (S2 : 2 ^ (23 - lk) > 0) by (apply Z.lt_gt, Z.pow_pos_nonneg; lia).
    assert (M1 : 2 ^ 23 <= K * 2 ^ (23 - lk)) by nia.
    assert (M2 : K * 2 ^ (23 - lk) < 2 ^ 24).
    { assert (K * 2 ^ (23 - lk) < 2 ^ (lk + 1) * 2 ^ (23 - lk)) by nia.
      rewrite <- Z.pow_add_r in H by lia. replace (lk + 1 + (23 - lk)) with 24 in H by lia. exact H. }
    assert (HE : lk + j - 22 <= 254).
    { destruct (Z_le_gt_dec (lk + j - 22) 254) as [|G]; [lia|].
      assert (2 ^ lk * 2 ^ j <= K * 2 ^ j) by (apply Z.mul_le_mono_nonneg_r; [apply Z.pow_nonneg; lia | lia]).
      rewrite <- Z.pow_add_r in H by lia. assert (2 ^ 211 <= 2 ^ (lk + j)) by (apply Z.pow_le_mono_r; lia). lia. }
    repeat split; try lia.
    unfold FO.value. destruct (lk + j - 22 =? 0) eqn:Z0; [apply Z.eqb_eq in Z0; lia|].
    replace (2 ^ 23 + (K * 2 ^ (23 - lk) - 2 ^ 23)) with (K * 2 ^ (23 - lk)) by ring.
    rewrite <- Z.mul_assoc, <- Z.pow_add_r by lia. f_equal. f_equal. lia.
Qed.

Lemma shift_round_small : forall w, 0 <= w <= 2 ^ 149 -> TA.shift_round w 275 = 0.
Proof.
  intros w Hw. unfold TA.shift_round.
  assert (L : 2 ^ 149 < 2 ^ 274) by (apply Z.pow_lt_mono_r; lia).
  assert (L2 : 2 ^ 274 < 2 ^ 275) by (apply Z.pow_lt_mono_r; lia).
  rewrite Z.div_small, Z.mod_small by lia. replace (275 - 1) with 274 by lia.
  destruct (2 ^ 274 <? w) eqn:A; [apply Z.ltb_lt in A; lia|].
  destruct (w =? 2 ^ 274) eqn:B; [apply Z.eqb_eq in B; lia|]. reflexivity.
Qed.

Section ScaledRound.
  Variable rn : Z -> Z.        (** a float32 rounding, on values in units of 2^-275 *)
  Hypothesis rn_fix : forall E M, 0 <= E <= 254 -> 0 <= M < 2 ^ 23 ->
    rn (FO.value 23 E M * 2 ^ 126) = FO.value 23 E M * 2 ^ 126.
  Hypothesis rn_mono : forall v w, v <= w -> rn v <= rn w.

  (** `round((x * 2^a) * 2^b)`, x the float32 with fields (E, M). *)
  Definition formula (E M ns : Z) : Z :=
    let a := clamp (-126) 127 ns in
    let b := clamp (-126) 127 (ns - a) in
    TA.shift_round (rn (rn (FO.value 23 E M * 2 ^ (126 + a)) * 2 ^ b)) 275.

  Lemma rn_float : forall K j, 0 <= K < 2 ^ 24 -> 0 <= j -> K * 2 ^ j < 2 ^ 211 ->
    rn (K * 2 ^ (j + 126)) = K * 2 ^ (j + 126).
  Proof.
    intros K j HK Hj Hb. destruct (an_integer_times_a_power_of_two_is_a_float32 K j HK Hj Hb) as [E [M [HE [HM V]]]].
    rewrite Z.pow_add_r by lia. rewrite Z.mul_assoc, <- V. apply rn_fix; lia.
  Qed.

  Lemma rn_small : forall v, 0 <= v <= 2 ^ 149 -> 0 <= rn v <= 2 ^ 149.
  Proof.
    intros v Hv.
    assert (Z0 : rn 0 = 0) by (exact (rn_fix 0 0 ltac:(lia) ltac:(lia))).
    assert (T : rn (2 ^ 149) = 2 ^ 149).
    { assert (E : 2 ^ 149 = FO.value 23 1 0 * 2 ^ 126) by reflexivity.
      rewrite E. apply rn_fix; lia. }
    pose proof (rn_mono 0 v ltac:(lia)). pose proof (rn_mono v (2 ^ 149) ltac:(lia)). lia.
  Qed.

  Theorem the_float32_formula_is_the_exact_rounding : forall E M ns,
    0 <= E <= 254 -> 0 <= M < 2 ^ 23 -> -126 <= ns <= 254 ->
    FO.value 23 E M * 2 ^ (126 + ns) < 2 ^ 62 * 2 ^ 275 ->
    formula E M ns = TA.shift_round (FO.value 23 E M * 2 ^ (126 + ns)) 275.
  Proof.
    intros E M ns HE HM Hns Hb. unfold formula.
    destruct (the_two_factors_are_the_scale ns Hns) as [Sab [Hb0 [Ha [Hb1 Hlow]]]].
    set (a := clamp (-126) 127 ns) in *. set (b := clamp (-126) 127 (ns - a)) in *.
    destruct (split_value E M ltac:(lia) HM) as [V HK].
    set (K := if E =? 0 then M else 2 ^ 23 + M) in *. set (c := Z.max (E - 1) 0) in *.
    assert (Hc : 0 <= c) by (unfold c; lia).
    rewrite V. rewrite <- !Z.mul_assoc, <- !Z.pow_add_r by lia.
    assert (Hb' : (0 <= c + ns /\ K * 2 ^ (c + ns) < 2 ^ 211) \/ c + ns < 0).
    { destruct (Z_lt_le_dec (c + ns) 0) as [|G]; [right; lia | left; split; [lia|]].
      rewrite V, <- Z.mul_assoc, <- Z.pow_add_r in Hb by lia.
      replace (c + (126 + ns)) with ((c + ns) + 126) in Hb by lia. rewrite Z.pow_add_r in Hb by lia.
      assert (P : 2 ^ 62 * 2 ^ 275 = 2 ^ 211 * 2 ^ 126) by (rewrite <- !Z.pow_add_r by lia; reflexivity).
      rewrite P, Z.mul_assoc in Hb. assert (H126 : 0 < 2 ^ 126) by (apply Z.pow_pos_nonneg; lia).
      exact (proj2 (Z.mul_lt_mono_pos_r _ _ _ H126) Hb). }
    destruct (Z_le_gt_dec ns 127) as [Lo|Hi].
    - (* one factor: a = ns, b = 0 *)
      destruct (Hlow Lo) as [Ea Eb]. rewrite Ea, Eb, Z.pow_0_r, Z.mul_1_r.
      destruct Hb' as [[Nn Fit]|Neg].
      + replace (c + (126 + ns)) with ((c + ns) + 126) by lia.
        rewrite rn_float, rn_float by lia. reflexivity.
      + (* the product is below 2^-126: both round to 0 *)
        assert (Small : 0 <= K * 2 ^ (c + (126 + ns)) <= 2 ^ 149).
        { split; [apply Z.mul_nonneg_nonneg; [lia | apply Z.pow_nonneg; lia]|].
          assert (2 ^ (c + (126 + ns)) <= 2 ^ 125) by (apply Z.pow_le_mono_r; lia).
          assert (2 ^ 24 * 2 ^ 125 = 2 ^ 149) by (rewrite <- Z.pow_add_r by lia; reflexivity).
          assert (0 <= 2 ^ (c + (126 + ns))) by (apply Z.pow_nonneg; lia). nia. }
        pose proof (rn_small _ Small) as R1. pose proof (rn_small _ R1) as R2.
        rewrite !shift_round_small by lia. reflexivity.
    - (* two factors: a = 127, b = ns - 127, both products exact *)
      assert (Ea : a = 127) by (unfold a, clamp; lia). assert (Eb : b = ns - 127) by (unfold b; rewrite Ea; unfold clamp; lia).
      rewrite Ea, Eb. destruct Hb' as [[Nn Fit]|Neg]; [|lia].
      assert (F1 : K * 2 ^ (c + 127) < 2 ^ 211).
      { assert (2 ^ (c + 127) <= 2 ^ (c + ns)) by (apply Z.pow_le_mono_r; lia). nia. }
      replace (c + (126 + 127)) with ((c + 127) + 126) by lia. rewrite rn_float by lia.
      rewrite <- Z.mul_assoc, <- Z.pow_add_r by lia.
      replace (c + 127 + 126 + (ns - 127)) with ((c + ns) + 126) by lia. rewrite rn_float by lia.
      f_equal. f_equal. f_equal. lia.
  Qed.
End ScaledRound.

(** ** Refutations. *)

(** x = 2^-130: the subnormal with M = 2^19. *)
Theorem one_factor_alone_loses_the_value :
  TA.shift_round (FO.value 23 0 (2 ^ 19) * 2 ^ (126 + 127)) 275 = 0 /\
  TA.shift_round (FO.value 23 0 (2 ^ 19) * 2 ^ (126 + 140)) 275 = 1024.
Proof. vm_compute. split; reflexivity. Qed.

(** x = (1 + 2^-23) / 2 (fields 126, 1) at ns = -126: the product
    x * 2^-126 needs one bit more than a subnormal has, 2^148 + 2^125 in
    these units. [bad_rn] keeps every float32 (those are multiples of 2^126)
    and sends that product to 1. *)
Definition p_bad : Z := 2 ^ 148 + 2 ^ 125.
Definition bad_rn (v : Z) : Z := if v =? p_bad then 2 ^ 275 else v.

Theorem a_rounding_that_is_not_monotone_breaks_it :
  (forall E M, bad_rn (FO.value 23 E M * 2 ^ 126) = FO.value 23 E M * 2 ^ 126) /\
  p_bad < 2 ^ 149 /\ bad_rn p_bad > bad_rn (2 ^ 149) /\
  formula bad_rn 126 1 (-126) = 1 /\
  TA.shift_round (FO.value 23 126 1 * 2 ^ (126 + -126)) 275 = 0.
Proof.
  split.
  - intros E M. unfold bad_rn. destruct (FO.value 23 E M * 2 ^ 126 =? p_bad) eqn:X; [|reflexivity].
    apply Z.eqb_eq in X. exfalso.
    assert (A : (FO.value 23 E M * 2 ^ 126) mod 2 ^ 126 = 0) by (apply Z_mod_mult).
    rewrite X in A. vm_compute in A. discriminate A.
  - vm_compute. repeat split; try reflexivity; discriminate.
Qed.

Print Assumptions the_two_factors_are_the_scale.
Print Assumptions an_integer_times_a_power_of_two_is_a_float32.
Print Assumptions the_float32_formula_is_the_exact_rounding.
Print Assumptions one_factor_alone_loses_the_value.
Print Assumptions a_rounding_that_is_not_monotone_breaks_it.
