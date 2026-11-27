(** * `_scaled_round_f64`, the float64 formula (`python/y_lang/exact/train_lm.py`).

    On a GPU whose float64 runs at half float32's rate (`FAST_FLOAT64`:
    P100, V100, A100, H100, B200), `scaled_round` of a float64 x computes in
    float64 what `scaled_round_int` computes in integers:

        a = clamp(ns, -1022, 1023);  b = clamp(ns - a, -1022, 1023)
        y = round((x * 2^a) * 2^b)

    Its docstring argues it is the same function, for every ns the callers
    make (q8 |s| <= 511, sums -s >= -1022, det_sqrt -s <= 1084). This file
    proves it, as `ExactScaledRound.v` does for float32, for every rounding
    [rn] that keeps a float64 unchanged and is monotone.

    Units: integers times 2^-2096 (float64's step 2^-1074 times the smallest
    factor 2^-1022), so a float64 with fields (E, M) is
    `ExactFloatOrder.value 52 E M * 2^1022`. Magnitudes: x >= 0.

    [the_float64_formula_is_the_exact_rounding]: for every finite float64
    x >= 0 and -1022 <= ns <= 2046 with x * 2^ns below 2^62, the formula is
    `ExactTrainArith.shift_round (x * 2^ns) 2096`, the round-half-even.
    [the_two_factors_are_the_scale]: a + b = ns with 0 <= b, and b = 0 up
    to ns = 1023 (b is at most 61 for the callers' 1084).
    [rep64]: K * 2^j with K below 2^53 is a finite float64 below 2^1136.

    ** Refutation.

    [one_factor_alone_loses_the_value]: without b, at ns = 1084 (det_sqrt's
    largest) x = 2^-1072 gives round(x * 2^1023) = 0 where x * 2^1084 is
    2^12.

    ** Not proved here.

    That the GPU's float64 `*` rounds to nearest and `round` is half to even
    (IEEE; the tests compare the formula with `scaled_round_int` under
    `FORCE_F64_FORMULA`); the sign.

    Checked with Rocq 9.1:  coqc -Q . "" ExactScaledRoundF64.v  *)

From Stdlib Require Import ZArith Lia.
Require ExactFloatOrder ExactTrainArith ExactVFold.
Open Scope Z_scope.

Module FO := ExactFloatOrder.
Module TA := ExactTrainArith.
Module VF := ExactVFold.

Definition clamp (lo hi v : Z) : Z := Z.min (Z.max v lo) hi.

Theorem the_two_factors_are_the_scale : forall ns, -1022 <= ns <= 2046 ->
  let a := clamp (-1022) 1023 ns in
  let b := clamp (-1022) 1023 (ns - a) in
  a + b = ns /\ 0 <= b /\ -1022 <= a <= 1023 /\ b <= 1023 /\ (ns <= 1023 -> a = ns /\ b = 0).
Proof. intros ns H. cbv zeta. unfold clamp. lia. Qed.

(** A finite float64 is K * 2^c with K below 2^53. *)
Lemma split_value : forall E M, 0 <= E -> 0 <= M < 2 ^ 52 ->
  FO.value 52 E M = (if E =? 0 then M else 2 ^ 52 + M) * 2 ^ (Z.max (E - 1) 0) /\
  0 <= (if E =? 0 then M else 2 ^ 52 + M) < 2 ^ 53.
Proof.
  intros E M HE HM. unfold FO.value. destruct (E =? 0) eqn:Z0.
  - apply Z.eqb_eq in Z0. subst E. split; [|lia]. rewrite Z.max_r by lia. ring.
  - apply Z.eqb_neq in Z0. split; [|lia]. rewrite Z.max_l by lia. reflexivity.
Qed.

Theorem rep64 : forall K j, 0 <= K < 2 ^ 53 -> 0 <= j -> K * 2 ^ j < 2 ^ 1136 ->
  exists E M, 0 <= E <= 2046 /\ 0 <= M < 2 ^ 52 /\ FO.value 52 E M = K * 2 ^ j.
Proof.
  intros K j HK Hj Hb. destruct (Z.eq_dec K 0) as [->|HK0].
  - exists 0, 0. repeat split; try lia; try reflexivity.
  - destruct (VF.an_integer_times_a_power_of_two_is_a_float 52 K j ltac:(lia)
                ltac:(replace (52 + 1) with 53 by lia; lia) Hj) as [E [M [HE [HM V]]]].
    exists E, M. repeat split; try lia; try exact V.
    assert (2 ^ j < 2 ^ 1136) by (assert (2 ^ j <= K * 2 ^ j) by (pose proof (Z.pow_pos_nonneg 2 j ltac:(lia) Hj); nia); lia).
    assert (j < 1136) by (destruct (Z_lt_le_dec j 1136) as [|G]; [lia|]; assert (2 ^ 1136 <= 2 ^ j) by (apply Z.pow_le_mono_r; lia); lia).
    lia.
Qed.

Lemma shift_round_small : forall w, 0 <= w <= 2 ^ 1074 -> TA.shift_round w 2096 = 0.
Proof.
  intros w Hw. unfold TA.shift_round.
  assert (L : 2 ^ 1074 < 2 ^ 2095) by (apply Z.pow_lt_mono_r; lia).
  assert (L2 : 2 ^ 2095 < 2 ^ 2096) by (apply Z.pow_lt_mono_r; lia).
  rewrite Z.div_small, Z.mod_small by lia. replace (2096 - 1) with 2095 by lia.
  destruct (2 ^ 2095 <? w) eqn:A; [apply Z.ltb_lt in A; lia|].
  destruct (w =? 2 ^ 2095) eqn:B; [apply Z.eqb_eq in B; lia|]. reflexivity.
Qed.

Section ScaledRound.
  Variable rn : Z -> Z.        (** a float64 rounding, on values in units of 2^-2096 *)
  Hypothesis rn_fix : forall E M, 0 <= E <= 2046 -> 0 <= M < 2 ^ 52 ->
    rn (FO.value 52 E M * 2 ^ 1022) = FO.value 52 E M * 2 ^ 1022.
  Hypothesis rn_mono : forall v w, v <= w -> rn v <= rn w.

  (** `round((x * 2^a) * 2^b)`, x the float64 with fields (E, M). *)
  Definition formula (E M ns : Z) : Z :=
    let a := clamp (-1022) 1023 ns in
    let b := clamp (-1022) 1023 (ns - a) in
    TA.shift_round (rn (rn (FO.value 52 E M * 2 ^ (1022 + a)) * 2 ^ b)) 2096.

  Lemma rn_float : forall K j, 0 <= K < 2 ^ 53 -> 0 <= j -> K * 2 ^ j < 2 ^ 1136 ->
    rn (K * 2 ^ (j + 1022)) = K * 2 ^ (j + 1022).
  Proof.
    intros K j HK Hj Hb. destruct (rep64 K j HK Hj Hb) as [E [M [HE [HM V]]]].
    rewrite Z.pow_add_r by lia. rewrite Z.mul_assoc, <- V. apply rn_fix; lia.
  Qed.

  Lemma rn_small : forall v, 0 <= v <= 2 ^ 1074 -> 0 <= rn v <= 2 ^ 1074.
  Proof.
    intros v Hv.
    assert (Z0 : rn 0 = 0) by (exact (rn_fix 0 0 ltac:(lia) ltac:(lia))).
    assert (T : rn (2 ^ 1074) = 2 ^ 1074).
    { assert (E : 2 ^ 1074 = FO.value 52 1 0 * 2 ^ 1022) by reflexivity.
      rewrite E. apply rn_fix; lia. }
    pose proof (rn_mono 0 v ltac:(lia)). pose proof (rn_mono v (2 ^ 1074) ltac:(lia)). lia.
  Qed.

  Theorem the_float64_formula_is_the_exact_rounding : forall E M ns,
    0 <= E <= 2046 -> 0 <= M < 2 ^ 52 -> -1022 <= ns <= 2046 ->
    FO.value 52 E M * 2 ^ (1022 + ns) < 2 ^ 62 * 2 ^ 2096 ->
    formula E M ns = TA.shift_round (FO.value 52 E M * 2 ^ (1022 + ns)) 2096.
  Proof.
    intros E M ns HE HM Hns Hb. unfold formula.
    destruct (the_two_factors_are_the_scale ns Hns) as [Sab [Hb0 [Ha [Hb1 Hlow]]]].
    set (a := clamp (-1022) 1023 ns) in *. set (b := clamp (-1022) 1023 (ns - a)) in *.
    destruct (split_value E M ltac:(lia) HM) as [V HK].
    set (K := if E =? 0 then M else 2 ^ 52 + M) in *. set (c := Z.max (E - 1) 0) in *.
    assert (Hc : 0 <= c) by (unfold c; lia).
    rewrite V. rewrite <- !Z.mul_assoc, <- !Z.pow_add_r by lia.
    assert (Hb' : (0 <= c + ns /\ K * 2 ^ (c + ns) < 2 ^ 1136) \/ c + ns < 0).
    { destruct (Z_lt_le_dec (c + ns) 0) as [|G]; [right; lia | left; split; [lia|]].
      rewrite V, <- Z.mul_assoc, <- Z.pow_add_r in Hb by lia.
      replace (c + (1022 + ns)) with ((c + ns) + 1022) in Hb by lia. rewrite Z.pow_add_r in Hb by lia.
      assert (P : 2 ^ 62 * 2 ^ 2096 = 2 ^ 1136 * 2 ^ 1022) by (rewrite <- !Z.pow_add_r by lia; reflexivity).
      rewrite P, Z.mul_assoc in Hb. assert (H126 : 0 < 2 ^ 1022) by (apply Z.pow_pos_nonneg; lia).
      exact (proj2 (Z.mul_lt_mono_pos_r _ _ _ H126) Hb). }
    destruct (Z_le_gt_dec ns 1023) as [Lo|Hi].
    - (* one factor: a = ns, b = 0 *)
      destruct (Hlow Lo) as [Ea Eb]. rewrite Ea, Eb, Z.pow_0_r, Z.mul_1_r.
      destruct Hb' as [[Nn Fit]|Neg].
      + replace (c + (1022 + ns)) with ((c + ns) + 1022) by lia.
        rewrite rn_float, rn_float by lia. reflexivity.
      + (* the product is below 2^-126: both round to 0 *)
        assert (Small : 0 <= K * 2 ^ (c + (1022 + ns)) <= 2 ^ 1074).
        { split; [apply Z.mul_nonneg_nonneg; [lia | apply Z.pow_nonneg; lia]|].
          assert (2 ^ (c + (1022 + ns)) <= 2 ^ 1021) by (apply Z.pow_le_mono_r; lia).
          assert (2 ^ 53 * 2 ^ 1021 = 2 ^ 1074) by (rewrite <- Z.pow_add_r by lia; reflexivity).
          assert (0 <= 2 ^ (c + (1022 + ns))) by (apply Z.pow_nonneg; lia). nia. }
        pose proof (rn_small _ Small) as R1. pose proof (rn_small _ R1) as R2.
        rewrite !shift_round_small by lia. reflexivity.
    - (* two factors: a = 1023, b = ns - 1023, both products exact *)
      assert (Ea : a = 1023) by (unfold a, clamp; lia). assert (Eb : b = ns - 1023) by (unfold b; rewrite Ea; unfold clamp; lia).
      rewrite Ea, Eb. destruct Hb' as [[Nn Fit]|Neg]; [|lia].
      assert (F1 : K * 2 ^ (c + 1023) < 2 ^ 1136).
      { assert (2 ^ (c + 1023) <= 2 ^ (c + ns)) by (apply Z.pow_le_mono_r; lia). nia. }
      replace (c + (1022 + 1023)) with ((c + 1023) + 1022) by lia. rewrite rn_float by lia.
      rewrite <- Z.mul_assoc, <- Z.pow_add_r by lia.
      replace (c + 1023 + 1022 + (ns - 1023)) with ((c + ns) + 1022) by lia. rewrite rn_float by lia.
      f_equal. f_equal. f_equal. lia.
  Qed.
End ScaledRound.

(** x = 2^-1072: the subnormal with M = 4 (units of 2^-1074). *)
Theorem one_factor_alone_loses_the_value :
  TA.shift_round (FO.value 52 0 4 * 2 ^ (1022 + 1023)) 2096 = 0 /\
  TA.shift_round (FO.value 52 0 4 * 2 ^ (1022 + 1084)) 2096 = 2 ^ 12.
Proof. vm_compute. split; reflexivity. Qed.

Print Assumptions the_two_factors_are_the_scale.
Print Assumptions rep64.
Print Assumptions the_float64_formula_is_the_exact_rounding.
Print Assumptions one_factor_alone_loses_the_value.
