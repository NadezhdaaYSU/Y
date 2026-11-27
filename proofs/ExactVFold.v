(** * Folding the V scales into the weights (`python/y_lang/exact/kv.py`).

    `fold_scale_into_weights(p, sv)` turns a query's softmax weights p
    (integers, Q0.28) and the keys' V scales sv (float32) into one integer
    weight per key and one common scale:

        w = float64(p) * sv;  wmax = max(w);  c = 2^28 / wmax
        W = round(w * c), clamped to 0 .. 2^28

    so the V product is an integer sum. Two facts make it what it says.

    [an_integer_times_a_power_of_two_is_a_float]: for any mantissa width p,
    K * 2^j with K below 2^(p + 1) is a float with an exponent field at most
    j + 1. [the_weight_product_is_exact_in_float64]: so p * sv, a 29-bit
    integer times a 24-bit significand, is a float64 itself: w and its
    maximum are exact, whatever the order of the max.

    [the_fold_stays_within_2_28]: with c and w * c each rounded to float64
    (|d| <= 2^-53 in the normal range), w * c is below 2^28 + 1/2, so every
    integer within 1/2 of it (the rounding, ties either way) is in
    0 .. 2^28: the clamp never changes a value.

    ** Refutation.

    [in_float32_the_fold_could_pass_2_28]: with float32's u = 2^-24 the same
    bound reaches 2^28 + 32, past 2^28 + 1/2: the fold needs float64 (or the
    clamp).

    ** Not proved here.

    IEEE's float64 multiply and divide (the standard model, in the normal
    range: wmax >= 2^28 * 2^-149 at the row's maximum key, so c and w * c are
    normal); that `round` rounds to nearest.

    Checked with Rocq 9.1:  coqc -Q . "" ExactVFold.v  *)

From Stdlib Require Import ZArith Lia QArith Qabs Lqa.
Require ExactFloatOrder.
Open Scope Z_scope.

Module FO := ExactFloatOrder.

Theorem an_integer_times_a_power_of_two_is_a_float : forall p K j,
  1 <= p -> 0 <= K < 2 ^ (p + 1) -> 0 <= j ->
  exists E M, 0 <= E <= j + 1 /\ 0 <= M < 2 ^ p /\ FO.value p E M = K * 2 ^ j.
Proof.
  intros p K j Hp HK Hj.
  destruct (Z_lt_le_dec (K * 2 ^ j) (2 ^ p)) as [Ls|Ge].
  - exists 0, (K * 2 ^ j). assert (0 <= K * 2 ^ j) by (apply Z.mul_nonneg_nonneg; [lia | apply Z.pow_nonneg; lia]).
    repeat split; try lia; try reflexivity.
  - assert (HK0 : 0 < K).
    { destruct (Z.eq_dec K 0) as [->|]; [|lia]. rewrite Z.mul_0_l in Ge.
      assert (0 < 2 ^ p) by (apply Z.pow_pos_nonneg; lia). lia. }
    set (lk := Z.log2 K).
    destruct (Z.log2_spec K HK0) as [L1 L2]. fold lk in L1, L2.
    assert (Hlk : 0 <= lk <= p).
    { split; [apply Z.log2_nonneg|]. destruct (Z_le_gt_dec lk p) as [|G]; [lia|].
      assert (2 ^ (p + 1) <= 2 ^ lk) by (apply Z.pow_le_mono_r; lia). lia. }
    assert (Hge : p <= lk + j).
    { destruct (Z_le_gt_dec p (lk + j)) as [|G]; [lia|].
      assert (K * 2 ^ j < 2 ^ (Z.succ lk) * 2 ^ j) by (apply Z.mul_lt_mono_pos_r; [apply Z.pow_pos_nonneg; lia | lia]).
      rewrite <- Z.pow_add_r in H by lia. assert (2 ^ (Z.succ lk + j) <= 2 ^ p) by (apply Z.pow_le_mono_r; lia). lia. }
    exists (lk + j - p + 1), (K * 2 ^ (p - lk) - 2 ^ p).
    assert (S1 : 2 ^ lk * 2 ^ (p - lk) = 2 ^ p) by (rewrite <- Z.pow_add_r by lia; f_equal; lia).
    assert (S2 : 0 < 2 ^ (p - lk)) by (apply Z.pow_pos_nonneg; lia).
    assert (M1 : 2 ^ p <= K * 2 ^ (p - lk)) by nia.
    assert (M2 : K * 2 ^ (p - lk) < 2 ^ (p + 1)).
    { assert (K * 2 ^ (p - lk) < 2 ^ (Z.succ lk) * 2 ^ (p - lk)) by nia.
      rewrite <- Z.pow_add_r in H by lia. replace (Z.succ lk + (p - lk)) with (p + 1) in H by lia. exact H. }
    assert (P1 : 2 ^ (p + 1) = 2 * 2 ^ p) by (rewrite Z.pow_add_r by lia; ring).
    repeat split; try lia.
    unfold FO.value. destruct (lk + j - p + 1 =? 0) eqn:Z0; [apply Z.eqb_eq in Z0; lia|].
    replace (2 ^ p + (K * 2 ^ (p - lk) - 2 ^ p)) with (K * 2 ^ (p - lk)) by ring.
    rewrite <- Z.mul_assoc, <- Z.pow_add_r by lia. f_equal. f_equal. lia.
Qed.

(** p * sv in units of 2^-1074: sv a finite float32 (fields e, m; in units
    of 2^-149 it is `value 23 e m`, so here `value 23 e m * 2^925`). *)
Theorem the_weight_product_is_exact_in_float64 : forall P e m,
  0 <= P <= 2 ^ 28 -> 0 <= e <= 254 -> 0 <= m < 2 ^ 23 ->
  exists E N, 0 <= E <= 2046 /\ 0 <= N < 2 ^ 52 /\
    FO.value 52 E N = P * (FO.value 23 e m * 2 ^ 925).
Proof.
  intros P e m HP He Hm.
  set (K32 := if e =? 0 then m else 2 ^ 23 + m).
  set (c := Z.max (e - 1) 0).
  assert (V : FO.value 23 e m = K32 * 2 ^ c).
  { unfold K32, c, FO.value. destruct (e =? 0) eqn:Z0.
    - apply Z.eqb_eq in Z0. subst e. rewrite Z.max_r by lia. ring.
    - apply Z.eqb_neq in Z0. rewrite Z.max_l by lia. reflexivity. }
  assert (HK : 0 <= K32 < 2 ^ 24) by (unfold K32; destruct (e =? 0); lia).
  assert (HKP : 0 <= P * K32 < 2 ^ 53).
  { split; [nia|]. assert (P * K32 <= 2 ^ 28 * K32) by nia.
    assert (2 ^ 28 * 2 ^ 24 < 2 ^ 53 + 2 ^ 28 * 2 ^ 24 - 2 ^ 28 * 2 ^ 24 + 1) by lia.
    replace (2 ^ 53) with (2 ^ 28 * 2 ^ 25) by reflexivity. nia. }
  destruct (an_integer_times_a_power_of_two_is_a_float 52 (P * K32) (c + 925) ltac:(lia)
              ltac:(replace (52 + 1) with 53 by lia; lia) ltac:(unfold c; lia)) as [E [N [HE [HN VE]]]].
  exists E, N. repeat split; try lia.
  rewrite VE, V. rewrite Z.pow_add_r by (unfold c; lia). ring.
Qed.

Section Fold.
  Open Scope Q_scope.
  Variables w wmax c y d1 d2 u : Q.
  Hypothesis Hu0 : 0 <= u.
  Hypothesis Hu : u * 9007199254740992 <= 1.              (* u <= 2^-53 *)
  Hypothesis Hm : 0 < wmax.
  Hypothesis Hw : 0 <= w <= wmax.
  Hypothesis H1 : - u <= d1 <= u.
  Hypothesis H2 : - u <= d2 <= u.
  Hypothesis Hc : c == 268435456 / wmax * (1 + d1).       (* RN64(2^28 / wmax) *)
  Hypothesis Hy : y == w * c * (1 + d2).                  (* RN64(w * c) *)

  Lemma the_scaled_weight_is_below_2_28_and_a_half : 0 <= y < 268435456 + (1 # 2).
  Proof.
    assert (Hcw : c * wmax == 268435456 * (1 + d1)) by (rewrite Hc; field; intro Z; lra).
    assert (Hc0 : 0 < c).
    { rewrite Hc. apply Qmult_lt_0_compat; [apply Qlt_shift_div_l; [exact Hm | lra] | lra]. }
    assert (A : w * c <= wmax * c) by nra.
    assert (B : wmax * c <= 268435456 * (1 + u)) by (rewrite Qmult_comm, Hcw; nra).
    split.
    - rewrite Hy. assert (0 <= w * c) by nra. nra.
    - rewrite Hy. assert (0 <= w * c) by nra.
      assert (w * c * (1 + d2) <= 268435456 * (1 + u) * (1 + u)) by nra.
      nra.
  Qed.

  Theorem the_fold_stays_within_2_28 : forall W : Z,
    Qabs (y - inject_Z W) <= 1 # 2 -> (0 <= W <= 2 ^ 28)%Z.
  Proof.
    intros W HW. pose proof the_scaled_weight_is_below_2_28_and_a_half as [B0 B1].
    apply Qabs_Qle_condition in HW.
    assert (U : inject_Z W < inject_Z (2 ^ 28 + 1)) by (rewrite inject_Z_plus; change (inject_Z (2 ^ 28)) with 268435456; change (inject_Z 1) with 1; lra).
    assert (D : inject_Z (-1) < inject_Z W) by (change (inject_Z (-1)) with (-1); lra).
    rewrite <- Zlt_Qlt in U, D. lia.
  Qed.
End Fold.

Theorem in_float32_the_fold_could_pass_2_28 :
  let u := 1 # 16777216 in
  (268435456 + (1 # 2) < 268435456 * (1 + u) * (1 + u))%Q.
Proof. vm_compute. reflexivity. Qed.

Print Assumptions an_integer_times_a_power_of_two_is_a_float.
Print Assumptions the_weight_product_is_exact_in_float64.
Print Assumptions the_scaled_weight_is_below_2_28_and_a_half.
Print Assumptions the_fold_stays_within_2_28.
Print Assumptions in_float32_the_fold_could_pass_2_28.
