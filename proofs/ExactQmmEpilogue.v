(** * qmm's float32 epilogue (`python/y_lang/exact/qmm.py`).

    `_qmm_kernel` (and `qmm_sum`'s two products) turn the exact integer sum
    c of a tile into `c.to(float32) * 2^e`, with 2^e built from its bits
    when -126 <= e <= 127, where `exact_product` computes float32(c * 2^e)
    (float64(c) * 2^e exact, then one rounding). The kernel's comment argues
    they agree: rounding to 24 significant bits commutes with a power of two,
    and a nonzero integer times 2^e >= 2^-126 is never subnormal.
    `ExactQmmSplit.v` left this to the tests. This file proves it.

    Units: values are integers times 2^-126, so c * 2^e is c * 2^(e + 126)
    for e >= -126. In the normal range float32's rounding to nearest is
    [sig_round]: the value rounded to 24 significant bits, half to even.

    [rounding_commutes_with_the_scale]: sig_round (c 2^j) = sig_round c 2^j
    for every integer c >= 0 and j >= 0.
    [the_epilogue_is_the_float32_of_the_product]: so for -126 <= e,
    float32(c 2^e) is c.to(float32) times 2^e (each side's overflow to
    infinity is decided on this same rounded value).
    [the_product_is_never_subnormal]: for c >= 1 and e >= -126 the value is
    at least 2^-126, the normal range the rounding model covers.
    [the_rounded_value_fits_24_bits]: sig_round c is s 2^t with s <= 2^24, so
    the kernel's multiply by 2^e has nothing to round;
    [a_24_bit_integer_is_kept]: and every integer below 2^24 is kept, so the
    precision is float32's 24 bits, not fewer.

    ** Refutations.

    [r_zero_is_not_a_rounding]: `shift_round m 0` adds 1 to an odd m, so the
    rounding must skip a shift of 0 (as [sig_round] does).
    [below_2_126_the_grid_is_fixed]: 3 * 2^-150 is no float32 (the subnormal
    grid is 2^-149) while 3 has 24 bits, so past e = -126 the shortcut fails
    and the kernel takes the float64 product.

    ** Not proved here.

    That float32 rounding to nearest is [sig_round] in the normal range and
    that IEEE overflow is decided on the rounded value (IEEE 754), that an
    int64's `.to(float32)` rounds to nearest even, the sign (the conversion
    rounds |c| and negates).

    Checked with Rocq 9.1:  coqc -Q . "" ExactQmmEpilogue.v  *)

From Stdlib Require Import ZArith Lia.
Require ExactTrainArith.
Open Scope Z_scope.

Module TA := ExactTrainArith.

Definition sig_round (c : Z) : Z :=
  let k := Z.log2 c + 1 - 24 in
  if k <=? 0 then c else TA.shift_round c k * 2 ^ k.

Lemma shift_round_scale : forall m r j, 0 <= m -> 1 <= r -> 0 <= j ->
  TA.shift_round (m * 2 ^ j) (r + j) = TA.shift_round m r.
Proof.
  intros m r j Hm Hr Hj. unfold TA.shift_round.
  assert (Pj : 0 < 2 ^ j) by (apply Z.pow_pos_nonneg; lia).
  assert (Pr : 0 < 2 ^ r) by (apply Z.pow_pos_nonneg; lia).
  rewrite Z.pow_add_r by lia.
  rewrite Z.div_mul_cancel_r by lia.
  rewrite Z.mul_mod_distr_r by lia.
  replace (r + j - 1) with ((r - 1) + j) by lia. rewrite Z.pow_add_r by lia.
  pose proof (Z.mod_pos_bound m (2 ^ r) Pr).
  assert (Ph : 0 <= 2 ^ (r - 1)) by (apply Z.pow_nonneg; lia).
  destruct (2 ^ (r - 1) <? m mod 2 ^ r) eqn:E1;
  destruct (2 ^ (r - 1) * 2 ^ j <? m mod 2 ^ r * 2 ^ j) eqn:E2;
  try (rewrite Z.ltb_lt in *; rewrite Z.ltb_ge in *; nia);
  [reflexivity|].
  destruct (m mod 2 ^ r =? 2 ^ (r - 1)) eqn:E3;
  destruct (m mod 2 ^ r * 2 ^ j =? 2 ^ (r - 1) * 2 ^ j) eqn:E4;
  try reflexivity; rewrite ?Z.eqb_eq, ?Z.eqb_neq in *; nia.
Qed.

Lemma shift_round_exact : forall m j r, 0 <= m -> 1 <= r <= j ->
  TA.shift_round (m * 2 ^ j) r = m * 2 ^ (j - r).
Proof.
  intros m j r Hm Hr. unfold TA.shift_round.
  replace (m * 2 ^ j) with (m * 2 ^ (j - r) * 2 ^ r)
    by (rewrite <- Z.mul_assoc, <- Z.pow_add_r by lia; f_equal; f_equal; lia).
  assert (Pr : 0 < 2 ^ r) by (apply Z.pow_pos_nonneg; lia).
  rewrite Z.div_mul by lia. rewrite Z.mod_mul by lia.
  assert (H1 : 1 <= 2 ^ (r - 1)) by (pose proof (Z.pow_le_mono_r 2 0 (r - 1) ltac:(lia) ltac:(lia)); cbn in *; lia).
  destruct (2 ^ (r - 1) <? 0) eqn:E1; [rewrite Z.ltb_lt in E1; lia|].
  destruct (0 =? 2 ^ (r - 1)) eqn:E2; [rewrite Z.eqb_eq in E2; lia|]. cbn. lia.
Qed.

Theorem rounding_commutes_with_the_scale : forall c j, 0 <= c -> 0 <= j ->
  sig_round (c * 2 ^ j) = sig_round c * 2 ^ j.
Proof.
  intros c j Hc Hj. unfold sig_round.
  destruct (Z.eq_dec c 0) as [->|Hc0].
  { cbn. reflexivity. }
  assert (Lg : Z.log2 (c * 2 ^ j) = Z.log2 c + j) by (rewrite Z.log2_mul_pow2 by lia; lia).
  rewrite Lg.
  destruct (Z.log2 c + 1 - 24 <=? 0) eqn:K.
  - apply Z.leb_le in K.
    destruct (Z.log2 c + j + 1 - 24 <=? 0) eqn:K'; [reflexivity|].
    apply Z.leb_gt in K'.
    rewrite shift_round_exact by lia.
    rewrite <- Z.mul_assoc, <- Z.pow_add_r by lia. f_equal. f_equal. lia.
  - apply Z.leb_gt in K.
    destruct (Z.log2 c + j + 1 - 24 <=? 0) eqn:K'; [apply Z.leb_le in K'; lia|].
    replace (Z.log2 c + j + 1 - 24) with ((Z.log2 c + 1 - 24) + j) by lia.
    rewrite shift_round_scale by lia.
    rewrite Z.pow_add_r by lia. ring.
Qed.

Theorem the_epilogue_is_the_float32_of_the_product : forall c e, 0 <= c -> -126 <= e ->
  sig_round (c * 2 ^ (e + 126)) = sig_round c * 2 ^ (e + 126).
Proof. intros. apply rounding_commutes_with_the_scale; lia. Qed.

Theorem the_product_is_never_subnormal : forall c e, 1 <= c -> -126 <= e ->
  1 <= c * 2 ^ (e + 126) /\ 1 <= sig_round c * 2 ^ (e + 126).
Proof.
  intros c e Hc He.
  assert (P : 1 <= 2 ^ (e + 126)) by (pose proof (Z.pow_le_mono_r 2 0 (e + 126) ltac:(lia) ltac:(lia)); cbn in *; lia).
  split; [nia|].
  assert (S : 1 <= sig_round c).
  { unfold sig_round. destruct (Z.log2 c + 1 - 24 <=? 0) eqn:K; [lia|].
    apply Z.leb_gt in K.
    assert (Pk : 0 < 2 ^ (Z.log2 c + 1 - 24)) by (apply Z.pow_pos_nonneg; lia).
    assert (Lc : 2 ^ Z.log2 c <= c) by (apply Z.log2_spec; lia).
    assert (Q : 1 <= c / 2 ^ (Z.log2 c + 1 - 24)).
    { apply Z.div_le_lower_bound; [lia|].
      assert (2 ^ (Z.log2 c + 1 - 24) <= 2 ^ Z.log2 c) by (apply Z.pow_le_mono_r; lia). lia. }
    unfold TA.shift_round. match goal with |- context [if ?b then _ else _] => destruct b end; nia. }
  nia.
Qed.

Theorem the_rounded_value_fits_24_bits : forall c, 0 <= c ->
  exists s t, 0 <= s <= 2 ^ 24 /\ 0 <= t /\ sig_round c = s * 2 ^ t.
Proof.
  intros c Hc. unfold sig_round.
  destruct (Z.log2 c + 1 - 24 <=? 0) eqn:K.
  - apply Z.leb_le in K. exists c, 0. split; [|split; [lia|ring]].
    destruct (Z.eq_dec c 0) as [->|Hc0]; [lia|].
    assert (c < 2 ^ (Z.log2 c + 1)) by (pose proof (Z.log2_spec c ltac:(lia)) as [_ L]; rewrite <- Z.add_1_r in L; exact L).
    assert (2 ^ (Z.log2 c + 1) <= 2 ^ 24) by (apply Z.pow_le_mono_r; lia). lia.
  - apply Z.leb_gt in K.
    assert (Hc0 : 0 < c) by (destruct (Z.eq_dec c 0) as [E|E]; [subst c; cbn in K; lia|lia]).
    set (k := Z.log2 c + 1 - 24) in *.
    exists (TA.shift_round c k), k. split; [|split; [lia|reflexivity]].
    assert (Pk : 0 < 2 ^ k) by (apply Z.pow_pos_nonneg; lia).
    assert (Hlt : c < 2 ^ 24 * 2 ^ k).
    { rewrite <- Z.pow_add_r by lia. replace (24 + k) with (Z.log2 c + 1) by (unfold k; lia).
      pose proof (Z.log2_spec c ltac:(lia)) as [_ L]; rewrite <- Z.add_1_r in L; exact L. }
    assert (Qd : c / 2 ^ k < 2 ^ 24) by (apply Z.div_lt_upper_bound; lia).
    assert (Q0 : 0 <= c / 2 ^ k) by (apply Z.div_pos; lia).
    unfold TA.shift_round. match goal with |- context [if ?b then _ else _] => destruct b end; lia.
Qed.

(** The precision is float32's: every integer below 2^24 is a float32 and
    is kept (with [the_rounded_value_fits_24_bits], the rounding is to 24
    significant bits, not fewer). *)
Theorem a_24_bit_integer_is_kept : forall c, 0 <= c < 2 ^ 24 -> sig_round c = c.
Proof.
  intros c Hc. unfold sig_round.
  destruct (Z.eq_dec c 0) as [->|Hc0]; [reflexivity|].
  assert (L : Z.log2 c < 24) by (apply Z.log2_lt_pow2; lia).
  destruct (Z.log2 c + 1 - 24 <=? 0) eqn:K; [reflexivity|]. apply Z.leb_gt in K. lia.
Qed.

(** ** Refutations. *)

Theorem r_zero_is_not_a_rounding : TA.shift_round 3 0 = 4.
Proof. reflexivity. Qed.

(** In units of 2^-149 (float32's smallest step), 3 * 2^-150 is 3/2: no
    integer, so no float32; 3 itself needs only 2 bits. *)
Theorem below_2_126_the_grid_is_fixed :
  sig_round 3 = 3 /\ ~ (exists n, 2 * n = 3).
Proof. split; [reflexivity|]. intros [n H]. lia. Qed.

Print Assumptions rounding_commutes_with_the_scale.
Print Assumptions the_epilogue_is_the_float32_of_the_product.
Print Assumptions the_product_is_never_subnormal.
Print Assumptions the_rounded_value_fits_24_bits.
Print Assumptions a_24_bit_integer_is_kept.
Print Assumptions r_zero_is_not_a_rounding.
Print Assumptions below_2_126_the_grid_is_fixed.
