(** * Reading |x| from its bits (`python/y_lang/exact/train_lm.py`).

    The exact quantizers never compare floats. `exponent(x, dim)` takes the
    bits of |x| as an integer (the sign bit masked off), their maximum, and
    reads the maximum's exponent field; `q8`'s scale, the fused MLP kernel's
    row maxima (`mlp_kernel.py`) and `ExactRowCopy.v`'s maxima all rest on
    two facts about IEEE binary formats, proved here for every mantissa
    width p (float32: p = 23; float64: p = 52).

    A finite non-negative float with exponent field E (0 <= E <= 2^w - 2)
    and mantissa field M (0 <= M < 2^p) has the bits E * 2^p + M and the
    value [value p E M] * 2^-(bias + p - 1): M for a subnormal (E = 0),
    (2^p + M) * 2^(E - 1) for a normal one.

    [the_bits_are_ordered_as_the_values]: for two such floats, the bits
    compare as their values do, so the maximum of the bits is the bits of
    the maximum ([the_max_of_the_bits_is_the_bits_of_the_max]).

    [the_field_bounds_the_value]: the exponent the code reads, field - bias
    + 1 for a nonzero maximum (frexp's for a normal float, the smallest
    normal one's for a subnormal), bounds the value from above: in the
    scaled units here, value < 2^(p + E) for every field E, the subnormals
    (E = 0) included.
    [a_normal_value_reaches_half_its_bound]: for a normal float the bound is
    tight within a factor of 2 (frexp's exponent).

    ** Refutations.

    [with_the_sign_bit_the_order_breaks]: without masking the sign bit, a
    negative number's bits exceed every positive number's.
    [an_infinite_field_is_not_ordered_by_the_formula]: the all-ones field
    is not a finite value, which is why `exponent` refuses it.

    ** Not proved here.

    That torch's `.view(int)` gives these bits (IEEE storage, checked by the
    quantizer tests on ties, zeros and subnormals); NaN and infinity, which
    `exponent` refuses before any scale is used.

    Checked with Rocq 9.1:  coqc -Q . "" ExactFloatOrder.v  *)

From Stdlib Require Import ZArith Lia.
Open Scope Z_scope.

(** The value of a finite non-negative float, scaled by 2^(bias + p - 1). *)
Definition value (p E M : Z) : Z := if E =? 0 then M else (2 ^ p + M) * 2 ^ (E - 1).

(** Its bits with the sign bit 0. *)
Definition bits (p E M : Z) : Z := E * 2 ^ p + M.

Definition finite (p E M : Z) : Prop := 0 <= E /\ 0 <= M < 2 ^ p.

Lemma value_bounds : forall p E M, 0 <= p -> finite p E M ->
  (E = 0 -> value p E M = M) /\
  (1 <= E -> 2 ^ (p + E - 1) <= value p E M < 2 ^ (p + E)).
Proof.
  intros p E M Hp [HE HM]. unfold value. split.
  - intros ->. reflexivity.
  - intro H1. destruct (E =? 0) eqn:E0; [apply Z.eqb_eq in E0; lia|].
    assert (P1 : 2 ^ (p + E - 1) = 2 ^ p * 2 ^ (E - 1)) by (rewrite <- Z.pow_add_r; [f_equal; lia | lia | lia]).
    assert (P2 : 2 ^ (p + E) = 2 ^ p * 2 ^ (E - 1) * 2) by
      (rewrite <- Z.pow_add_r by lia; rewrite <- (Z.pow_1_r 2) at 3; rewrite <- Z.pow_add_r by lia; f_equal; lia).
    assert (0 < 2 ^ (E - 1)) by (apply Z.pow_pos_nonneg; lia).
    rewrite P1, P2. nia.
Qed.

(** Each field step multiplies the scale by two, so a larger field always
    means a larger value, whatever the mantissas. *)
Lemma larger_field_larger_value : forall p E1 M1 E2 M2, 0 <= p ->
  finite p E1 M1 -> finite p E2 M2 -> E1 < E2 -> value p E1 M1 < value p E2 M2.
Proof.
  intros p E1 M1 E2 M2 Hp F1 F2 HE.
  destruct (value_bounds p E1 M1 Hp F1) as [Z1 N1]. destruct (value_bounds p E2 M2 Hp F2) as [Z2 N2].
  destruct F1 as [HE1 HM1].
  assert (Hlo : 2 ^ (p + E2 - 1) <= value p E2 M2) by (apply N2; lia).
  destruct (Z.eq_dec E1 0) as [->|Hne].
  - rewrite (Z1 eq_refl). assert (2 ^ p <= 2 ^ (p + E2 - 1)) by (apply Z.pow_le_mono_r; lia). lia.
  - assert (Hhi : value p E1 M1 < 2 ^ (p + E1)) by (apply N1; lia).
    assert (2 ^ (p + E1) <= 2 ^ (p + E2 - 1)) by (apply Z.pow_le_mono_r; lia). lia.
Qed.

Lemma same_field_mantissa_order : forall p E M1 M2, value p E M1 - value p E M2 = (M1 - M2) * (if E =? 0 then 1 else 2 ^ (E - 1)).
Proof. intros p E M1 M2. unfold value. destruct (E =? 0); ring. Qed.

Theorem the_bits_are_ordered_as_the_values : forall p E1 M1 E2 M2, 0 <= p ->
  finite p E1 M1 -> finite p E2 M2 ->
  (bits p E1 M1 <= bits p E2 M2 <-> value p E1 M1 <= value p E2 M2).
Proof.
  intros p E1 M1 E2 M2 Hp F1 F2. unfold bits.
  pose proof F1 as [HE1 HM1]. pose proof F2 as [HE2 HM2].
  assert (0 < 2 ^ p) by (apply Z.pow_pos_nonneg; lia).
  destruct (Z.lt_trichotomy E1 E2) as [Lt|[Eq|Gt]].
  - pose proof (larger_field_larger_value p E1 M1 E2 M2 Hp F1 F2 Lt). split; intros; nia.
  - subst E2. pose proof (same_field_mantissa_order p E1 M1 M2) as D.
    assert (0 < (if E1 =? 0 then 1 else 2 ^ (E1 - 1))) by (destruct (E1 =? 0) eqn:E0; [lia | apply Z.eqb_neq in E0; apply Z.pow_pos_nonneg; lia]).
    split; intros; nia.
  - pose proof (larger_field_larger_value p E2 M2 E1 M1 Hp F2 F1 Gt). split; intros; nia.
Qed.

Corollary the_max_of_the_bits_is_the_bits_of_the_max : forall p E1 M1 E2 M2, 0 <= p ->
  finite p E1 M1 -> finite p E2 M2 ->
  (Z.max (bits p E1 M1) (bits p E2 M2) = bits p E1 M1 <-> value p E2 M2 <= value p E1 M1).
Proof.
  intros p E1 M1 E2 M2 Hp F1 F2. rewrite <- (the_bits_are_ordered_as_the_values p E2 M2 E1 M1 Hp F2 F1). lia.
Qed.

Theorem the_field_bounds_the_value : forall p E M, 0 <= p -> finite p E M ->
  value p E M < 2 ^ (p + E).
Proof.
  intros p E M Hp F. destruct (value_bounds p E M Hp F) as [Z0 N]. destruct F as [HE HM].
  destruct (Z.eq_dec E 0) as [->|Hne].
  - rewrite (Z0 eq_refl), Z.add_0_r. lia.
  - apply N. lia.
Qed.

Theorem a_normal_value_reaches_half_its_bound : forall p E M, 0 <= p -> finite p E M -> 1 <= E ->
  2 ^ (p + E - 1) <= value p E M.
Proof. intros p E M Hp F H1. apply (value_bounds p E M Hp F). exact H1. Qed.

(** float32: the sign bit set makes the bits of -0.5 (2^31 + 126 * 2^23)
    exceed those of the largest finite float (254 * 2^23 + 2^23 - 1). *)
Theorem with_the_sign_bit_the_order_breaks :
  2 ^ 31 + 126 * 2 ^ 23 > 254 * 2 ^ 23 + (2 ^ 23 - 1).
Proof. lia. Qed.

(** The all-ones field read as a finite value would give 2^(p + 2^w - 2), a
    value past every finite one: an infinity has no such value, so the code
    refuses the field instead of reading it. Here: it is not below the
    largest finite float32's bound 2^(23 + 254). *)
Theorem an_infinite_field_is_not_ordered_by_the_formula :
  ~ (value 23 255 0 < 2 ^ (23 + 254)).
Proof. unfold value. intro H. vm_compute in H. discriminate H. Qed.

Print Assumptions the_bits_are_ordered_as_the_values.
Print Assumptions the_max_of_the_bits_is_the_bits_of_the_max.
Print Assumptions the_field_bounds_the_value.
Print Assumptions a_normal_value_reaches_half_its_bound.
Print Assumptions with_the_sign_bit_the_order_breaks.
Print Assumptions an_infinite_field_is_not_ordered_by_the_formula.
