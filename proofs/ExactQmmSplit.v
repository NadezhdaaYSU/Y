(** * The fused product's digit split (`python/y_lang/exact/qmm.py`).

    `_qmm_kernel` multiplies a 12-bit operand by an 8-bit one on int8
    tensor cores: each tile of qa splits in registers as
    `lo = ((a + 64) & 127) - 64`, `hi = (a - lo) >> 7`, two int8 dots
    accumulate lo and hi in int32, and `c = hi * 128 + lo` joins them in
    int64. That is a bitwise form of the digit split `train_lm.digits` uses
    (a floor division, proved in `ExactTrainArith`); this file proves the two
    are the same split, and that the kernel's accumulators hold their sums.

    [qmm_lo_is_the_digit] and [qmm_hi_is_the_carry]: the bitwise lo and hi
    are `ExactTrainArith`'s digit and carry, for every integer a (two's
    complement `&` and arithmetic `>>`, as Z's are). [the_join_is_the_value]:
    hi * 128 + lo = a. [the_12_bit_digits]: for |a| <= 2047, lo is in
    [-64, 63] and hi in [-16, 16], both int8.

    [the_accumulators_fit_int32]: with K <= `MAX_K` = 131,040 the two int32
    accumulators' partial sums (at most K terms of |lo| 64 or |hi| 16 times
    127) fit int32, and so does the one-digit mode's (|a| <= 127).
    [the_join_is_exact_in_float64]: |c| <= K * 2047 * 127 is below 2^53, so
    the kernel's float64 branch converts it exactly; that margin is wide
    ([the_join_fails_only_past_2_35]: |c| reaches 2^53 only between K = 2^35
    and 2^36).
    [one_digit_past_133144_can_overflow]: one-digit sums of 133,145 terms of
    127 * 127 leave int32, the bound `MAX_K` stays inside.

    ** Not proved here.

    The float32 epilogue's rounding (`c.to(float32) * 2^e`, the field reading
    of `ExactMlpKernel.the_float32_field_is_normal_exactly_when_ok`), and the
    int8 dot itself (`Int8GemmExact.v` for Y's; Triton's `tl.dot` is taken
    as an exact int32 dot product). Tested against `exact_product` bit for
    bit by python/tests/test_exact_qmm.py.

    Checked with Rocq 9.1:  coqc -Q . "" ExactQmmSplit.v  *)

From Stdlib Require Import ZArith Lia.
Require ExactTrainArith.
Open Scope Z_scope.

Module TA := ExactTrainArith.

Definition lo (a : Z) : Z := Z.land (a + 64) 127 - 64.
Definition hi (a : Z) : Z := Z.shiftr (a - lo a) 7.

Lemma land_127 : forall x, Z.land x 127 = x mod 128.
Proof. intro x. change 127 with (Z.ones 7). rewrite Z.land_ones by lia. reflexivity. Qed.

Theorem qmm_lo_is_the_digit : forall a, lo a = TA.digit a.
Proof.
  intro a. unfold lo, TA.digit, TA.carry. rewrite land_127.
  pose proof (Z.div_mod (a + 64) 128 ltac:(lia)). lia.
Qed.

Theorem qmm_hi_is_the_carry : forall a, hi a = TA.carry a.
Proof.
  intro a. unfold hi. rewrite qmm_lo_is_the_digit, Z.shiftr_div_pow2 by lia.
  unfold TA.digit. replace (a - (a - 128 * TA.carry a)) with (TA.carry a * 128) by ring.
  change (2 ^ 7) with 128. apply Z.div_mul. lia.
Qed.

Theorem the_join_is_the_value : forall a, hi a * 128 + lo a = a.
Proof. intro a. rewrite qmm_hi_is_the_carry, qmm_lo_is_the_digit. unfold TA.digit. ring. Qed.

Theorem the_12_bit_digits : forall a, -2047 <= a <= 2047 ->
  -64 <= lo a <= 63 /\ -16 <= hi a <= 16.
Proof.
  intros a H. rewrite qmm_lo_is_the_digit, qmm_hi_is_the_carry.
  pose proof (TA.digit_range a). destruct (TA.carry_spec a). lia.
Qed.

Definition in_int32 (z : Z) : Prop := - 2 ^ 31 <= z <= 2 ^ 31 - 1.

(** A partial sum of n <= K terms each at most [t] in magnitude is at most
    K * t; the cases are the accumulators' term bounds. *)
Theorem the_accumulators_fit_int32 : forall K s,
  0 <= K <= 131040 ->
  (- (K * (64 * 127)) <= s <= K * (64 * 127) -> in_int32 s) /\
  (- (K * (16 * 127)) <= s <= K * (16 * 127) -> in_int32 s) /\
  (- (K * (127 * 127)) <= s <= K * (127 * 127) -> in_int32 s).
Proof.
  intros K s HK. unfold in_int32. change (2 ^ 31) with 2147483648.
  repeat split; intros; nia.
Qed.

Theorem the_join_is_exact_in_float64 : forall K c,
  0 <= K <= 131040 -> - (K * (2047 * 127)) <= c <= K * (2047 * 127) -> Z.abs c < 2 ^ 53.
Proof. intros K c HK Hc. change (2 ^ 53) with 9007199254740992. nia. Qed.

(** The join's margin is wide: the bound above holds far past `MAX_K`; the
    largest |c| reaches 2^53 only between K = 2^35 and 2^36. *)
Theorem the_join_fails_only_past_2_35 :
  2 ^ 35 * (2047 * 127) < 2 ^ 53 /\ ~ (2 ^ 36 * (2047 * 127) < 2 ^ 53).
Proof. split; [lia | lia]. Qed.

Theorem one_digit_past_133144_can_overflow : ~ in_int32 (133145 * (127 * 127)).
Proof. unfold in_int32. change (2 ^ 31) with 2147483648. lia. Qed.

Print Assumptions qmm_lo_is_the_digit.
Print Assumptions qmm_hi_is_the_carry.
Print Assumptions the_join_is_the_value.
Print Assumptions the_12_bit_digits.
Print Assumptions the_accumulators_fit_int32.
Print Assumptions the_join_is_exact_in_float64.
Print Assumptions the_join_fails_only_past_2_35.
Print Assumptions one_digit_past_133144_can_overflow.
