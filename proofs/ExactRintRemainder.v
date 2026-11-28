(** * x - rint(x) in float32 (`python/y_lang/exact/attn_weights_kernel.py`).

    The attention kernel's `_rint32(x)` is the nearest integer k of a float32
    x ((|x| + 2^23) - 2^23 below 2^23, `ExactSoftmaxKernel.v`; x itself at or
    past 2^23), and the kernel forms `d_hi - k1` and `d_lo - k2` in float32.
    `ExactAttnDiff.v` takes both as exact. This file proves it.

    Units: a float32 is s 2^t units of 2^-L (0 <= s < 2^24, t >= 0; L >= 1
    fixes the unit, 149 for float32's smallest step), an integer k is k 2^L.

    [a_float32_past_2_23_is_an_integer]: s 2^t >= 2^23 2^L forces t >= L,
    so the value is a multiple of 2^L: `_rint32` may return it unchanged.
    [the_nearest_integer_is_not_negative]: for x >= 0, the integer within
    1/2 is >= 0 (the negative case is the mirror image).
    [the_remainder_is_a_float32]: for any integer k within 1/2 of x,
    |x - k| is s' 2^t' with s' < 2^24: a float32, so the subtraction has
    nothing to round. (k = 0 leaves x; otherwise x >= 1/2 makes t >= L - 24,
    and |x - k| <= 1/2 is at most 2^23 steps of 2^t.)

    ** Refutation.

    [a_far_integer_leaves_no_float32]: against an integer that is not the
    nearest the remainder can need more than 24 bits: (2^24 - 1) 2^-23
    minus -2^24 is 2^24 + 2 - 2^-23, odd in units of 2^-23 and past 2^24.

    ** Not proved here.

    That float32 subtraction of two float32s whose exact difference is a
    float32 returns it (IEEE), and the overflow range (no value here is
    near 2^128).

    Checked with Rocq 9.1:  coqc -Q . "" ExactRintRemainder.v  *)

From Stdlib Require Import ZArith Lia.
Open Scope Z_scope.

Lemma pow_pos : forall n, 0 <= n -> 0 < 2 ^ n.
Proof. intros. apply Z.pow_pos_nonneg; lia. Qed.

Theorem a_float32_past_2_23_is_an_integer : forall L s t, 1 <= L -> 0 <= s < 2 ^ 24 -> 0 <= t ->
  2 ^ 23 * 2 ^ L <= s * 2 ^ t -> L <= t /\ (s * 2 ^ t) mod 2 ^ L = 0.
Proof.
  intros L s t HL Hs Ht Hx.
  assert (Ht' : L <= t).
  { destruct (Z_lt_le_dec t L) as [Lt|Ge]; [|exact Ge].
    exfalso. assert (2 * 2 ^ t <= 2 ^ L).
    { rewrite <- Z.pow_succ_r by lia. apply Z.pow_le_mono_r; lia. }
    assert (2 ^ 24 = 2 ^ 23 * 2) by reflexivity.
    pose proof (pow_pos t Ht). pose proof (pow_pos L ltac:(lia)). nia. }
  split; [exact Ht'|].
  replace t with ((t - L) + L) by lia. rewrite Z.pow_add_r by lia.
  rewrite Z.mul_assoc. apply Z.mod_mul. pose proof (pow_pos L ltac:(lia)). lia.
Qed.

Theorem the_nearest_integer_is_not_negative : forall L X k, 1 <= L -> 0 <= X ->
  Z.abs (X - k * 2 ^ L) <= 2 ^ (L - 1) -> 0 <= k.
Proof.
  intros L X k HL HX Hd.
  assert (E : 2 ^ L = 2 * 2 ^ (L - 1)) by (rewrite <- Z.pow_succ_r by lia; f_equal; lia).
  pose proof (pow_pos (L - 1) ltac:(lia)). nia.
Qed.

Theorem the_remainder_is_a_float32 : forall L s t k, 1 <= L -> 0 <= s < 2 ^ 24 -> 0 <= t ->
  Z.abs (s * 2 ^ t - k * 2 ^ L) <= 2 ^ (L - 1) ->
  exists s' t', 0 <= s' < 2 ^ 24 /\ 0 <= t' /\ Z.abs (s * 2 ^ t - k * 2 ^ L) = s' * 2 ^ t'.
Proof.
  intros L s t k HL Hs Ht Hd.
  assert (Pt : 0 < 2 ^ t) by (apply pow_pos; lia).
  assert (PL1 : 0 < 2 ^ (L - 1)) by (apply pow_pos; lia).
  assert (E : 2 ^ L = 2 * 2 ^ (L - 1)) by (rewrite <- Z.pow_succ_r by lia; f_equal; lia).
  pose proof (the_nearest_integer_is_not_negative L (s * 2 ^ t) k HL ltac:(nia) Hd) as Hk.
  destruct (Z.eq_dec k 0) as [->|Hk0].
  { exists s, t. split; [lia|split; [lia|]]. rewrite Z.mul_0_l, Z.sub_0_r. apply Z.abs_eq. nia. }
  (* k >= 1, so x >= 1/2: s 2^t >= 2^(L-1). *)
  assert (Hx : 2 ^ (L - 1) <= s * 2 ^ t) by (apply Z.abs_le in Hd; nia).
  destruct (Z_lt_le_dec t L) as [Lt|Ge].
  - (* L - 24 <= t < L: the remainder is a multiple of 2^t, at most 2^(L-1). *)
    assert (Ht24 : L - 24 <= t).
    { destruct (Z_lt_le_dec t (L - 24)) as [Lo|Ok]; [exfalso|exact Ok].
      assert (2 ^ t * 2 ^ 24 <= 2 ^ (L - 1)).
      { rewrite <- Z.pow_add_r by lia. apply Z.pow_le_mono_r; lia. }
      nia. }
    set (m := s - k * 2 ^ (L - t)).
    assert (Ed : s * 2 ^ t - k * 2 ^ L = m * 2 ^ t).
    { unfold m. replace L with ((L - t) + t) at 1 by lia. rewrite Z.pow_add_r by lia. ring. }
    exists (Z.abs m), t. rewrite Ed, Z.abs_mul, (Z.abs_eq (2 ^ t)) by lia.
    split; [|split; [lia|reflexivity]].
    assert (Hm : Z.abs m * 2 ^ t <= 2 ^ (L - 1)) by (rewrite Ed, Z.abs_mul, (Z.abs_eq (2 ^ t)) in Hd by lia; exact Hd).
    assert (Hp : 2 ^ (L - 1) = 2 ^ (L - 1 - t) * 2 ^ t) by (rewrite <- Z.pow_add_r by lia; f_equal; lia).
    assert (Hq : 2 ^ (L - 1 - t) <= 2 ^ 23) by (apply Z.pow_le_mono_r; lia).
    rewrite Hp in Hm. assert (Z.abs m <= 2 ^ (L - 1 - t)) by nia.
    split; [lia|]. assert (2 ^ 23 < 2 ^ 24) by reflexivity. lia.
  - (* t >= L: x is an integer and the remainder a multiple of 2^L at most 2^(L-1): 0. *)
    assert (Ed : s * 2 ^ t - k * 2 ^ L = (s * 2 ^ (t - L) - k) * 2 ^ L).
    { replace t with ((t - L) + L) at 1 by lia. rewrite Z.pow_add_r by lia. ring. }
    rewrite Ed in *. rewrite Z.abs_mul, (Z.abs_eq (2 ^ L)) in Hd by lia.
    assert (Z0 : s * 2 ^ (t - L) - k = 0).
    { destruct (Z.eq_dec (s * 2 ^ (t - L) - k) 0) as [Z|NZ]; [exact Z|exfalso].
      assert (1 <= Z.abs (s * 2 ^ (t - L) - k)) by lia. nia. }
    exists 0, 0. rewrite Z0. split; [lia|split; [lia|reflexivity]].
Qed.

(** The refutation, in units of 2^-23: x = 2^24 - 1, the integer -2^24 is
    -2^47 units, and the difference 2^47 + 2^24 - 1 is odd and past 2^24. *)
Theorem a_far_integer_leaves_no_float32 :
  ~ (exists s' t', 0 <= s' < 2 ^ 24 /\ 0 <= t' /\ (2 ^ 24 - 1) - (- 2 ^ 24) * 2 ^ 23 = s' * 2 ^ t').
Proof.
  intros [s' [t' [Hs [Ht E]]]].
  destruct (Z.eq_dec t' 0) as [->|Ht0].
  - cbn in E. lia.
  - replace t' with (Z.succ (t' - 1)) in E by lia. rewrite Z.pow_succ_r in E by lia.
    assert (Odd : Z.odd ((2 ^ 24 - 1) - (- 2 ^ 24) * 2 ^ 23) = true) by reflexivity.
    rewrite E in Odd. rewrite Z.mul_comm, <- Z.mul_assoc, Z.odd_mul in Odd. cbn in Odd. discriminate.
Qed.

Print Assumptions a_float32_past_2_23_is_an_integer.
Print Assumptions the_nearest_integer_is_not_negative.
Print Assumptions the_remainder_is_a_float32.
Print Assumptions a_far_integer_leaves_no_float32.
