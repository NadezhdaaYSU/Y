(** * The quantizer's storage types (`python/y_lang/exact/train_lm.py`).

    Since 2026-11-27 `q8` returns the narrowest integer type for its width
    (`q_dtype`): int8 up to 8 bits, int16 up to 15, int32 above. `digits`
    then splits a 12-bit value in int16 arithmetic, and `exact_product`
    stacks the digits as int8. The digit split itself is proved over the
    integers in `ExactTrainArith`; this file proves that the machine types
    hold every value the code forms, so the narrower types change no bits.

    [q8_fits_its_type]: a clamped value, |q| <= 2^(bits-1) - 1, fits the
    type `q_dtype` gives for its width, for every width from 1 to 32 bits.

    [int16_holds_every_step]: for |r| <= 2^14 - 1, every value each step of
    the split forms (r, r + 64, the shifted carry 128 * hi, the digit) is
    in int16, for any number of digits, because each carry is in [-128,
    128] and so is again in reach. [a_15_bit_split_stays_in_int16]: every
    clamped value of 9 to 15 bits starts in that range.

    [the_split_is_safe_wherever_q_dtype_gives_int16] joins the two at the
    cut: wherever `q_dtype` gives int16 to a split width, the split stays in
    int16.

    [the_stacked_digits_fit_int8]: every digit is in int8, so the int8
    stack loses nothing.

    ** Refutations, the bounds being load-bearing.

    [int16_wraps_at_16_bits]: at 16 bits the largest clamped value plus 64
    leaves int16, which is why `q_dtype` gives int32 there.
    [int8_cannot_hold_9_bits]: the largest 9-bit value leaves int8.

    ** Not proved here.

    That torch's int8 and int16 arithmetic is two's complement with an
    arithmetic right shift (the tests compare the digits on the CPU and the
    GPU), and that every value reaching `digits` was clamped first (the clamp
    is in `q8`, in int32, before the cast).

    Checked with Rocq 9.1:  coqc -Q . "" ExactTrainWidths.v  *)

From Stdlib Require Import ZArith Lia List.
Require ExactTrainArith.
Import ListNotations.
Open Scope Z_scope.

Module TA := ExactTrainArith.

(** A two's complement integer of [w] bits. *)
Definition fits (w z : Z) : Prop := - 2 ^ (w - 1) <= z <= 2 ^ (w - 1) - 1.

(** `q_dtype`: int8 to 8 bits, int16 to 15, int32 above. *)
Definition q_width (bits : Z) : Z :=
  if bits <=? 8 then 8 else if bits <=? 15 then 16 else 32.

Lemma pow2_le : forall a b, 0 <= a <= b -> 2 ^ a <= 2 ^ b.
Proof. intros a b H. apply Z.pow_le_mono_r; lia. Qed.

(** A clamped value of [bits] fits every type at least that wide. *)
Lemma top_fits : forall bits w q,
  1 <= bits <= w -> - TA.top bits <= q <= TA.top bits -> fits w q.
Proof.
  intros bits w q Hb Hq. unfold fits, TA.top in *.
  pose proof (pow2_le (bits - 1) (w - 1) ltac:(lia)). lia.
Qed.

Theorem q8_fits_its_type : forall bits q,
  1 <= bits <= 32 -> - TA.top bits <= q <= TA.top bits -> fits (q_width bits) q.
Proof.
  intros bits q Hb Hq. unfold q_width.
  (* Split on whatever tests the definition makes, so a changed cut is
     judged by the theorem, not by this script. *)
  repeat match goal with |- context [if ?c then _ else _] => destruct c eqn:? end;
    rewrite ?Z.leb_le, ?Z.leb_gt in *; apply (top_fits bits); first [exact Hq | lia].
Qed.

Definition in_int16 (z : Z) : Prop := -32768 <= z <= 32767.
Definition in_int8 (z : Z) : Prop := -128 <= z <= 127.

(** Every value the split forms, step by step: `hi = (r + 64) >> 7`,
    `d = r - (hi << 7)`, `r = hi`. *)
Fixpoint split_in_int16 (n : nat) (r : Z) : Prop :=
  match n with
  | O => in_int16 r
  | S m => in_int16 r /\ in_int16 (r + 64) /\ in_int16 (128 * TA.carry r)
           /\ in_int16 (TA.digit r) /\ split_in_int16 m (TA.carry r)
  end.

Lemma carry_small : forall r, -16383 <= r <= 16383 -> -128 <= TA.carry r <= 128.
Proof. intros r H. destruct (TA.carry_spec r) as [E B]. lia. Qed.

Theorem int16_holds_every_step : forall n r,
  -16383 <= r <= 16383 -> split_in_int16 n r.
Proof.
  induction n as [|n IH]; intros r H.
  - cbn [split_in_int16]. unfold in_int16. lia.
  - cbn [split_in_int16]. pose proof (carry_small r H) as C. pose proof (TA.digit_range r) as D.
    unfold in_int16. split; [lia|]. split; [lia|]. split; [lia|]. split; [lia|].
    apply IH. lia.
Qed.

Corollary a_15_bit_split_stays_in_int16 : forall bits q n,
  9 <= bits <= 15 -> - TA.top bits <= q <= TA.top bits -> split_in_int16 n q.
Proof.
  intros bits q n Hb Hq. apply int16_holds_every_step. unfold TA.top in Hq.
  pose proof (pow2_le (bits - 1) 14 ltac:(lia)) as P.
  change (2 ^ 14) with 16384 in P. lia.
Qed.

(** The two facts joined at `q_dtype`'s cut: wherever it gives int16 to a
    width that `digits` splits, the split stays in int16. Moving the cut to
    16 bits breaks this theorem ([int16_wraps_at_16_bits]). *)
Theorem the_split_is_safe_wherever_q_dtype_gives_int16 : forall bits q n,
  9 <= bits -> q_width bits = 16 -> - TA.top bits <= q <= TA.top bits -> split_in_int16 n q.
Proof.
  intros bits q n H9 W Hq. unfold q_width in W.
  repeat match type of W with context [if ?c then _ else _] => destruct c eqn:? end;
    try discriminate; rewrite ?Z.leb_le, ?Z.leb_gt in *.
  apply (a_15_bit_split_stays_in_int16 bits); [lia | exact Hq].
Qed.

Lemma int8_of_digit_range : forall l,
  Forall (fun d => -64 <= d <= 63) l -> Forall in_int8 l.
Proof.
  intros l H. induction H as [|d l Hd _ IH]; constructor; [unfold in_int8; lia | exact IH].
Qed.

Theorem the_stacked_digits_fit_int8 : forall n r, Forall in_int8 (TA.digits n r).
Proof. intros n r. apply int8_of_digit_range, TA.every_digit_is_int8. Qed.

Theorem int16_wraps_at_16_bits : ~ in_int16 (TA.top 16 + 64).
Proof. unfold in_int16, TA.top. change (2 ^ (16 - 1)) with 32768. lia. Qed.

Theorem int8_cannot_hold_9_bits : ~ in_int8 (TA.top 9).
Proof. unfold in_int8, TA.top. change (2 ^ (9 - 1)) with 256. lia. Qed.

Print Assumptions top_fits.
Print Assumptions q8_fits_its_type.
Print Assumptions int16_holds_every_step.
Print Assumptions a_15_bit_split_stays_in_int16.
Print Assumptions the_split_is_safe_wherever_q_dtype_gives_int16.
Print Assumptions the_stacked_digits_fit_int8.
Print Assumptions int16_wraps_at_16_bits.
Print Assumptions int8_cannot_hold_9_bits.
