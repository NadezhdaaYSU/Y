(** * `q8`'s scale, rounding and clamp (`python/y_lang/exact/train_lm.py`).

    `q8(x, bits)` takes e with every |x| below 2^e (`exponent`, from the
    bits of the maximum; `ExactFloatOrder.the_field_bounds_the_value`), puts
    s = e - (bits - 1), rounds each x * 2^-s to the nearest integer (ties to
    even) and clamps it to top = 2^(bits-1) - 1. Here x is m * 2^-t for an
    integer m (every binary float is), so x * 2^-s is m / 2^r with
    r = t + s, and |x| < 2^e is m < 2^(r + bits - 1); the rounding is
    `ExactTrainArith.shift_round`, the nearest-even integer
    (`shift_round_is_nearest`). Magnitudes only: the code rounds |x| and
    puts the sign back (`rounding_commutes_with_sign`).

    [the_rounding_reaches_at_most_one_past_top]: the rounded value is at
    most 2^(bits-1), one past top. [the_clamped_value_is_in_range]: after
    the clamp, 0 <= q <= top. [the_clamped_error_is_below_one_step]: the
    error |m - q 2^r| stays below one step 2^r, clamped or not (below half
    a step where the clamp does nothing).

    ** Refutation, the clamp being load-bearing.

    [without_the_clamp_top_is_passed]: 8 bits, m = 255 at r = 1 (x =
    127.5 * 2^s, below 2^7 * 2^s): the rounding gives 128, past top 127.

    ** Not proved here.

    The scale range check (|s| <= 511,
    `ExactTrainArith.a_scale_of_512_could_leave_float64`). The float formula
    `round(x * 2^a * 2^b)` the GPU path computes is this rounding by
    `ExactScaledRound.v`.

    Checked with Rocq 9.1:  coqc -Q . "" ExactQ8Clamp.v  *)

From Stdlib Require Import ZArith Lia.
Require ExactTrainArith.
Open Scope Z_scope.

Module TA := ExactTrainArith.

Definition top (bits : Z) : Z := 2 ^ (bits - 1) - 1.
Definition q8_mag (bits m r : Z) : Z := Z.min (TA.shift_round m r) (top bits).

Section Q8.
  Variables bits m r : Z.
  Hypothesis Hbits : 1 <= bits.
  Hypothesis Hr : 1 <= r.
  Hypothesis Hm : 0 <= m < 2 ^ (r + bits - 1).

  Lemma pow_rb : 2 ^ (r + bits - 1) = 2 ^ (bits - 1) * 2 ^ r.
  Proof. rewrite <- Z.pow_add_r by lia. f_equal. lia. Qed.

  Lemma half_step : 2 ^ r = 2 * 2 ^ (r - 1) /\ 0 < 2 ^ (r - 1).
  Proof.
    split; [|apply Z.pow_pos_nonneg; lia].
    rewrite <- (Z.pow_1_r 2) at 2. rewrite <- Z.pow_add_r by lia. f_equal. lia.
  Qed.

  Lemma rounding_nonneg : 0 <= TA.shift_round m r.
  Proof.
    destruct (TA.shift_round_is_nearest m r ltac:(lia) Hr) as [A _]. destruct half_step as [E H].
    assert (0 < 2 ^ r) by lia. nia.
  Qed.

  Theorem the_rounding_reaches_at_most_one_past_top : TA.shift_round m r <= 2 ^ (bits - 1).
  Proof.
    destruct (TA.shift_round_is_nearest m r ltac:(lia) Hr) as [A _]. destruct half_step as [E H].
    pose proof pow_rb as P. assert (0 < 2 ^ (bits - 1)) by (apply Z.pow_pos_nonneg; lia).
    set (q := TA.shift_round m r) in *. set (R := 2 ^ r) in *. set (H2 := 2 ^ (r - 1)) in *.
    set (T := 2 ^ (bits - 1)) in *.
    (* q R <= m + R/2 < T R + R/2, so q < T + 1/2 *)
    assert (q * R < T * R + R) by nia. nia.
  Qed.

  Theorem the_clamped_value_is_in_range : 0 <= q8_mag bits m r <= top bits.
  Proof.
    unfold q8_mag, top. pose proof rounding_nonneg.
    assert (1 <= 2 ^ (bits - 1)) by (change 1 with (2 ^ 0); apply Z.pow_le_mono_r; lia).
    lia.
  Qed.

  Theorem the_clamped_error_is_below_one_step : Z.abs (m - q8_mag bits m r * 2 ^ r) < 2 ^ r.
  Proof.
    destruct (TA.shift_round_is_nearest m r ltac:(lia) Hr) as [A _]. destruct half_step as [E H].
    pose proof the_rounding_reaches_at_most_one_past_top as U. pose proof pow_rb as P.
    unfold q8_mag, top.
    set (q := TA.shift_round m r) in *. set (R := 2 ^ r) in *. set (H2 := 2 ^ (r - 1)) in *.
    set (T := 2 ^ (bits - 1)) in *.
    destruct (Z.le_gt_cases q (T - 1)) as [L|G].
    - rewrite Z.min_l by exact L. lia.
    - rewrite Z.min_r by lia. assert (q = T) by lia. subst q.
      (* m >= T R - R/2, m < T R: the clamped error is in (R/2, R) *)
      assert (m - (T - 1) * R = (m - T * R) + R) by ring. nia.
  Qed.
End Q8.

Theorem without_the_clamp_top_is_passed :
  255 < 2 ^ (1 + 8 - 1) /\ TA.shift_round 255 1 = 128 /\ 128 > top 8.
Proof. vm_compute. repeat split; reflexivity. Qed.

Print Assumptions the_rounding_reaches_at_most_one_past_top.
Print Assumptions the_clamped_value_is_in_range.
Print Assumptions the_clamped_error_is_below_one_step.
Print Assumptions without_the_clamp_top_is_passed.
