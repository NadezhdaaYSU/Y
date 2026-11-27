(** * The exact SiLU's interpolation (`python/y_lang/exact/model.py`, `ExactSiLU`).

    `ExactSiLU.sigmoid` quantizes x (clamped to -16 .. 16) to q = round(x *
    2^20), splits it into a knot index i and an offset f (12 bits), and
    interpolates the host-built table: s = t[i] + (d[i] * f) >> 12, with
    d[i] = t[i+1] - t[i]; the sigmoid is s * 2^-24. Its docstring claims
    q's rounding is exact, the interpolation cannot overflow int32 (that is
    `ExactMlpKernel.the_interpolation_fits_int32`), and "the sum is at most
    2^24, so its conversion to float32 and the 2^-24 scale are exact".

    [the_interpolation_is_between_the_knots]: for a non-decreasing table
    step (d >= 0) and 0 <= f < 2^12, t[i] <= s <= t[i+1]; so with the
    table in 0 .. 2^24, s is too. [the_interpolated_sigmoid_is_a_float32]:
    an integer in 0 .. 2^24 times 2^-24 is a float32 (a normal one or 0),
    so the conversion and the scale are exact.
    [the_quantized_x_is_the_exact_rounding]: q = round(x * 2^20) for a
    float32 |x| <= 16 is the round-half-even of x * 2^20
    (`ExactScaledRound.the_float32_formula_is_the_exact_rounding` at
    ns = 20, one factor).

    ** Refutation.

    [without_the_mask_the_offset_overshoots]: the offset is `q & (2^12 - 1)`;
    an offset of 2^13 would put s at t + 2 d, past the next knot, so the
    bound needs the mask.

    ** Not proved here.

    The table's values (built in float64 on the host, checked by
    `exact_selftest.check_silu_table` and python/tests/test_exact_mlp_kernel.py:
    0 <= t <= 2^24, non-decreasing); the final multiply x * sigmoid (one
    IEEE rounding, the same everywhere).

    Checked with Rocq 9.1:  coqc -Q . "" ExactSiluTable.v  *)

From Stdlib Require Import ZArith Lia.
Require ExactFloatOrder ExactScaledRound.
Open Scope Z_scope.

Module FO := ExactFloatOrder.
Module SR := ExactScaledRound.

Definition interp (t d f : Z) : Z := t + Z.shiftr (d * f) 12.

Theorem the_interpolation_is_between_the_knots : forall t d f,
  0 <= d -> 0 <= f < 2 ^ 12 ->
  t <= interp t d f <= t + d /\ (0 <= t -> t + d <= 2 ^ 24 -> 0 <= interp t d f <= 2 ^ 24).
Proof.
  intros t d f Hd Hf. unfold interp. rewrite Z.shiftr_div_pow2 by lia.
  assert (0 <= d * f / 2 ^ 12) by (apply Z.div_pos; nia).
  assert (d * f / 2 ^ 12 <= d) by (apply Z.div_le_upper_bound; nia).
  split; [lia|]. intros; lia.
Qed.

(** s in 0 .. 2^24, times 2^-24: in float32's units of 2^-149, s * 2^125. *)
Theorem the_interpolated_sigmoid_is_a_float32 : forall s, 0 <= s <= 2 ^ 24 ->
  exists E M, 0 <= E <= 254 /\ 0 <= M < 2 ^ 23 /\ FO.value 23 E M = s * 2 ^ 125.
Proof.
  intros s Hs. destruct (Z.eq_dec s (2 ^ 24)) as [->|Hne].
  - exists 127, 0. repeat split; try lia; try reflexivity.
  - apply SR.an_integer_times_a_power_of_two_is_a_float32; lia.
Qed.

Section Quantize.
  Variable rn : Z -> Z.
  Hypothesis rn_fix : forall E M, 0 <= E <= 254 -> 0 <= M < 2 ^ 23 ->
    rn (FO.value 23 E M * 2 ^ 126) = FO.value 23 E M * 2 ^ 126.
  Hypothesis rn_mono : forall v w, v <= w -> rn v <= rn w.

  (** |x| <= 16: in units of 2^-149, value <= 2^153. *)
  Theorem the_quantized_x_is_the_exact_rounding : forall E M,
    0 <= E <= 254 -> 0 <= M < 2 ^ 23 -> FO.value 23 E M <= 2 ^ 153 ->
    SR.formula rn E M 20 = ExactTrainArith.shift_round (FO.value 23 E M * 2 ^ (126 + 20)) 275.
  Proof.
    intros E M HE HM Hx. apply SR.the_float32_formula_is_the_exact_rounding; try assumption; lia.
  Qed.
End Quantize.

(** An offset of 2^13 (13 bits, past the mask): t = 0, d = 4096. *)
Theorem without_the_mask_the_offset_overshoots :
  interp 0 4096 (2 ^ 13) = 8192 /\ 8192 > 0 + 4096.
Proof. vm_compute. split; reflexivity. Qed.

Print Assumptions the_interpolation_is_between_the_knots.
Print Assumptions the_interpolated_sigmoid_is_a_float32.
Print Assumptions the_quantized_x_is_the_exact_rounding.
Print Assumptions without_the_mask_the_offset_overshoots.
