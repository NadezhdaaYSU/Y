(** * Powers of two from their bits (`python/y_lang/exact/train_lm.py`).

    The scaled roundings multiply by 2^a and 2^b built from bits, not by a
    float power: `pow2(k)` is `(k + 1023) << 52` read as a float64 and
    `_pow2_f32(k)` is `(k + 127) << 23` read as a float32. `ExactScaledRound.v`
    and `ExactScaledRoundF64.v` take each factor to be the exact power of
    two; this file proves the bits are it, for the ranges the clamps keep.

    [pow2_is_two_to_the_k]: for -1022 <= k <= 1023 the bits are those of the
    normal float64 with field k + 1023 and mantissa 0, whose value is 2^k
    (`ExactFloatOrder.value 52` in units of 2^-1074: 2^(k + 1074)).
    [pow2_f32_is_two_to_the_k]: for -126 <= k <= 127 the same for float32
    (units of 2^-149: 2^(k + 149)).

    ** Refutations.

    [past_the_clamps_the_bits_are_not_the_power]: k = -1023 gives the field
    0, the bits of +0, not 2^-1023; k = 1024 gives the field 2047, the bits
    of infinity. So `check_exp`'s range and the clamps at -1022 and 1023
    (-126 and 127) are what make the factors exact.

    ** Not proved here.

    That `.view(torch.float64)` reads the bits as IEEE storage.

    Checked with Rocq 9.1:  coqc -Q . "" ExactPow2Bits.v  *)

From Stdlib Require Import ZArith Lia.
Require ExactFloatOrder.
Open Scope Z_scope.

Module FO := ExactFloatOrder.

Theorem pow2_is_two_to_the_k : forall k, -1022 <= k <= 1023 ->
  Z.shiftl (k + 1023) 52 = FO.bits 52 (k + 1023) 0 /\ 1 <= k + 1023 <= 2046 /\
  FO.value 52 (k + 1023) 0 = 2 ^ (k + 1074).
Proof.
  intros k Hk. unfold FO.bits, FO.value. rewrite Z.shiftl_mul_pow2 by lia.
  destruct (k + 1023 =? 0) eqn:Z0; [apply Z.eqb_eq in Z0; lia|].
  repeat split; try lia. rewrite Z.add_0_r, <- Z.pow_add_r by lia. f_equal. lia.
Qed.

Theorem pow2_f32_is_two_to_the_k : forall k, -126 <= k <= 127 ->
  Z.shiftl (k + 127) 23 = FO.bits 23 (k + 127) 0 /\ 1 <= k + 127 <= 254 /\
  FO.value 23 (k + 127) 0 = 2 ^ (k + 149).
Proof.
  intros k Hk. unfold FO.bits, FO.value. rewrite Z.shiftl_mul_pow2 by lia.
  destruct (k + 127 =? 0) eqn:Z0; [apply Z.eqb_eq in Z0; lia|].
  repeat split; try lia. rewrite Z.add_0_r, <- Z.pow_add_r by lia. f_equal. lia.
Qed.

Theorem past_the_clamps_the_bits_are_not_the_power :
  FO.value 52 (-1023 + 1023) 0 = 0 /\ 2 ^ (-1023 + 1074) <> 0 /\ 1024 + 1023 = 2047 /\
  FO.value 23 (-127 + 127) 0 = 0 /\ 128 + 127 = 255.
Proof. vm_compute. repeat split; try reflexivity; discriminate. Qed.

Print Assumptions pow2_is_two_to_the_k.
Print Assumptions pow2_f32_is_two_to_the_k.
Print Assumptions past_the_clamps_the_bits_are_not_the_power.
