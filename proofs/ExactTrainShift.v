(** * The integer helpers of exact training T1 (`python/y_lang/exact/train.py`).

    `train.py` keeps every value an int64 with a power-of-two scale. Three
    helpers carry its arithmetic:

    - `bit_length(m)`: the number of powers 2^0 .. 2^62 at most m.
    - `shift(v, e)`: v * 2^e; for e < 0, `(v + 2^(d-1)) >> d` with d = -e.
    - `quantize_rows(v)`: s = max(bit_length(max |v|) - 7, 0) and
      q = shift(v, -s) clamped to -127 .. 127.

    [bit_length_is_the_binary_length]: for 0 <= m < 2^63 the count is 0 at
    0 and floor(log2 m) + 1 above. [the_right_shift_rounds_half_up]: for
    d >= 1, q = (v + 2^(d-1)) >> d is the nearest integer to v / 2^d with
    ties toward +inf: q 2^d - 2^(d-1) <= v < q 2^d + 2^(d-1), for every v,
    negative ones included. [the_quantized_row_is_within_128]: each
    rounded value is in -128 .. 128, so after the clamp in -127 .. 127.

    ** Refutations.

    [the_clamp_acts_at_both_ends]: v = 511 rounds to 128 and v = -511 to
    -128 (s = 2), so the clamp is needed on both sides; the docstring's "one
    clamp of a rounded-up 128" names only the top one.
    [a_right_shift_can_leave_int64]: `shift` refuses a left shift that
    would leave int64 but checks nothing for a right one, and v + 2^(d-1)
    passes 2^63 - 1 for v near the top of int64 (v = 2^63 - 1, d = 62). The
    callers' values are far smaller; the bound is theirs, not `shift`'s.

    ** Not proved here.

    That torch's int64 `>>` is an arithmetic shift (floor division) and
    `<<` a multiply; the callers' magnitudes.

    Checked with Rocq 9.1:  coqc -Q . "" ExactTrainShift.v  *)

From Stdlib Require Import ZArith Lia.
Require ExactDrawSearch.
Open Scope Z_scope.

Module DS := ExactDrawSearch.

(** `(m.unsqueeze(-1) >= p).sum(-1)` with p = 2^0 .. 2^62. *)
Definition bit_length (m : Z) : nat := DS.count 63 (fun i => 2 ^ Z.of_nat i <=? m).

Theorem bit_length_is_the_binary_length : forall m, 0 <= m < 2 ^ 63 ->
  Z.of_nat (bit_length m) = if m =? 0 then 0 else Z.log2 m + 1.
Proof.
  intros m Hm. unfold bit_length.
  assert (HP : DS.prefix_pred 63 (fun i => 2 ^ Z.of_nat i <=? m)).
  { intros i i' H Hi'. apply Z.leb_le in Hi'. apply Z.leb_le.
    assert (2 ^ Z.of_nat i <= 2 ^ Z.of_nat i') by (apply Z.pow_le_mono_r; lia). lia. }
  destruct (m =? 0) eqn:Z0.
  - apply Z.eqb_eq in Z0. subst m.
    rewrite (DS.count_unique 63 _ 0 HP ltac:(lia)); [reflexivity|].
    intros i Hi. split; [|lia]. intro H. apply Z.leb_le in H.
    assert (0 < 2 ^ Z.of_nat i) by (apply Z.pow_pos_nonneg; lia). lia.
  - apply Z.eqb_neq in Z0.
    assert (Hlog : 0 <= Z.log2 m < 63).
    { split; [apply Z.log2_nonneg|]. apply Z.log2_lt_pow2; lia. }
    rewrite (DS.count_unique 63 _ (Z.to_nat (Z.log2 m + 1)) HP ltac:(lia)); [lia|].
    intros i Hi. destruct (Z.log2_spec m ltac:(lia)) as [L1 L2]. split.
    + intro H. apply Z.leb_le in H. destruct (Nat.lt_ge_cases i (Z.to_nat (Z.log2 m + 1))) as [|G]; [assumption|].
      assert (2 ^ Z.succ (Z.log2 m) <= 2 ^ Z.of_nat i) by (apply Z.pow_le_mono_r; lia). lia.
    + intro H. apply Z.leb_le.
      assert (2 ^ Z.of_nat i <= 2 ^ Z.log2 m) by (apply Z.pow_le_mono_r; lia). lia.
Qed.

(** `shift(v, -d)` for d >= 1. *)
Definition shift_down (v d : Z) : Z := Z.shiftr (v + 2 ^ (d - 1)) d.

Theorem the_right_shift_rounds_half_up : forall v d, 1 <= d ->
  let q := shift_down v d in
  q * 2 ^ d - 2 ^ (d - 1) <= v < q * 2 ^ d + 2 ^ (d - 1).
Proof.
  intros v d Hd q. unfold q, shift_down. rewrite Z.shiftr_div_pow2 by lia.
  assert (P : 2 ^ d = 2 * 2 ^ (d - 1)) by (rewrite <- Z.pow_succ_r by lia; f_equal; lia).
  assert (0 < 2 ^ (d - 1)) by (apply Z.pow_pos_nonneg; lia).
  pose proof (Z.div_mod (v + 2 ^ (d - 1)) (2 ^ d) ltac:(lia)).
  pose proof (Z.mod_pos_bound (v + 2 ^ (d - 1)) (2 ^ d) ltac:(lia)). lia.
Qed.

(** The row's shift s and one value v with |v| <= m, m the row's max. *)
Definition row_shift (m : Z) : Z := Z.max (Z.of_nat (bit_length m) - 7) 0.

Definition quantized (v m : Z) : Z :=
  let s := row_shift m in if s =? 0 then v else shift_down v s.

Theorem the_quantized_row_is_within_128 : forall v m, 0 <= m < 2 ^ 63 -> Z.abs v <= m ->
  -128 <= quantized v m <= 128 /\ -127 <= Z.max (-127) (Z.min 127 (quantized v m)) <= 127.
Proof.
  intros v m Hm Hv. split; [|lia]. unfold quantized, row_shift.
  pose proof (bit_length_is_the_binary_length m Hm) as BL.
  destruct (m =? 0) eqn:Z0.
  - apply Z.eqb_eq in Z0. subst m. rewrite BL. cbn. lia.
  - apply Z.eqb_neq in Z0. rewrite BL. destruct (Z.log2_spec m ltac:(lia)) as [L1 L2].
    set (s := Z.max (Z.log2 m + 1 - 7) 0).
    destruct (s =? 0) eqn:S0.
    + apply Z.eqb_eq in S0. unfold s in S0.
      assert (Z.log2 m <= 6) by lia. assert (2 ^ Z.succ (Z.log2 m) <= 2 ^ 7) by (apply Z.pow_le_mono_r; lia).
      change (2 ^ 7) with 128 in *. lia.
    + apply Z.eqb_neq in S0. assert (Hs : s = Z.log2 m - 6) by (unfold s in *; lia).
      pose proof (the_right_shift_rounds_half_up v s ltac:(lia)) as R. cbv zeta in R.
      set (q := shift_down v s) in *.
      (* |v| <= m < 2^(log2 m + 1) = 2^(s + 7) = 128 * 2^s *)
      assert (B : 2 ^ Z.succ (Z.log2 m) = 128 * 2 ^ s).
      { rewrite Hs. change 128 with (2 ^ 7). rewrite <- Z.pow_add_r by lia. f_equal. lia. }
      assert (P : 2 ^ s = 2 * 2 ^ (s - 1)) by (rewrite <- Z.pow_succ_r by lia; f_equal; lia).
      assert (0 < 2 ^ (s - 1)) by (apply Z.pow_pos_nonneg; lia).
      nia.
Qed.

Theorem the_clamp_acts_at_both_ends :
  quantized 511 511 = 128 /\ quantized (-511) 511 = -128.
Proof. vm_compute. split; reflexivity. Qed.

Theorem a_right_shift_can_leave_int64 :
  (2 ^ 63 - 1) + 2 ^ (62 - 1) > 2 ^ 63 - 1.
Proof. vm_compute. reflexivity. Qed.

Print Assumptions bit_length_is_the_binary_length.
Print Assumptions the_right_shift_rounds_half_up.
Print Assumptions the_quantized_row_is_within_128.
Print Assumptions the_clamp_acts_at_both_ends.
Print Assumptions a_right_shift_can_leave_int64.
