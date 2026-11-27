(** * The MLP kernel's integer arithmetic (`python/y_lang/exact/mlp_kernel.py`).

    `_mlp_kernel` computes the MLP's gate and up projections and their glue
    in one Triton kernel (2026-11-27). Its float32 steps are the separate
    path's, operation for operation; that is checked bit for bit by
    python/tests/test_exact_mlp_kernel.py, not proved here. What is proved
    here is the integer and bit-field arithmetic around them, where an
    overflow or an out-of-range index would be silent:

    ** The K loop ([the_blocked_loop_is_the_dot]).

    The kernel adds `tl.dot` of BK-wide blocks of x8 and the weight, with
    every load past K masked to 0, over ceil(K / BK) blocks. That is the
    dot product sum_{k<K} x_k w_k exactly, for every BK > 0
    ([ceil_blocks_cover]: ceil(K / BK) blocks reach K).

    ** No int32 overflow ([every_partial_sum_fits_int32]).

    The int32 accumulator and `tl.dot`'s tensor cores add the terms in an
    order the kernel does not choose. Every partial sum is a sum of at most
    K terms, each |x w| <= 127 * 127, so it fits int32 whenever
    K <= 133,144 ([the_guard_is_inside]: `usable` refuses K past
    `qmm.MAX_K` = 131,040). [past_133144_a_sum_can_overflow]: at one more
    term, 127s everywhere leave int32.

    ** The sigmoid table ([the_table_index_is_in_bounds]).

    `q = rint(xf * 2^20) + 16 * 2^20` with xf clamped to [-16, 16], then the
    index `min(q >> 12, TN - 1)`: always in [0, TN - 1]. The lower clamp is
    the load-bearing one ([without_the_lower_clamp_the_index_is_negative]:
    the `min` bounds the index only from above).
    [the_interpolation_fits_int32]: `t + ((d * f) >> 12)`, f = q & 4095,
    stays in int32 for a table with 0 <= t <= 2^24 and 0 <= d <= 2^14.

    ** The scales' bit fields ([the_float32_field_is_normal_exactly_when_ok]).

    The adapter product's scale 2^e is built as the float32 bits
    (e + 127) << 23 where `ok`, -126 <= e <= 127: exactly the e whose
    biased field is a normal one, 1 to 254 ([the_shifted_field_fits_int32]:
    the shift stays in int32). [at_minus_127_the_field_is_zero] and
    [at_128_the_field_is_all_ones]: one step outside, the bits are a zero or
    an infinity, not 2^e, which is why those tiles take the float64 product,
    whose field ([the_float64_field_is_normal]) is normal for every
    |sa|, |sb| <= 511 (`q8` refuses others).

    ** Not proved here.

    The float32 roundings (`acc * sx * sw + z * scale`, `g * sig`,
    `a * yu`) equal the separate path's: the same IEEE operations on the
    same operands, FMA contraction off (tested bit for bit, with mutants of
    the order, of contraction and of the rounding mode failing). That the
    built table meets the bounds [the_interpolation_fits_int32] assumes:
    python/tests/test_exact_mlp_kernel.py checks the table. That the decoded
    bits are IEEE's (the field reading, as `ExactTrainArith` takes it).

    Checked with Rocq 9.1:  coqc -Q . "" ExactMlpKernel.v  *)

From Stdlib Require Import ZArith Lia List.
Import ListNotations.
Open Scope Z_scope.

(* ------------------------------------------------------------------ *)
(** ** The K loop                                                      *)
(* ------------------------------------------------------------------ *)

Fixpoint sumto (n : nat) (f : nat -> Z) : Z :=
  match n with
  | O => 0
  | S m => sumto m f + f m
  end.

Lemma sumto_ext : forall n f g, (forall k, (k < n)%nat -> f k = g k) -> sumto n f = sumto n g.
Proof.
  induction n as [|n IH]; intros f g H; cbn [sumto]; [reflexivity|].
  rewrite (IH f g) by (intros; apply H; lia). rewrite (H n) by lia. reflexivity.
Qed.

Lemma sumto_add : forall a b f, sumto (a + b) f = sumto a f + sumto b (fun j => f (a + j)%nat).
Proof.
  intros a b f. induction b as [|b IH].
  - rewrite Nat.add_0_r. cbn [sumto]. lia.
  - rewrite Nat.add_succ_r. cbn [sumto]. rewrite IH. lia.
Qed.

Lemma sumto_blocks : forall nb bk f,
  sumto nb (fun b => sumto bk (fun j => f (b * bk + j)%nat)) = sumto (nb * bk) f.
Proof.
  induction nb as [|nb IH]; intros bk f; cbn [sumto]; [reflexivity|].
  rewrite IH, Nat.mul_succ_l, sumto_add. reflexivity.
Qed.

(** A load past K reads 0 (`mask=..., other=0`). *)
Definition masked (K : nat) (f : nat -> Z) (k : nat) : Z := if (k <? K)%nat then f k else 0.

Lemma sumto_masked : forall n K f, (K <= n)%nat -> sumto n (masked K f) = sumto K f.
Proof.
  intros n K f H. replace n with (K + (n - K))%nat by lia. rewrite sumto_add.
  assert (Z0 : sumto (n - K) (fun j => masked K f (K + j)%nat) = 0).
  { generalize (n - K)%nat as m. induction m as [|m IH]; cbn [sumto]; [reflexivity|].
    rewrite IH. unfold masked. destruct (Nat.ltb_spec (K + m) K); [lia | reflexivity]. }
  rewrite Z0, Z.add_0_r. apply sumto_ext. intros k Hk. unfold masked.
  destruct (Nat.ltb_spec k K); [reflexivity | lia].
Qed.

Lemma ceil_blocks_cover : forall K BK, (0 < BK)%nat -> (K <= ((K + BK - 1) / BK) * BK)%nat.
Proof.
  intros K BK H. pose proof (Nat.div_mod_eq (K + BK - 1) BK) as E.
  pose proof (Nat.mod_upper_bound (K + BK - 1) BK ltac:(lia)) as B. nia.
Qed.

Theorem the_blocked_loop_is_the_dot : forall K BK f, (0 < BK)%nat ->
  sumto ((K + BK - 1) / BK) (fun b => sumto BK (fun j => masked K f (b * BK + j)%nat)) = sumto K f.
Proof.
  intros K BK f H. rewrite (sumto_blocks _ _ (masked K f)).
  apply sumto_masked, ceil_blocks_cover, H.
Qed.

(* ------------------------------------------------------------------ *)
(** ** No int32 overflow                                               *)
(* ------------------------------------------------------------------ *)

Definition in_int32 (z : Z) : Prop := - 2 ^ 31 <= z <= 2 ^ 31 - 1.

Lemma a_term_is_bounded : forall x w, -127 <= x <= 127 -> -127 <= w <= 127 -> -16129 <= x * w <= 16129.
Proof. intros x w Hx Hw. nia. Qed.

Lemma sum_bounded : forall ts : list Z, Forall (fun t => -16129 <= t <= 16129) ts ->
  - (Z.of_nat (length ts) * 16129) <= fold_right Z.add 0 ts <= Z.of_nat (length ts) * 16129.
Proof.
  intros ts H. induction H as [|t ts Ht _ IH]; cbn [fold_right length]; [lia|].
  rewrite Nat2Z.inj_succ. lia.
Qed.

(** Every partial sum, in any order, is a sum of a list of at most K terms. *)
Theorem every_partial_sum_fits_int32 : forall ts : list Z,
  Forall (fun t => -16129 <= t <= 16129) ts -> Z.of_nat (length ts) <= 133144 ->
  in_int32 (fold_right Z.add 0 ts).
Proof.
  intros ts H L. pose proof (sum_bounded ts H) as B. unfold in_int32.
  change (2 ^ 31) with 2147483648. lia.
Qed.

(** `usable`'s guard, K <= `qmm.MAX_K`. *)
Theorem the_guard_is_inside : 131040 <= 133144.
Proof. lia. Qed.

(** One more term: 133,145 products of 127 * 127, the largest sum, leaves int32. *)
Theorem past_133144_a_sum_can_overflow : ~ in_int32 (133145 * (127 * 127)).
Proof. unfold in_int32. change (2 ^ 31) with 2147483648. lia. Qed.

(* ------------------------------------------------------------------ *)
(** ** The sigmoid table                                               *)
(* ------------------------------------------------------------------ *)

(** r = rint(xf * 2^20): |xf| <= 16 puts it in [-2^24, 2^24]. The index is
    min(q >> 12, TN - 1), the shift a floor division. *)
Definition table_index (r TN : Z) : Z := Z.min ((r + 16 * 2 ^ 20) / 2 ^ 12) (TN - 1).

Theorem the_table_index_is_in_bounds : forall r TN,
  - 2 ^ 24 <= r <= 2 ^ 24 -> 1 <= TN -> 0 <= table_index r TN <= TN - 1.
Proof.
  intros r TN Hr HT. unfold table_index.
  assert (0 <= (r + 16 * 2 ^ 20) / 2 ^ 12) by (apply Z.div_pos; lia).
  lia.
Qed.

Theorem without_the_lower_clamp_the_index_is_negative :
  table_index (- 2 ^ 24 - 2 ^ 12) 8193 < 0.
Proof. unfold table_index. cbv. reflexivity. Qed.

Theorem the_interpolation_fits_int32 : forall t d f,
  0 <= t <= 2 ^ 24 -> 0 <= d <= 2 ^ 14 -> 0 <= f < 2 ^ 12 ->
  in_int32 (d * f) /\ in_int32 (t + (d * f) / 2 ^ 12).
Proof.
  intros t d f Ht Hd Hf. unfold in_int32.
  assert (0 <= d * f <= 2 ^ 26) by nia.
  assert (0 <= (d * f) / 2 ^ 12 <= 2 ^ 14).
  { split; [apply Z.div_pos; lia|]. apply Z.div_le_upper_bound; [lia|]. nia. }
  change (2 ^ 31) with 2147483648 in *. change (2 ^ 26) with 67108864 in *.
  change (2 ^ 24) with 16777216 in *. change (2 ^ 14) with 16384 in *. lia.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The scales' bit fields                                          *)
(* ------------------------------------------------------------------ *)

(** float32's normal biased exponent fields are 1 .. 254 (0: zero and
    subnormals; 255: infinities and NaNs). *)
Theorem the_float32_field_is_normal_exactly_when_ok : forall e,
  (1 <= e + 127 <= 254) <-> (-126 <= e <= 127).
Proof. intro e. lia. Qed.

Theorem the_shifted_field_fits_int32 : forall e,
  -126 <= e <= 127 -> 0 <= (e + 127) * 2 ^ 23 /\ in_int32 ((e + 127) * 2 ^ 23).
Proof. intros e H. unfold in_int32. change (2 ^ 23) with 8388608. change (2 ^ 31) with 2147483648. lia. Qed.

Theorem at_minus_127_the_field_is_zero : -127 + 127 = 0.
Proof. reflexivity. Qed.

Theorem at_128_the_field_is_all_ones : 128 + 127 = 255.
Proof. reflexivity. Qed.

(** float64's normal fields are 1 .. 2046. *)
Theorem the_float64_field_is_normal : forall sa sb,
  -511 <= sa <= 511 -> -511 <= sb <= 511 -> 1 <= sa + sb + 1023 <= 2046.
Proof. intros sa sb Ha Hb. lia. Qed.

Print Assumptions the_blocked_loop_is_the_dot.
Print Assumptions every_partial_sum_fits_int32.
Print Assumptions the_guard_is_inside.
Print Assumptions past_133144_a_sum_can_overflow.
Print Assumptions the_table_index_is_in_bounds.
Print Assumptions without_the_lower_clamp_the_index_is_negative.
Print Assumptions the_interpolation_fits_int32.
Print Assumptions the_float32_field_is_normal_exactly_when_ok.
Print Assumptions the_shifted_field_fits_int32.
Print Assumptions the_float64_field_is_normal.
