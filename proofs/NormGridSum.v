(** * The norm's sum of squares on an integer grid.

    `y_lang.exact.model.sum_of_squares` (torch) and
    `norm_kernel._grid_sum_of_squares` (Triton) compute ExactRMSNorm's
    sum of x^2 without a float64 add. A float32 x is m * 2^(p/2 - 150), so
    x^2 = m2 * 2^(p - 300) with m2 = m * m < 2^48. With e the row's largest
    p and s = e - p >= 0, each square goes onto the grid 2^(e - 300 - 32):

      grid m2 s = floor(m2 * 2^(32 - s))

    The code never forms that 80-bit integer. It splits it into two 40-bit
    limbs with shifts and masks (the Triton kernel's spelling, modelled
    below as [hi] and [lo], shift amounts clamped at 63 as the code clamps
    them), sums each limb over the row in int64, moves the low limb's carry
    into the high one, and converts the high limb to float64.

    ** What is proved.

    - [the_limbs_are_the_grid_value]: hi * 2^40 + lo is the grid value, for
      every m2 below 2^48 and every s >= 0 (both clamps included).
    - [the_grid_value_is_the_floor]: that grid value is floor(m2 * 2^(32-s)),
      the definition the code's comments and tests state.
    - [the_torch_mask_is_the_kernel_mask]: the torch spelling of the low
      limb's mask, (1 << (s + 8)) - 1, is the kernel's, (2^40 - 1) >> (32 - s).
    - [the_carried_limbs_are_the_row_sum]: after the carry, big * 2^40 +
      small is the exact sum of the row's grid values.
    - [the_sum_does_not_depend_on_the_order]: any permutation of the row
      gives the same big and small.
    - [the_limb_sums_fit]: for rows of at most 8192 values every limb sum is
      below 2^53 (so int64 never wraps), small is below 2^40 and big below
      2^53: both convert to float64 exactly.
    - [past_8192_values_big_can_reach_2_53]: the 8192 is load-bearing; 8193
      squares of the largest float32 significand put big at 2^53 or more.

    ** What is NOT proved.

    - The floating-point steps: m2 = m * m in int64 (a 48-bit product), the
      final float64(big) * 2^40 + float64(small) (one IEEE add of two exact
      operands, so one correctly rounded result), the scaling by the power of
      two 2^(e - 332) (exact while the result is normal, which it is for
      every float32 input), and the float64 sum taken instead for a row with
      an infinite or NaN value.
    - That the torch and Triton code compute [hi], [lo] and the sums: the tie
      is by transcription, checked by behaviour in
      `python/y_lang/tests/test_exact_sum_of_squares.py` (a Python-integer
      oracle, permuted rows, 19 mutants).
    - That e is the row's largest p (a max reduction) and so s >= 0.

    Checked with Rocq 9.1:  coqc proofs/NormGridSum.v  *)

Require Import ZArith Lia List Permutation.
Import ListNotations.
Open Scope Z_scope.

(* ------------------------------------------------------------------ *)
(** ** One square                                                      *)
(* ------------------------------------------------------------------ *)

(** The grid value of m2 * 2^(p - 300) on 2^(e - 332), s = e - p. *)
Definition grid (m2 s : Z) : Z :=
  if s <=? 32 then m2 * 2 ^ (32 - s) else m2 / 2 ^ (s - 32).

(** The kernel's high limb: `m2 >> min(s + 8, 63)`. *)
Definition hi (m2 s : Z) : Z := Z.shiftr m2 (Z.min (s + 8) 63).

(** The kernel's low limb:
    `s <= 32 ? (m2 & ((2^40 - 1) >> (32 - sa))) << (32 - sa)`
    `        : (m2 >> min(max(s - 32, 0), 63)) & (2^40 - 1)`, sa = min(s, 32). *)
Definition lo (m2 s : Z) : Z :=
  let sa := Z.min s 32 in
  if s <=? 32
  then Z.shiftl (Z.land m2 (Z.shiftr (Z.ones 40) (32 - sa))) (32 - sa)
  else Z.land (Z.shiftr m2 (Z.min (Z.max (s - 32) 0) 63)) (Z.ones 40).

Lemma ones_shiftr : forall s, 0 <= s <= 32 ->
  Z.shiftr (Z.ones 40) (32 - s) = Z.ones (8 + s).
Proof.
  intros s Hs. rewrite Z.shiftr_div_pow2 by lia.
  rewrite Z.ones_div_pow2 by lia. f_equal. lia.
Qed.

(** The torch code writes the mask as `(1 << (s + 8)) - 1`. *)
Theorem the_torch_mask_is_the_kernel_mask : forall s, 0 <= s <= 32 ->
  Z.shiftl 1 (s + 8) - 1 = Z.shiftr (Z.ones 40) (32 - s).
Proof.
  intros s Hs. rewrite ones_shiftr by lia.
  rewrite Z.shiftl_1_l, Z.ones_equiv, <- Z.sub_1_r.
  replace (8 + s) with (s + 8) by lia. reflexivity.
Qed.

Lemma shiftr_big : forall m2 k, 0 <= m2 < 2 ^ 48 -> 48 <= k -> Z.shiftr m2 k = 0.
Proof.
  intros m2 k Hm Hk. rewrite Z.shiftr_div_pow2 by lia.
  apply Z.div_small. split; [lia|].
  apply Z.lt_le_trans with (2 ^ 48); [lia|]. apply Z.pow_le_mono_r; lia.
Qed.

(** The clamped shift is the unclamped one: past 47 both give 0. *)
Lemma hi_unclamped : forall m2 s, 0 <= m2 < 2 ^ 48 -> 0 <= s ->
  hi m2 s = m2 / 2 ^ (s + 8).
Proof.
  intros m2 s Hm Hs. unfold hi.
  destruct (Z.le_gt_cases (s + 8) 63) as [H|H].
  - rewrite Z.min_l by lia. apply Z.shiftr_div_pow2. lia.
  - rewrite Z.min_r by lia. rewrite shiftr_big by lia.
    symmetry. apply Z.div_small. split; [lia|].
    apply Z.lt_le_trans with (2 ^ 48); [lia|]. apply Z.pow_le_mono_r; lia.
Qed.

Theorem the_limbs_are_the_grid_value : forall m2 s,
  0 <= m2 < 2 ^ 48 -> 0 <= s ->
  hi m2 s * 2 ^ 40 + lo m2 s = grid m2 s.
Proof.
  intros m2 s Hm Hs. rewrite hi_unclamped by assumption.
  unfold lo, grid.
  destruct (Z.leb_spec s 32) as [Hle|Hgt].
  - rewrite Z.min_l by lia. rewrite ones_shiftr by lia.
    rewrite Z.land_ones by lia. rewrite Z.shiftl_mul_pow2 by lia.
    (* m2 = 2^(8+s) * (m2 / 2^(8+s)) + m2 mod 2^(8+s), and 2^40 = 2^(32-s) * 2^(8+s) *)
    assert (E : 2 ^ 40 = 2 ^ (8 + s) * 2 ^ (32 - s)).
    { rewrite <- Z.pow_add_r by lia. f_equal. lia. }
    replace (s + 8) with (8 + s) by lia.
    rewrite E. pose proof (Z.div_mod m2 (2 ^ (8 + s))) as D.
    assert (2 ^ (8 + s) <> 0) by (apply Z.pow_nonzero; lia). specialize (D H).
    rewrite D at 3. ring.
  - rewrite Z.max_l by lia. rewrite Z.land_ones by lia.
    destruct (Z.le_gt_cases (s - 32) 63) as [H63|H63].
    + rewrite Z.min_l by lia. rewrite Z.shiftr_div_pow2 by lia.
      assert (Hdiv : m2 / 2 ^ (s + 8) = m2 / 2 ^ (s - 32) / 2 ^ 40).
      { rewrite Z.div_div by (try apply Z.pow_pos_nonneg; lia).
        rewrite <- Z.pow_add_r by lia. f_equal. f_equal. lia. }
      rewrite Hdiv. rewrite Z.mul_comm. symmetry. apply Z.div_mod.
      apply Z.pow_nonzero; lia.
    + rewrite Z.min_r by lia. rewrite shiftr_big by lia.
      assert (Z0 : m2 / 2 ^ (s - 32) = 0).
      { apply Z.div_small. split; [lia|].
        apply Z.lt_le_trans with (2 ^ 48); [lia|]. apply Z.pow_le_mono_r; lia. }
      assert (Z1 : m2 / 2 ^ (s + 8) = 0).
      { apply Z.div_small. split; [lia|].
        apply Z.lt_le_trans with (2 ^ 48); [lia|]. apply Z.pow_le_mono_r; lia. }
      rewrite Z0, Z1. reflexivity.
Qed.

(** The grid value is the floor the code's comments define:
    grid * 2^s <= m2 * 2^32 < (grid + 1) * 2^s. *)
Theorem the_grid_value_is_the_floor : forall m2 s,
  0 <= m2 -> 0 <= s ->
  grid m2 s * 2 ^ s <= m2 * 2 ^ 32 < (grid m2 s + 1) * 2 ^ s.
Proof.
  intros m2 s Hm Hs. unfold grid.
  destruct (Z.leb_spec s 32) as [Hle|Hgt].
  - assert (E : 2 ^ 32 = 2 ^ (32 - s) * 2 ^ s).
    { rewrite <- Z.pow_add_r by lia. f_equal. lia. }
    rewrite E. assert (0 < 2 ^ s) by (apply Z.pow_pos_nonneg; lia). nia.
  - assert (E : 2 ^ s = 2 ^ (s - 32) * 2 ^ 32).
    { rewrite <- Z.pow_add_r by lia. f_equal. lia. }
    rewrite E. assert (P : 0 < 2 ^ (s - 32)) by (apply Z.pow_pos_nonneg; lia).
    assert (P32 : 0 < 2 ^ 32) by lia.
    pose proof (Z.div_mod m2 (2 ^ (s - 32)) ltac:(lia)) as D.
    pose proof (Z.mod_pos_bound m2 (2 ^ (s - 32)) P) as B.
    split; nia.
Qed.

Lemma limbs_bounded : forall m2 s, 0 <= m2 < 2 ^ 48 -> 0 <= s ->
  0 <= lo m2 s < 2 ^ 40 /\ 0 <= hi m2 s < 2 ^ 40.
Proof.
  intros m2 s Hm Hs. split.
  - unfold lo. destruct (Z.leb_spec s 32) as [Hle|Hgt].
    + rewrite Z.min_l by lia. rewrite ones_shiftr by lia.
      rewrite Z.land_ones by lia. rewrite Z.shiftl_mul_pow2 by lia.
      pose proof (Z.mod_pos_bound m2 (2 ^ (8 + s)) ltac:(apply Z.pow_pos_nonneg; lia)).
      assert (E : 2 ^ 40 = 2 ^ (8 + s) * 2 ^ (32 - s)).
      { rewrite <- Z.pow_add_r by lia. f_equal. lia. }
      assert (0 < 2 ^ (32 - s)) by (apply Z.pow_pos_nonneg; lia). nia.
    + rewrite Z.land_ones by lia. apply Z.mod_pos_bound. lia.
  - rewrite hi_unclamped by assumption. split.
    + apply Z.div_pos; [lia | apply Z.pow_pos_nonneg; lia].
    + apply Z.div_lt_upper_bound; [apply Z.pow_pos_nonneg; lia|].
      rewrite <- Z.pow_add_r by lia.
      apply Z.lt_le_trans with (2 ^ 48); [lia|]. apply Z.pow_le_mono_r; lia.
Qed.

(* ------------------------------------------------------------------ *)
(** ** A row                                                           *)
(* ------------------------------------------------------------------ *)

(** A row's squares, each (m2, s). *)
Definition zsum (f : Z * Z -> Z) (l : list (Z * Z)) : Z :=
  fold_right (fun t acc => f t + acc) 0 l.

Definition ok (t : Z * Z) : Prop := 0 <= fst t < 2 ^ 48 /\ 0 <= snd t.

Definition fhi (t : Z * Z) := hi (fst t) (snd t).
Definition flo (t : Z * Z) := lo (fst t) (snd t).
Definition fgrid (t : Z * Z) := grid (fst t) (snd t).

(** The two int64 limb sums, and the carry the code moves. *)
Definition big (l : list (Z * Z)) := zsum fhi l + zsum flo l / 2 ^ 40.
Definition small (l : list (Z * Z)) := zsum flo l mod 2 ^ 40.

Lemma zsum_split : forall l, Forall ok l ->
  zsum fhi l * 2 ^ 40 + zsum flo l = zsum fgrid l.
Proof.
  induction l as [|t l IH]; intros H; [reflexivity|].
  inversion H as [|? ? Ht Hl]; subst.
  destruct Ht as [Hm Hs].
  pose proof (the_limbs_are_the_grid_value (fst t) (snd t) Hm Hs) as E.
  unfold zsum in *. cbn [fold_right].
  unfold fhi, flo, fgrid in *. rewrite <- (IH Hl), <- E. ring.
Qed.

Theorem the_carried_limbs_are_the_row_sum : forall l, Forall ok l ->
  big l * 2 ^ 40 + small l = zsum fgrid l.
Proof.
  intros l H. rewrite <- zsum_split by assumption. unfold big, small.
  pose proof (Z.div_mod (zsum flo l) (2 ^ 40) ltac:(lia)). lia.
Qed.

Lemma zsum_perm : forall f l l', Permutation l l' -> zsum f l = zsum f l'.
Proof.
  intros f l l' P. induction P; unfold zsum in *; cbn [fold_right]; lia.
Qed.

Theorem the_sum_does_not_depend_on_the_order : forall l l',
  Permutation l l' -> big l = big l' /\ small l = small l'.
Proof.
  intros l l' P. unfold big, small.
  rewrite (zsum_perm fhi l l' P), (zsum_perm flo l l' P). split; reflexivity.
Qed.

Lemma zsum_bound : forall f l, (forall t, In t l -> 0 <= f t < 2 ^ 40) ->
  0 <= zsum f l <= Z.of_nat (length l) * (2 ^ 40 - 1).
Proof.
  intros f l. induction l as [|t l IH]; intros H; unfold zsum in *; cbn [fold_right length];
    [lia|].
  assert (Ht := H t (or_introl eq_refl)).
  assert (Hl : forall u, In u l -> 0 <= f u < 2 ^ 40) by (intros u Hu; apply H; right; exact Hu).
  specialize (IH Hl). rewrite Nat2Z.inj_succ. lia.
Qed.

Theorem the_limb_sums_fit : forall l, Forall ok l -> Z.of_nat (length l) <= 8192 ->
  0 <= zsum fhi l < 2 ^ 53 /\ 0 <= zsum flo l < 2 ^ 53 /\
  0 <= small l < 2 ^ 40 /\ 0 <= big l < 2 ^ 53.
Proof.
  intros l H Hn.
  assert (Hin : forall t, In t l -> ok t) by (rewrite Forall_forall in H; exact H).
  assert (Bh : forall t, In t l -> 0 <= fhi t < 2 ^ 40).
  { intros t Ht. destruct (Hin t Ht). apply (limbs_bounded (fst t) (snd t)); assumption. }
  assert (Bl : forall t, In t l -> 0 <= flo t < 2 ^ 40).
  { intros t Ht. destruct (Hin t Ht). apply (limbs_bounded (fst t) (snd t)); assumption. }
  pose proof (zsum_bound fhi l Bh) as SH. pose proof (zsum_bound flo l Bl) as SL.
  unfold big, small.
  set (n := Z.of_nat (length l)) in *.
  set (L := zsum flo l) in *. set (Hs := zsum fhi l) in *.
  assert (Q : 2 ^ 40 * (L / 2 ^ 40) <= L) by (apply Z.mul_div_le; lia).
  assert (Q0 : 0 <= L / 2 ^ 40) by (apply Z.div_pos; lia).
  assert (M : 0 <= L mod 2 ^ 40 < 2 ^ 40) by (apply Z.mod_pos_bound; lia).
  set (q := L / 2 ^ 40) in *.
  (* With the powers as numerals every remaining step is linear: the carry
     q satisfies q * 2^40 <= L <= n * (2^40 - 1), so q < n when n > 0, and
     big = Hs + q <= n * 2^40 - 1. *)
  change (2 ^ 40) with 1099511627776 in *. change (2 ^ 53) with 9007199254740992.
  lia.
Qed.

(** The 8192 is load-bearing: 8193 squares of the largest significand
    (m = 2^24 - 1, all at s = 0) carry big to 2^53 or past it, where its
    float64 conversion would round. *)
Lemma zsum_repeat : forall f t n, zsum f (repeat t n) = Z.of_nat n * f t.
Proof.
  intros f t n. induction n as [|n IH]; unfold zsum in *; cbn [repeat fold_right]; [lia|].
  rewrite IH. lia.
Qed.

Theorem past_8192_values_big_can_reach_2_53 :
  let l := repeat ((2 ^ 24 - 1) * (2 ^ 24 - 1), 0) (Z.to_nat 8193) in
  Forall ok l /\ Z.of_nat (length l) = 8193 /\ 2 ^ 53 <= big l.
Proof.
  intros l. split; [|split].
  - apply Forall_forall. intros t Ht. apply repeat_spec in Ht. subst t.
    unfold ok. simpl. lia.
  - unfold l. rewrite repeat_length. apply Z2Nat.id. lia.
  - unfold big, l. rewrite !zsum_repeat. rewrite Z2Nat.id by lia.
    apply Z.leb_le. vm_compute. reflexivity.
Qed.

Print Assumptions the_limbs_are_the_grid_value.
Print Assumptions the_grid_value_is_the_floor.
Print Assumptions the_torch_mask_is_the_kernel_mask.
Print Assumptions the_carried_limbs_are_the_row_sum.
Print Assumptions the_sum_does_not_depend_on_the_order.
Print Assumptions the_limb_sums_fit.
Print Assumptions past_8192_values_big_can_reach_2_53.
