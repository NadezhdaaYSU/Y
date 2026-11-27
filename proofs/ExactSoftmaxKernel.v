(** * Exact training's softmax kernel (`python/y_lang/exact/softmax_kernel.py`).

    The kernel computes `train_lm.Ops.softmax` and `q8(P, bits=8)` in one
    pass, with the Python's operations in its order. Its docstring rests on
    three claims about float32, proved here.

    ** The rounding to an integer (`_rne_nonneg`).

    For x >= 0 below 2^23 the kernel rounds by `(x + 2^23) - 2^23`, and
    keeps x above. Values are integers times 2^-149 (float32's step), so a
    float32 with fields (E, M) is `ExactFloatOrder.value 23 E M`; [rn] is
    the float32 rounding, assumed only to keep float32s ([rn_fix]) and, in
    the binade [2^23, 2^24) where the step is 1, to round to the nearest
    integer with ties to even ([rn_grid]: IEEE's rule there).
    [the_magic_add_rounds_half_even]: for every finite float32 x >= 0 the
    result is round-half-even(x) (`ExactTrainArith.shift_round`, at 149
    bits), in both branches.

    ** No subnormal reaches the divide (`div_rn`).

    The row's weights p are `exp2_neg_q16_16` of t >= 0, one of them of
    t = 0 (the row's maximum; a blocked key adds 0), at most 1024 keys.
    [the_row_sum_is_between_2_28_and_2_38]: each p is in 0 .. 2^28 and their
    sum in 2^28 .. 2^38. [the_converted_operands_are_normal]: for any
    monotone conversion to float32 that keeps float32 integers, float32(p) is
    0 or in 1 .. 2^28 and float32(sum) in 2^28 .. 2^38.
    [the_rounded_quotient_is_zero_or_normal]: then the quotient, rounded by
    any monotone rounding that keeps 0, 2^-38, 2^-10 and 1, is 0 or in
    2^-38 .. 1, and the row maximum's in 2^-10 .. 1. Every operand and result
    is 0 or at least 2^-38, far above 2^-126: libdevice's subnormal flush
    has nothing to flush.

    ** The row's quantizer scale.

    [the_softmax_row_scale_is_small]:
    the row maximum of P in 2^-10 .. 1 has the field 117 .. 127, so q8's
    s = field - 126 - 7 is in -16 .. -6: the range flag (|s| > 511) cannot
    fire and `scaled_round`'s two factors are a = -s, b = 0.

    ** Refutations.

    [without_the_branch_an_odd_integer_moves]: adding 2^23 to x = 2^23 + 1
    lands in [2^24, 2^25), where the step is 2, and the tie rounds to even:
    (x + 2^23) - 2^23 is 2^23, not x. [a_row_of_2048_keys_can_pass_2_38]:
    the 2^38 bound needs the kernel's limit of 1024 keys.

    ** Not proved here.

    IEEE's rounding itself (that float32 `+`, `-`, the int64 to float32
    conversion and `div_rn` round to nearest, ties to even); that the row's
    maximum key gets t = 0 (m - m is +0 in IEEE, and the clamp keeps 0);
    the sign of a negative x, which the callers clamp to 0.

    Checked with Rocq 9.1:  coqc -Q . "" ExactSoftmaxKernel.v  *)

From Stdlib Require Import ZArith Lia QArith Lqa List.
Import ListNotations.
Require ExactFloatOrder ExactTrainArith ExactDecodeWidths ExactScaledRound.
Open Scope Z_scope.

Module FO := ExactFloatOrder.
Module TA := ExactTrainArith.
Module DW := ExactDecodeWidths.
Module SR := ExactScaledRound.

(** ** `_rne_nonneg` *)

Lemma shift_round_add_even : forall V K r, 0 <= V -> 1 <= r -> Z.even K = true ->
  TA.shift_round (V + K * 2 ^ r) r = TA.shift_round V r + K.
Proof.
  intros V K r HV Hr HK. unfold TA.shift_round.
  assert (0 < 2 ^ r) by (apply Z.pow_pos_nonneg; lia).
  rewrite Z.div_add, Z_mod_plus_full by lia. rewrite Z.odd_add.
  assert (Z.odd K = false) by (rewrite <- Z.negb_even, HK; reflexivity). rewrite H0, Bool.xorb_false_r.
  ring.
Qed.

Lemma shift_round_exact : forall q r, 1 <= r -> TA.shift_round (q * 2 ^ r) r = q.
Proof.
  intros q r Hr. unfold TA.shift_round.
  assert (0 < 2 ^ (r - 1)) by (apply Z.pow_pos_nonneg; lia).
  rewrite Z.div_mul, Z_mod_mult by (apply Z.pow_nonzero; lia).
  destruct (2 ^ (r - 1) <? 0) eqn:A; [apply Z.ltb_lt in A; lia|].
  destruct (0 =? 2 ^ (r - 1)) eqn:B; [apply Z.eqb_eq in B; lia|]. cbn. ring.
Qed.

Section Rne.
  Variable rn : Z -> Z.
  Hypothesis rn_fix : forall E M, 0 <= E <= 254 -> 0 <= M < 2 ^ 23 ->
    rn (FO.value 23 E M) = FO.value 23 E M.
  Hypothesis rn_grid : forall S, 2 ^ 172 <= S < 2 ^ 173 -> rn S = TA.shift_round S 149 * 2 ^ 149.

  (** `where(x < 2^23, (x + 2^23) - 2^23, x)`, each operation rounded. *)
  Definition rne_nonneg (V : Z) : Z :=
    if V <? 2 ^ 172 then rn (rn (V + 2 ^ 172) - 2 ^ 172) else V.

  Theorem the_magic_add_rounds_half_even : forall E M, 0 <= E <= 254 -> 0 <= M < 2 ^ 23 ->
    rne_nonneg (FO.value 23 E M) = TA.shift_round (FO.value 23 E M) 149 * 2 ^ 149.
  Proof.
    intros E M HE HM. unfold rne_nonneg. set (V := FO.value 23 E M).
    assert (HV0 : 0 <= V).
    { unfold V, FO.value. destruct (E =? 0); [lia|]. apply Z.mul_nonneg_nonneg; [lia | apply Z.pow_nonneg; lia]. }
    destruct (V <? 2 ^ 172) eqn:L.
    - apply Z.ltb_lt in L.
      assert (P173 : 2 ^ 173 = 2 * 2 ^ 172) by reflexivity.
      rewrite (rn_grid (V + 2 ^ 172)) by lia.
      replace (V + 2 ^ 172) with (V + 2 ^ 23 * 2 ^ 149) by (rewrite <- Z.pow_add_r by lia; reflexivity).
      rewrite shift_round_add_even by (try lia; reflexivity).
      replace ((TA.shift_round V 149 + 2 ^ 23) * 2 ^ 149 - 2 ^ 172) with (TA.shift_round V 149 * 2 ^ 149)
        by (replace (2 ^ 172) with (2 ^ 23 * 2 ^ 149) by reflexivity; ring).
      (* k = the rounding of V, at most 2^23: an integer float32 *)
      destruct (TA.shift_round_is_nearest V 149 HV0 ltac:(lia)) as [A _].
      set (k := TA.shift_round V 149) in *.
      assert (P : 2 ^ 149 = 2 * 2 ^ 148) by (rewrite <- (Z.pow_1_r 2) at 2; rewrite <- Z.pow_add_r by lia; reflexivity).
      replace (149 - 1) with 148 in A by lia.
      assert (Q : 2 ^ 172 = 2 ^ 23 * 2 ^ 149) by (rewrite <- Z.pow_add_r by lia; reflexivity).
      assert (0 < 2 ^ 148) by (apply Z.pow_pos_nonneg; lia).
      assert (Hk : 0 <= k <= 2 ^ 23) by nia.
      assert (Hk2 : k * 2 ^ 149 < 2 ^ 211).
      { assert (2 ^ 211 = 2 ^ 62 * 2 ^ 149) by (rewrite <- Z.pow_add_r by lia; reflexivity). nia. }
      destruct (SR.an_integer_times_a_power_of_two_is_a_float32 k 149 ltac:(lia) ltac:(lia) Hk2) as [E' [M' [HE' [HM' VE]]]].
      rewrite <- VE. apply rn_fix; assumption.
    - apply Z.ltb_ge in L.
      (* x >= 2^23 is an integer: its field is at least 150 *)
      pose proof (FO.the_field_bounds_the_value 23 E M ltac:(lia) ltac:(unfold FO.finite; lia)) as B. fold V in B.
      assert (HE150 : 150 <= E).
      { destruct (Z_le_gt_dec 150 E) as [|G]; [lia|].
        assert (2 ^ (23 + E) <= 2 ^ 172) by (apply Z.pow_le_mono_r; lia). lia. }
      unfold V, FO.value. destruct (E =? 0) eqn:Z0; [apply Z.eqb_eq in Z0; lia|].
      replace (2 ^ (E - 1)) with (2 ^ (E - 150) * 2 ^ 149) by (rewrite <- Z.pow_add_r by lia; f_equal; lia).
      rewrite Z.mul_assoc, shift_round_exact by lia. reflexivity.
  Qed.
End Rne.

(** x = 2^23 + 1, one step past the branch: its sum with 2^23 is 2^24 + 1,
    a tie between 2^24 and 2^24 + 2 where the step is 2^150 in these units. *)
Theorem without_the_branch_an_odd_integer_moves :
  let V := (2 ^ 23 + 1) * 2 ^ 149 in
  TA.shift_round (V + 2 ^ 172) 150 * 2 ^ 150 - 2 ^ 172 = 2 ^ 23 * 2 ^ 149 /\ V <> 2 ^ 23 * 2 ^ 149.
Proof. vm_compute. split; [reflexivity | discriminate]. Qed.

(** ** The row's weights and their sum *)

Definition zsum (l : list Z) : Z := fold_right Z.add 0 l.

Lemma exp_at_zero : DW.exp2_neg_q16_16 0 = 2 ^ 28.
Proof. vm_compute. reflexivity. Qed.

Lemma zsum_cons : forall h t, zsum (h :: t) = h + zsum t.
Proof. reflexivity. Qed.

Lemma zsum_nonneg : forall l, Forall (fun p => 0 <= p) l -> 0 <= zsum l.
Proof.
  induction l as [|h t IH]; intro F; [reflexivity|]. rewrite zsum_cons.
  inversion F; subst. specialize (IH H2). lia.
Qed.

Lemma zsum_ge_member : forall l x, Forall (fun p => 0 <= p) l -> In x l -> x <= zsum l.
Proof.
  induction l as [|h t IH]; intros x F I; [destruct I|]. inversion F; subst. rewrite zsum_cons.
  destruct I as [<-|I].
  - pose proof (zsum_nonneg t H2). lia.
  - specialize (IH x H2 I). lia.
Qed.

Lemma zsum_le : forall l c, Forall (fun p => p <= c) l -> zsum l <= Z.of_nat (length l) * c.
Proof.
  induction l as [|h t IH]; intros c F; [cbn; lia|]. inversion F; subst. specialize (IH c H2).
  rewrite zsum_cons. change (length (h :: t)) with (S (length t)). rewrite Nat2Z.inj_succ. lia.
Qed.

Theorem the_row_sum_is_between_2_28_and_2_38 : forall ts,
  Forall (fun t => 0 <= t) ts -> In 0 ts -> (length ts <= 1024)%nat ->
  let ps := map DW.exp2_neg_q16_16 ts in
  Forall (fun p => 0 <= p <= 2 ^ 28) ps /\ 2 ^ 28 <= zsum ps <= 2 ^ 38.
Proof.
  intros ts Ht I0 L ps.
  assert (F : Forall (fun p => 0 <= p <= 2 ^ 28) ps).
  { apply Forall_forall. intros p Hp. unfold ps in Hp. apply in_map_iff in Hp as [t [<- _]].
    apply DW.the_integer_exp_is_a_q28_weight. }
  split; [exact F|]. split.
  - rewrite <- exp_at_zero. apply zsum_ge_member.
    + eapply Forall_impl; [|exact F]. cbn. lia.
    + unfold ps. apply in_map. exact I0.
  - assert (S : zsum ps <= Z.of_nat (length ps) * 2 ^ 28) by (apply zsum_le; eapply Forall_impl; [|exact F]; cbn; lia).
    assert (Ln : length ps = length ts) by (unfold ps; apply length_map). rewrite Ln in S.
    assert (Z.of_nat (length ts) <= 1024) by lia.
    assert (P38 : 2 ^ 38 = 1024 * 2 ^ 28) by reflexivity. rewrite P38. nia.
Qed.

(** A row of 2048 keys at t = 0 sums to 2^39. *)
Theorem a_row_of_2048_keys_can_pass_2_38 :
  zsum (map DW.exp2_neg_q16_16 (repeat 0 2048)) = 2 ^ 39.
Proof. vm_compute. reflexivity. Qed.

Section Convert.
  (** The int64 to float32 conversion, on integers. *)
  Variable rz : Z -> Z.
  Hypothesis rz_fix : forall K j, 0 <= K < 2 ^ 24 -> 0 <= j -> rz (K * 2 ^ j) = K * 2 ^ j.
  Hypothesis rz_mono : forall v w, v <= w -> rz v <= rz w.

  Lemma rz_at : forall K j, 0 <= K < 2 ^ 24 -> 0 <= j -> rz (K * 2 ^ j) = K * 2 ^ j.
  Proof. exact rz_fix. Qed.

  Theorem the_converted_operands_are_normal : forall p s,
    0 <= p <= 2 ^ 28 -> 2 ^ 28 <= s <= 2 ^ 38 ->
    (rz p = 0 \/ 1 <= rz p <= 2 ^ 28) /\ 2 ^ 28 <= rz s <= 2 ^ 38 /\ (p = 2 ^ 28 -> rz p = 2 ^ 28).
  Proof.
    intros p s Hp Hs.
    assert (R0 : rz 0 = 0) by (exact (rz_fix 0 0 ltac:(lia) ltac:(lia))).
    assert (R1 : rz 1 = 1) by (exact (rz_fix 1 0 ltac:(lia) ltac:(lia))).
    assert (R28 : rz (2 ^ 28) = 2 ^ 28) by (rewrite <- (Z.mul_1_l (2 ^ 28)); apply rz_fix; lia).
    assert (R38 : rz (2 ^ 38) = 2 ^ 38) by (rewrite <- (Z.mul_1_l (2 ^ 38)); apply rz_fix; lia).
    split; [|split].
    - destruct (Z.eq_dec p 0) as [->|Hne]; [left; exact R0 | right].
      pose proof (rz_mono 1 p ltac:(lia)). pose proof (rz_mono p (2 ^ 28) ltac:(lia)). lia.
    - pose proof (rz_mono (2 ^ 28) s ltac:(lia)). pose proof (rz_mono s (2 ^ 38) ltac:(lia)). lia.
    - intros ->. exact R28.
  Qed.
End Convert.

Section Quotient.
  Open Scope Q_scope.
  Definition q38 : Q := 1 # 274877906944.     (* 2^-38 *)
  Definition q10 : Q := 1 # 1024.             (* 2^-10 *)

  Variable rq : Q -> Q.
  Hypothesis rq_mono : forall a b, a <= b -> rq a <= rq b.
  Hypothesis rq_0 : rq 0 == 0.
  Hypothesis rq_38 : rq q38 == q38.
  Hypothesis rq_10 : rq q10 == q10.
  Hypothesis rq_1 : rq 1 == 1.
  Hypothesis rq_ext : forall a b, a == b -> rq a == rq b.

  Theorem the_rounded_quotient_is_zero_or_normal : forall num den : Z,
    (num = 0%Z \/ (1 <= num <= 2 ^ 28)%Z) -> (2 ^ 28 <= den <= 2 ^ 38)%Z ->
    let x := rq (inject_Z num / inject_Z den) in
    (x == 0 \/ q38 <= x <= 1) /\ (num = (2 ^ 28)%Z -> q10 <= x <= 1).
  Proof.
    intros num den Hn Hd x.
    assert (Dpos : 0 < inject_Z den) by (change 0 with (inject_Z 0); rewrite <- Zlt_Qlt; lia).
    assert (Dlo : inject_Z (2 ^ 28) <= inject_Z den) by (rewrite <- Zle_Qle; lia).
    assert (Dhi : inject_Z den <= inject_Z (2 ^ 38)) by (rewrite <- Zle_Qle; lia).
    change (inject_Z (2 ^ 28)) with (268435456 # 1) in Dlo. change (inject_Z (2 ^ 38)) with (274877906944 # 1) in Dhi.
    assert (Up : forall n : Z, (n <= 2 ^ 28)%Z -> inject_Z n / inject_Z den <= 1).
    { intros n Hn'. apply Qle_shift_div_r; [exact Dpos|].
      assert (inject_Z n <= inject_Z (2 ^ 28)) by (rewrite <- Zle_Qle; lia).
      change (inject_Z (2 ^ 28)) with (268435456 # 1) in H. lra. }
    split.
    - destruct Hn as [->|Hn].
      + left. unfold x. apply (Qeq_trans _ (rq 0)); [apply rq_ext | exact rq_0].
        unfold Qdiv. change (inject_Z 0) with 0. ring.
      + right. assert (L : q38 <= inject_Z num / inject_Z den).
        { apply Qle_shift_div_l; [exact Dpos|].
          assert (1 <= inject_Z num) by (change 1 with (inject_Z 1); rewrite <- Zle_Qle; lia).
          unfold q38. lra. }
        pose proof (rq_mono _ _ L). pose proof (rq_mono _ _ (Up num ltac:(lia))). unfold x.
        rewrite rq_38 in *. rewrite rq_1 in *. split; assumption.
    - intros ->. assert (L : q10 <= inject_Z (2 ^ 28) / inject_Z den).
      { apply Qle_shift_div_l; [exact Dpos|]. change (inject_Z (2 ^ 28)) with (268435456 # 1). unfold q10. lra. }
      pose proof (rq_mono _ _ L). pose proof (rq_mono _ _ (Up (2 ^ 28)%Z ltac:(lia))). unfold x.
      rewrite rq_10 in *. rewrite rq_1 in *. split; assumption.
  Qed.
End Quotient.

(** ** The row's quantizer scale *)

Theorem the_softmax_row_scale_is_small : forall E M, 0 <= E -> 0 <= M < 2 ^ 23 ->
  2 ^ 139 <= FO.value 23 E M <= 2 ^ 149 ->
  let s := E - 126 - 7 in
  117 <= E <= 127 /\ -16 <= s <= -6 /\ ~ (s < -511 \/ 511 < s) /\
  SR.clamp (-126) 127 (- s) = - s /\ SR.clamp (-126) 127 (- s - SR.clamp (-126) 127 (- s)) = 0.
Proof.
  intros E M HE HM Hv s.
  pose proof (FO.the_field_bounds_the_value 23 E M ltac:(lia) ltac:(unfold FO.finite; lia)) as B.
  assert (E1 : 1 <= E).
  { destruct (Z.eq_dec E 0) as [->|]; [|lia]. unfold FO.value in Hv. cbn -[Z.pow] in Hv.
    assert (2 ^ 23 < 2 ^ 139) by (apply Z.pow_lt_mono_r; lia). lia. }
  pose proof (FO.a_normal_value_reaches_half_its_bound 23 E M ltac:(lia) ltac:(unfold FO.finite; lia) E1) as L.
  assert (117 <= E).
  { destruct (Z_le_gt_dec 117 E) as [|G]; [lia|].
    assert (2 ^ (23 + E) <= 2 ^ 139) by (apply Z.pow_le_mono_r; lia). lia. }
  assert (E <= 127).
  { destruct (Z_le_gt_dec E 127) as [|G]; [lia|].
    assert (2 ^ 150 <= 2 ^ (23 + E - 1)) by (apply Z.pow_le_mono_r; lia).
    assert (2 ^ 149 < 2 ^ 150) by (apply Z.pow_lt_mono_r; lia). lia. }
  unfold s, SR.clamp. lia.
Qed.

Print Assumptions the_magic_add_rounds_half_even.
Print Assumptions without_the_branch_an_odd_integer_moves.
Print Assumptions the_row_sum_is_between_2_28_and_2_38.
Print Assumptions a_row_of_2048_keys_can_pass_2_38.
Print Assumptions the_converted_operands_are_normal.
Print Assumptions the_rounded_quotient_is_zero_or_normal.
Print Assumptions the_softmax_row_scale_is_small.
