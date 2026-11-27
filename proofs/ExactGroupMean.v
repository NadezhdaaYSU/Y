(** * GRPO's group advantages (`python/y_lang/exact/rl.py`, `advantages`).

    Dr. GRPO's advantage is reward - the mean reward of the row's group,
    computed on the host in float64: `m = sum(g) / group`, then `x - m`.
    The docstring says "the mean of rewards in {0, 1} over a power-of-two
    group is exact". This file proves it, and that the advantages are exact
    too.

    Units: float64 values are integers times 2^-1074 (`ExactFloatOrder`'s
    units); [rn] is the float64 rounding, assumed only to keep every finite
    float64 and its negation. The sum runs in ExactReduce's float64 carrier
    (round to nearest even at 53 bits) on the rewards themselves, as
    Python's `sum` does, left to right from 0.

    [the_reward_sum_is_exact]: for rewards in {0, 1} and at most 2^53 of
    them, the float64 sum is the integer count of ones.
    [the_mean_is_exact]: for a group of 2^k rows (k <= 52), s / 2^k is a
    float64, so the divide returns it exactly.
    [every_advantage_is_exact]: and each x - m is (x 2^k - s) / 2^k
    exactly, a float64 too.

    ** Refutation.

    [a_group_of_three_rounds_the_mean]: the code does not check that the
    group is a power of two, and with 3 rows and one reward of 1 the mean
    1/3 is no float64: the claim needs the power of two. (The host's
    float64 is the same everywhere, so this costs exactness, not
    determinism.)

    ** Not proved here.

    That Python's float64 arithmetic is IEEE round to nearest; rewards
    outside {0, 1}.

    Checked with Rocq 9.1:  coqc -Q . "" ExactGroupMean.v  *)

From Stdlib Require Import ZArith Lia List.
Import ListNotations.
Require ExactFloatOrder ExactReduce ExactVFold.
Open Scope Z_scope.

Module FO := ExactFloatOrder.
Module ER := ExactReduce.
Module VF := ExactVFold.

Definition is01 (x : Z) : Prop := x = 0 \/ x = 1.

(** Python's `sum(g)`: left to right from 0, each addition in float64. *)
Definition fsum (xs : list Z) : Z := fold_left (fun a x => ER.cadd ER.F64 a x) xs 0.

Lemma fold_exact : forall xs a, Forall is01 xs -> 0 <= a ->
  a + Z.of_nat (length xs) <= 2 ^ 53 ->
  fold_left (fun a x => ER.cadd ER.F64 a x) xs a = a + ER.zsum xs.
Proof.
  induction xs as [|x t IH]; intros a F Ha Hb; cbn [fold_left ER.zsum length] in *; [lia|].
  inversion F as [|? ? Hx Ht]; subst.
  rewrite Nat2Z.inj_succ in Hb.
  assert (E : ER.cadd ER.F64 a x = a + x).
  { cbn [ER.cadd]. apply ER.rne_exact; [lia|]. destruct Hx as [->| ->]; lia. }
  rewrite E, IH; [ring | exact Ht | destruct Hx as [->| ->]; lia | destruct Hx as [->| ->]; lia].
Qed.

Lemma zsum_01 : forall xs, Forall is01 xs -> 0 <= ER.zsum xs <= Z.of_nat (length xs).
Proof.
  induction xs as [|x t IH]; intro F; cbn [ER.zsum length]; [lia|].
  inversion F as [|? ? Hx Ht]; subst. specialize (IH Ht). rewrite Nat2Z.inj_succ.
  destruct Hx as [->| ->]; lia.
Qed.

Theorem the_reward_sum_is_exact : forall xs, Forall is01 xs ->
  Z.of_nat (length xs) <= 2 ^ 53 -> fsum xs = ER.zsum xs.
Proof. intros xs F L. unfold fsum. rewrite fold_exact by (try exact F; lia). ring. Qed.

Section Mean.
  Variable rn : Z -> Z.
  Hypothesis rn_fix : forall E N, 0 <= E <= 2046 -> 0 <= N < 2 ^ 52 ->
    rn (FO.value 52 E N) = FO.value 52 E N /\ rn (- FO.value 52 E N) = - FO.value 52 E N.

  (** An integer below 2^53 times 2^j is a float64 (and so is its negation). *)
  Lemma rn_keeps : forall K j, Z.abs K < 2 ^ 53 -> 0 <= j <= 1074 -> rn (K * 2 ^ j) = K * 2 ^ j.
  Proof.
    intros K j HK Hj.
    destruct (VF.an_integer_times_a_power_of_two_is_a_float 52 (Z.abs K) j ltac:(lia)
                ltac:(replace (52 + 1) with 53 by lia; lia) ltac:(lia)) as [E [N [HE [HN V]]]].
    destruct (rn_fix E N ltac:(lia) HN) as [P M].
    destruct (Z_le_gt_dec 0 K) as [Pos|Neg].
    - rewrite Z.abs_eq in V by lia. rewrite <- V. exact P.
    - rewrite Z.abs_neq in V by lia.
      replace (K * 2 ^ j) with (- (- K * 2 ^ j)) by ring. rewrite <- V. exact M.
  Qed.

  Variable k : Z.
  Hypothesis Hk : 0 <= k <= 52.
  Variable xs : list Z.
  Hypothesis F : Forall is01 xs.
  Hypothesis L : Z.of_nat (length xs) = 2 ^ k.

  Definition s : Z := fsum xs.

  Lemma s_range : s = ER.zsum xs /\ 0 <= s <= 2 ^ k.
  Proof.
    assert (2 ^ k <= 2 ^ 53) by (apply Z.pow_le_mono_r; lia).
    unfold s. rewrite the_reward_sum_is_exact by (try exact F; lia).
    pose proof (zsum_01 xs F). lia.
  Qed.

  (** m = rn(s / 2^k): in units of 2^-1074, s * 2^(1074 - k). *)
  Theorem the_mean_is_exact : rn (s * 2 ^ (1074 - k)) = s * 2 ^ (1074 - k).
  Proof.
    destruct s_range as [_ Hs]. apply rn_keeps; [|lia].
    assert (2 ^ k < 2 ^ 53) by (apply Z.pow_lt_mono_r; lia). lia.
  Qed.

  (** x - m: x * 2^1074 - s * 2^(1074 - k) = (x 2^k - s) * 2^(1074 - k). *)
  Theorem every_advantage_is_exact : forall x, In x xs ->
    rn (x * 2 ^ 1074 - s * 2 ^ (1074 - k)) = (x * 2 ^ k - s) * 2 ^ (1074 - k).
  Proof.
    intros x Hx. destruct s_range as [_ Hs].
    assert (Hx01 : is01 x) by (rewrite Forall_forall in F; exact (F x Hx)).
    assert (P : 2 ^ 1074 = 2 ^ k * 2 ^ (1074 - k)) by (rewrite <- Z.pow_add_r by lia; f_equal; lia).
    replace (x * 2 ^ 1074 - s * 2 ^ (1074 - k)) with ((x * 2 ^ k - s) * 2 ^ (1074 - k)) by (rewrite P; ring).
    apply rn_keeps; [|lia].
    assert (2 ^ k < 2 ^ 53) by (apply Z.pow_lt_mono_r; lia).
    assert (0 < 2 ^ k) by (apply Z.pow_pos_nonneg; lia).
    destruct Hx01 as [->| ->]; lia.
  Qed.
End Mean.

(** Every float64 is an integer in units of 2^-1074; 1/3 is 2^1074 / 3 of
    them, which is no integer. *)
Theorem a_group_of_three_rounds_the_mean : 2 ^ 1074 mod 3 <> 0.
Proof. vm_compute. discriminate. Qed.

Print Assumptions the_reward_sum_is_exact.
Print Assumptions the_mean_is_exact.
Print Assumptions every_advantage_is_exact.
Print Assumptions a_group_of_three_rounds_the_mean.
