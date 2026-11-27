(** * The exact sampler's draw probabilities (`python/y_lang/exact/rl.py`).

    The sampler draws a token from integer weights p (the exact softmax's
    Q0.28 numerators, all >= 0, sum l) by integer inverse CDF: u = r mod l
    for a 62-bit integer r (`uniforms`: blake2b, shifted to 62 bits), and the
    token is the first index whose running sum passes u (the count of
    indices whose inclusive running sum is at most u, as
    `ExactDrawSearch.v` defines it). The module docstring says
    P(i) = p_i / l "to within the modulo's bias, below l / 2^62 < 2^-16
    (l < 2^28 times the vocabulary)". This file proves it, counting the
    62-bit values r that draw each token.

    [the_token_is_the_block_holding_u]: for 0 <= u < l, the token is t
    exactly when S_t <= u < S_(t+1), S the running sums: each token owns
    p_t consecutive values of u.
    [the_count_in_a_residue_block]: for 0 <= a <= b <= l, the number of r
    in 0 .. N - 1 with a <= r mod l < b is (b - a) (N / l) plus at most
    b - a more.
    [each_token_is_drawn_p_over_l_up_to_the_bias]: so the count c_t of
    r drawing token t has |c_t l - p_t N| <= p_t l, which is
    |c_t / N - p_t / l| <= p_t / N <= l / N.
    [the_bias_is_below_2_minus_16]: at N = 2^62, for l below 2^46 (2^28 per
    entry, a vocabulary below 2^18: Qwen2.5's 151,936, Llama 3's 128,256),
    the bias is below 2^-16.

    ** Refutation.

    [the_modulo_has_a_bias]: three weights of 1 and N = 4: token 0 is
    drawn by 2 of the 4 values, the others by 1 each, so the bias is real
    and the bound is what can be said.

    ** Not proved here.

    That blake2b's 62-bit values are uniform (the hash's assumption); that
    the sampler's p and l are `softmax_ints`' (`ExactDrawSearch.v` and the
    tests).

    Checked with Rocq 9.1:  coqc -Q . "" ExactDrawBias.v  *)

From Stdlib Require Import ZArith Lia Bool.
Require ExactDrawSearch.
Open Scope Z_scope.

Module DS := ExactDrawSearch.

Section Draw.
  Variable p : nat -> Z.
  Variable V : nat.                              (** the vocabulary *)
  Hypothesis Hp : forall i, 0 <= p i.

  Definition S (i : nat) : Z := DS.sumto i p.
  Definition l : Z := S V.

  (** The flat count: indices whose inclusive running sum is at most u. *)
  Definition token (u : Z) : nat := DS.count V (fun i => S (Datatypes.S i) <=? u).

  Lemma S_succ : forall i, S (Datatypes.S i) = S i + p i.
  Proof. intro i. reflexivity. Qed.

  Lemma S_mono : forall i j, (i <= j)%nat -> S i <= S j.
  Proof. intros i j H. apply DS.sumto_mono; [exact H | exact Hp]. Qed.

  Lemma pred_prefix : forall u, DS.prefix_pred V (fun i => S (Datatypes.S i) <=? u).
  Proof.
    intros u i i' H Hi'. apply Z.leb_le in Hi'. apply Z.leb_le.
    pose proof (S_mono (Datatypes.S i) (Datatypes.S i') ltac:(lia)). lia.
  Qed.

  Theorem the_token_is_the_block_holding_u : forall u t, 0 <= u < l -> (t < V)%nat ->
    (token u = t <-> S t <= u < S (Datatypes.S t)).
  Proof.
    intros u t Hu Ht. unfold token. split.
    - intro E. pose proof (DS.count_spec V _ (pred_prefix u)) as C. rewrite E in C. split.
      + destruct t as [|t']; [cbn; unfold S; cbn; lia|].
        assert (Ct : (t' < V)%nat) by lia. specialize (C t' Ct). apply proj2 in C.
        specialize (C ltac:(lia)). apply Z.leb_le in C. exact C.
      + specialize (C t Ht). destruct (S (Datatypes.S t) <=? u) eqn:B.
        * apply proj1 in C. specialize (C eq_refl). lia.
        * apply Z.leb_gt in B. exact B.
    - intros [A B]. apply DS.count_unique; [apply pred_prefix | lia |].
      intros i Hi. split.
      + intro H. apply Z.leb_le in H. destruct (Nat.lt_ge_cases i t) as [|G]; [assumption|].
        pose proof (S_mono (Datatypes.S t) (Datatypes.S i) ltac:(lia)). lia.
      + intro H. apply Z.leb_le. pose proof (S_mono (Datatypes.S i) t ltac:(lia)). lia.
  Qed.
End Draw.

(** ** Counting the r in 0 .. N - 1 by their residue *)

Definition in_block (l a b : Z) (r : nat) : bool :=
  (a <=? Z.of_nat r mod l) && (Z.of_nat r mod l <? b).

Lemma count_in_block_exact : forall l a b N, 0 < l -> 0 <= a <= b -> b <= l ->
  Z.of_nat (DS.count N (in_block l a b))
  = (b - a) * (Z.of_nat N / l) + Z.max 0 (Z.min (Z.of_nat N mod l) b - a).
Proof.
  intros l a b N Hl Ha Hb. induction N as [|N IH].
  - cbn. lia.
  - cbn [DS.count]. rewrite Nat2Z.inj_add, IH.
    pose proof (Z.div_mod (Z.of_nat N) l ltac:(lia)) as DM. pose proof (Z.mod_pos_bound (Z.of_nat N) l Hl) as MB.
    set (q := Z.of_nat N / l) in *. set (m := Z.of_nat N mod l) in *.
    unfold in_block. fold m.
    rewrite Nat2Z.inj_succ.
    destruct (Z.eq_dec (m + 1) l) as [Wrap|NoWrap].
    + assert (Q1 : (Z.succ (Z.of_nat N)) / l = q + 1) by (symmetry; apply Z.div_unique with 0; lia).
      assert (M1 : (Z.succ (Z.of_nat N)) mod l = 0) by (symmetry; apply Z.mod_unique with (q + 1); lia).
      rewrite Q1, M1. destruct (a <=? m) eqn:A; destruct (m <? b) eqn:B; cbn;
        rewrite ?Z.leb_le, ?Z.leb_gt, ?Z.ltb_lt, ?Z.ltb_ge in *; lia.
    + assert (Q1 : (Z.succ (Z.of_nat N)) / l = q) by (symmetry; apply Z.div_unique with (m + 1); lia).
      assert (M1 : (Z.succ (Z.of_nat N)) mod l = m + 1) by (symmetry; apply Z.mod_unique with q; lia).
      rewrite Q1, M1. destruct (a <=? m) eqn:A; destruct (m <? b) eqn:B; cbn;
        rewrite ?Z.leb_le, ?Z.leb_gt, ?Z.ltb_lt, ?Z.ltb_ge in *; lia.
Qed.

Theorem the_count_in_a_residue_block : forall l a b N, 0 < l -> 0 <= a <= b -> b <= l ->
  (b - a) * (Z.of_nat N / l) <= Z.of_nat (DS.count N (in_block l a b))
                              <= (b - a) * (Z.of_nat N / l + 1).
Proof. intros l a b N Hl Ha Hb. rewrite count_in_block_exact by assumption. lia. Qed.

(** ** Each token's share *)

Section Share.
  Variable p : nat -> Z.
  Variable V : nat.
  Hypothesis Hp : forall i, 0 <= p i.
  Hypothesis Hl : 0 < l p V.
  Variable N : nat.                               (** 2^62, the r's *)

  (** The number of r in 0 .. N - 1 whose u = r mod l draws token t. *)
  Definition drawn (t : nat) : nat :=
    DS.count N (fun r => Nat.eqb (token p V (Z.of_nat r mod l p V)) t).

  Lemma drawn_is_a_block : forall t, (t < V)%nat ->
    drawn t = DS.count N (in_block (l p V) (S p t) (S p (Datatypes.S t))).
  Proof.
    intros t Ht. unfold drawn, in_block.
    assert (E : forall r, Nat.eqb (token p V (Z.of_nat r mod l p V)) t
                  = (S p t <=? Z.of_nat r mod l p V) && (Z.of_nat r mod l p V <? S p (Datatypes.S t))).
    { intro r. pose proof (Z.mod_pos_bound (Z.of_nat r) (l p V) Hl) as MB.
      pose proof (the_token_is_the_block_holding_u p V Hp (Z.of_nat r mod l p V) t MB Ht) as T.
      destruct (Nat.eqb (token p V (Z.of_nat r mod l p V)) t) eqn:Q.
      - apply Nat.eqb_eq in Q. apply T in Q as [A B]. symmetry. apply andb_true_iff.
        split; [apply Z.leb_le | apply Z.ltb_lt]; lia.
      - apply Nat.eqb_neq in Q. symmetry. apply not_true_iff_false. intro H.
        apply andb_true_iff in H as [A B]. apply Z.leb_le in A. apply Z.ltb_lt in B.
        apply Q, T. lia. }
    clear Ht. induction N as [|n IH]; [reflexivity|]. cbn [DS.count]. rewrite IH, E. reflexivity.
  Qed.

  Theorem each_token_is_drawn_p_over_l_up_to_the_bias : forall t, (t < V)%nat ->
    Z.abs (Z.of_nat (drawn t) * l p V - p t * Z.of_nat N) <= p t * l p V.
  Proof.
    intros t Ht. rewrite drawn_is_a_block by exact Ht.
    assert (Hb : S p (Datatypes.S t) <= l p V) by (apply S_mono; [exact Hp | lia]).
    assert (Ha : 0 <= S p t) by (apply DS.sumto_nonneg; intros; apply Hp).
    pose proof (S_mono p Hp t (Datatypes.S t) ltac:(lia)).
    pose proof (the_count_in_a_residue_block (l p V) (S p t) (S p (Datatypes.S t)) N Hl ltac:(lia) Hb) as [Lo Hi].
    rewrite S_succ in Lo, Hi. replace (S p t + p t - S p t) with (p t) in Lo, Hi by ring.
    pose proof (Z.div_mod (Z.of_nat N) (l p V) ltac:(lia)) as DM.
    pose proof (Z.mod_pos_bound (Z.of_nat N) (l p V) Hl) as MB.
    set (q := Z.of_nat N / l p V) in *. set (m := Z.of_nat N mod l p V) in *.
    set (c := Z.of_nat (DS.count N (in_block (l p V) (S p t) (S p t + p t)))) in *.
    pose proof (Hp t). rewrite S_succ in *.
    assert (E : p t * Z.of_nat N = p t * q * l p V + p t * m) by (rewrite DM; ring).
    rewrite E. nia.
  Qed.

  Theorem the_bias_is_below_2_minus_16 : forall t, (t < V)%nat ->
    Z.of_nat N = 2 ^ 62 -> l p V < 2 ^ 46 ->
    Z.abs (Z.of_nat (drawn t) * l p V - p t * 2 ^ 62) * 2 ^ 16 < l p V * 2 ^ 62.
  Proof.
    intros t Ht HN HL. pose proof (each_token_is_drawn_p_over_l_up_to_the_bias t Ht) as B.
    rewrite HN in B.
    assert (Pt : p t <= l p V).
    { pose proof (S_mono p Hp (Datatypes.S t) V ltac:(lia)). pose proof (S_mono p Hp O t ltac:(lia)).
      rewrite S_succ in *. unfold l. unfold S at 1 in H0. cbn in H0. lia. }
    assert (0 <= p t) by apply Hp.
    assert (p t * l p V * 2 ^ 16 < l p V * 2 ^ 62).
    { assert (2 ^ 62 = 2 ^ 46 * 2 ^ 16) by reflexivity. nia. }
    nia.
  Qed.
End Share.

(** Three weights of 1, N = 4: r = 0, 3 draw token 0, r = 1 token 1, r = 2 token 2. *)
Definition ones3 (i : nat) : Z := 1.

Theorem the_modulo_has_a_bias :
  drawn ones3 3 4 0 = 2%nat /\ drawn ones3 3 4 1 = 1%nat /\ drawn ones3 3 4 2 = 1%nat.
Proof. vm_compute. repeat split; reflexivity. Qed.

Print Assumptions the_token_is_the_block_holding_u.
Print Assumptions the_count_in_a_residue_block.
Print Assumptions each_token_is_drawn_p_over_l_up_to_the_bias.
Print Assumptions the_bias_is_below_2_minus_16.
Print Assumptions the_modulo_has_a_bias.
