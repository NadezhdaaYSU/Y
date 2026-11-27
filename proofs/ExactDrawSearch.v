(** * The fused draw's two-level search (`python/y_lang/exact/vocab_draw.py`).

    The exact sampler draws token t from integer weights p (the exact
    softmax's p, all >= 0) with u = r mod l, l the sum of p: t is the first
    index whose running sum passes u, which `rl.draw_rows` computes as the
    count of indices whose inclusive running sum is at most u.

    `_draw_kernel` never stores p. It sums each block of B entries (entries
    past the vocabulary are 0), counts the blocks k whose inclusive running
    block sum is at most u, computes block k's p again, and counts the
    entries j of that block whose running sum, started at the sum of the
    blocks before it, is at most u. The token is k * B + j.

    [the_two_level_search_is_the_flat_one]: for every block size B > 0,
    every number of blocks and every p >= 0 with 0 <= u < l, k * B + j is
    the flat count, and so the same token. [the_block_is_in_range] and
    [the_offset_is_in_range]: k < the number of blocks and j < B, so the
    kernel's second pass reads a block that exists and the token is inside
    it.

    The general fact underneath, [count_spec]: for a predicate that holds
    on a prefix of 0 .. n - 1 (a running sum of non-negative terms is at
    most u up to some index, then never again), index i satisfies it exactly
    when i is below the count.

    ** Refutations.

    [a_negative_weight_breaks_it]: with one negative p the running sums are
    not monotone, and the two counts differ. [u_at_the_total_leaves_the_blocks]:
    without the `mod l` (u = l), every block counts, so k is the number of
    blocks, past the last one.

    ** Not proved here.

    That the kernel's p equal `softmax_ints`' (the same integer and float32
    steps; tested bit for bit by python/tests/test_exact_vocab_draw.py), and
    that `tl.cumsum` and `tl.sum` of int64 are the integer sums (no
    overflow: p <= 2^28 and the vocabulary below 2^34 entries).

    Checked with Rocq 9.1:  coqc -Q . "" ExactDrawSearch.v  *)

From Stdlib Require Import ZArith Lia Bool.
Open Scope Z_scope.

Fixpoint sumto (n : nat) (f : nat -> Z) : Z :=
  match n with
  | O => 0
  | S m => sumto m f + f m
  end.

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

Lemma sumto_nonneg : forall n f, (forall i, (i < n)%nat -> 0 <= f i) -> 0 <= sumto n f.
Proof.
  induction n as [|n IH]; intros f H; cbn [sumto]; [lia|].
  specialize (IH f (fun i Hi => H i ltac:(lia))). specialize (H n ltac:(lia)). lia.
Qed.

Lemma sumto_mono : forall a b f, (a <= b)%nat -> (forall i, 0 <= f i) -> sumto a f <= sumto b f.
Proof.
  intros a b f Hab Hf. replace b with (a + (b - a))%nat by lia. rewrite sumto_add.
  pose proof (sumto_nonneg (b - a) (fun j => f (a + j)%nat) (fun i _ => Hf _)). lia.
Qed.

(* ------------------------------------------------------------------ *)
(** ** Counting a prefix predicate                                     *)
(* ------------------------------------------------------------------ *)

Fixpoint count (n : nat) (P : nat -> bool) : nat :=
  match n with
  | O => O
  | S m => (count m P + (if P m then 1 else 0))%nat
  end.

(** P holds on a prefix of 0 .. n - 1. *)
Definition prefix_pred (n : nat) (P : nat -> bool) : Prop :=
  forall i i', (i <= i' < n)%nat -> P i' = true -> P i = true.

Lemma count_le : forall n P, (count n P <= n)%nat.
Proof. induction n as [|n IH]; intro P; cbn [count]; [lia|]. specialize (IH P). destruct (P n); lia. Qed.

Lemma count_all : forall n P, (forall i, (i < n)%nat -> P i = true) -> count n P = n.
Proof.
  induction n as [|n IH]; intros P H; cbn [count]; [reflexivity|].
  rewrite IH by (intros; apply H; lia). rewrite (H n) by lia. lia.
Qed.

Theorem count_spec : forall n P, prefix_pred n P ->
  forall i, (i < n)%nat -> (P i = true <-> (i < count n P)%nat).
Proof.
  induction n as [|n IH]; intros P HP i Hi; [lia|]. cbn [count].
  assert (HPn : prefix_pred n P) by (intros a b Hab; apply HP; lia).
  destruct (P n) eqn:En.
  - rewrite (count_all n P) by (intros x Hx; apply (HP x n); [lia | exact En]).
    split; intros; [lia|]. destruct (Nat.eq_dec i n) as [->|Hne]; [exact En|].
    apply (HP i n); [lia | exact En].
  - rewrite Nat.add_0_r. destruct (Nat.eq_dec i n) as [->|Hne].
    + rewrite En. pose proof (count_le n P). split; [discriminate | lia].
    + apply IH; [exact HPn | lia].
Qed.

Lemma count_unique : forall n P t, prefix_pred n P -> (t <= n)%nat ->
  (forall i, (i < n)%nat -> (P i = true <-> (i < t)%nat)) -> count n P = t.
Proof.
  intros n P t HP Ht H. pose proof (count_le n P) as Hc.
  destruct (Nat.lt_trichotomy (count n P) t) as [Hlt|[Heq|Hgt]]; [|exact Heq|].
  - pose proof (count_spec n P HP (count n P) ltac:(lia)) as S1. pose proof (H (count n P) ltac:(lia)) as S2.
    destruct (P (count n P)); [destruct S1 as [S1 _]; specialize (S1 eq_refl); lia|].
    destruct S2 as [_ S2]. specialize (S2 Hlt). discriminate.
  - pose proof (count_spec n P HP t ltac:(lia)) as S1. pose proof (H t ltac:(lia)) as S2.
    destruct (P t); [destruct S2 as [S2 _]; specialize (S2 eq_refl); lia|].
    destruct S1 as [_ S1]. specialize (S1 Hgt). discriminate.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The two searches                                                *)
(* ------------------------------------------------------------------ *)

Section Search.
  Variable p : nat -> Z.              (** p of the vocabulary, 0 past it *)
  Variable B nb : nat.                (** block size, number of blocks *)
  Variable u : Z.                     (** r mod l *)

  (** The flat search: the count of indices whose inclusive running sum is
      at most u. *)
  Definition run (i : nat) : Z := sumto (S i) p.
  Definition flat : nat := count (nb * B) (fun i => run i <=? u).

  (** The kernel: block sums, the count of blocks, then the count inside
      block k from the sum of the blocks before it. *)
  Definition bsum (b : nat) : Z := sumto B (fun x => p (b * B + x)%nat).
  Definition cum (b : nat) : Z := sumto (S b) bsum.
  Definition k : nat := count nb (fun b => cum b <=? u).
  Definition before : Z := sumto k bsum.
  Definition within (x : nat) : Z := sumto (S x) (fun y => p (k * B + y)%nat).
  Definition j : nat := count B (fun x => before + within x <=? u).

  Hypothesis Hp : forall i, 0 <= p i.
  Hypothesis HB : (0 < B)%nat.
  Hypothesis Hu : 0 <= u < sumto (nb * B) p.

  Lemma bsum_nonneg : forall b, 0 <= bsum b.
  Proof. intro b. apply sumto_nonneg. intros; apply Hp. Qed.

  Lemma blocks_prefix : forall b, sumto b bsum = sumto (b * B) p.
  Proof. intro b. unfold bsum. apply sumto_blocks. Qed.

  Lemma run_split : forall b x, (x < B)%nat -> run (b * B + x) = sumto b bsum + sumto (S x) (fun y => p (b * B + y)%nat).
  Proof.
    intros b x Hx. unfold run. replace (S (b * B + x)) with (b * B + S x)%nat by lia.
    rewrite sumto_add, blocks_prefix. reflexivity.
  Qed.

  Lemma cum_prefix : prefix_pred nb (fun b => cum b <=? u).
  Proof.
    intros a b Hab H. apply Z.leb_le in H. apply Z.leb_le. unfold cum in *.
    pose proof (sumto_mono (S a) (S b) bsum ltac:(lia) bsum_nonneg). lia.
  Qed.

  Lemma total_is_last_cum : (0 < nb)%nat -> cum (nb - 1) = sumto (nb * B) p.
  Proof. intro H. unfold cum. replace (S (nb - 1)) with nb by lia. apply blocks_prefix. Qed.

  Lemma nb_pos : (0 < nb)%nat.
  Proof. destruct nb as [|n]; [cbn in Hu; lia | lia]. Qed.

  Theorem the_block_is_in_range : (k < nb)%nat.
  Proof.
    pose proof nb_pos as Hn. pose proof (count_spec nb _ cum_prefix (nb - 1) ltac:(lia)) as S.
    assert (H : (cum (nb - 1) <=? u) = false) by (apply Z.leb_gt; rewrite total_is_last_cum by exact Hn; lia).
    cbv beta in S. rewrite H in S. unfold k.
    destruct (Nat.lt_ge_cases (count nb (fun b => (cum b <=? u)%Z)) nb) as [L|L]; [exact L|].
    exfalso. destruct S as [_ S]. specialize (S ltac:(lia)). discriminate.
  Qed.

  Lemma within_prefix : prefix_pred B (fun x => before + within x <=? u).
  Proof.
    intros a b Hab H. apply Z.leb_le in H. apply Z.leb_le. unfold within in *.
    pose proof (sumto_mono (S a) (S b) (fun y => p (k * B + y)%nat) ltac:(lia) (fun i => Hp _)). lia.
  Qed.

  Lemma before_plus_last : before + within (B - 1) = cum k.
  Proof.
    unfold before, within, cum. replace (S (B - 1)) with B by lia. cbn [sumto]. unfold bsum. reflexivity.
  Qed.

  Lemma k_fails : cum k > u.
  Proof.
    pose proof the_block_is_in_range as Hk. pose proof (count_spec nb _ cum_prefix k Hk) as S.
    destruct (cum k <=? u) eqn:E; [destruct S as [S _]; specialize (S eq_refl); unfold k in S; lia|].
    apply Z.leb_gt in E. lia.
  Qed.

  Theorem the_offset_is_in_range : (j < B)%nat.
  Proof.
    pose proof (count_spec B _ within_prefix (B - 1) ltac:(lia)) as S.
    assert (H : (before + within (B - 1) <=? u) = false) by (apply Z.leb_gt; rewrite before_plus_last; pose proof k_fails; lia).
    cbv beta in S. rewrite H in S. unfold j.
    destruct (Nat.lt_ge_cases (count B (fun x => (before + within x <=? u)%Z)) B) as [L|L]; [exact L|].
    exfalso. destruct S as [_ S]. specialize (S ltac:(lia)). discriminate.
  Qed.

  Lemma run_prefix : prefix_pred (nb * B) (fun i => run i <=? u).
  Proof.
    intros a b Hab H. apply Z.leb_le in H. apply Z.leb_le. unfold run in *.
    pose proof (sumto_mono (S a) (S b) p ltac:(lia) Hp). lia.
  Qed.

  Theorem the_two_level_search_is_the_flat_one : (k * B + j)%nat = flat.
  Proof.
    pose proof the_block_is_in_range as Hk. pose proof the_offset_is_in_range as Hj.
    symmetry. unfold flat. apply count_unique; [exact run_prefix | nia |].
    intros i Hi.
    pose proof (Nat.div_mod_eq i B) as Ei. pose proof (Nat.mod_upper_bound i B ltac:(lia)) as Bi.
    set (b := (i / B)%nat) in *. set (x := (i mod B)%nat) in *.
    assert (Ri : run i = sumto b bsum + sumto (S x) (fun y => p (b * B + y)%nat)).
    { rewrite <- run_split by exact Bi. f_equal. lia. }
    assert (Wx : 0 <= sumto (S x) (fun y => p (b * B + y)%nat) <= bsum b).
    { split; [apply sumto_nonneg; intros; apply Hp|]. unfold bsum.
      apply (sumto_mono (S x) B (fun y => p (b * B + y)%nat) ltac:(lia) (fun _ => Hp _)). }
    rewrite Ri. destruct (Nat.lt_trichotomy b k) as [Hb|[Hb|Hb]].
    - (* a block before k: all of it counts *)
      assert (cum b <= u).
      { pose proof (count_spec nb _ cum_prefix b ltac:(lia)) as S. apply Z.leb_le, S. unfold k in Hb. exact Hb. }
      unfold cum in H. cbn [sumto] in H. split; intros; [nia|]. apply Z.leb_le. lia.
    - (* block k: the offsets below j *)
      rewrite Hb in Ri, Ei, Wx |- *.
      change (sumto k bsum + sumto (S x) (fun y => p (k * B + y)%nat)) with (before + within x).
      pose proof (count_spec B _ within_prefix x Bi) as S. cbv beta in S. rewrite S.
      change (count B (fun x0 => (before + within x0 <=? u)%Z)) with j. split; intro; nia.
    - (* a block after k: none of it *)
      assert (cum k <= sumto b bsum) by (apply (sumto_mono (S k) b bsum ltac:(lia) bsum_nonneg)).
      pose proof k_fails. split; intro H1; [apply Z.leb_le in H1; lia | nia].
  Qed.
End Search.

(* ------------------------------------------------------------------ *)
(** ** Refutations                                                     *)
(* ------------------------------------------------------------------ *)

(** p = [3, -2, 4, 0] in blocks of 2, u = 2: the running sums 3, 1, 5, 5
    are not monotone; the flat count is 1 (only index 1), the two-level
    search finds block 0's sum 1 <= 2, so k = 1, then j = 0: token 2. *)
Definition bad_p (i : nat) : Z := match i with O => 3 | S O => -2 | S (S O) => 4 | _ => 0 end.

Theorem a_negative_weight_breaks_it :
  (k bad_p 2 2 2 * 2 + j bad_p 2 2 2)%nat <> flat bad_p 2 2 2.
Proof. vm_compute. discriminate. Qed.

(** u = l, the total, without `mod l`: every block's running sum is at
    most u, so k is the number of blocks. *)
Definition ones (i : nat) : Z := 1.

Theorem u_at_the_total_leaves_the_blocks : k ones 2 3 6 = 3%nat.
Proof. vm_compute. reflexivity. Qed.

Print Assumptions count_spec.
Print Assumptions the_block_is_in_range.
Print Assumptions the_offset_is_in_range.
Print Assumptions the_two_level_search_is_the_flat_one.
Print Assumptions a_negative_weight_breaks_it.
Print Assumptions u_at_the_total_leaves_the_blocks.
