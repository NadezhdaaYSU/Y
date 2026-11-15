(** * Any order, any association: a reduction of integers in a bounded carrier.

    The kernel proofs in this directory each take one kernel's schedule (its
    tiles, its K split, its grid stride) and show that schedule sums to the
    plain fold. The exact decode path in `python/y_lang/exact/` mostly does not
    run Y's kernels: by default its sums go through Triton's `tl.dot`,
    `torch._int_mm`, a float32 `matmul` and `torch.sum`, whose schedules are
    not Y's to state. This file proves the one fact that covers all of them.

    ** The claim.

    A SCHEDULE is a binary tree whose leaves name the terms, each exactly
    once, with any number of zero leaves ([tree], [covers]). That is every
    association and every order: a sequential loop, a pairwise tree, a tiling,
    a K split, per-thread partials summed at the end, atomics landing in any
    order, a contraction padded with zeros to a multiple of 32.

    A CARRIER is the machine type the additions run in ([carrier]): int32 and
    int64 with two's-complement wrap, float32 and float64 with round to
    nearest even. Each is EXACT WITHIN its cap ([exact_within]): if the true
    sum of two values has magnitude at most the cap (2^31 - 1, 2^63 - 1, 2^24,
    2^53), the machine addition returns it.

    [any_schedule_gives_the_sum]: if the absolute values of the terms sum to
    at most the cap, every schedule in that carrier returns the same integer,
    the mathematical sum. [no_partial_leaves_the_range] adds that no partial
    sum anywhere in the tree leaves the range, so the result does not depend
    on what the machine does on overflow.

    The premise is the bound the Python code asserts at every site, for
    example `in_features * 127 * 127 < 2^31` in `ExactLinear` and
    `d * q_levels * k_levels < 2^24` in `assert_score_budget`: a bound on the
    sum of absolute values, which is what bounds every partial sum in every
    order.

    ** The refutations.

    [float32_addition_rounds_past_its_cap] and
    [past_the_cap_two_schedules_disagree]: one unit past 2^24 the same three
    terms give different float32 results under two associations. That is the
    bound being load-bearing, and it is what a float accumulation planted in
    a reduction does.

    ** Max and argmax.

    A max reduction is modelled by its specification, not by a schedule:
    [is_max_of] says the result is one of the inputs and no input has a larger
    order key. Any comparison-based max in any order meets it
    ([any_max_tree_is_a_max]). Two results that meet it have equal keys
    ([max_key_unique]); they are the same value where the key decides the
    value. [is_first_argmax] is torch's documented rule (the first index of a
    maximal value), and it has exactly one answer ([first_argmax_unique]).

    ** What is NOT proved here.

    - That a given kernel's additions form such a tree. For Y's kernels the
      per-kernel files prove more (the emitted indices); for Triton, cuBLASLt
      and oneDNN it is an assumption.
    - That a device's float32 add is [rne 24] on integers. [rne] is IEEE 754
      round-to-nearest-even restricted to integer values below overflow; the
      tie to hardware is an assumption.

    Checked with Rocq 9.1:  coqc -Q . "" ExactReduce.v  *)

From Stdlib Require Import ZArith Arith Lia List Permutation Bool.
Require Decomposition ExactGemmMicro ExactPvExact.
Import ListNotations.
Open Scope Z_scope.

Module D := Decomposition.
Module MC := ExactGemmMicro.
Module EP := ExactPvExact.

(* ------------------------------------------------------------------ *)
(** ** Sums of lists                                                   *)
(* ------------------------------------------------------------------ *)

Fixpoint zsum (l : list Z) : Z :=
  match l with
  | [] => 0
  | x :: r => x + zsum r
  end.

Definition abs_sum (l : list Z) : Z := zsum (map Z.abs l).

Lemma zsum_app : forall a b, zsum (a ++ b) = zsum a + zsum b.
Proof. induction a as [|x a IH]; intros b; cbn; [reflexivity|]. rewrite IH. ring. Qed.

Lemma zsum_perm : forall a b, Permutation a b -> zsum a = zsum b.
Proof. induction 1; cbn; lia. Qed.

Lemma abs_sum_app : forall a b, abs_sum (a ++ b) = abs_sum a + abs_sum b.
Proof. intros a b. unfold abs_sum. rewrite map_app. apply zsum_app. Qed.

Lemma abs_sum_perm : forall a b, Permutation a b -> abs_sum a = abs_sum b.
Proof. intros a b H. unfold abs_sum. apply zsum_perm, Permutation_map, H. Qed.

Lemma abs_sum_nonneg : forall l, 0 <= abs_sum l.
Proof. induction l as [|x l IH]; unfold abs_sum in *; cbn; lia. Qed.

Lemma zsum_le_abs_sum : forall l, Z.abs (zsum l) <= abs_sum l.
Proof. induction l as [|x l IH]; unfold abs_sum in *; cbn; lia. Qed.

Lemma zsum_nonneg : forall l, Forall (fun x => 0 <= x) l -> 0 <= zsum l.
Proof. induction 1; cbn; lia. Qed.

Lemma abs_sum_of_nonneg : forall l, Forall (fun x => 0 <= x) l -> abs_sum l = zsum l.
Proof.
  induction 1 as [|x l Hx Hl IH]; [reflexivity|].
  unfold abs_sum in *. cbn. rewrite IH. lia.
Qed.

Lemma zsum_bounded : forall c l, 0 <= c -> Forall (fun x => Z.abs x <= c) l ->
  abs_sum l <= Z.of_nat (length l) * c.
Proof.
  intros c l Hc H. induction H as [|x l Hx Hl IH]; [cbn; lia|].
  unfold abs_sum in *. cbn [map zsum length]. rewrite Nat2Z.inj_succ. lia.
Qed.

Lemma map_nth_seq : forall (l : list Z),
  map (fun i => nth i l 0) (seq 0 (length l)) = l.
Proof.
  induction l as [|x l IH]; [reflexivity|].
  cbn [length seq map nth]. f_equal.
  rewrite <- seq_shift, map_map. exact IH.
Qed.

(* ------------------------------------------------------------------ *)
(** ** Schedules                                                       *)
(* ------------------------------------------------------------------ *)

Inductive tree : Type :=
| Leaf (i : nat)
| Zero
| Node (l r : tree).

Fixpoint leaves (t : tree) : list nat :=
  match t with
  | Leaf i => [i]
  | Zero => []
  | Node l r => leaves l ++ leaves r
  end.

(** The schedule uses each of the [n] terms exactly once. *)
Definition covers (t : tree) (n : nat) : Prop := Permutation (leaves t) (seq 0 n).

Fixpoint teval (add : Z -> Z -> Z) (terms : list Z) (t : tree) : Z :=
  match t with
  | Leaf i => nth i terms 0
  | Zero => 0
  | Node l r => add (teval add terms l) (teval add terms r)
  end.

(** The machine addition agrees with the integers while the true sum has
    magnitude at most [B]. *)
Definition exact_within (B : Z) (add : Z -> Z -> Z) : Prop :=
  forall a b, Z.abs (a + b) <= B -> add a b = a + b.

Definition picked (terms : list Z) (t : tree) : list Z :=
  map (fun i => nth i terms 0) (leaves t).

Lemma picked_node : forall terms l r,
  picked terms (Node l r) = picked terms l ++ picked terms r.
Proof. intros. unfold picked. cbn. apply map_app. Qed.

Lemma teval_picked : forall B add terms t,
  exact_within B add ->
  abs_sum (picked terms t) <= B ->
  teval add terms t = zsum (picked terms t).
Proof.
  intros B add terms t Hadd. induction t as [i| |l IHl r IHr]; intros Hb.
  - cbn. lia.
  - reflexivity.
  - rewrite picked_node in *. rewrite abs_sum_app in Hb.
    pose proof (abs_sum_nonneg (picked terms l)).
    pose proof (abs_sum_nonneg (picked terms r)).
    cbn [teval]. rewrite IHl by lia. rewrite IHr by lia.
    rewrite zsum_app. apply Hadd.
    pose proof (zsum_le_abs_sum (picked terms l)).
    pose proof (zsum_le_abs_sum (picked terms r)). lia.
Qed.

Lemma picked_covers : forall terms t,
  covers t (length terms) -> Permutation (picked terms t) terms.
Proof.
  intros terms t H. unfold picked.
  pose proof (Permutation_map (fun i => nth i terms 0) H) as P.
  rewrite map_nth_seq in P. exact P.
Qed.

(** Every association and every order of the terms gives their sum. *)
Theorem any_schedule_gives_the_sum : forall B add terms t,
  exact_within B add ->
  covers t (length terms) ->
  abs_sum terms <= B ->
  teval add terms t = zsum terms.
Proof.
  intros B add terms t Hadd Hc Hb.
  pose proof (picked_covers terms t Hc) as P.
  rewrite (teval_picked B) by (try assumption; rewrite (abs_sum_perm _ _ P); exact Hb).
  apply zsum_perm, P.
Qed.

Corollary any_two_schedules_agree : forall B add1 add2 terms t1 t2,
  exact_within B add1 -> exact_within B add2 ->
  covers t1 (length terms) -> covers t2 (length terms) ->
  abs_sum terms <= B ->
  teval add1 terms t1 = teval add2 terms t2.
Proof.
  intros. rewrite (any_schedule_gives_the_sum B add1) by assumption.
  rewrite (any_schedule_gives_the_sum B add2) by assumption. reflexivity.
Qed.

(** No partial sum anywhere in the tree leaves the range, so nothing
    depends on what the machine does when an addition overflows. *)
Fixpoint partials_within (B : Z) (terms : list Z) (t : tree) : Prop :=
  match t with
  | Node l r => partials_within B terms l /\ partials_within B terms r
                /\ Z.abs (teval Z.add terms t) <= B
  | _ => True
  end.

Lemma partials_picked : forall B terms t,
  abs_sum (picked terms t) <= B -> partials_within B terms t.
Proof.
  intros B terms t. induction t as [i| |l IHl r IHr]; intros Hb; cbn [partials_within]; auto.
  rewrite picked_node, abs_sum_app in Hb.
  pose proof (abs_sum_nonneg (picked terms l)).
  pose proof (abs_sum_nonneg (picked terms r)).
  assert (E : exact_within (abs_sum (picked terms l) + abs_sum (picked terms r)) Z.add)
    by (intros a b _; reflexivity).
  repeat split; [apply IHl; lia | apply IHr; lia |].
  rewrite (teval_picked _ Z.add terms (Node l r) E)
    by (rewrite picked_node, abs_sum_app; lia).
  rewrite picked_node, zsum_app.
  pose proof (zsum_le_abs_sum (picked terms l)).
  pose proof (zsum_le_abs_sum (picked terms r)). lia.
Qed.

Theorem no_partial_leaves_the_range : forall B terms t,
  covers t (length terms) -> abs_sum terms <= B -> partials_within B terms t.
Proof.
  intros B terms t Hc Hb. apply partials_picked.
  rewrite (abs_sum_perm _ _ (picked_covers terms t Hc)). exact Hb.
Qed.

(* ------------------------------------------------------------------ *)
(** ** Carriers                                                        *)
(* ------------------------------------------------------------------ *)

(** Round to nearest, ties to even, of the integer [z] to [p] significant
    bits: IEEE 754's default rounding on integer values below overflow. *)
Definition rne (p z : Z) : Z :=
  let a := Z.abs z in
  let n := Z.log2 a + 1 in
  if (a =? 0) || (n <=? p) then z
  else
    let s := n - p in
    let q := a / 2 ^ s in
    let r := a mod 2 ^ s in
    let h := 2 ^ (s - 1) in
    let q' := if r <? h then q
              else if h <? r then q + 1
              else if Z.even q then q else q + 1 in
    Z.sgn z * (q' * 2 ^ s).

Lemma rne_exact : forall p z, 0 < p -> Z.abs z <= 2 ^ p -> rne p z = z.
Proof.
  intros p z Hp Hz. unfold rne.
  destruct (Z.eqb_spec (Z.abs z) 0) as [E0|N0]; [reflexivity|]. cbn [orb].
  destruct (Z.leb_spec (Z.log2 (Z.abs z) + 1) p) as [Hn|Hn]; [reflexivity|].
  assert (Ha : 0 < Z.abs z) by lia.
  pose proof (Z.log2_spec (Z.abs z) Ha) as [Hlo _].
  assert (Hpow : 2 ^ p <= 2 ^ Z.log2 (Z.abs z)) by (apply Z.pow_le_mono_r; lia).
  assert (Ea : Z.abs z = 2 ^ p) by lia.
  rewrite Ea. rewrite Z.log2_pow2 by lia.
  replace (p + 1 - p) with 1 by lia. replace (1 - 1) with 0 by lia.
  change (2 ^ 1) with 2. change (2 ^ 0) with 1.
  assert (E2 : 2 ^ p = 2 * 2 ^ (p - 1)).
  { rewrite <- Z.pow_succ_r by lia. f_equal. lia. }
  assert (Hm : 2 ^ p mod 2 = 0).
  { rewrite E2, Z.mul_comm. apply Z.mod_mul. lia. }
  rewrite Hm. cbn [Z.ltb Z.compare].
  assert (Hd : 2 ^ p / 2 * 2 = 2 ^ p).
  { pose proof (Z.div_mod (2 ^ p) 2 ltac:(lia)) as DM. rewrite Hm in DM. lia. }
  rewrite Hd, <- Ea. clear. destruct z; cbn; lia.
Qed.

Inductive carrier : Type := I32 | I64 | F32 | F64.

Definition cadd (c : carrier) (a b : Z) : Z :=
  match c with
  | I32 => MC.wrap32 (a + b)
  | I64 => EP.wrap64 (a + b)
  | F32 => rne 24 (a + b)
  | F64 => rne 53 (a + b)
  end.

Definition cap (c : carrier) : Z :=
  match c with
  | I32 => MC.I32MAX
  | I64 => EP.I64MAX
  | F32 => 2 ^ 24
  | F64 => 2 ^ 53
  end.

Theorem every_carrier_is_exact_within_its_cap : forall c,
  exact_within (cap c) (cadd c).
Proof.
  intros c a b H. destruct c; cbn [cadd cap] in *.
  - unfold MC.wrap32, MC.I32MAX, MC.I32MIN in *.
    rewrite Z.mod_small by lia. lia.
  - unfold EP.wrap64, EP.I64MAX, EP.I64MIN in *.
    rewrite Z.mod_small by lia. lia.
  - apply rne_exact; lia.
  - apply rne_exact; lia.
Qed.

Lemma cap_pos : forall c, 0 < cap c.
Proof. destruct c; cbn; unfold MC.I32MAX, EP.I64MAX; lia. Qed.

(** A product formed in the carrier before it is added: exact for the same
    reason, and with a fused multiply-add it is not rounded at all. *)
Definition cmul (c : carrier) (fused : bool) (a b : Z) : Z :=
  if fused then a * b else cadd c (a * b) 0.

Lemma cmul_exact : forall c fused a b,
  Z.abs (a * b) <= cap c -> cmul c fused a b = a * b.
Proof.
  intros c fused a b H. unfold cmul. destruct fused; [reflexivity|].
  rewrite (every_carrier_is_exact_within_its_cap c) by (rewrite Z.add_0_r; exact H). lia.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The refutations: the cap is load-bearing                        *)
(* ------------------------------------------------------------------ *)

Theorem float32_addition_rounds_past_its_cap :
  cadd F32 (2 ^ 24) 1 = 2 ^ 24 /\ 2 ^ 24 + 1 <> 2 ^ 24.
Proof. split; [vm_compute; reflexivity | lia]. Qed.

Definition spike : list Z := [2 ^ 24; 1; 1].
Definition left_first : tree := Node (Node (Leaf 0) (Leaf 1)) (Leaf 2).
Definition right_first : tree := Node (Leaf 0) (Node (Leaf 1) (Leaf 2)).

(** The same three terms, one unit past the float32 cap, under two
    associations: two different answers, and neither order is wrong. *)
Theorem past_the_cap_two_schedules_disagree :
  covers left_first 3 /\ covers right_first 3
  /\ abs_sum spike = cap F32 + 2
  /\ teval (cadd F32) spike left_first = 2 ^ 24
  /\ teval (cadd F32) spike right_first = 2 ^ 24 + 2
  /\ zsum spike = 2 ^ 24 + 2.
Proof.
  repeat split; try (vm_compute; reflexivity); unfold covers; cbn; apply Permutation_refl.
Qed.

(** In int32 the same three terms are far inside the cap, and agree. *)
Theorem inside_the_cap_the_same_schedules_agree :
  teval (cadd I32) spike left_first = teval (cadd I32) spike right_first.
Proof. vm_compute. reflexivity. Qed.

(** One past the int32 cap the sum wraps to the other end of the range. *)
Theorem int32_addition_wraps_past_its_cap :
  cadd I32 MC.I32MAX 1 = MC.I32MIN.
Proof. vm_compute. reflexivity. Qed.

(* ------------------------------------------------------------------ *)
(** ** Dot products                                                    *)
(* ------------------------------------------------------------------ *)

Fixpoint zip_mul (a b : list Z) : list Z :=
  match a, b with
  | x :: a', y :: b' => x * y :: zip_mul a' b'
  | _, _ => []
  end.

Definition dot (a b : list Z) : Z := zsum (zip_mul a b).

Lemma zip_mul_length : forall a b,
  length a = length b -> length (zip_mul a b) = length a.
Proof.
  induction a as [|x a IH]; destruct b as [|y b]; cbn; intros H; try lia.
  rewrite IH by lia. reflexivity.
Qed.

Lemma zip_mul_nth : forall a b k,
  length a = length b -> nth k (zip_mul a b) 0 = nth k a 0 * nth k b 0.
Proof.
  induction a as [|x a IH]; destruct b as [|y b]; cbn; intros k H; try lia.
  - destruct k; reflexivity.
  - destruct k; [reflexivity|]. apply IH. lia.
Qed.

Lemma abs_sum_zip_mul : forall ma mb a b,
  0 <= ma -> 0 <= mb ->
  Forall (fun x => Z.abs x <= ma) a ->
  Forall (fun y => Z.abs y <= mb) b ->
  abs_sum (zip_mul a b) <= Z.of_nat (length a) * (ma * mb).
Proof.
  intros ma mb a b Hma Hmb Ha. revert b.
  induction Ha as [|x a Hx Ha IH]; intros b Hb; [cbn; lia|].
  destruct b as [|y b].
  - cbn [zip_mul]. unfold abs_sum. cbn [map zsum length].
    rewrite Nat2Z.inj_succ. nia.
  - inversion Hb as [|? ? Hy Hb']; subst.
    specialize (IH b Hb'). unfold abs_sum in *. cbn [zip_mul map zsum length].
    rewrite Nat2Z.inj_succ, Z.abs_mul. nia.
Qed.

(** The list form is the [nat -> Z] form the kernel proofs are stated in. *)
Lemma acc_range_ext : forall (f g : nat -> Z) lo n,
  (forall k, (lo <= k < lo + n)%nat -> f k = g k) ->
  D.acc_range Z.add f lo n = D.acc_range Z.add g lo n.
Proof.
  intros f g lo n. induction n as [|n IH]; intros H; [reflexivity|].
  cbn [D.acc_range]. rewrite IH by (intros k Hk; apply H; lia).
  rewrite H by lia. reflexivity.
Qed.

Lemma acc_range_is_zsum : forall l,
  D.acc_range Z.add (fun k => nth k l 0) 0 (length l) = zsum l.
Proof.
  induction l as [|x l IH] using rev_ind; [reflexivity|].
  rewrite length_app. cbn [length]. rewrite Nat.add_1_r. cbn [D.acc_range].
  rewrite (acc_range_ext _ (fun k => nth k l 0))
    by (intros k Hk; apply app_nth1; lia).
  rewrite IH, zsum_app. cbn [zsum Nat.add].
  rewrite app_nth2 by lia. rewrite Nat.sub_diag. cbn. ring.
Qed.

Theorem dot_is_the_kernel_proofs_sum : forall a b,
  length a = length b ->
  dot a b = D.acc_range Z.add (fun k => nth k a 0 * nth k b 0) 0 (length a).
Proof.
  intros a b H. unfold dot.
  rewrite <- (acc_range_is_zsum (zip_mul a b)), (zip_mul_length a b H).
  apply acc_range_ext. intros k _. apply zip_mul_nth, H.
Qed.

(** A dot product of bounded operands, by any schedule in any carrier whose
    cap admits [length * ma * mb]. *)
Theorem any_schedule_gives_the_dot_product : forall c ma mb a b t,
  0 <= ma -> 0 <= mb ->
  Forall (fun x => Z.abs x <= ma) a ->
  Forall (fun y => Z.abs y <= mb) b ->
  length a = length b ->
  Z.of_nat (length a) * (ma * mb) <= cap c ->
  covers t (length a) ->
  teval (cadd c) (zip_mul a b) t = dot a b.
Proof.
  intros c ma mb a b t Hma Hmb Ha Hb Hl Hcap Hc.
  apply (any_schedule_gives_the_sum (cap c)).
  - apply every_carrier_is_exact_within_its_cap.
  - rewrite zip_mul_length by assumption. exact Hc.
  - pose proof (abs_sum_zip_mul ma mb a b Hma Hmb Ha Hb). lia.
Qed.

(* ------------------------------------------------------------------ *)
(** ** Max and argmax                                                  *)
(* ------------------------------------------------------------------ *)

Section Max.
  Variable X : Type.
  Variable key : X -> Z.

  (** What a max reduction returns: one of the inputs, with no input above
      it. Ties may go either way. *)
  Definition is_max_of (l : list X) (r : X) : Prop :=
    In r l /\ forall y, In y l -> key y <= key r.

  Theorem max_key_unique : forall l r1 r2,
    is_max_of l r1 -> is_max_of l r2 -> key r1 = key r2.
  Proof.
    intros l r1 r2 [I1 M1] [I2 M2].
    pose proof (M1 r2 I2). pose proof (M2 r1 I1). lia.
  Qed.

  (** A binary max that returns one of its arguments, with the larger key. *)
  Variable mx : X -> X -> X.
  Hypothesis mx_picks : forall a b,
    (mx a b = a \/ mx a b = b) /\ key (mx a b) = Z.max (key a) (key b).

  Fixpoint nozero (t : tree) : Prop :=
    match t with
    | Leaf _ => True
    | Zero => False
    | Node l r => nozero l /\ nozero r
    end.

  Fixpoint xeval (d : X) (xs : list X) (t : tree) : X :=
    match t with
    | Leaf i => nth i xs d
    | Zero => d
    | Node l r => mx (xeval d xs l) (xeval d xs r)
    end.

  Lemma xeval_max : forall d xs t,
    nozero t ->
    (forall i, In i (leaves t) -> (i < length xs)%nat) ->
    In (xeval d xs t) (map (fun i => nth i xs d) (leaves t))
    /\ forall i, In i (leaves t) -> key (nth i xs d) <= key (xeval d xs t).
  Proof.
    intros d xs t. induction t as [i| |l IHl r IHr]; intros Hz Hin.
    - cbn. split; [left; reflexivity|]. intros j [E|[]]. subst. lia.
    - destruct Hz.
    - destruct Hz as [Hzl Hzr]. cbn [leaves xeval] in *.
      assert (Hl : forall i, In i (leaves l) -> (i < length xs)%nat)
        by (intros i Hi; apply Hin, in_or_app; left; exact Hi).
      assert (Hr : forall i, In i (leaves r) -> (i < length xs)%nat)
        by (intros i Hi; apply Hin, in_or_app; right; exact Hi).
      destruct (IHl Hzl Hl) as [Il Ml]. destruct (IHr Hzr Hr) as [Ir Mr].
      destruct (mx_picks (xeval d xs l) (xeval d xs r)) as [Hpick Hkey].
      split.
      + rewrite map_app. apply in_or_app.
        destruct Hpick as [E|E]; rewrite E; [left|right]; assumption.
      + intros i Hi. rewrite Hkey. apply in_app_or in Hi. destruct Hi as [Hi|Hi].
        * specialize (Ml i Hi). lia.
        * specialize (Mr i Hi). lia.
  Qed.

  (** A comparison-based max, associated and ordered any way, meets the
      specification. *)
  Theorem any_max_tree_is_a_max : forall d xs t,
    nozero t -> covers t (length xs) -> is_max_of xs (xeval d xs t).
  Proof.
    intros d xs t Hz Hc.
    assert (Hin : forall i, In i (leaves t) -> (i < length xs)%nat).
    { intros i Hi. apply (Permutation_in _ Hc) in Hi. apply in_seq in Hi. lia. }
    destruct (xeval_max d xs t Hz Hin) as [I M]. split.
    - apply in_map_iff in I. destruct I as [i [E Hi]]. rewrite <- E.
      apply nth_In, Hin, Hi.
    - intros y Hy. destruct (In_nth xs y d Hy) as [i [Hi E]]. rewrite <- E.
      apply M. apply (Permutation_in _ (Permutation_sym Hc)), in_seq. lia.
  Qed.

  (** torch's rule for argmax: the first index of a maximal value. *)
  Definition is_first_argmax (d : X) (l : list X) (i : nat) : Prop :=
    (i < length l)%nat
    /\ (forall j, (j < length l)%nat -> key (nth j l d) <= key (nth i l d))
    /\ (forall j, (j < i)%nat -> key (nth j l d) < key (nth i l d)).

  Theorem first_argmax_unique : forall d l i1 i2,
    is_first_argmax d l i1 -> is_first_argmax d l i2 -> i1 = i2.
  Proof.
    intros d l i1 i2 [L1 [M1 F1]] [L2 [M2 F2]].
    destruct (Nat.lt_trichotomy i1 i2) as [H|[H|H]]; [|exact H|].
    - pose proof (F2 i1 H). pose proof (M1 i2 L2). lia.
    - pose proof (F1 i2 H). pose proof (M2 i1 L1). lia.
  Qed.

  (** A left-to-right scan that replaces its candidate only on a strictly
      larger value is one implementation, so the rule is satisfiable. *)
  Fixpoint scan (d : X) (l : list X) (base best : nat) (bv : X) : nat :=
    match l with
    | [] => best
    | x :: r => if key bv <? key x then scan d r (S base) base x
                else scan d r (S base) best bv
    end.

  Definition argmax_scan (d : X) (l : list X) : nat :=
    match l with
    | [] => O
    | x :: r => scan d r 1 0 x
    end.

  Lemma scan_spec : forall d r pre best bv,
    (best < length pre)%nat -> nth best pre d = bv ->
    (forall j, (j < length pre)%nat -> key (nth j pre d) <= key bv) ->
    (forall j, (j < best)%nat -> key (nth j pre d) < key bv) ->
    is_first_argmax d (pre ++ r) (scan d r (length pre) best bv).
  Proof.
    intros d r. induction r as [|x r IH]; intros pre best bv Hb Hbv Hmax Hfirst.
    - cbn [scan]. rewrite app_nil_r. subst bv. repeat split; assumption.
    - cbn [scan].
      assert (Eapp : pre ++ x :: r = (pre ++ [x]) ++ r)
        by (rewrite <- app_assoc; reflexivity).
      assert (Elen : S (length pre) = length (pre ++ [x]))
        by (rewrite length_app; cbn; lia).
      destruct (Z.ltb_spec (key bv) (key x)) as [Hlt|Hge];
        rewrite Eapp, Elen; apply IH.
      + rewrite length_app. cbn. lia.
      + rewrite app_nth2 by lia. rewrite Nat.sub_diag. reflexivity.
      + intros j Hj. rewrite length_app in Hj. cbn in Hj.
        destruct (Nat.eq_dec j (length pre)) as [E|N].
        * subst j. rewrite app_nth2 by lia. rewrite Nat.sub_diag. cbn. lia.
        * rewrite app_nth1 by lia. specialize (Hmax j ltac:(lia)). lia.
      + intros j Hj. rewrite app_nth1 by lia. specialize (Hmax j ltac:(lia)). lia.
      + rewrite length_app. cbn. lia.
      + rewrite app_nth1 by lia. exact Hbv.
      + intros j Hj. rewrite length_app in Hj. cbn in Hj.
        destruct (Nat.eq_dec j (length pre)) as [E|N].
        * subst j. rewrite app_nth2 by lia. rewrite Nat.sub_diag. cbn. lia.
        * rewrite app_nth1 by lia. apply Hmax. lia.
      + intros j Hj. rewrite app_nth1 by lia. apply Hfirst, Hj.
  Qed.

  Theorem the_scan_is_a_first_argmax : forall d l,
    l <> [] -> is_first_argmax d l (argmax_scan d l).
  Proof.
    intros d [|x r] H; [contradiction|]. cbn [argmax_scan].
    apply (scan_spec d r [x] 0%nat x); cbn; try lia; try reflexivity.
    - intros j Hj. assert (j = 0)%nat by lia. subst. cbn. lia.
  Qed.
End Max.

(** A max of integers: here the key is the value, so the result is unique. *)
Theorem integer_max_is_unique : forall l r1 r2,
  is_max_of Z (fun z => z) l r1 -> is_max_of Z (fun z => z) l r2 -> r1 = r2.
Proof. intros l r1 r2 H1 H2. exact (max_key_unique Z (fun z => z) l r1 r2 H1 H2). Qed.


(* ------------------------------------------------------------------ *)
(** ** Scaled schedules: a partial sum multiplied mid-tree             *)
(* ------------------------------------------------------------------ *)

(** Triton's two-digit GEMM (`_w8a8_gemm_hilo`) accumulates, per K block,
    `acc += dot(hi) * 128; acc += dot(lo)` in one int32: a partial sum is
    multiplied by 128 and then added. A [stree] adds that node. Its leaves
    carry the product of the scales above them ([sleaves]); covering means
    each term index appears once, with the multiplier [mu] gives it. *)

Inductive stree : Type :=
| SLeaf (i : nat)
| SZero
| SNode (l r : stree)
| SScale (k : Z) (t : stree).

Fixpoint sleaves (t : stree) : list (nat * Z) :=
  match t with
  | SLeaf i => [(i, 1)]
  | SZero => []
  | SNode l r => sleaves l ++ sleaves r
  | SScale k t => map (fun p => (fst p, k * snd p)) (sleaves t)
  end.

Definition scovers (t : stree) (mu : nat -> Z) (n : nat) : Prop :=
  Permutation (sleaves t) (map (fun i => (i, mu i)) (seq 0 n)).

(** The machine: additions in [add], a scale in [mul]. *)
Fixpoint steval (add mul : Z -> Z -> Z) (terms : list Z) (t : stree) : Z :=
  match t with
  | SLeaf i => nth i terms 0
  | SZero => 0
  | SNode l r => add (steval add mul terms l) (steval add mul terms r)
  | SScale k t => mul k (steval add mul terms t)
  end.

Definition exact_mul_within (B : Z) (mul : Z -> Z -> Z) : Prop :=
  forall k v, Z.abs (k * v) <= B -> mul k v = k * v.

Definition wterms (terms : list Z) (ps : list (nat * Z)) : list Z :=
  map (fun p => snd p * nth (fst p) terms 0) ps.

Lemma wterms_app : forall terms a b, wterms terms (a ++ b) = wterms terms a ++ wterms terms b.
Proof. intros. unfold wterms. apply map_app. Qed.

Lemma zsum_map_scale : forall (ps : list (nat * Z)) (terms : list Z) k,
  zsum (wterms terms (map (fun p => (fst p, k * snd p)) ps)) = k * zsum (wterms terms ps).
Proof.
  intros ps terms k. unfold wterms. rewrite map_map. cbn [fst snd].
  induction ps as [|p ps IH]; cbn [map zsum]; [ring|]. rewrite IH. ring.
Qed.

Lemma abs_sum_map_scale : forall (ps : list (nat * Z)) (terms : list Z) k,
  abs_sum (wterms terms (map (fun p => (fst p, k * snd p)) ps)) = Z.abs k * abs_sum (wterms terms ps).
Proof.
  intros ps terms k. unfold abs_sum, wterms. rewrite !map_map. cbn [fst snd].
  induction ps as [|p ps IH]; cbn [map zsum]; [ring|]. rewrite IH, !Z.abs_mul. ring.
Qed.

Lemma steval_sleaves : forall B add mul terms t,
  exact_within B add -> exact_mul_within B mul ->
  abs_sum (wterms terms (sleaves t)) <= B ->
  steval add mul terms t = zsum (wterms terms (sleaves t)).
Proof.
  intros B add mul terms t Hadd Hmul. induction t as [i| |l IHl r IHr|k t IH]; intros Hb.
  - unfold wterms. cbn [sleaves steval map zsum fst snd]. ring.
  - reflexivity.
  - cbn [sleaves steval] in *. rewrite wterms_app, abs_sum_app in Hb.
    pose proof (abs_sum_nonneg (wterms terms (sleaves l))).
    pose proof (abs_sum_nonneg (wterms terms (sleaves r))).
    rewrite IHl, IHr by lia. rewrite wterms_app, zsum_app. apply Hadd.
    pose proof (zsum_le_abs_sum (wterms terms (sleaves l))).
    pose proof (zsum_le_abs_sum (wterms terms (sleaves r))). lia.
  - cbn [sleaves steval] in *. rewrite zsum_map_scale. rewrite abs_sum_map_scale in Hb.
    pose proof (abs_sum_nonneg (wterms terms (sleaves t))) as H0.
    destruct (Z.eq_dec k 0) as [Hk|Hk].
    + (* a zero scale: the product is 0 whatever the inner value *)
      subst k. rewrite Hmul by (rewrite Z.mul_0_l; cbn; nia). ring.
    + assert (Hin : abs_sum (wterms terms (sleaves t)) <= B) by nia.
      rewrite (IH Hin). apply Hmul. rewrite Z.abs_mul.
      pose proof (zsum_le_abs_sum (wterms terms (sleaves t))).
      pose proof (Z.abs_nonneg k). nia.
Qed.

Definition weighted (mu : nat -> Z) (terms : list Z) : list Z :=
  map (fun i => mu i * nth i terms 0) (seq 0 (length terms)).

Lemma sleaves_weighted : forall t mu terms,
  scovers t mu (length terms) -> Permutation (wterms terms (sleaves t)) (weighted mu terms).
Proof.
  intros t mu terms H. unfold wterms, weighted.
  pose proof (Permutation_map (fun p => snd p * nth (fst p) terms 0) H) as P.
  rewrite map_map in P. exact P.
Qed.

(** Every scaled schedule in a carrier whose additions and scalings are exact
    within B gives the weighted sum, while the absolute values of the
    weighted terms sum to at most B. *)
Theorem any_scaled_schedule_gives_the_weighted_sum : forall B add mul mu terms t,
  exact_within B add -> exact_mul_within B mul ->
  scovers t mu (length terms) ->
  abs_sum (weighted mu terms) <= B ->
  steval add mul terms t = zsum (weighted mu terms).
Proof.
  intros B add mul mu terms t Hadd Hmul Hc Hb.
  pose proof (sleaves_weighted t mu terms Hc) as P.
  rewrite (steval_sleaves B add mul terms t Hadd Hmul) by (rewrite (abs_sum_perm _ _ P); exact Hb).
  apply zsum_perm, P.
Qed.

(** A scaling in a carrier: the product formed, then rounded or wrapped. *)
Definition cscale (c : carrier) (k v : Z) : Z := cadd c (k * v) 0.

Theorem every_carrier_scales_exactly_within_its_cap : forall c,
  exact_mul_within (cap c) (cscale c).
Proof.
  intros c k v H. unfold cscale.
  rewrite (every_carrier_is_exact_within_its_cap c) by (rewrite Z.add_0_r; exact H). ring.
Qed.

(** The kernel's shape at two terms, `acc = dot(hi) * 128 + dot(lo)`: the
    premise is satisfiable, and in int32 it gives the weighted sum. *)
Example a_scaled_schedule :
  scovers (SNode (SScale 128 (SLeaf 0)) (SLeaf 1)) (fun i => if (i =? 0)%nat then 128 else 1) 2
  /\ steval (cadd I32) (cscale I32) [3; 5] (SNode (SScale 128 (SLeaf 0)) (SLeaf 1)) = 128 * 3 + 5.
Proof. split; [unfold scovers; cbn; apply Permutation_refl | vm_compute; reflexivity]. Qed.

Print Assumptions any_schedule_gives_the_sum.
Print Assumptions any_two_schedules_agree.
Print Assumptions no_partial_leaves_the_range.
Print Assumptions every_carrier_is_exact_within_its_cap.
Print Assumptions float32_addition_rounds_past_its_cap.
Print Assumptions past_the_cap_two_schedules_disagree.
Print Assumptions inside_the_cap_the_same_schedules_agree.
Print Assumptions int32_addition_wraps_past_its_cap.
Print Assumptions dot_is_the_kernel_proofs_sum.
Print Assumptions any_schedule_gives_the_dot_product.
Print Assumptions max_key_unique.
Print Assumptions any_max_tree_is_a_max.
Print Assumptions first_argmax_unique.
Print Assumptions the_scan_is_a_first_argmax.
Print Assumptions integer_max_is_unique.
Print Assumptions any_scaled_schedule_gives_the_weighted_sum.
Print Assumptions every_carrier_scales_exactly_within_its_cap.
Print Assumptions a_scaled_schedule.
