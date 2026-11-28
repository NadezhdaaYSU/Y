(** * The fused draw's int64 sums (`python/y_lang/exact/vocab_draw.py`).

    `_draw_kernel` sums the softmax weights p of each block with `tl.sum`
    and their running sums with `tl.cumsum`, in int64, in whatever order
    Triton's reductions take. `ExactDrawSearch.v` proves the search over
    those sums and leaves their exactness to "no overflow: p <= 2^28 and the
    vocabulary below 2^34 entries". This file proves that step, with
    `ExactReduce.v`'s int64 carrier (two's complement wrap).

    [every_order_of_the_total_is_exact]: for weights in 0 .. 2^28 and at most
    2^34 of them, every schedule of their int64 sum gives the integer sum.
    [every_running_sum_is_exact]: so does every schedule of every prefix
    (each element of a cumsum is the sum of a prefix).

    ** Refutation.

    [past_2_35_weights_the_sum_wraps]: 2^35 weights of 2^28 reach 2^63, and
    int64's last addition wraps to -2^63.

    ** Not proved here.

    That Triton's `tl.sum` and `tl.cumsum` are such schedules of int64
    additions.

    Checked with Rocq 9.1:  coqc -Q . "" ExactDrawSums.v  *)

From Stdlib Require Import ZArith Lia List.
Import ListNotations.
Require ExactReduce ExactPvExact.
Open Scope Z_scope.

Module ER := ExactReduce.
Module EP := ExactPvExact.

Definition weight (p : Z) : Prop := 0 <= p <= 2 ^ 28.

Lemma abs_sum_weights : forall ps, Forall weight ps ->
  ER.abs_sum ps <= Z.of_nat (length ps) * 2 ^ 28.
Proof.
  induction ps as [|p ps IH]; intro F; [cbn; lia|].
  inversion F as [|? ? Hp Hps]; subst. unfold ER.abs_sum in *. cbn [map ER.zsum length].
  specialize (IH Hps). unfold weight in Hp. rewrite Nat2Z.inj_succ. lia.
Qed.

Theorem every_order_of_the_total_is_exact : forall ps t,
  Forall weight ps -> Z.of_nat (length ps) <= 2 ^ 34 -> ER.covers t (length ps) ->
  ER.teval (ER.cadd ER.I64) ps t = ER.zsum ps.
Proof.
  intros ps t F L C. apply (ER.any_schedule_gives_the_sum EP.I64MAX).
  - apply (ER.every_carrier_is_exact_within_its_cap ER.I64).
  - exact C.
  - pose proof (abs_sum_weights ps F). unfold EP.I64MAX.
    assert (Z.of_nat (length ps) * 2 ^ 28 <= 2 ^ 34 * 2 ^ 28) by nia. lia.
Qed.

Theorem every_running_sum_is_exact : forall ps n t,
  Forall weight ps -> Z.of_nat (length ps) <= 2 ^ 34 -> (n <= length ps)%nat ->
  ER.covers t (length (firstn n ps)) ->
  ER.teval (ER.cadd ER.I64) (firstn n ps) t = ER.zsum (firstn n ps).
Proof.
  intros ps n t F L Hn C. apply every_order_of_the_total_is_exact; [| |exact C].
  - rewrite <- (firstn_skipn n ps) in F. apply Forall_app in F. exact (proj1 F).
  - rewrite length_firstn. lia.
Qed.

Theorem past_2_35_weights_the_sum_wraps :
  2 ^ 35 * 2 ^ 28 = 2 ^ 63 /\
  ER.cadd ER.I64 ((2 ^ 35 - 1) * 2 ^ 28) (2 ^ 28) = - 2 ^ 63.
Proof. vm_compute. split; reflexivity. Qed.

Print Assumptions every_order_of_the_total_is_exact.
Print Assumptions every_running_sum_is_exact.
Print Assumptions past_2_35_weights_the_sum_wraps.
