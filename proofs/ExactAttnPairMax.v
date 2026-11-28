(** * The attention kernel's row maximum of double floats
      (`python/y_lang/exact/attn_weights_kernel.py`).

    `_attn_p_kernel` holds each logit as a pair (hi, lo) from
    `_fast_two_sum`, and forms the row maximum without adding the parts:
    m_hi = max hi, then m_lo = max lo over the keys whose hi is m_hi.
    `ExactAttnWeights.v`'s [the_maximum_moves_by_at_most_the_error] assumes
    the computed maximum mf is the largest of the pairs' VALUES hi + lo
    (Lf <= mf for every key, mf one of them). This file proves the kernel's
    pair is that value, from what `_fast_two_sum` gives: hi is the rounding
    of hi + lo, for any monotone rounding.

    [a_smaller_hi_has_a_value_no_larger]: hi_j < hi_i forces
    hi_j + lo_j <= hi_i + lo_i (a larger value would round to at least hi_i).
    [the_pair_maximum_is_the_maximum_value]: m_hi + m_lo is at least every
    key's value and is one key's value, the two facts the decision proof
    takes.

    ** Refutations.

    [without_the_rounding_the_pair_maximum_misses]: pairs (1, 0) and (0, 5)
    give m_hi + m_lo = 1 below the value 5; no monotone rounding takes 1 to 1
    and 5 to 0, so the hypothesis is what rules this out.
    [the_low_parts_need_their_own_maximum]: the lo of the first key with the
    largest hi, instead of the largest such lo, misses by 2^-30 under a
    monotone rounding (two values that round alike).

    ** Not proved here.

    That `_fast_two_sum` is exact (hi + lo is q1 + f1, with |f1| far below
    |q1|) and that float32 addition is a monotone rounding: the IEEE facts
    `ExactAttnWeights.v` also takes; that `tl.max` returns a largest element.

    Checked with Rocq 9.1:  coqc -Q . "" ExactAttnPairMax.v  *)

From Stdlib Require Import QArith Qround Lqa List.
Import ListNotations.
Open Scope Q_scope.

Section PairMax.
  Variable rn : Q -> Q.
  Hypothesis rn_mono : forall x y, x <= y -> rn x <= rn y.

  (** The row's unblocked keys, as pairs (hi, lo). *)
  Variable ps : list (Q * Q).
  Hypothesis Hrn : forall h l, In (h, l) ps -> rn (h + l) == h.

  Lemma a_smaller_hi_has_a_value_no_larger : forall hi li hj lj,
    In (hi, li) ps -> In (hj, lj) ps -> hj < hi -> hj + lj <= hi + li.
  Proof.
    intros hi li hj lj Hi Hj Hlt.
    destruct (Qlt_le_dec (hi + li) (hj + lj)) as [Hgt|Hle]; [|exact Hle].
    exfalso. pose proof (rn_mono _ _ (Qlt_le_weak _ _ Hgt)) as Hm.
    rewrite (Hrn _ _ Hi), (Hrn _ _ Hj) in Hm. lra.
  Qed.

  (** m_hi: a largest hi; m_lo: a largest lo among the keys with that hi. *)
  Variables m_hi m_lo : Q.
  Hypothesis HM : forall h l, In (h, l) ps -> h <= m_hi.
  Hypothesis Hm : forall h l, In (h, l) ps -> h == m_hi -> l <= m_lo.
  Hypothesis Hmi : exists h l, In (h, l) ps /\ h == m_hi /\ l == m_lo.

  Theorem the_pair_maximum_is_the_maximum_value :
    (forall h l, In (h, l) ps -> h + l <= m_hi + m_lo) /\
    (exists h l, In (h, l) ps /\ h + l == m_hi + m_lo).
  Proof.
    destruct Hmi as [h0 [l0 [H0 [Eh El]]]]. split.
    - intros h l Hin. destruct (Qlt_le_dec h m_hi) as [Hlt|Hge].
      + assert (Hlt0 : h < h0) by lra.
        pose proof (a_smaller_hi_has_a_value_no_larger h0 l0 h l H0 Hin Hlt0). lra.
      + assert (Heq : h == m_hi) by (pose proof (HM h l Hin); lra).
        pose proof (Hm h l Hin Heq). lra.
    - exists h0, l0. split; [exact H0|lra].
  Qed.
End PairMax.

(** The refutations. *)

Theorem without_the_rounding_the_pair_maximum_misses :
  let ps := [(1, 0); (0, 5)] in
  (forall h l, In (h, l) ps -> h <= 1) /\
  (forall h l, In (h, l) ps -> h == 1 -> l <= 0) /\
  In (0, 5) ps /\ 1 + 0 < 0 + 5 /\
  ~ (exists rn : Q -> Q, (forall x y, x <= y -> rn x <= rn y) /\
       rn (1 + 0) == 1 /\ rn (0 + 5) == 0).
Proof.
  cbv zeta. split; [|split; [|split; [|split]]].
  - intros h l [E|[E|[]]]; injection E as <- <-; lra.
  - intros h l [E|[E|[]]] Hh; injection E as <- <-; lra.
  - right. left. reflexivity.
  - lra.
  - intros [rn [Hmono [H1 H5]]].
    assert (Hle : 1 + 0 <= 0 + 5) by lra.
    pose proof (Hmono _ _ Hle). lra.
Qed.

Theorem the_low_parts_need_their_own_maximum :
  let rn := fun _ : Q => 1 in
  let ps := [(1, 0); (1, 1 # 1073741824)] in
  (forall x y, x <= y -> rn x <= rn y) /\
  (forall h l, In (h, l) ps -> rn (h + l) == h) /\
  1 + 0 < 1 + (1 # 1073741824).
Proof.
  cbv zeta. split; [|split].
  - intros. lra.
  - intros h l [E|[E|[]]]; injection E as <- <-; reflexivity.
  - lra.
Qed.

Print Assumptions a_smaller_hi_has_a_value_no_larger.
Print Assumptions the_pair_maximum_is_the_maximum_value.
Print Assumptions without_the_rounding_the_pair_maximum_misses.
Print Assumptions the_low_parts_need_their_own_maximum.
