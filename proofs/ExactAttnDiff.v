(** * The attention kernel's difference M - L in float32
      (`python/y_lang/exact/attn_weights_kernel.py`).

    From the row maximum (m_hi, m_lo) and a logit (l_hi, l_lo), each a
    double float32 with |hi| < 2^24 (larger logits take the float64 path)
    and |lo| <= u |hi|, `_attn_p_kernel` forms

      d_hi, d_e = _two_sum(m_hi, -l_hi)       d_hi = RN(m_hi - l_hi), d_e its error
      d_lo = d_e + (m_lo - l_lo)               two float32 roundings
      k1 = rint(d_hi), k2 = rint(d_lo)          nearest integers
      r = (d_hi - k1) + (d_lo - k2)             one rounding (each part exact)

    and decides with K = k1 + k2 and r. `ExactAttnWeights.v`'s
    [the_kernel_integer_is_the_rounding] takes |K + r - (mf - Lf)| <= F and
    -1 <= r <= 1, with F = 2^-20 in [the_budget_fits_the_guard]; its "Not
    proved here" left F to the kernel. This file proves both.

    [the_lo_parts_are_at_most_1]: |lo| <= u |hi| < 1 for |hi| < 2^24.
    [the_float32_difference_is_within_2_21]: with RN any rounding that is
    monotone, keeps 1 and -1, and has relative error at most u = 2^-24,
    |K + r - ((m_hi + m_lo) - (l_hi + l_lo))| <= 7u + 2u^2 < 2^-21 and
    -1 <= r <= 1. [the_budget_value_holds]: so F = 2^-20 holds twice over.

    ** Refutation.

    [a_plain_subtraction_would_lose_1]: without _two_sum's error term the
    difference of two logits below 2^24 can be off by 1 (3 * 2^23 + 1 is
    odd, and every float32 between 2^24 and 2^25 is even), 65,536 times the
    guard.

    ** Not proved here.

    That `_two_sum` returns RN(a + b) and its exact error (Knuth's TwoSum,
    classical, for float32 without FMA contraction), that `_rint32` is a
    nearest integer (the 2^23 trick: `ExactSoftmaxKernel.v`; past 2^23 a
    float32 is an integer: `ExactRintRemainder.v`), that a float32
    subtraction with a float32 result is exact (x - rint(x) is a float32:
    `ExactRintRemainder.v`, so the kernel's d_hi - k1 and d_lo - k2 are
    exact), and that float32 addition is such a rounding (IEEE round to
    nearest, no overflow below 2^26).

    Checked with Rocq 9.1:  coqc -Q . "" ExactAttnDiff.v  *)

From Stdlib Require Import ZArith Lia QArith Qabs Lqa.
Open Scope Q_scope.

Definition u : Q := 1 # 16777216.          (* 2^-24 *)

Lemma round_bound : forall X d B, - u <= d <= u -> - B <= X <= B -> - (u * B) <= X * d <= u * B.
Proof. intros X d B Hd HX. unfold u in *. nra. Qed.

Theorem the_lo_parts_are_at_most_1 : forall hi lo,
  Qabs lo <= u * Qabs hi -> Qabs hi < 16777216 -> Qabs lo <= 1.
Proof.
  intros hi lo Hl Hh. unfold u in Hl.
  assert (Qabs hi * (1 # 16777216) <= 1) by (pose proof (Qabs_nonneg hi); nra). lra.
Qed.

Section Difference.
  Variable rn : Q -> Q.
  Hypothesis rn_mono : forall x y, x <= y -> rn x <= rn y.
  Hypothesis rn_one : rn 1 == 1.
  Hypothesis rn_mone : rn (-1) == -1.
  Hypothesis rn_err : forall y, exists d, - u <= d <= u /\ rn y == y * (1 + d).

  Variables m_hi m_lo l_hi l_lo : Q.
  Hypothesis Hmhi : Qabs m_hi < 16777216.
  Hypothesis Hlhi : Qabs l_hi < 16777216.
  Hypothesis Hmlo : Qabs m_lo <= 1.
  Hypothesis Hllo : Qabs l_lo <= 1.

  (** `_two_sum`: d_hi the rounding, d_e the exact error. *)
  Definition d_hi := rn (m_hi - l_hi).
  Definition d_e := (m_hi - l_hi) - d_hi.
  Definition w := rn (m_lo - l_lo).
  Definition d_lo := rn (d_e + w).

  (** k1, k2 nearest integers; the two parts exact; one rounding. *)
  Variables k1 k2 : Z.
  Hypothesis Hk1 : Qabs (d_hi - inject_Z k1) <= 1 # 2.
  Hypothesis Hk2 : Qabs (d_lo - inject_Z k2) <= 1 # 2.
  Definition x := (d_hi - inject_Z k1) + (d_lo - inject_Z k2).
  Definition r := rn x.

  Lemma d_e_small : - 2 <= d_e <= 2.
  Proof.
    unfold d_e, d_hi. destruct (rn_err (m_hi - l_hi)) as [d [Hd E]]. rewrite E.
    apply Qabs_Qlt_condition in Hmhi. apply Qabs_Qlt_condition in Hlhi.
    assert (HB : - 33554432 <= m_hi - l_hi <= 33554432) by lra.
    pose proof (round_bound (m_hi - l_hi) d 33554432 Hd HB). unfold u in *. lra.
  Qed.

  Lemma w_close : exists d1, - u <= d1 <= u /\ w == (m_lo - l_lo) * (1 + d1).
  Proof. apply rn_err. Qed.

  Lemma the_identity : forall d1 d2 d3,
    w == (m_lo - l_lo) * (1 + d1) -> d_lo == (d_e + w) * (1 + d2) -> r == x * (1 + d3) ->
    inject_Z (k1 + k2) + r - ((m_hi + m_lo) - (l_hi + l_lo)) ==
    (m_lo - l_lo) * d1 + (d_e + w) * d2 + x * d3.
  Proof.
    intros d1 d2 d3 Ew Elo Er. rewrite Er. unfold x. rewrite Elo.
    rewrite inject_Z_plus. unfold d_e. rewrite Ew. unfold d_e. ring.
  Qed.

  Lemma the_bound : forall a b c d1 d2 d3,
    - (u * 2) <= a * d1 <= u * 2 -> - (u * (4 + 2 * u)) <= b * d2 <= u * (4 + 2 * u) ->
    - (u * 1) <= c * d3 <= u * 1 ->
    Qabs (a * d1 + b * d2 + c * d3) <= 7 * u + 2 * (u * u).
  Proof. intros. apply Qabs_Qle_condition. unfold u in *. split; lra. Qed.

  Theorem the_float32_difference_is_within_2_21 :
    Qabs (inject_Z (k1 + k2) + r - ((m_hi + m_lo) - (l_hi + l_lo))) <= 7 * u + 2 * (u * u) /\
    -1 <= r <= 1.
  Proof.
    pose proof d_e_small as Hde.
    destruct w_close as [d1 [Hd1 Ew]].
    destruct (rn_err (d_e + w)) as [d2 [Hd2 Elo]].
    destruct (rn_err x) as [d3 [Hd3 Er]].
    assert (HB1 : - 2 <= m_lo - l_lo <= 2).
    { clear -Hmlo Hllo. apply Qabs_Qle_condition in Hmlo. apply Qabs_Qle_condition in Hllo. lra. }
    assert (Hx : -1 <= x <= 1).
    { clear -Hk1 Hk2. unfold x. apply Qabs_Qle_condition in Hk1. apply Qabs_Qle_condition in Hk2. lra. }
    pose proof (round_bound (m_lo - l_lo) d1 2 Hd1 HB1) as R1.
    assert (Hw : - (2 + 2 * u) <= w <= 2 + 2 * u).
    { clear -Ew R1 HB1. rewrite Ew. unfold u in *. lra. }
    assert (HB2 : - (4 + 2 * u) <= d_e + w <= 4 + 2 * u) by (clear -Hde Hw; lra).
    pose proof (round_bound (d_e + w) d2 (4 + 2 * u) Hd2 HB2) as R2.
    pose proof (round_bound x d3 1 Hd3 Hx) as R3.
    split.
    - rewrite (the_identity d1 d2 d3 Ew Elo Er). apply the_bound; assumption.
    - unfold r. split.
      + rewrite <- rn_mone. apply rn_mono. clear -Hx. lra.
      + rewrite <- rn_one. apply rn_mono. clear -Hx. lra.
  Qed.
End Difference.

Theorem the_budget_value_holds : 7 * u + 2 * (u * u) < 1 # 2097152 /\ (1 # 2097152) < 1 # 1048576.
Proof. unfold u. split; reflexivity. Qed.

(** The refutation: a plain float32 subtraction of the hi parts, without
    _two_sum's error term. The logits 2^24 - 1 and -(2^23 + 2) are float32s
    below 2^24; their difference 3 * 2^23 + 1 is odd, and every float32 in
    [2^24, 2^25) is a multiple of 2, so the rounded difference is off by at
    least 1, which is 2^16 times the guard 2^-16. *)
Theorem a_plain_subtraction_would_lose_1 :
  (2 ^ 24 - 1) - (- (2 ^ 23 + 2)) = 3 * 2 ^ 23 + 1 /\
  (forall n : Z, 1 <= Z.abs (2 * n - (3 * 2 ^ 23 + 1)))%Z /\
  (1 # 65536) * 65536 == 1.
Proof.
  split; [reflexivity|split; [|reflexivity]].
  intro n. lia.
Qed.

Print Assumptions the_lo_parts_are_at_most_1.
Print Assumptions the_float32_difference_is_within_2_21.
Print Assumptions the_budget_value_holds.
Print Assumptions a_plain_subtraction_would_lose_1.
