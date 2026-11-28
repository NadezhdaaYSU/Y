(** * The attention softmax's integers without float64 (`python/y_lang/exact/attn_weights_kernel.py`).

    `attention.exact_softmax_weights` forms L = RN64(RN64(s * a) * sk) per
    score, the row maximum M of the unblocked L, and tq = round(RN64(M - L))
    in float64. The kernel forms each L as a pair of float32s hi + lo and the
    maximum of those pairs, writes the difference as K + r (K an integer,
    r in [-1, 1]), and takes K + step(r) unless r is within the guard band G
    of a half-integer, a logit is 2^24 or more, or K puts the key past
    30 * 2^16 (weight 0 either way); a row block with an undetermined element
    recomputes in float64. This file proves the two facts the shortcut rests
    on, over the rationals.

    ** The double float32 logit.

    With u = 2^-24 (float32's unit roundoff), each float32 rounding the
    standard model's x (1 + d), |d| <= u, and the two classical error-free
    steps as hypotheses (Dekker's product: p + e = a b exactly with
    |e| <= u |p|; Fast2Sum: s + t = a + b exactly with |t| <= u |s|):
    [the_double_float_logit_is_within_5e_14]: |hi + lo - s a sk| <=
    5e-14 s a sk for s, a, sk > 0 (about 10 u^2; the negative case is the
    mirror image: negating s negates every step exactly).

    ** The decision.

    [the_maximum_moves_by_at_most_the_error]: if each approximated logit is
    within E of its float64 value, the approximated maximum is within E of
    the float64 maximum. [the_kernel_integer_is_the_rounding]: if the computed difference K + r
    is within F of the approximations' difference, the float64 subtraction
    within R of the exact one, 2E + F + R < G and r is not within G of a
    half-integer, then K + step(r) is within 1/2 of RN64(M - L): it is the
    rounding, whatever the tie rule (no tie is possible).
    [far_keys_have_weight_zero]: if K >= 30 * 2^16 + 2, every integer within
    1/2 of RN64(M - L) gives `exp2_neg_q16_16` 0, so the weight is 0 both
    ways. [the_budget_fits_the_guard]: for |L| < 2^24 the double float's
    5e-14 relative is below 2^-20 (with float64's own 2^-52, E <= 2^-20),
    and with F = 2^-20 and R = 2^-28 the kernel's G = 2^-16 holds them with
    room to spare.

    ** Refutation.

    [without_the_guard_a_near_tie_rounds_the_other_way]: a difference
    2^-30 above 3/2 approximated 2^-28 below it: the approximation's
    nearest integer is 1, the float64 one 2.

    ** Not proved here.

    Dekker's product and Fast2Sum themselves (classical, for float32 without
    FMA contraction and without overflow); that each float32 and float64
    operation rounds to nearest (IEEE). The float32 roundings' share F
    (below 2^-21 for |hi| < 2^24, with r in -1 .. 1) is `ExactAttnDiff.v`,
    and that the kernel's pair maximum is the largest value hi + lo (the
    premise of [the_maximum_moves_by_at_most_the_error]) is
    `ExactAttnPairMax.v`; the tests compare the kernel with the float64
    path on ties, near ties and random rows.

    Checked with Rocq 9.1:  coqc -Q . "" ExactAttnWeights.v  *)

From Stdlib Require Import ZArith Lia QArith Qabs Lqa.
Require ExactDecodeWidths.
Open Scope Q_scope.

Module DW := ExactDecodeWidths.

Definition u : Q := 1 # 16777216.          (* 2^-24 *)

Lemma scale_bound : forall c X B, 0 < c -> - B <= X <= B -> - (c * B) <= c * X <= c * B.
Proof. intros c X B Hc [H1 H2]. split; nra. Qed.

Lemma round_bound : forall X d B, - u <= d <= u -> - B <= X <= B -> - (u * B) <= X * d <= u * B.
Proof. intros X d B [D1 D2] [H1 H2]. unfold u in *. split; nra. Qed.

Section DoubleFloat.
  Variables s a sk : Q.
  Hypothesis Hs : 0 < s.
  Hypothesis Ha : 0 < a.
  Hypothesis Hsk : 0 < sk.

  (** a = a_hi + a_lo: a_hi = RN32(a), a_lo = RN32(a - a_hi). *)
  Variables a_hi a_lo d0 d1 : Q.
  Hypothesis Hd0 : - u <= d0 <= u.
  Hypothesis Hd1 : - u <= d1 <= u.
  Hypothesis Hahi : a_hi == a * (1 + d0).
  Hypothesis Halo : a_lo == (a - a_hi) * (1 + d1).

  (** s * a: (p1, e1) = two_prod(s, a_hi); e1' = RN(e1 + RN(s * a_lo)); then Fast2Sum. *)
  Variables p1 e1 sal e1p d2 d3 x_hi x_lo : Q.
  Hypothesis Htp1 : p1 + e1 == s * a_hi.
  Hypothesis He1 : - (u * p1) <= e1 <= u * p1.
  Hypothesis Hp1 : 0 <= p1.
  Hypothesis Hd2 : - u <= d2 <= u.
  Hypothesis Hd3 : - u <= d3 <= u.
  Hypothesis Hsal : sal == s * a_lo * (1 + d2).
  Hypothesis He1p : e1p == (e1 + sal) * (1 + d3).
  Hypothesis Hx : x_hi + x_lo == p1 + e1p.
  Hypothesis Hxlo : - (u * x_hi) <= x_lo <= u * x_hi.
  Hypothesis Hxhi : 0 <= x_hi.

  (** times sk: (q1, f1) = two_prod(x_hi, sk); f1' = RN(f1 + RN(x_lo * sk)); then Fast2Sum. *)
  Variables q1 f1 xls f1p d4 d5 l_hi l_lo : Q.
  Hypothesis Htp2 : q1 + f1 == x_hi * sk.
  Hypothesis Hf1 : - (u * q1) <= f1 <= u * q1.
  Hypothesis Hq1 : 0 <= q1.
  Hypothesis Hd4 : - u <= d4 <= u.
  Hypothesis Hd5 : - u <= d5 <= u.
  Hypothesis Hxls : xls == x_lo * sk * (1 + d4).
  Hypothesis Hf1p : f1p == (f1 + xls) * (1 + d5).
  Hypothesis Hl : l_hi + l_lo == q1 + f1p.

  Lemma a_parts : - (u * a) <= a - a_hi <= u * a /\ - (u * (u * a)) <= a_hi + a_lo - a <= u * (u * a).
  Proof.
    assert (D : - (u * a) <= a - a_hi <= u * a).
    { assert (E : a - a_hi == - (a * d0)) by (rewrite Hahi; ring). rewrite E. unfold u in *. split; nra. }
    split; [exact D|].
    assert (E : a_hi + a_lo - a == (a - a_hi) * d1) by (rewrite Halo; ring).
    rewrite E. apply round_bound; [exact Hd1 | exact D].
  Qed.

  Lemma x_close : - ((2 # 100000000000000) * (s * a)) <= x_hi + x_lo - s * a <= (2 # 100000000000000) * (s * a).
  Proof.
    destruct a_parts as [D A].
    assert (E : x_hi + x_lo - s * a == s * (a_hi + a_lo - a) + s * a_lo * d2 + (e1 + sal) * d3).
    { rewrite Hx, He1p, Hsal. assert (p1 == s * a_hi - e1) by lra. rewrite H. ring. }
    assert (Alo : - (u * a * (1 + u)) <= a_lo <= u * a * (1 + u)).
    { assert (E2 : a_lo == (a - a_hi) + (a - a_hi) * d1) by (rewrite Halo; ring).
      pose proof (round_bound (a - a_hi) d1 (u * a) Hd1 D). rewrite E2. unfold u in *. lra. }
    pose proof (scale_bound s _ _ Hs A) as T1.
    pose proof (scale_bound s _ _ Hs Alo) as SA.
    pose proof (round_bound (s * a_lo) d2 _ Hd2 SA) as T2.
    assert (Ahi : a_hi <= a * (1 + u)) by (rewrite Hahi; unfold u in *; nra).
    assert (P1 : p1 <= s * a * (1 + 3 * u)).
    { assert (p1 * (1 - u) <= s * a_hi) by (unfold u in *; lra).
      assert (s * a_hi <= s * (a * (1 + u))) by (apply Qmult_le_l; [exact Hs | exact Ahi]).
      unfold u in *. lra. }
    assert (SL : - (s * (u * a * (1 + u)) + u * (s * (u * a * (1 + u)))) <= sal
                 <= s * (u * a * (1 + u)) + u * (s * (u * a * (1 + u)))).
    { assert (sal == s * a_lo + s * a_lo * d2) by (rewrite Hsal; ring). lra. }
    assert (ES : - (u * (s * a * (1 + 3 * u)) + (s * (u * a * (1 + u)) + u * (s * (u * a * (1 + u)))))
                 <= e1 + sal <= u * (s * a * (1 + 3 * u)) + (s * (u * a * (1 + u)) + u * (s * (u * a * (1 + u))))).
    { unfold u in *. lra. }
    pose proof (round_bound (e1 + sal) d3 _ Hd3 ES) as T3.
    rewrite E. unfold u in *. split; lra.
  Qed.

  Theorem the_double_float_logit_is_within_5e_14 :
    - ((5 # 100000000000000) * (s * a * sk)) <= l_hi + l_lo - s * a * sk <= (5 # 100000000000000) * (s * a * sk).
  Proof.
    pose proof x_close as [X1 X2].
    assert (E : l_hi + l_lo - s * a * sk == (x_hi + x_lo - s * a) * sk + x_lo * sk * d4 + (f1 + xls) * d5).
    { rewrite Hl, Hf1p, Hxls. assert (q1 == x_hi * sk - f1) by lra. rewrite H. ring. }
    pose proof (scale_bound sk _ _ Hsk (conj X1 X2)) as T1.
    assert (Xh : x_hi <= s * a * 2).
    { assert (x_hi * (1 - u) <= x_hi + x_lo) by (unfold u in *; lra). unfold u in *. lra. }
    pose proof (scale_bound sk _ _ Hsk Hxlo) as XL.
    pose proof (round_bound (sk * x_lo) d4 _ Hd4 XL) as T2.
    assert (Q1 : q1 <= x_hi * sk * (1 + 3 * u)).
    { assert (q1 * (1 - u) <= x_hi * sk) by (unfold u in *; lra). unfold u in *. assert (0 <= x_hi * sk) by nra. lra. }
    assert (XLS : - (sk * (u * x_hi) + u * (sk * (u * x_hi))) <= xls <= sk * (u * x_hi) + u * (sk * (u * x_hi))).
    { assert (xls == sk * x_lo + sk * x_lo * d4) by (rewrite Hxls; ring). lra. }
    assert (FX : - (u * (x_hi * sk * (1 + 3 * u)) + (sk * (u * x_hi) + u * (sk * (u * x_hi)))) <= f1 + xls
                 <= u * (x_hi * sk * (1 + 3 * u)) + (sk * (u * x_hi) + u * (sk * (u * x_hi)))).
    { unfold u in *. lra. }
    pose proof (round_bound (f1 + xls) d5 _ Hd5 FX) as T3.
    assert (XS : x_hi * sk <= s * a * sk * 2).
    { assert (x_hi * sk <= (s * a * 2) * sk) by (apply Qmult_le_r; [exact Hsk | exact Xh]). lra. }
    assert (0 <= x_hi * sk) by nra.
    assert (E4 : sk * x_lo * d4 == x_lo * sk * d4) by ring.
    rewrite E. unfold u in *. split; lra.
  Qed.
End DoubleFloat.

(** ** The decision *)

Section Decision.
  Variables E F R G : Q.
  Variables M64 mf L64 Lf Li64 Lif Lk64 Lkf : Q.
  (** mf is the max of the approximations, attained at key i; M64 the max
      of the float64 logits, attained at key k. *)
  Hypothesis HmfLf : Lf <= mf.
  Hypothesis HmfLk : Lkf <= mf.
  Hypothesis Hmfi : mf == Lif.
  Hypothesis HM64L : L64 <= M64.
  Hypothesis HM64i : Li64 <= M64.
  Hypothesis HM64k : M64 == Lk64.
  Hypothesis Herr : Qabs (Lf - L64) <= E.
  Hypothesis Herri : Qabs (Lif - Li64) <= E.
  Hypothesis Herrk : Qabs (Lkf - Lk64) <= E.

  Theorem the_maximum_moves_by_at_most_the_error : Qabs (mf - M64) <= E.
  Proof.
    apply Qabs_Qle_condition in Herri. apply Qabs_Qle_condition in Herrk.
    apply Qabs_Qle_condition. split; lra.
  Qed.

  (** D64 = RN64(M64 - L64) within R of the exact difference; the computed
      K + r within F of mf - Lf. *)
  Variables D64 r : Q.
  Variable K : Z.
  Hypothesis HR : Qabs (D64 - (M64 - L64)) <= R.
  Hypothesis HF : Qabs (inject_Z K + r - (mf - Lf)) <= F.
  Hypothesis Hr : - 1 <= r <= 1.
  Hypothesis Hbudget : 2 * E + F + R < G.
  Hypothesis HG : 0 < G < 1 # 4.

  Definition step (r : Q) : Z := if Qlt_le_dec (1 # 2) r then 1%Z else if Qlt_le_dec r (- (1 # 2)) then (-1)%Z else 0%Z.

  Lemma close : Qabs (inject_Z K + r - D64) < G.
  Proof.
    pose proof the_maximum_moves_by_at_most_the_error as Mm.
    apply Qabs_Qle_condition in Mm. apply Qabs_Qle_condition in Herr.
    apply Qabs_Qle_condition in HR. apply Qabs_Qle_condition in HF.
    apply Qabs_Qlt_condition. split; lra.
  Qed.

  Theorem the_kernel_integer_is_the_rounding :
    G < Qabs (Qabs r - (1 # 2)) ->
    Qabs (D64 - inject_Z (K + step r)) < 1 # 2.
  Proof.
    intro Hg. pose proof close as C. apply Qabs_Qlt_condition in C.
    rewrite inject_Z_plus. unfold step.
    destruct (Qlt_le_dec (1 # 2) r) as [P|P].
    - change (inject_Z 1) with 1. apply Qabs_Qlt_condition.
      assert (Qabs r == r) by (apply Qabs_pos; lra). rewrite H in Hg.
      assert (Qabs (r - (1 # 2)) == r - (1 # 2)) by (apply Qabs_pos; lra). rewrite H0 in Hg. split; lra.
    - destruct (Qlt_le_dec r (- (1 # 2))) as [N|N].
      + change (inject_Z (-1)) with (-1). apply Qabs_Qlt_condition.
        assert (Qabs r == - r) by (apply Qabs_neg; lra). rewrite H in Hg.
        assert (Qabs (- r - (1 # 2)) == - r - (1 # 2)) by (apply Qabs_pos; lra). rewrite H0 in Hg. split; lra.
      + change (inject_Z 0) with 0. apply Qabs_Qlt_condition.
        assert (Hle : Qabs r <= 1 # 2) by (apply Qabs_Qle_condition; split; lra).
        assert (H : Qabs (Qabs r - (1 # 2)) == - (Qabs r - (1 # 2))) by (apply Qabs_neg; lra).
        rewrite H in Hg. assert (H0 : Qabs r < (1 # 2) - G) by lra. apply Qabs_Qlt_condition in H0. split; lra.
  Qed.

  Theorem far_keys_have_weight_zero : (30 * 65536 + 2 <= K)%Z ->
    forall z : Z, Qabs (D64 - inject_Z z) <= 1 # 2 -> DW.exp2_neg_q16_16 z = 0%Z.
  Proof.
    intros HK z Hz. pose proof close as C. apply Qabs_Qlt_condition in C. apply Qabs_Qle_condition in Hz.
    assert (Kq : inject_Z (30 * 65536 + 2) <= inject_Z K) by (rewrite <- Zle_Qle; exact HK).
    assert (Zq : inject_Z (30 * 65536) < inject_Z z).
    { change (inject_Z (30 * 65536 + 2)) with (1966082 # 1) in Kq. change (inject_Z (30 * 65536)) with (1966080 # 1).
      lra. }
    rewrite <- Zlt_Qlt in Zq. unfold DW.exp2_neg_q16_16.
    assert (N : (30 <=? Z.shiftr (Z.max z 0) 16)%Z = true).
    { apply Z.leb_le. rewrite Z.shiftr_div_pow2 by lia. apply Z.div_le_lower_bound; lia. }
    cbv zeta. rewrite N. reflexivity.
  Qed.
End Decision.

Theorem the_budget_fits_the_guard :
  (5 # 100000000000000) * 16777216 + (1 # 4503599627370496) * 16777216 <= 1 # 1048576 /\
  2 * (1 # 1048576) + (1 # 1048576) + (1 # 268435456) < 1 # 65536.
Proof. split; lra. Qed.

(** D64 = 3/2 + 2^-30, approximated as 3/2 - 2^-28: nearest integers 1 and 2. *)
Theorem without_the_guard_a_near_tie_rounds_the_other_way :
  let d64 := (3 # 2) + (1 # 1073741824) in
  let df := (3 # 2) - (1 # 268435456) in
  Qabs (df - 1) < 1 # 2 /\ Qabs (d64 - 2) < 1 # 2.
Proof. cbv zeta. split; apply Qabs_Qlt_condition; split; lra. Qed.

Print Assumptions the_double_float_logit_is_within_5e_14.
Print Assumptions the_maximum_moves_by_at_most_the_error.
Print Assumptions the_kernel_integer_is_the_rounding.
Print Assumptions far_keys_have_weight_zero.
Print Assumptions the_budget_fits_the_guard.
Print Assumptions without_the_guard_a_near_tie_rounds_the_other_way.
