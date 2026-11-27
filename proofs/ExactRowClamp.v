(** * The per-row quantizers' clamp (`kv.quantize_rows`, `w8a8_gemv`).

    Decode quantizes each row x to integers by float32 divides: s = amax / L
    (amax the row's largest |x|, L the levels: 127, or a wider K width),
    q = round(x / s), clamped to -L .. L. `kv.quantize_rows` and the W8A8
    kernel's `_quant_rows` do it with IEEE divides (`div_rn` in the Triton
    kernel), and the W8A8 kernel's comment records the case the clamp is
    for: a row whose largest |x| is below 127 * 2^-126, where the scale is
    subnormal.

    Values here are rationals; a float32 divide whose result is normal is the
    exact quotient times (1 + d), |d| <= u, u <= 2^-24 (round to nearest).

    [the_quotient_stays_below_L_and_a_half]: for every x of a row whose scale
    is normal, and L <= 2^21, |RN(x / s)| < L + 1/2.
    [the_clamp_does_nothing_when_the_scale_is_normal]: so every integer
    within 1/2 of it (the rounding, ties either way) is in -L .. L: the clamp
    never changes a value there.

    ** Refutations.

    [a_subnormal_scale_needs_the_clamp]: amax = 190 * 2^-149 and L = 127:
    amax / L is 1.496 of float32's smallest step, so s rounds to that step
    (a subnormal), and x = amax gives x / s = 190, past 127. The clamp is
    load-bearing where the scale is subnormal.
    [the_margin_is_gone_at_L_2_23]: at L = 2^23 the bound L (1 + u) does not
    stay below (L + 1/2) (1 - u), so the argument needs L well below 2^22.

    ** Not proved here.

    IEEE's division (the standard model above, in the normal range); that
    `round` and `rint` round to nearest; the replacement of a zero scale by 1
    (a row of zeros, or one whose scale underflows to 0, whose |x| are then
    below 2^-128 and round to 0).

    Checked with Rocq 9.1:  coqc -Q . "" ExactRowClamp.v  *)

From Stdlib Require Import ZArith Lia QArith Qabs Lqa.
Open Scope Q_scope.

Definition u32 : Q := 1 # 16777216.          (* 2^-24 *)

Section Clamp.
  Variables amax x s q d1 d2 u : Q.
  Variable L : Z.
  Hypothesis HL : (1 <= L <= 2 ^ 21)%Z.
  Hypothesis Hu0 : 0 <= u.
  Hypothesis Hu : u <= u32.
  Hypothesis Ha : 0 < amax.
  Hypothesis Hx : Qabs x <= amax.
  Hypothesis H1 : - u <= d1 <= u.
  Hypothesis H2 : - u <= d2 <= u.
  Hypothesis Hs : s == amax / inject_Z L * (1 + d1).      (* RN(amax / L), normal *)
  Hypothesis Hq : q == x / s * (1 + d2).                  (* RN(x / s) *)

  Lemma L_pos : 0 < inject_Z L.
  Proof. change 0 with (inject_Z 0). rewrite <- Zlt_Qlt. lia. Qed.

  Lemma L_le : inject_Z L <= 2097152.
  Proof. change 2097152 with (inject_Z (2 ^ 21)). rewrite <- Zle_Qle. lia. Qed.

  Theorem the_quotient_stays_below_L_and_a_half : Qabs q < inject_Z L + (1 # 2).
  Proof.
    pose proof L_pos as HLq. pose proof L_le as HLe. unfold u32 in Hu.
    set (Lq := inject_Z L) in *.
    set (a := amax / Lq).
    assert (Ha0 : 0 < a) by (apply Qlt_shift_div_l; [exact HLq | lra]).
    assert (AL : a * Lq == amax) by (unfold a; field; intro Z; lra).
    assert (Hsp : 0 < s) by (rewrite Hs; fold a; nra).
    assert (Hslo : a * (1 - u) <= s) by (rewrite Hs; fold a; nra).
    set (r := x / s).
    assert (RS : r * s == x) by (unfold r; field; intro Z; lra).
    assert (AR : Qabs r * s == Qabs x).
    { rewrite <- RS, Qabs_Qmult, (Qabs_pos s) by lra. reflexivity. }
    assert (R0 : 0 <= Qabs r) by apply Qabs_nonneg.
    (* |r| a (1 - u) <= |r| s = |x| <= a L, so |r| (1 - u) <= L *)
    assert (R1 : Qabs r * (a * (1 - u)) <= a * Lq).
    { apply Qle_trans with (Qabs r * s); [nra | lra]. }
    assert (R2 : Qabs r * (1 - u) <= Lq) by nra.
    assert (Q1 : Qabs q == Qabs r * (1 + d2)).
    { rewrite Hq. fold r. rewrite Qabs_Qmult, (Qabs_pos (1 + d2)) by lra. reflexivity. }
    assert (Q2 : Qabs q * (1 - u) <= Lq * (1 + u)) by (rewrite Q1; nra).
    assert (M : Lq * (1 + u) < (Lq + (1 # 2)) * (1 - u)) by nra.
    nra.
  Qed.

  Theorem the_clamp_does_nothing_when_the_scale_is_normal : forall k : Z,
    Qabs (q - inject_Z k) <= 1 # 2 -> (- L <= k <= L)%Z.
  Proof.
    intros k Hk. pose proof the_quotient_stays_below_L_and_a_half as B.
    apply Qabs_Qle_condition in Hk. apply Qabs_Qlt_condition in B.
    assert (U : inject_Z k < inject_Z (L + 1)) by (rewrite inject_Z_plus; change (inject_Z 1) with 1; lra).
    assert (D : inject_Z (- L - 1) < inject_Z k).
    { unfold Z.sub. rewrite inject_Z_plus, !inject_Z_opp. change (inject_Z 1) with 1. lra. }
    rewrite <- Zlt_Qlt in U, D. lia.
  Qed.
End Clamp.

(** In units of 2^-149: amax = 190, L = 127. The nearest integer to 190 / 127
    is 1 (190 is 63 from 127 and 64 from 254), a subnormal step; 190 / 1 is
    past 127. *)
Theorem a_subnormal_scale_needs_the_clamp :
  (2 * Z.abs (190 - 127 * 1) < 127 /\ 1 < 2 ^ 23 /\ 190 / 1 > 127)%Z.
Proof. vm_compute. repeat split; reflexivity. Qed.

Theorem the_margin_is_gone_at_L_2_23 :
  ~ (inject_Z (2 ^ 23) * (1 + u32) < (inject_Z (2 ^ 23) + (1 # 2)) * (1 - u32)).
Proof. intro H. vm_compute in H. discriminate H. Qed.

Print Assumptions the_quotient_stays_below_L_and_a_half.
Print Assumptions the_clamp_does_nothing_when_the_scale_is_normal.
Print Assumptions a_subnormal_scale_needs_the_clamp.
Print Assumptions the_margin_is_gone_at_L_2_23.
