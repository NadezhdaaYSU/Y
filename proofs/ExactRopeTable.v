(** * The RoPE table's guard (`python/y_lang/exact/model.py`, `cos_sin_f32`).

    `ExactRotary` reads cos and sin from a table built once on the host:
    each entry should be the correctly rounded float32 of cos(arg) * scale,
    whatever the host's libm. `cos_sin_f32` computes y = fn(arg) * scale in
    float64 with the host's cos or sin, rounds y to float32, and sends to
    mpmath every entry whose y has its low 29 fraction bits within
    `ROPE_MARGIN` = 2 * `ROPE_LIBM_ULPS` + 2 = 10 of 2^28 (a float32
    midpoint's pattern), is below 2^-126, or is not finite. Its comment
    claims every entry is then correctly rounded on any host whose libm errs
    by less than `ROPE_LIBM_ULPS` = 4 ulps. This file proves it.

    Units: values are rationals in units of 2^-1074; y has the float64
    fields (E, N), so it is `ExactNormKernel.v64 E N`, its ulp
    `ExactNormKernel.D E`, and its float32 rounding the float32
    `ExactNormKernel.G E N` (`the_rounding_is_a_normal_float32` there).
    Magnitudes: cos(arg) * scale > 0 (the sign is symmetric).

    [the_float32_is_the_nearest_to_every_fraction_within_the_margin]: if the
    guard's distance is at least w (w <= 2^26), G is strictly the nearest
    float32 to every fraction vn / vd within w ulps of y.
    [the_libm_error_moves_y_by_less_than_eleven_ulps]: with the true value
    cs, a computed c within 4 ulps of it (|c - cs| <= 4 * 2^-52 * cs, which
    holds for 4 ulps of cs in float64's normal range) and y = c * s rounded
    to float64, |y - cs * s| < 11 ulps of y (it is below 9 and a little).
    [the_rope_entry_is_correctly_rounded]: so for every y the guard passes,
    the float32 the table keeps is strictly the nearest float32 to the true
    cs * s: the correctly rounded value, with no tie to break.

    ** Refutation.

    [a_margin_of_the_libm_error_alone_admits_a_tie]: with the distance 4, a
    libm error within the bound can put the true value on the midpoint,
    where the two neighbours are equally near: a margin of
    `ROPE_LIBM_ULPS` alone would not do.

    ** Not proved here.

    That the host's libm meets its bound (the comment's assumption; libms
    claim 1 ulp); IEEE's float64 multiply and its rounding to float32; that
    mpmath decides the entries the guard sends to it; scale below 2^126.

    Checked with Rocq 9.1:  coqc -Q . "" ExactRopeTable.v  *)

From Stdlib Require Import ZArith Lia QArith Qabs Lqa.
Require ExactFloatOrder ExactNormKernel.
Open Scope Z_scope.

Module FO := ExactFloatOrder.
Module NK := ExactNormKernel.

(** `ROPE_MARGIN = 2 * ROPE_LIBM_ULPS + 2`. *)
Definition rope_libm_ulps : Z := 4.
Definition rope_margin : Z := 2 * rope_libm_ulps + 2.

Section Nearest.
  Variables E N : Z.
  Hypothesis HE : 897 <= E <= 1149.
  Hypothesis HN : 0 <= N < 2 ^ 52.

  Theorem the_float32_is_the_nearest_to_every_fraction_within_the_margin : forall w vn vd,
    0 <= w <= NK.dist (FO.bits 52 E N) -> w <= 2 ^ 26 -> 0 < vd ->
    Z.abs (vn - NK.v64 E N * vd) < w * NK.D E * vd ->
    forall e' m', FO.finite 23 e' m' -> FO.bits 23 e' m' <> NK.f32 (FO.bits 52 E N) ->
    Z.abs (vn - NK.G E N * vd) < Z.abs (vn - NK.v32 e' m' * vd).
  Proof.
    intros w vn vd Hw Hw2 Hvd Hv e' m' F' Ne.
    pose proof (NK.D_pos E HE) as HD. pose proof (NK.the_gap_is_the_guard_distance E N HE HN) as Gap.
    set (V := NK.v64 E N) in *. set (G := NK.G E N) in *. set (O := NK.v32 e' m') in *.
    set (D := NK.D E) in *. set (dd := NK.dist (FO.bits 52 E N)) in *.
    set (X := D * vd). assert (HX : 0 < X) by (unfold X; nia).
    set (W := w * X). assert (HW : 0 <= W <= 2 ^ 26 * X) by (unfold W; nia).
    assert (HV : Z.abs (vn - V * vd) < W) by (unfold W, X; rewrite Z.mul_assoc; exact Hv).
    assert (HG : Z.abs (V * vd - G * vd) <= 2 ^ 28 * X - W).
    { replace (V * vd - G * vd) with ((V - G) * vd) by ring. rewrite Z.abs_mul, (Z.abs_eq vd) by lia.
      rewrite Gap. unfold W, X. assert (dd * (D * vd) >= w * (D * vd)) by nia. nia. }
    destruct (Z_lt_le_dec (NK.f32 (FO.bits 52 E N)) (FO.bits 23 e' m')) as [Lt|Le].
    - pose proof (NK.above E N HE HN e' m' F' Lt) as A. fold G O D in A.
      assert (O * vd >= G * vd + 2 ^ 29 * X) by (unfold X; nia). lia.
    - assert (Lt : FO.bits 23 e' m' < NK.f32 (FO.bits 52 E N)) by lia.
      destruct (NK.below E N HE HN e' m' F' Lt) as [B|[B Gv]]; fold G O D V in B; [|fold G V in Gv].
      + assert (O * vd <= G * vd - 2 ^ 29 * X) by (unfold X; nia). lia.
      + assert (O * vd <= G * vd - 2 ^ 28 * X) by (unfold X; nia).
        assert (G * vd <= V * vd) by nia. lia.
  Qed.
End Nearest.

(** A rational's distance, as integers over its denominator. *)
Lemma q_window : forall (v : Q) (A B : Z),
  (Qabs (v - inject_Z A) < inject_Z B)%Q <-> Z.abs (Qnum v - A * Zpos (Qden v)) < B * Zpos (Qden v).
Proof.
  intros [n d] A B. unfold Qabs, Qminus, Qplus, Qopp, Qlt, inject_Z. cbn [Qnum Qden].
  rewrite !Z.mul_1_r, Pos.mul_1_r. replace (n + - A * Z.pos d) with (n - A * Z.pos d) by ring.
  cbn. reflexivity.
Qed.

Lemma q_compare : forall (v : Q) (A B : Z),
  Z.abs (Qnum v - A * Zpos (Qden v)) < Z.abs (Qnum v - B * Zpos (Qden v)) ->
  (Qabs (v - inject_Z A) < Qabs (v - inject_Z B))%Q.
Proof.
  intros [n d] A B H. cbn [Qnum Qden] in H. unfold Qabs, Qminus, Qplus, Qopp, Qlt, inject_Z. cbn [Qnum Qden].
  rewrite !Z.mul_1_r, !Pos.mul_1_r. replace (n + - A * Z.pos d) with (n - A * Z.pos d) by ring.
  replace (n + - B * Z.pos d) with (n - B * Z.pos d) by ring. cbn [Qnum Qden Z.abs].
  apply Z.mul_lt_mono_pos_r; [lia | exact H].
Qed.

(** ** The libm's error after the scale and the rounding to float64 *)
Section Budget.
  Open Scope Q_scope.
  Variables c cs s d u Dy y : Q.
  Hypothesis Hu0 : 0 <= u.
  Hypothesis Hu : u * 9007199254740992 <= 1.          (* u <= 2^-53 *)
  Hypothesis Hd : - u <= d <= u.
  Hypothesis Hcs : 0 < cs.
  Hypothesis Hs : 0 < s.
  Hypothesis Hc : Qabs (c - cs) <= 4 * (2 * u) * cs.  (* 4 ulps of cs: 4 * 2^-52 * cs *)
  Hypothesis Hy : y == c * s * (1 + d).
  Hypothesis HDy : 0 < Dy.
  Hypothesis Hbin : y < 9007199254740992 * Dy.       (* y below 2^53 of its ulps *)

  Theorem the_libm_error_moves_y_by_less_than_eleven_ulps : Qabs (y - cs * s) < 11 * Dy.
  Proof.
    assert (Hc' : - (8 * u * cs) <= c - cs <= 8 * u * cs).
    { apply Qabs_Qle_condition. setoid_replace (8 * u * cs) with (4 * (2 * u) * cs) by ring. exact Hc. }
    assert (Hu1 : u <= 1 # 1000) by lra.
    assert (Hc0 : cs * (1 - 8 * u) <= c) by lra.
    assert (Hcp : 0 < c) by nra.
    assert (Hcsp : 0 < c * s) by nra.
    assert (Hlow : c * s * (1 - u) <= y) by (rewrite Hy; nra).
    assert (A1 : u * (c * s) * (1 - u) <= Dy) by nra.
    assert (A2 : cs * s * (1 - 8 * u) <= c * s) by nra.
    assert (Hsplit : y - cs * s == (c - cs) * s + c * s * d) by (rewrite Hy; ring).
    assert (B1 : - (u * (c * s)) <= c * s * d <= u * (c * s)) by (split; nra).
    assert (B2 : - (8 * u * (cs * s)) <= (c - cs) * s <= 8 * u * (cs * s)) by (split; nra).
    set (K := (1 - u) * (1 - 8 * u)).
    assert (HK : 9 # 10 < K) by (unfold K; nra).
    assert (P1 : 0 <= u * (c * s)) by nra.
    assert (P2 : 0 <= u * (cs * s)) by nra.
    assert (KU : K <= 1 - u) by (unfold K; nra).
    assert (T1 : u * (c * s) * K <= Dy) by nra.
    assert (P3 : 0 <= 8 * u * (1 - u)) by nra.
    assert (T2 : 8 * u * (cs * s) * K <= 8 * Dy).
    { assert (E2 : 8 * u * (cs * s) * K == (8 * u * (1 - u)) * (cs * s * (1 - 8 * u))) by (unfold K; ring).
      rewrite E2.
      assert (L : (8 * u * (1 - u)) * (cs * s * (1 - 8 * u)) <= (8 * u * (1 - u)) * (c * s))
        by (apply Qmult_le_l_compat; assumption) || nra.
      assert (E3 : (8 * u * (1 - u)) * (c * s) == 8 * (u * (c * s) * (1 - u))) by ring.
      lra. }
    assert (C : (u * (c * s) + 8 * u * (cs * s)) * K <= 9 * Dy) by lra.
    assert (S0 : 0 <= u * (c * s) + 8 * u * (cs * s)) by lra.
    assert (T : u * (c * s) + 8 * u * (cs * s) < 11 * Dy) by nra.
    apply Qabs_Qlt_condition. rewrite Hsplit. split; lra.
  Qed.
End Budget.

Definition u64 : Q := 1 # 9007199254740992.

Theorem the_rope_entry_is_correctly_rounded : forall E N (c cs s d : Q),
  897 <= E <= 1149 -> 0 <= N < 2 ^ 52 -> rope_margin < NK.dist (FO.bits 52 E N) ->
  (0 < cs)%Q -> (0 < s)%Q -> (- u64 <= d <= u64)%Q ->
  (Qabs (c - cs) <= 4 * (2 * u64) * cs)%Q ->
  (inject_Z (NK.v64 E N) == c * s * (1 + d))%Q ->
  forall e' m', FO.finite 23 e' m' -> FO.bits 23 e' m' <> NK.f32 (FO.bits 52 E N) ->
  (Qabs (cs * s - inject_Z (NK.G E N)) < Qabs (cs * s - inject_Z (NK.v32 e' m')))%Q.
Proof.
  intros E N c cs s d HE HN Hm Hcs Hs Hd Hc Hy e' m' F' Ne.
  pose proof (NK.D_pos E HE) as HD.
  assert (VE : NK.v64 E N = (2 ^ 52 + N) * NK.D E) by (apply NK.V_eq; assumption).
  assert (Hbin : (inject_Z (NK.v64 E N) < 9007199254740992 * inject_Z (NK.D E))%Q).
  { rewrite <- (inject_Z_mult 9007199254740992). rewrite <- Zlt_Qlt. rewrite VE. nia. }
  assert (HDq : (0 < inject_Z (NK.D E))%Q) by (change 0%Q with (inject_Z 0); rewrite <- Zlt_Qlt; lia).
  pose proof (the_libm_error_moves_y_by_less_than_eleven_ulps c cs s d u64 (inject_Z (NK.D E)) (inject_Z (NK.v64 E N))
    ltac:(unfold u64; lra) ltac:(unfold u64; lra) Hd Hcs Hs Hc Hy HDq Hbin) as B.
  (* the true value cs * s, as a fraction within 11 ulps of y *)
  set (v := (cs * s)%Q).
  assert (W : (Qabs (v - inject_Z (NK.v64 E N)) < inject_Z (11 * NK.D E))%Q).
  { rewrite inject_Z_mult. unfold v. rewrite Qabs_Qminus. exact B. }
  apply q_window in W.
  apply q_compare.
  unfold rope_margin, rope_libm_ulps in Hm.
  apply (the_float32_is_the_nearest_to_every_fraction_within_the_margin E N HE HN 11); try assumption; try lia.
Qed.

(** ** Refutation *)

Definition Nt : Z := 2 ^ 28 + 4.

Theorem a_margin_of_the_libm_error_alone_admits_a_tie :
  let V := NK.v64 1023 Nt in
  let D := NK.D 1023 in
  let cs := V - 4 * D in
  NK.dist (FO.bits 52 1023 Nt) = rope_libm_ulps /\
  (inject_Z (4 * D) <= 4 * (2 * u64) * inject_Z cs)%Q /\
  Z.abs (cs - NK.G 1023 Nt) = Z.abs (cs - (NK.G 1023 Nt - 2 ^ 29 * D)).
Proof.
  intros V D cs. split; [vm_compute; reflexivity|]. split.
  - apply Qle_bool_imp_le. vm_compute. reflexivity.
  - vm_compute. reflexivity.
Qed.

Print Assumptions the_float32_is_the_nearest_to_every_fraction_within_the_margin.
Print Assumptions the_libm_error_moves_y_by_less_than_eleven_ulps.
Print Assumptions the_rope_entry_is_correctly_rounded.
Print Assumptions a_margin_of_the_libm_error_alone_admits_a_tie.
