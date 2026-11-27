(** * The RMS norm kernel's float64 work on bits (`python/y_lang/exact/norm_kernel.py`).

    `ExactRMSNorm` gives `float32(x / r)` per element, `r` the row's float64
    root, the divide correctly rounded in float64. The kernel divides once per
    row (`rinv = 1/r`), multiplies per element (`q = x * rinv`) and rounds `q`
    to float32 by integer arithmetic on its bits, sending to the true divide
    every element whose bits fail a guard. Its module docstring argues that the
    guarded path gives the divide's float32 everywhere. This file proves that
    argument, and the other bit formulas the kernel uses, for every input.

    Units: every value is an integer multiple of 2^-1074 (float64's smallest
    step), so a float64 with fields (E, N) is [v64 E N] =
    `ExactFloatOrder.value 52 E N` and a float32 with fields (e, m) is
    [v32 e m] = `ExactFloatOrder.value 23 e m * 2^925`. Magnitudes: the code
    works on |q|'s bits `abits` and puts the sign back; the sign bit (bit 63)
    does not reach the low 29 bits the guard reads
    ([the_sign_does_not_reach_the_guard_bits]).

    ** The guard and the rounding.

    [the_guard_admits_only_zero_and_the_normal_binades]: an `abits` the guard
    passes (`tiny`, `dist <= 3` and `big` all false) is 0, or has a field E
    in 897 .. 1149 (|q| in [2^-126, 2^127)) and `dist > 3`.
    [zero_passes_and_rounds_to_zero]: `dist` of 0 is 2^28, and the rounding
    gives the bits 0.

    [the_rounding_is_a_normal_float32]: for E in 897 .. 1149,
    `max(((abits + 2^28) >> 29) - (896 << 23), 0)` is a normal float32's bits
    (field 1 .. 254) whose value is k * 2^29 ulp64(q), k the significand
    2^52 + N plus 2^28, shifted down 29: round half up to float32's 24 bits.
    [the_gap_is_the_guard_distance]: q is (2^28 - dist) ulp64(q) from it.

    [the_kernel_float32_is_the_nearest_to_every_value_within_four_ulps]: if
    `dist > 3`, that float32 is strictly nearer than every other finite
    float32 to every value within 4 ulp64(q) of q. Strictly: no tie, so any
    rounding to nearest picks it.

    ** The multiply against the divide.

    [the_multiply_is_within_three_ulps_of_the_divide]: with t = x/r exact,
    rinv = (1/r)(1 + d1), q = x rinv (1 + d2) and p = RN64(x/r) = t(1 + d3),
    every |di| <= u <= 2^-53: |q - p| < 3.0001 ulp64(q), the docstring's
    bound.
    [the_multiply_rounds_as_the_divide]: so for a q the guard passes, the
    code's float32 is the nearest float32 to q and to p alike: the float32 of
    the divide.

    ** The other bit formulas.

    [a_normal_x_widens_by_its_bits]: for a normal float32 x, `x`'s bits
    (sign off) shifted up 29 plus 896 << 52 are the bits of a finite float64
    of the same value. [the_scale_is_a_power_of_two]: `(e + 691) << 52` read
    as a float64 is 2^(e - 332), a finite float64 for every e the row can
    have (2 .. 508).

    ** Refutations.

    [a_guard_of_two_would_miss_a_tie]: with `dist` 3, a p three ulps from q
    (within the bound) is a float32 midpoint, and round to nearest even
    takes the even neighbor while q's rounding takes the odd one: the guard
    must send `dist <= 3`, not only `dist <= 2`, to the divide.
    [a_subnormal_x_does_not_widen_by_its_bits]: the shift gives a subnormal
    x another value, with or without the 896, which is why a row with one
    converts in float64.

    ** Not proved here.

    That each float64 operation is IEEE round to nearest, which gives
    |d| <= 2^-53 in float64's normal range (1/r, x * rinv and x/r stay in it:
    r is at least sqrt(eps) and at most about 2^128, and the guard keeps
    |q| >= 2^-126); that `q == 0` exactly when `x == 0`; Triton's `>>`,
    `&` and bitcasts as IEEE storage (the tests run the kernel against the
    divide on inputs built to sit on midpoints).

    Checked with Rocq 9.1:  coqc -Q . "" ExactNormKernel.v  *)

From Stdlib Require Import ZArith Lia QArith Lqa.
Require ExactFloatOrder.
Open Scope Z_scope.

Module FO := ExactFloatOrder.

Definition v64 (E N : Z) : Z := FO.value 52 E N.
Definition v32 (e m : Z) : Z := FO.value 23 e m * 2 ^ 925.

(** The kernel's integer steps on `abits`, with its literals. *)
Definition dist (b : Z) : Z := Z.abs (Z.land b 536870911 - 268435456).
Definition f32 (b : Z) : Z := Z.max (Z.shiftr (b + 268435456) 29 - 7516192768) 0.
Definition tiny (b : Z) : bool := (b - 1) mod 2 ^ 64 <? 0x380FFFFFFFFFFFFF.
Definition big (b : Z) : bool := 0x47E0000000000000 <=? b.
Definition unsafe (b : Z) : bool := tiny b || (dist b <=? 3) || big b.

Theorem the_kernel_literals :
  536870911 = 2 ^ 29 - 1 /\ 268435456 = 2 ^ 28 /\ 7516192768 = 896 * 2 ^ 23 /\
  0x380FFFFFFFFFFFFF = 897 * 2 ^ 52 - 1 /\ 0x47E0000000000000 = 1150 * 2 ^ 52 /\
  4035225266123964416 = 896 * 2 ^ 52.
Proof. repeat split; reflexivity. Qed.

Lemma land_low : forall b, Z.land b 536870911 = b mod 2 ^ 29.
Proof.
  intro b. replace 536870911 with (Z.ones 29) by reflexivity. apply Z.land_ones. lia.
Qed.

(** The signed int64 of a negative q is abits - 2^63. *)
Theorem the_sign_does_not_reach_the_guard_bits : forall abits,
  dist (abits - 2 ^ 63) = dist abits.
Proof.
  intro abits. unfold dist. rewrite !land_low. f_equal. f_equal.
  replace (abits - 2 ^ 63) with (abits + (- 2 ^ 34) * 2 ^ 29) by lia.
  apply Z_mod_plus_full.
Qed.

Definition carry (N : Z) : Z := (N + 2 ^ 28) / 2 ^ 29.
Definition D (E : Z) : Z := 2 ^ (E - 1).
Definition G (E N : Z) : Z :=
  v32 (f32 (FO.bits 52 E N) / 2 ^ 23) (f32 (FO.bits 52 E N) mod 2 ^ 23).

Section Round.
  Variables E N : Z.
  Hypothesis HE : 897 <= E <= 1149.
  Hypothesis HN : 0 <= N < 2 ^ 52.

  Lemma D_pos : 0 < D E.
  Proof. unfold D. apply Z.pow_pos_nonneg; lia. Qed.

  Lemma carry_range : 0 <= carry N <= 2 ^ 23.
  Proof.
    unfold carry. split; [apply Z.div_pos; lia|].
    assert (carry N < 2 ^ 23 + 1); [|unfold carry in *; lia].
    unfold carry. apply Z.div_lt_upper_bound; lia.
  Qed.

  Lemma f32_eq : f32 (FO.bits 52 E N) = (E - 896) * 2 ^ 23 + carry N.
  Proof.
    pose proof carry_range. unfold f32, FO.bits, carry in *. rewrite Z.shiftr_div_pow2 by lia.
    replace (E * 2 ^ 52 + N + 268435456) with (N + 2 ^ 28 + (E * 2 ^ 23) * 2 ^ 29) by lia.
    rewrite Z.div_add by lia. lia.
  Qed.

  Lemma shape :
    (carry N < 2 ^ 23 /\ f32 (FO.bits 52 E N) / 2 ^ 23 = E - 896 /\ f32 (FO.bits 52 E N) mod 2 ^ 23 = carry N) \/
    (carry N = 2 ^ 23 /\ f32 (FO.bits 52 E N) / 2 ^ 23 = E - 895 /\ f32 (FO.bits 52 E N) mod 2 ^ 23 = 0).
  Proof.
    pose proof carry_range. rewrite f32_eq. destruct (Z.eq_dec (carry N) (2 ^ 23)) as [Hc|Hc].
    - right. split; [exact Hc|]. rewrite Hc. split.
      + symmetry. apply Z.div_unique with 0; lia.
      + symmetry. apply Z.mod_unique with (E - 895); lia.
    - left. split; [lia|]. split.
      + symmetry. apply Z.div_unique with (carry N); lia.
      + symmetry. apply Z.mod_unique with (E - 896); lia.
  Qed.

  Lemma f32_fields : let f := f32 (FO.bits 52 E N) in
    f = FO.bits 23 (f / 2 ^ 23) (f mod 2 ^ 23) /\ FO.finite 23 (f / 2 ^ 23) (f mod 2 ^ 23).
  Proof.
    intro f. unfold FO.bits, FO.finite. pose proof (Z.div_mod f (2 ^ 23) ltac:(lia)).
    pose proof (Z.mod_pos_bound f (2 ^ 23) ltac:(lia)). destruct shape as [[_ [A _]]|[_ [A _]]];
    subst f; rewrite A in *; lia.
  Qed.

  Theorem the_rounding_is_a_normal_float32 : let f := f32 (FO.bits 52 E N) in
    1 <= f / 2 ^ 23 <= 254 /\ 0 <= f mod 2 ^ 23 < 2 ^ 23 /\
    G E N = (2 ^ 52 + N + 2 ^ 28) / 2 ^ 29 * 2 ^ 29 * D E.
  Proof.
    intro f. assert (K : (2 ^ 52 + N + 2 ^ 28) / 2 ^ 29 = 2 ^ 23 + carry N).
    { unfold carry. replace (2 ^ 52 + N + 2 ^ 28) with (N + 2 ^ 28 + 2 ^ 23 * 2 ^ 29) by lia.
      rewrite Z.div_add by lia. lia. }
    rewrite K. pose proof carry_range. unfold G, v32, FO.value, D. subst f.
    destruct shape as [[Hc [A B]]|[Hc [A B]]]; rewrite A, B; (destruct (_ =? 0) eqn:Z0; [apply Z.eqb_eq in Z0; lia|]).
    - repeat split; try lia.
      assert (P : 2 ^ (E - 896 - 1) * 2 ^ 925 = 2 ^ 29 * 2 ^ (E - 1)) by (rewrite <- !Z.pow_add_r by lia; f_equal; lia).
      rewrite <- Z.mul_assoc, P. ring.
    - repeat split; try lia.
      assert (P : 2 ^ (E - 895 - 1) * 2 ^ 925 = 2 * 2 ^ 29 * 2 ^ (E - 1)).
      { rewrite <- (Z.pow_1_r 2) at 3. rewrite <- !Z.pow_add_r by lia. f_equal. lia. }
      rewrite Hc, <- Z.mul_assoc, P. ring.
  Qed.

  Lemma G_eq : G E N = (2 ^ 23 + carry N) * 2 ^ 29 * D E.
  Proof.
    destruct the_rounding_is_a_normal_float32 as [_ [_ H]]. rewrite H. f_equal. f_equal.
    unfold carry. replace (2 ^ 52 + N + 2 ^ 28) with (N + 2 ^ 28 + 2 ^ 23 * 2 ^ 29) by lia.
    rewrite Z.div_add by lia. lia.
  Qed.

  Lemma V_eq : v64 E N = (2 ^ 52 + N) * D E.
  Proof.
    unfold v64, FO.value, D. destruct (E =? 0) eqn:Z0; [apply Z.eqb_eq in Z0; lia | reflexivity].
  Qed.

  Lemma dist_eq : dist (FO.bits 52 E N) = Z.abs (N mod 2 ^ 29 - 2 ^ 28).
  Proof.
    unfold dist, FO.bits. rewrite land_low. f_equal. f_equal.
    replace (E * 2 ^ 52 + N) with (N + (E * 2 ^ 23) * 2 ^ 29) by lia. apply Z_mod_plus_full.
  Qed.

  Theorem the_gap_is_the_guard_distance :
    Z.abs (v64 E N - G E N) = (2 ^ 28 - dist (FO.bits 52 E N)) * D E.
  Proof.
    rewrite V_eq, G_eq, dist_eq. pose proof D_pos.
    pose proof (Z.div_mod N (2 ^ 29) ltac:(lia)) as DM. pose proof (Z.mod_pos_bound N (2 ^ 29) ltac:(lia)) as MB.
    set (a := N / 2 ^ 29) in *. set (r := N mod 2 ^ 29) in *.
    assert (C : carry N = a \/ carry N = a + 1).
    { unfold carry. destruct (Z_lt_le_dec r (2 ^ 28)).
      - left. symmetry. apply Z.div_unique with (r + 2 ^ 28); lia.
      - right. symmetry. apply Z.div_unique with (r - 2 ^ 28); lia. }
    replace ((2 ^ 52 + N) * D E - (2 ^ 23 + carry N) * 2 ^ 29 * D E) with ((N - 2 ^ 29 * carry N) * D E) by ring.
    rewrite Z.abs_mul, (Z.abs_eq (D E)) by lia. f_equal.
    assert (C2 : (carry N = a /\ r < 2 ^ 28) \/ (carry N = a + 1 /\ 2 ^ 28 <= r)).
    { unfold carry. destruct (Z_lt_le_dec r (2 ^ 28)).
      - left. split; [|lia]. symmetry. apply Z.div_unique with (r + 2 ^ 28); lia.
      - right. split; [|lia]. symmetry. apply Z.div_unique with (r - 2 ^ 28); lia. }
    destruct C2 as [[-> L]|[-> L]]; lia.
  Qed.

  (** The float32 just above the code's, and every one past it. *)
  Lemma above : forall e' m', FO.finite 23 e' m' ->
    f32 (FO.bits 52 E N) < FO.bits 23 e' m' -> G E N + 2 ^ 29 * D E <= v32 e' m'.
  Proof.
    intros e' m' F' Lt. destruct f32_fields as [Hf Ff]. cbv zeta in Hf, Ff.
    set (f := f32 (FO.bits 52 E N)) in *. set (e := f / 2 ^ 23) in *. set (m := f mod 2 ^ 23) in *.
    assert (He : 1 <= e) by (destruct the_rounding_is_a_normal_float32 as [X _]; exact (proj1 X)).
    destruct Ff as [_ Hm].
    set (e2 := if m + 1 <? 2 ^ 23 then e else e + 1).
    set (m2 := if m + 1 <? 2 ^ 23 then m + 1 else 0).
    assert (B2 : FO.bits 23 e2 m2 = f + 1) by (unfold e2, m2, FO.bits in *; destruct (m + 1 <? 2 ^ 23) eqn:X; [|apply Z.ltb_ge in X]; lia).
    assert (F2 : FO.finite 23 e2 m2) by (unfold e2, m2, FO.finite; destruct (m + 1 <? 2 ^ 23) eqn:X; [apply Z.ltb_lt in X|]; lia).
    assert (V2 : FO.value 23 e2 m2 = FO.value 23 e m + 2 ^ (e - 1)).
    { unfold e2, m2, FO.value. destruct (e =? 0) eqn:Z0; [apply Z.eqb_eq in Z0; lia|].
      destruct (m + 1 <? 2 ^ 23) eqn:X.
      - rewrite Z0. ring.
      - apply Z.ltb_ge in X. destruct (e + 1 =? 0) eqn:Z1; [apply Z.eqb_eq in Z1; lia|].
        replace (e + 1 - 1) with ((e - 1) + 1) by lia. rewrite Z.pow_add_r by lia.
        assert (m = 2 ^ 23 - 1) by lia. subst m. lia. }
    assert (Ord : FO.value 23 e2 m2 <= FO.value 23 e' m')
      by (apply (FO.the_bits_are_ordered_as_the_values 23); [lia | exact F2 | exact F' | lia]).
    assert (Step : 2 ^ 29 * D E <= 2 ^ (e - 1) * 2 ^ 925).
    { unfold D. destruct shape as [[_ [A _]]|[_ [A _]]]; fold f in A; fold e in A; rewrite A.
      - rewrite <- !Z.pow_add_r by lia. apply Z.eq_le_incl. f_equal. lia.
      - rewrite <- !Z.pow_add_r by lia. apply Z.pow_le_mono_r; lia. }
    unfold G, v32. fold f. fold e m. nia.
  Qed.

  (** The float32 just below the code's, and every one before it: a full
      float32 step below, or half of one where the code's float32 is the
      binade's first and q is not below it. *)
  Lemma below : forall e' m', FO.finite 23 e' m' ->
    FO.bits 23 e' m' < f32 (FO.bits 52 E N) ->
    v32 e' m' <= G E N - 2 ^ 29 * D E \/ (v32 e' m' <= G E N - 2 ^ 28 * D E /\ G E N <= v64 E N).
  Proof.
    intros e' m' F' Lt. destruct f32_fields as [Hf Ff]. cbv zeta in Hf, Ff.
    pose proof G_eq as GE. pose proof V_eq as VE. pose proof D_pos.
    set (f := f32 (FO.bits 52 E N)) in *. set (e := f / 2 ^ 23) in *. set (m := f mod 2 ^ 23) in *.
    assert (He : 1 <= e) by (destruct the_rounding_is_a_normal_float32 as [X _]; exact (proj1 X)).
    destruct Ff as [_ Hm].
    set (e3 := if 1 <=? m then e else e - 1).
    set (m3 := if 1 <=? m then m - 1 else 2 ^ 23 - 1).
    assert (B3 : FO.bits 23 e3 m3 = f - 1) by (unfold e3, m3, FO.bits in *; destruct (1 <=? m) eqn:X; [|apply Z.leb_gt in X]; lia).
    assert (F3 : FO.finite 23 e3 m3) by (unfold e3, m3, FO.finite; destruct (1 <=? m) eqn:X; [apply Z.leb_le in X|]; lia).
    assert (Ord : FO.value 23 e' m' <= FO.value 23 e3 m3)
      by (apply (FO.the_bits_are_ordered_as_the_values 23); [lia | exact F' | exact F3 | lia]).
    unfold G, v32 in *. fold f e m in GE |- *.
    assert (Ve : FO.value 23 e m = (2 ^ 23 + m) * 2 ^ (e - 1))
      by (unfold FO.value; destruct (e =? 0) eqn:Z0; [apply Z.eqb_eq in Z0; lia | reflexivity]).
    destruct (1 <=? m) eqn:X.
    - (* a full step below, in the same binade *)
      apply Z.leb_le in X. left.
      assert (V3 : FO.value 23 e3 m3 = FO.value 23 e m - 2 ^ (e - 1)).
      { unfold e3, m3. unfold FO.value. destruct (e =? 0) eqn:Z0; [apply Z.eqb_eq in Z0; lia|]. ring. }
      assert (Step : 2 ^ 29 * D E <= 2 ^ (e - 1) * 2 ^ 925).
      { unfold D. destruct shape as [[_ [A _]]|[_ [A _]]]; fold f e in A; rewrite A;
        rewrite <- !Z.pow_add_r by lia; apply Z.pow_le_mono_r; lia. }
      nia.
    - apply Z.leb_gt in X. assert (M0 : m = 0) by lia.
      destruct shape as [[Hc [A B]]|[Hc [A B]]]; fold f e m in A, B.
      + (* no carry: the binade's first float32; the one below is half a step down, or a full one at e = 1 *)
        destruct (Z.eq_dec e 1) as [E1|E1].
        * left. assert (V3 : FO.value 23 e3 m3 = 2 ^ 23 - 1) by (unfold e3, m3; rewrite E1; reflexivity).
          assert (E = 897) by lia. subst E.
          rewrite Ve, E1 in *. unfold D in *. replace (897 - 1) with 896 in * by lia.
          assert (P : 2 ^ 925 = 2 ^ 29 * 2 ^ 896) by (rewrite <- Z.pow_add_r by lia; reflexivity).
          rewrite P in *. nia.
        * right. split.
          -- assert (V3 : FO.value 23 e3 m3 = (2 ^ 24 - 1) * 2 ^ (e - 2)).
             { unfold e3, m3. unfold FO.value. destruct (e - 1 =? 0) eqn:Z0; [apply Z.eqb_eq in Z0; lia|].
               replace (e - 1 - 1) with (e - 2) by lia. ring. }
             assert (Ve2 : FO.value 23 e m = 2 ^ 24 * 2 ^ (e - 2)).
             { rewrite Ve, M0. replace (e - 1) with ((e - 2) + 1) by lia. rewrite Z.pow_add_r by lia. ring. }
             assert (Step : 2 ^ (e - 2) * 2 ^ 925 = 2 ^ 28 * D E).
             { unfold D. rewrite A. rewrite <- !Z.pow_add_r by lia. f_equal. lia. }
             nia.
          -- rewrite GE, VE. assert (carry N = 0) by lia. nia.
      + (* carry: the next binade's first float32; the one below is a full step of q's binade down *)
        left. assert (V3 : FO.value 23 e3 m3 = (2 ^ 24 - 1) * 2 ^ (e - 2)).
        { unfold e3, m3. unfold FO.value. destruct (e - 1 =? 0) eqn:Z0; [apply Z.eqb_eq in Z0; lia|].
          replace (e - 1 - 1) with (e - 2) by lia. ring. }
        assert (Ve2 : FO.value 23 e m = 2 ^ 24 * 2 ^ (e - 2)).
        { rewrite Ve, M0. replace (e - 1) with ((e - 2) + 1) by lia. rewrite Z.pow_add_r by lia. ring. }
        assert (Step : 2 ^ (e - 2) * 2 ^ 925 = 2 ^ 29 * D E).
        { unfold D. rewrite A. rewrite <- !Z.pow_add_r by lia. f_equal. lia. }
        nia.
  Qed.

  Theorem the_kernel_float32_is_the_nearest_to_every_value_within_four_ulps :
    3 < dist (FO.bits 52 E N) ->
    forall v, Z.abs (v - v64 E N) < 4 * D E ->
    forall e' m', FO.finite 23 e' m' -> FO.bits 23 e' m' <> f32 (FO.bits 52 E N) ->
    Z.abs (v - G E N) < Z.abs (v - v32 e' m').
  Proof.
    intros Hd v Hv e' m' F' Ne. pose proof the_gap_is_the_guard_distance as Gap. pose proof D_pos.
    assert (Gap4 : Z.abs (v64 E N - G E N) <= (2 ^ 28 - 4) * D E) by (rewrite Gap; apply Z.mul_le_mono_nonneg_r; lia).
    destruct (Z_lt_le_dec (f32 (FO.bits 52 E N)) (FO.bits 23 e' m')) as [Lt|Le].
    - pose proof (above e' m' F' Lt). lia.
    - assert (Lt : FO.bits 23 e' m' < f32 (FO.bits 52 E N)) by lia.
      destruct (below e' m' F' Lt) as [B|[B Gv]]; lia.
  Qed.
End Round.

(** ** The rounding error of the multiply, over the rationals. *)
Section ErrorBound.
  Open Scope Q_scope.
  Variables t d1 d2 d3 u w q p : Q.
  Hypothesis Hu0 : 0 <= u.
  Hypothesis Hu : u * 9007199254740992 <= 1.                       (* u <= 2^-53 *)
  Hypothesis H1 : -u <= d1 <= u.
  Hypothesis H2 : -u <= d2 <= u.
  Hypothesis H3 : -u <= d3 <= u.
  Hypothesis Ht : 0 < t.
  Hypothesis Hq : q == t * (1 + d1) * (1 + d2).                    (* RN64(x * RN64(1/r)) *)
  Hypothesis Hp : p == t * (1 + d3).                               (* RN64(x / r) *)
  Hypothesis Hw : q < 9007199254740992 * w.                        (* q below 2^53 ulp64(q) *)

  Theorem the_multiply_is_within_three_ulps_of_the_divide :
    - (30001 # 10000) * w < q - p < (30001 # 10000) * w.
  Proof.
    assert (Hd : - (3 * u + u * u) <= d1 + d2 + d1 * d2 - d3 <= 3 * u + u * u) by (split; nra).
    assert (Hqp : q - p == t * (d1 + d2 + d1 * d2 - d3)) by (rewrite Hq, Hp; ring).
    assert (Hf : (1 - u) * (1 - u) <= (1 + d1) * (1 + d2)) by nra.
    assert (Hlow : t * ((1 - u) * (1 - u)) <= q) by (rewrite Hq; nra).
    assert (Hu1 : u < 1 # 2) by lra.
    assert (Hpos : 0 < (1 - u) * (1 - u)) by nra.
    assert (Hq0 : 0 < q) by nra.
    assert (Hw0 : 0 < w) by lra.
    assert (Hs : 0 <= 3 * u + u * u) by nra.
    assert (Hb : t * (3 * u + u * u) * ((1 - u) * (1 - u)) <= q * (3 * u + u * u)) by nra.
    assert (Hc1 : q * (3 * u + u * u) <= 9007199254740992 * w * (3 * u + u * u)) by nra.
    assert (X : 0 <= w * (3 + u) * (1 - u * 9007199254740992)).
    { apply Qmult_le_0_compat; [apply Qmult_le_0_compat|]; lra. }
    assert (Hc2 : 9007199254740992 * w * (3 * u + u * u) <= w * (3 + u)) by nra.
    assert (Y : 3 + u < (30001 # 10000) * ((1 - u) * (1 - u))) by nra.
    assert (Hk : w * (3 + u) < (30001 # 10000) * ((1 - u) * (1 - u)) * w).
    { rewrite (Qmult_comm w). apply Qmult_lt_r; [exact Hw0 | exact Y]. }
    assert (He : t * (3 * u + u * u) < (30001 # 10000) * w).
    { apply (Qmult_lt_r _ _ ((1 - u) * (1 - u)) Hpos).
      apply (Qle_lt_trans _ (w * (3 + u))); [lra|].
      setoid_replace ((30001 # 10000) * w * ((1 - u) * (1 - u))) with ((30001 # 10000) * ((1 - u) * (1 - u)) * w) by ring.
      exact Hk. }
    assert (Tl : t * (d1 + d2 + d1 * d2 - d3) <= t * (3 * u + u * u)) by nra.
    assert (Tg : - (t * (3 * u + u * u)) <= t * (d1 + d2 + d1 * d2 - d3)) by nra.
    rewrite Hqp. split; lra.
  Qed.
End ErrorBound.

Definition u64 : Q := 1 # 9007199254740992.

Theorem the_multiply_rounds_as_the_divide : forall E N P (t d1 d2 d3 : Q),
  897 <= E <= 1149 -> 0 <= N < 2 ^ 52 -> 3 < dist (FO.bits 52 E N) ->
  (0 < t)%Q -> (- u64 <= d1 <= u64)%Q -> (- u64 <= d2 <= u64)%Q -> (- u64 <= d3 <= u64)%Q ->
  (inject_Z (v64 E N) == t * (1 + d1) * (1 + d2))%Q ->
  (inject_Z P == t * (1 + d3))%Q ->
  forall e' m', FO.finite 23 e' m' -> FO.bits 23 e' m' <> f32 (FO.bits 52 E N) ->
  Z.abs (v64 E N - G E N) < Z.abs (v64 E N - v32 e' m') /\
  Z.abs (P - G E N) < Z.abs (P - v32 e' m').
Proof.
  intros E N P t d1 d2 d3 HE HN Hd Ht H1 H2 H3 Hq Hp e' m' F' Ne.
  pose proof (D_pos E HE) as HD.
  assert (VE : v64 E N = (2 ^ 52 + N) * D E) by (apply V_eq; assumption).
  assert (Hw : (inject_Z (v64 E N) < 9007199254740992 * inject_Z (D E))%Q).
  { rewrite <- (inject_Z_mult 9007199254740992). rewrite <- Zlt_Qlt. rewrite VE. nia. }
  destruct (the_multiply_is_within_three_ulps_of_the_divide t d1 d2 d3 u64 (inject_Z (D E)) (inject_Z (v64 E N)) (inject_Z P))
    as [L R]; try assumption; try (unfold u64; lra).
  assert (HDq : (0 < inject_Z (D E))%Q) by (change 0%Q with (inject_Z 0); rewrite <- Zlt_Qlt; lia).
  assert (Gp : Z.abs (P - v64 E N) < 4 * D E).
  { assert (A : (inject_Z (P - v64 E N) < inject_Z (4 * D E))%Q).
    { unfold Z.sub. rewrite inject_Z_plus, inject_Z_opp, inject_Z_mult. change (inject_Z 4) with (4 # 1). lra. }
    assert (B : (inject_Z (v64 E N - P) < inject_Z (4 * D E))%Q).
    { unfold Z.sub. rewrite inject_Z_plus, inject_Z_opp, inject_Z_mult. change (inject_Z 4) with (4 # 1). lra. }
    rewrite <- Zlt_Qlt in A, B. lia. }
  split.
  - apply the_kernel_float32_is_the_nearest_to_every_value_within_four_ulps; try assumption.
    replace (v64 E N - v64 E N) with 0 by ring. lia.
  - apply the_kernel_float32_is_the_nearest_to_every_value_within_four_ulps; assumption.
Qed.

(** ** The guard. *)

Lemma tiny_spec : forall b, 0 <= b < 2 ^ 63 -> (tiny b = true <-> 0 < b < 897 * 2 ^ 52).
Proof.
  intros b Hb. unfold tiny. rewrite Z.ltb_lt.
  replace 0x380FFFFFFFFFFFFF with (897 * 2 ^ 52 - 1) by reflexivity. destruct (Z.eq_dec b 0) as [->|Hne].
  - replace ((0 - 1) mod 2 ^ 64) with (2 ^ 64 - 1) by reflexivity. lia.
  - rewrite Z.mod_small by lia. lia.
Qed.

Theorem the_guard_admits_only_zero_and_the_normal_binades : forall b, 0 <= b < 2 ^ 63 ->
  unsafe b = false ->
  b = 0 \/ (897 <= b / 2 ^ 52 <= 1149 /\ 0 <= b mod 2 ^ 52 < 2 ^ 52 /\
            b = FO.bits 52 (b / 2 ^ 52) (b mod 2 ^ 52) /\ 3 < dist b).
Proof.
  intros b Hb Hu. unfold unsafe in Hu. apply Bool.orb_false_iff in Hu as [Hu Hbig].
  apply Bool.orb_false_iff in Hu as [Ht Hd]. apply Z.leb_gt in Hd. unfold big in Hbig. apply Z.leb_gt in Hbig.
  replace 0x47E0000000000000 with (1150 * 2 ^ 52) in Hbig by reflexivity.
  destruct (Z.eq_dec b 0) as [->|Hne]; [left; reflexivity | right].
  assert (T : ~ (0 < b < 897 * 2 ^ 52)) by (rewrite <- tiny_spec by exact Hb; congruence).
  pose proof (Z.div_mod b (2 ^ 52) ltac:(lia)). pose proof (Z.mod_pos_bound b (2 ^ 52) ltac:(lia)).
  unfold FO.bits. repeat split; lia.
Qed.

Theorem zero_passes_and_rounds_to_zero : dist 0 = 2 ^ 28 /\ unsafe 0 = false /\ f32 0 = 0.
Proof. vm_compute. repeat split; reflexivity. Qed.

(** ** The other bit formulas. *)

Theorem a_normal_x_widens_by_its_bits : forall e m, 1 <= e <= 254 -> 0 <= m < 2 ^ 23 ->
  Z.shiftl (FO.bits 23 e m) 29 + 4035225266123964416 = FO.bits 52 (e + 896) (m * 2 ^ 29) /\
  FO.finite 52 (e + 896) (m * 2 ^ 29) /\ e + 896 <= 2046 /\
  v64 (e + 896) (m * 2 ^ 29) = v32 e m.
Proof.
  intros e m He Hm. unfold FO.bits, FO.finite, v64, v32, FO.value.
  rewrite Z.shiftl_mul_pow2 by lia.
  destruct (e + 896 =? 0) eqn:Z0; [apply Z.eqb_eq in Z0; lia|]. destruct (e =? 0) eqn:Z1; [apply Z.eqb_eq in Z1; lia|].
  repeat split; try lia.
  assert (P : 2 ^ (e + 896 - 1) = 2 ^ (e - 1) * 2 ^ 896) by (rewrite <- Z.pow_add_r by lia; f_equal; lia).
  assert (Q : 2 ^ 925 = 2 ^ 29 * 2 ^ 896) by (rewrite <- Z.pow_add_r by lia; reflexivity).
  rewrite P, Q. replace (2 ^ 52) with (2 ^ 23 * 2 ^ 29) by reflexivity. ring.
Qed.

Theorem the_scale_is_a_power_of_two : forall e, 2 <= e <= 508 ->
  FO.finite 52 (e + 691) 0 /\ e + 691 <= 2046 /\ v64 (e + 691) 0 = 2 ^ (e + 742).
Proof.
  intros e He. unfold FO.finite, v64, FO.value.
  destruct (e + 691 =? 0) eqn:Z0; [apply Z.eqb_eq in Z0; lia|].
  repeat split; try lia. rewrite Z.add_0_r, <- Z.pow_add_r by lia. f_equal. lia.
Qed.

(** ** Refutations. *)

Definition Nt : Z := 2 ^ 28 + 3.

Theorem a_guard_of_two_would_miss_a_tie :
  dist (FO.bits 52 1023 Nt) = 3 /\
  Z.abs (v64 1023 Nt - (v64 1023 Nt - 3 * D 1023)) < 4 * D 1023 /\
  Z.abs ((v64 1023 Nt - 3 * D 1023) - G 1023 Nt) =
    Z.abs ((v64 1023 Nt - 3 * D 1023) - (G 1023 Nt - 2 ^ 29 * D 1023)) /\
  Z.odd (f32 (FO.bits 52 1023 Nt)) = true.
Proof. vm_compute. repeat split; reflexivity. Qed.

Theorem a_subnormal_x_does_not_widen_by_its_bits :
  v64 0 (1 * 2 ^ 29) <> v32 0 1 /\ v64 896 (1 * 2 ^ 29) <> v32 0 1.
Proof. vm_compute. split; discriminate. Qed.

Print Assumptions the_sign_does_not_reach_the_guard_bits.
Print Assumptions the_rounding_is_a_normal_float32.
Print Assumptions the_gap_is_the_guard_distance.
Print Assumptions the_kernel_float32_is_the_nearest_to_every_value_within_four_ulps.
Print Assumptions the_multiply_is_within_three_ulps_of_the_divide.
Print Assumptions the_multiply_rounds_as_the_divide.
Print Assumptions the_guard_admits_only_zero_and_the_normal_binades.
Print Assumptions zero_passes_and_rounds_to_zero.
Print Assumptions a_normal_x_widens_by_its_bits.
Print Assumptions the_scale_is_a_power_of_two.
Print Assumptions a_guard_of_two_would_miss_a_tie.
Print Assumptions a_subnormal_x_does_not_widen_by_its_bits.
