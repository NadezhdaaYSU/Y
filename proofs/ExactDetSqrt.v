(** * Adam's deterministic square root (`python/y_lang/exact/train_lm.py`).

    Exact training's Adam divides by sqrt(v), and a float sqrt need not be
    the same on every device's libm, so `det_sqrt` builds it from integers:
    k = frexp's exponent - 61, made even; n = v * 2^-k, an integer;
    s = `isqrt(n)`; the result s * 2^(k/2). `isqrt` takes any device's float
    estimate x0 of sqrt(n) and corrects it twice:

        x = x - (x * x > n);  x = x + ((x + 1) * (x + 1) <= n)

    [one_correction_moves_one_step]: a correction moves x one step toward
    floor(sqrt n), or keeps it there. [isqrt_is_the_floor_square_root]: so
    two give floor(sqrt n) for every estimate within 2 of it (the docstring
    promises within 1), on every device, and for n < 2^62 no product leaves
    int64.
    [the_scaled_value_is_an_integer_in_2_60_2_62]: for a normal float64 v at
    least 2^-962 (the docstring's range), n is (2^52 + M) * 2^8 or * 2^9,
    in [2^60, 2^62), v = n * 2^k exactly, and k is even.
    [det_sqrt_is_within_2_minus_30]: s^2 <= n < (s + 1)^2 with s >= 2^30,
    so sqrt(v) = sqrt(n) * 2^(k/2) lies in [r, r * (1 + 2^-30)) for the
    result r = s * 2^(k/2): the docstring's 2^-30 relative bound.

    ** Refutation.

    [one_correction_is_not_enough_at_two]: an estimate 2 above the answer
    (n = 100, x0 = 12) is still 11 after one correction; the code's two
    corrections are what tolerate it.

    ** Not proved here.

    That the float estimate `torch.sqrt(float64(n))` truncated is within 2
    of floor(sqrt n) (a correctly rounded sqrt of n rounded to float64 is
    within 2^-21 of sqrt n for n < 2^62); that s * 2^(k/2) is a float64
    multiply by a normal power of two (exact).

    Checked with Rocq 9.1:  coqc -Q . "" ExactDetSqrt.v  *)

From Stdlib Require Import ZArith Lia.
Require ExactFloatOrder.
Open Scope Z_scope.

Module FO := ExactFloatOrder.

(** One round of `isqrt`'s corrections. *)
Definition correct (n x : Z) : Z :=
  let x1 := x - (if n <? x * x then 1 else 0) in
  x1 + (if (x1 + 1) * (x1 + 1) <=? n then 1 else 0).

Definition isqrt (n x0 : Z) : Z := correct n (correct n x0).

Theorem one_correction_moves_one_step : forall n x, 0 <= n -> 0 <= x ->
  let r := Z.sqrt n in
  correct n x = (if x <? r then x + 1 else if r <? x then x - 1 else x).
Proof.
  intros n x Hn Hx r. destruct (Z.sqrt_spec n Hn) as [L U]. fold r in L, U.
  assert (Hr : 0 <= r) by (apply Z.sqrt_nonneg).
  unfold correct.
  destruct (Z.lt_trichotomy x r) as [Lt|[Eq|Gt]].
  - assert (A : (n <? x * x) = false) by (apply Z.ltb_ge; nia).
    rewrite A, Z.sub_0_r.
    assert (B : ((x + 1) * (x + 1) <=? n) = true) by (apply Z.leb_le; nia).
    rewrite B. destruct (x <? r) eqn:C; [reflexivity | apply Z.ltb_ge in C; lia].
  - subst x. assert (A : (n <? r * r) = false) by (apply Z.ltb_ge; nia).
    rewrite A, Z.sub_0_r.
    assert (B : ((r + 1) * (r + 1) <=? n) = false) by (apply Z.leb_gt; nia).
    rewrite B, Z.ltb_irrefl. ring.
  - assert (A : (n <? x * x) = true) by (apply Z.ltb_lt; nia).
    rewrite A.
    assert (B : ((x - 1 + 1) * (x - 1 + 1) <=? n) = false) by (apply Z.leb_gt; nia).
    rewrite B. destruct (x <? r) eqn:C; [apply Z.ltb_lt in C; lia|].
    destruct (r <? x) eqn:D; [ring | apply Z.ltb_ge in D; lia].
Qed.

Theorem isqrt_is_the_floor_square_root : forall n x0, 0 <= n < 2 ^ 62 -> 0 <= x0 ->
  Z.abs (x0 - Z.sqrt n) <= 2 ->
  isqrt n x0 = Z.sqrt n /\ (x0 + 1) * (x0 + 1) < 2 ^ 63.
Proof.
  intros n x0 Hn Hx Hd. set (r := Z.sqrt n).
  assert (Hr : r < 2 ^ 31).
  { unfold r. apply Z.sqrt_lt_square; [lia | apply Z.pow_nonneg; lia |].
    change (2 ^ 31 * 2 ^ 31) with (2 ^ 62). lia. }
  assert (Hr0 : 0 <= r) by (apply Z.sqrt_nonneg).
  split; [|nia].
  unfold isqrt. rewrite (one_correction_moves_one_step n x0) by lia. fold r.
  destruct (x0 <? r) eqn:A; [apply Z.ltb_lt in A | apply Z.ltb_ge in A].
  - rewrite one_correction_moves_one_step by lia. fold r.
    destruct (x0 + 1 <? r) eqn:B; [apply Z.ltb_lt in B; lia | apply Z.ltb_ge in B].
    destruct (r <? x0 + 1) eqn:C; [apply Z.ltb_lt in C; lia | apply Z.ltb_ge in C; lia].
  - destruct (r <? x0) eqn:B; [apply Z.ltb_lt in B | apply Z.ltb_ge in B].
    + rewrite one_correction_moves_one_step by lia. fold r.
      destruct (x0 - 1 <? r) eqn:C; [apply Z.ltb_lt in C; lia | apply Z.ltb_ge in C].
      destruct (r <? x0 - 1) eqn:D; [apply Z.ltb_lt in D; lia | apply Z.ltb_ge in D; lia].
    + rewrite one_correction_moves_one_step by lia. fold r.
      rewrite (proj2 (Z.ltb_ge x0 r)) by lia. rewrite (proj2 (Z.ltb_ge r x0)) by lia. lia.
Qed.

Theorem one_correction_is_not_enough_at_two : correct 100 12 = 11 /\ Z.sqrt 100 = 10.
Proof. vm_compute. split; reflexivity. Qed.

(** ** The scaling *)

Section Scale.
  Variables field M : Z.
  Hypothesis Hf : 61 <= field <= 2046.        (* normal, v >= 2^-962 *)
  Hypothesis HM : 0 <= M < 2 ^ 52.

  Definition k0 : Z := field - 1022 - 61.
  Definition k : Z := k0 - Z.land k0 1.
  Definition n : Z := (2 ^ 52 + M) * 2 ^ (field - 1075 - k).

  Lemma land1 : Z.land k0 1 = k0 mod 2.
  Proof. replace 1 with (Z.ones 1) by reflexivity. rewrite Z.land_ones by lia. reflexivity. Qed.

  Theorem the_scaled_value_is_an_integer_in_2_60_2_62 :
    (field - 1075 - k = 8 \/ field - 1075 - k = 9) /\ 2 ^ 60 <= n < 2 ^ 62 /\
    Z.even k = true /\ - 1023 <= k /\
    FO.value 52 field M = n * 2 ^ (k + 1074).
  Proof.
    pose proof (Z.mod_pos_bound k0 2 ltac:(lia)) as MB. pose proof (Z.div_mod k0 2 ltac:(lia)) as DM.
    assert (Ek : k = k0 - k0 mod 2) by (unfold k; rewrite land1; reflexivity).
    assert (Hx : field - 1075 - k = 8 \/ field - 1075 - k = 9) by (unfold k0 in *; lia).
    split; [exact Hx|]. split.
    - unfold n. destruct Hx as [E|E]; rewrite E.
      + change (2 ^ 60) with (2 ^ 52 * 2 ^ 8). change (2 ^ 62) with (2 ^ 54 * 2 ^ 8).
        assert (2 ^ 52 + M < 2 ^ 54) by (change (2 ^ 54) with (4 * 2 ^ 52); lia). nia.
      + change (2 ^ 60) with (2 ^ 51 * 2 ^ 9). change (2 ^ 62) with (2 ^ 53 * 2 ^ 9).
        assert (2 ^ 51 <= 2 ^ 52 + M < 2 ^ 53) by (change (2 ^ 53) with (2 * 2 ^ 52); change (2 ^ 52) with (2 * 2 ^ 51) at 1; lia). nia.
    - split; [|split].
      + rewrite Ek. rewrite DM at 1. rewrite Z.add_simpl_r, Z.even_mul. reflexivity.
      + unfold k0 in *. lia.
      + unfold FO.value, n. destruct (field =? 0) eqn:Z0; [apply Z.eqb_eq in Z0; lia|].
        rewrite <- Z.mul_assoc, <- Z.pow_add_r by (unfold k0 in *; lia). f_equal. f_equal. lia.
  Qed.

  Theorem det_sqrt_is_within_2_minus_30 : forall x0, 0 <= x0 -> Z.abs (x0 - Z.sqrt n) <= 2 ->
    let s := isqrt n x0 in
    s * s <= n < (s + 1) * (s + 1) /\ 2 ^ 30 <= s < 2 ^ 31.
  Proof.
    intros x0 Hx Hd s. destruct the_scaled_value_is_an_integer_in_2_60_2_62 as [_ [Hn _]].
    destruct (isqrt_is_the_floor_square_root n x0 ltac:(lia) Hx Hd) as [E _].
    unfold s. rewrite E. destruct (Z.sqrt_spec n ltac:(lia)) as [L U]. split; [lia|].
    set (r := Z.sqrt n) in *. assert (0 <= r) by (apply Z.sqrt_nonneg).
    change (2 ^ 60) with (2 ^ 30 * 2 ^ 30) in Hn. change (2 ^ 62) with (2 ^ 31 * 2 ^ 31) in Hn.
    split; nia.
  Qed.
End Scale.

Print Assumptions one_correction_moves_one_step.
Print Assumptions isqrt_is_the_floor_square_root.
Print Assumptions one_correction_is_not_enough_at_two.
Print Assumptions the_scaled_value_is_an_integer_in_2_60_2_62.
Print Assumptions det_sqrt_is_within_2_minus_30.
