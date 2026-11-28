(** * isqrt's float estimate (`python/y_lang/exact/train_lm.py`).

    `isqrt(n)` starts from `torch.sqrt(n.to(torch.float64)).to(torch.int64)`
    and corrects it twice. `ExactDetSqrt.v` proves the corrections give
    floor(sqrt n) from any estimate within 2 of it and leaves the estimate's
    distance to the float arithmetic. This file proves it, for a conversion
    and a square root that may each be off by a relative 2^-50 (8 ulps of
    float64): torch's CPU float64 sqrt is not always correctly rounded, so
    the bound does not assume it is.

    Model: x = n (1 + d0) is float64(n); s >= 0 with s^2 = x (1 + d1)^2 is
    the computed sqrt(x) (s = sqrt(x) (1 + d1)); k is s truncated
    (k <= s < k + 1, k >= 0, the int64 conversion of a nonnegative value).

    [the_error_factor_is_within_4_dl]: (1 + d0)(1 + d1)^2 is within 4 * 2^-50
    of 1. [the_truncated_estimate_is_within_1]: for 0 <= n < 2^62,
    floor(sqrt n) - 1 <= k <= floor(sqrt n) + 1: within 1, the docstring's
    promise, and so within the 2 that `isqrt_is_the_floor_square_root`
    takes.

    ** Refutation.

    [a_cruder_sqrt_misses_by_more]: a square root off by a relative 2^-20
    leaves the bound: at n = 2^62 - 1 (floor(sqrt n) = 2^31 - 1) such an
    s may reach 2^31 + 2^11 - 2, 2047 past the answer.

    ** Not proved here.

    That torch's float64 conversion and sqrt have these relative errors
    (IEEE gives 2^-53 for both; the CPU's sqrt is within a few ulps), and
    that `.to(torch.int64)` truncates.

    Checked with Rocq 9.1:  coqc -Q . "" ExactSqrtEstimate.v  *)

From Stdlib Require Import ZArith Lia QArith Lqa.
Open Scope Q_scope.

Definition dl : Q := 1 # 1125899906842624.     (* 2^-50 *)

Lemma prod_bounds : forall a b la ha lb hb, 0 <= la -> 0 <= lb -> la <= a <= ha -> lb <= b <= hb ->
  la * lb <= a * b <= ha * hb.
Proof. intros a b la ha lb hb H1 H2 [Ha1 Ha2] [Hb1 Hb2]. split; nra. Qed.

Lemma mul_le_l : forall z a b, 0 <= z -> a <= b -> z * a <= z * b.
Proof. intros. nra. Qed.

Theorem the_error_factor_is_within_4_dl : forall d0 d1, - dl <= d0 <= dl -> - dl <= d1 <= dl ->
  1 - 4 * dl <= (1 + d0) * ((1 + d1) * (1 + d1)) <= 1 + 4 * dl.
Proof.
  intros d0 d1 H0 H1.
  assert (B1 : 1 - dl <= 1 + d1 <= 1 + dl) by lra.
  assert (B0 : 1 - dl <= 1 + d0 <= 1 + dl) by lra.
  assert (P : 0 <= 1 - dl) by (unfold dl; lra).
  pose proof (prod_bounds _ _ _ _ _ _ P P B1 B1) as P1.
  assert (P2 : 0 <= (1 - dl) * (1 - dl)) by (apply Qmult_le_0_compat; exact P).
  pose proof (prod_bounds _ _ _ _ _ _ P P2 B0 P1) as P3.
  assert (C1 : 1 - 4 * dl <= (1 - dl) * ((1 - dl) * (1 - dl))) by (unfold dl; vm_compute; discriminate).
  assert (C2 : (1 + dl) * ((1 + dl) * (1 + dl)) <= 1 + 4 * dl) by (unfold dl; vm_compute; discriminate).
  lra.
Qed.

Section Estimate.
  Variable n : Z.
  Hypothesis Hn : (0 <= n < 2 ^ 62)%Z.
  Variables x s d0 d1 : Q.
  Hypothesis Hd0 : - dl <= d0 <= dl.
  Hypothesis Hd1 : - dl <= d1 <= dl.
  Hypothesis Hx : x == inject_Z n * (1 + d0).
  Hypothesis Hs0 : 0 <= s.
  Hypothesis Hs : s * s == x * ((1 + d1) * (1 + d1)).
  Variable k : Z.
  Hypothesis Hk0 : (0 <= k)%Z.
  Hypothesis Hk : inject_Z k <= s < inject_Z k + 1.

  Theorem the_truncated_estimate_is_within_1 : (Z.sqrt n - 1 <= k <= Z.sqrt n + 1)%Z.
  Proof.
    set (m := Z.sqrt n).
    destruct (Z.sqrt_spec n (proj1 Hn)) as [Lm Um]. fold m in Lm, Um.
    assert (Hm0 : (0 <= m)%Z) by apply Z.sqrt_nonneg.
    assert (Hm31 : (m + 1 <= 2147483648)%Z).
    { destruct (Z_le_gt_dec (m + 1) 2147483648) as [ok|big]; [exact ok|].
      assert (2147483648 * 2147483648 <= m * m)%Z by nia.
      assert (2 ^ 62 = 2147483648 * 2147483648)%Z by reflexivity. lia. }
    set (M := inject_Z m). set (K := inject_Z k). set (N := inject_Z n).
    assert (QLm : M * M <= N).
    { unfold M, N. rewrite <- inject_Z_mult, <- Zle_Qle. exact Lm. }
    assert (QUm : N <= M * M + 2 * M).
    { unfold M, N. assert (Z1 : (n <= m * m + 2 * m)%Z) by nia. rewrite Zle_Qle in Z1.
      rewrite inject_Z_plus, !inject_Z_mult in Z1. exact Z1. }
    assert (QM0 : 0 <= M) by (unfold M; change 0 with (inject_Z 0); rewrite <- Zle_Qle; exact Hm0).
    assert (QM31 : M + 1 <= 2147483648).
    { unfold M. change 2147483648 with (inject_Z 2147483648). change 1 with (inject_Z 1).
      rewrite <- inject_Z_plus, <- Zle_Qle. exact Hm31. }
    assert (QMM : M * M <= M * 2147483648)
      by (apply mul_le_l; lra).
    assert (QK0 : 0 <= K) by (unfold K; change 0 with (inject_Z 0); rewrite <- Zle_Qle; exact Hk0).
    pose proof (the_error_factor_is_within_4_dl d0 d1 Hd0 Hd1) as EB.
    set (E := (1 + d0) * ((1 + d1) * (1 + d1))) in EB.
    assert (SS : s * s == N * E) by (rewrite Hs, Hx; unfold N, E; ring).
    assert (QN0 : 0 <= N) by (pose proof (Qmult_le_0_compat M M QM0 QM0); lra).
    assert (SU : s * s <= N * (1 + 4 * dl)) by (rewrite SS; apply mul_le_l; lra).
    assert (SL : N * (1 - 4 * dl) <= s * s) by (rewrite SS; apply mul_le_l; lra).
    destruct Hk as [Hk1 Hk2]. fold K in Hk1, Hk2.
    pose proof (prod_bounds s s K s K s QK0 QK0 ltac:(lra) ltac:(lra)) as [KS _].
    assert (SK : s * s < (K + 1) * (K + 1)) by (clear - Hs0 Hk2; nra).
    clear SS EB Hs Hx E.
    split.
    - destruct (Z_le_gt_dec (m - 1) k) as [ok|low]; [exact ok|exfalso].
      assert (Zk : (k + 2 <= m)%Z) by lia.
      assert (QK : K + 1 <= M - 1).
      { assert (Q2 : K + 2 <= M).
        { unfold K, M. change 2 with (inject_Z 2). rewrite <- inject_Z_plus, <- Zle_Qle. exact Zk. }
        lra. }
      assert (P0 : 0 <= K + 1) by lra.
      pose proof (prod_bounds (K + 1) (K + 1) (K + 1) (M - 1) (K + 1) (M - 1) P0 P0 ltac:(lra) ltac:(lra)) as KM.
      assert (QM2 : 2 <= M) by lra.
      unfold dl in *. nra.
    - destruct (Z_le_gt_dec k (m + 1)) as [ok|high]; [exact ok|exfalso].
      assert (Zk : (m + 2 <= k)%Z) by lia.
      assert (QK : M + 2 <= K).
      { unfold K, M. change 2 with (inject_Z 2). rewrite <- inject_Z_plus, <- Zle_Qle. exact Zk. }
      assert (P0 : 0 <= M + 2) by lra.
      pose proof (prod_bounds (M + 2) (M + 2) (M + 2) K (M + 2) K P0 P0 ltac:(lra) ltac:(lra)) as KM.
      unfold dl in *. nra.
  Qed.
End Estimate.

(** The refutation: with a relative error of 2^-20 the sqrt of
    n = 2^62 - 1 may be (2^31 - 1)(1 + 2^-20) >= 2^31 + 2^11 - 2: past the
    answer 2^31 - 1 by 2047, so the corrections could not reach it. *)
Theorem a_cruder_sqrt_misses_by_more :
  Z.sqrt (2 ^ 62 - 1) = (2 ^ 31 - 1)%Z /\
  inject_Z (2 ^ 31 - 1) * (1 + (1 # 1048576)) >= inject_Z (2 ^ 31 + 2 ^ 11 - 2).
Proof. split; [vm_compute; reflexivity|vm_compute; discriminate]. Qed.

Print Assumptions the_error_factor_is_within_4_dl.
Print Assumptions the_truncated_estimate_is_within_1.
Print Assumptions a_cruder_sqrt_misses_by_more.
