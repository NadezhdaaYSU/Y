(** * `int8_bmm`'s K bound (`python/y_lang/exact/int8_bmm.py`).

    Exact training's attention products run on tensor cores as int8 dot
    products with an int32 accumulator, in whatever order the hardware adds.
    The module refuses K past `MAX_K = (2^31 - 1) // (128 * 128)` = 131,071
    and says that within it c is the integer sums whatever the order. The
    operands are any int8, so -128 * -128 = 2^14 is a product; the bound uses
    128 * 128, not 127 * 127.

    [an_int8_product_is_at_most_2_14]: |a b| <= 2^14 for a, b in -128 .. 127.
    [max_k_is_the_largest_safe_k]: 131,071 is (2^31 - 1) / 2^14 and
    131,071 * 2^14 <= 2^31 - 1 < 131,072 * 2^14.
    [every_order_gives_the_int32_sum]: for K <= MAX_K, every schedule of the
    K products in ExactReduce's int32 carrier (two's complement wrap) gives
    the integer dot product (`ExactReduce.any_schedule_gives_the_sum`).

    ** Refutations.

    [one_more_term_wraps]: at K = 131,072 the products of -128 and -128 sum
    to 2^31, and int32's last addition wraps to -2^31.
    [the_127_bound_would_admit_a_wrap]: a bound from 127 * 127 (133,144)
    admits K = 131,072, where that wrap happens.

    ** Not proved here.

    That Triton's `tl.dot` with an int32 accumulator is such a schedule of
    exact int8 products (the tests compare it with the float32 product).

    Checked with Rocq 9.1:  coqc -Q . "" ExactInt8BmmBound.v  *)

From Stdlib Require Import ZArith Lia List.
Import ListNotations.
Require ExactReduce ExactGemmMicro.
Open Scope Z_scope.

Module ER := ExactReduce.
Module MC := ExactGemmMicro.

Definition int8 (x : Z) : Prop := -128 <= x <= 127.
Definition MAX_K : Z := (2 ^ 31 - 1) / (128 * 128).

Theorem an_int8_product_is_at_most_2_14 : forall a b, int8 a -> int8 b -> Z.abs (a * b) <= 2 ^ 14.
Proof. intros a b Ha Hb. unfold int8 in *. rewrite Z.abs_mul. change (2 ^ 14) with (128 * 128). nia. Qed.

Theorem max_k_is_the_largest_safe_k :
  MAX_K = 131071 /\ MAX_K * 2 ^ 14 <= 2 ^ 31 - 1 < (MAX_K + 1) * 2 ^ 14.
Proof. vm_compute. repeat split; try reflexivity; discriminate. Qed.

Definition products (a b : list Z) : list Z := map (fun p => fst p * snd p) (combine a b).

Lemma abs_sum_products : forall a b, Forall int8 a -> Forall int8 b ->
  ER.abs_sum (products a b) <= Z.of_nat (length (combine a b)) * 2 ^ 14.
Proof.
  induction a as [|x a IH]; intros b Fa Fb; [cbn; lia|].
  destruct b as [|y b]; [cbn; lia|].
  inversion Fa as [|? ? Hx Ha]; inversion Fb as [|? ? Hy Hb]; subst.
  unfold products, ER.abs_sum in *. cbn [combine map ER.zsum length].
  specialize (IH b Ha Hb). pose proof (an_int8_product_is_at_most_2_14 x y Hx Hy). cbn [fst snd]. lia.
Qed.

Theorem every_order_gives_the_int32_sum : forall a b t,
  Forall int8 a -> Forall int8 b -> length a = length b -> Z.of_nat (length a) <= MAX_K ->
  ER.covers t (length (products a b)) ->
  ER.teval (ER.cadd ER.I32) (products a b) t = ER.zsum (products a b).
Proof.
  intros a b t Fa Fb L K C. apply (ER.any_schedule_gives_the_sum MC.I32MAX).
  - apply (ER.every_carrier_is_exact_within_its_cap ER.I32).
  - exact C.
  - pose proof (abs_sum_products a b Fa Fb) as B. rewrite length_combine, L, Nat.min_id in B.
    rewrite <- L in B. destruct max_k_is_the_largest_safe_k as [_ [M _]].
    unfold MC.I32MAX. nia.
Qed.

Theorem one_more_term_wraps :
  (MAX_K + 1) * ((-128) * (-128)) = 2 ^ 31 /\
  ER.cadd ER.I32 (MAX_K * ((-128) * (-128))) ((-128) * (-128)) = - 2 ^ 31.
Proof. vm_compute. split; reflexivity. Qed.

Theorem the_127_bound_would_admit_a_wrap :
  (2 ^ 31 - 1) / (127 * 127) = 133144 /\ MAX_K + 1 <= 133144.
Proof. vm_compute. split; [reflexivity | discriminate]. Qed.

Print Assumptions an_int8_product_is_at_most_2_14.
Print Assumptions max_k_is_the_largest_safe_k.
Print Assumptions every_order_gives_the_int32_sum.
Print Assumptions one_more_term_wraps.
Print Assumptions the_127_bound_would_admit_a_wrap.
