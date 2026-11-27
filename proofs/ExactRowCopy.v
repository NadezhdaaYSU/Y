(** * A piece of one row runs as two (`python/y_lang/exact/lora.py`).

    Since 2026-11-26 compiled GPU training runs a piece of one row as two
    rows, the row and a copy of it that no target reads
    (`SFT._accumulate`), so that PyTorch never compiles the size 1 as its
    own case. The claim is that every result for the real row is the one
    the row alone gives. This file proves it for the three kinds of
    operation a step is made of, on rows of integers (a float's |x| bits are
    ordered as |x| is for finite x, which is how `exponent` reads a maximum):

    - Row-wise operations, each output row a function of its own input row
      (norms, attention within a sequence, a row's quantizer, the products
      of rows with a weight): the real row's output is unchanged
      ([rowwise_keeps_the_rows]).
    - A maximum over the rows, the scale of a quantizer along the token
      dimension: the copy equals a row already there, so the maximum is the
      same ([a_copy_keeps_the_max]), and so is every quantized value, a
      function of its element and that maximum only
      ([a_copy_keeps_the_quantized]).
    - An integer sum over the rows of products with a gradient, the
      adapters' gradients: the copy has no target, so its gradient row is
      0, and the sum is the same ([a_zero_row_keeps_the_sum]).

    ** Refutations, the conditions being load-bearing.

    [a_length_scale_does_not_survive]: a quantizer whose scale depends on
    the number of rows as well as their maximum, as `exact_sum`'s
    s = e - 61 + ceil_log2(n) does, gives another value when a row is
    added (ceil_log2 1 = 0, ceil_log2 2 = 1). [a_different_copy_moves_the_max]:
    a second row that is not a copy (a row of zeros' token ids makes other
    activations) can raise the maximum.

    ** Not proved here.

    That the step is made only of these three kinds over its rows: that no
    `exact_sum`, nor anything else whose result depends on the row count,
    runs over the rows of a training piece. That is a fact about the code,
    and it is checked, not proved: the tiny-model test
    (`test_a_piece_of_one_row_compiles_nothing_new`) compares the adapters
    and every p(target) with the CPU, which runs the row alone, and fails
    with a row of zeros as the copy; the T4 run of 2026-11-26 had the
    record's bits at step 5, whose last piece is one row.

    Checked with Rocq 9.1:  coqc -Q . "" ExactRowCopy.v  *)

From Stdlib Require Import ZArith Lia List.
Import ListNotations.
Open Scope Z_scope.

(* ------------------------------------------------------------------ *)
(** ** Row-wise operations                                             *)
(* ------------------------------------------------------------------ *)

Theorem rowwise_keeps_the_rows : forall {A B : Type} (f : A -> B) (rows : list A) (c : A),
  firstn (length rows) (map f (rows ++ [c])) = map f rows.
Proof.
  intros A B f rows c. rewrite map_app.
  rewrite firstn_app, length_map, Nat.sub_diag, firstn_O, app_nil_r.
  apply firstn_all2. rewrite length_map. lia.
Qed.

(* ------------------------------------------------------------------ *)
(** ** A maximum over the rows                                         *)
(* ------------------------------------------------------------------ *)

(** The maximum of non-negative values (|x| bits), 0 for none, as `amax`
    of a slice with a 0 start reads it. *)
Definition maxl (xs : list Z) : Z := fold_right Z.max 0 xs.

Lemma maxl_nonneg : forall xs, 0 <= maxl xs.
Proof. induction xs as [|x xs IH]; unfold maxl in *; cbn [fold_right]; lia. Qed.

Lemma maxl_app : forall xs ys, maxl (xs ++ ys) = Z.max (maxl xs) (maxl ys).
Proof.
  induction xs as [|x xs IH]; intro ys.
  - cbn [app]. pose proof (maxl_nonneg ys). unfold maxl at 2. cbn [fold_right]. lia.
  - cbn [app]. unfold maxl in *. cbn [fold_right]. rewrite IH. lia.
Qed.

Lemma maxl_ge : forall xs c, In c xs -> c <= maxl xs.
Proof.
  induction xs as [|x xs IH]; intros c H; [destruct H|].
  cbn [maxl fold_right]. destruct H as [<-|H]; [lia|].
  specialize (IH c H). unfold maxl in IH. lia.
Qed.

Theorem a_copy_keeps_the_max : forall xs c, In c xs -> maxl (xs ++ [c]) = maxl xs.
Proof.
  intros xs c H. rewrite maxl_app. pose proof (maxl_ge xs c H). pose proof (maxl_nonneg xs).
  change (maxl [c]) with (Z.max c 0). lia.
Qed.

(** A quantizer along the rows: each element's result a function [q] of
    the element and the slice's maximum (q8's scale is the max's exponent). *)
Theorem a_copy_keeps_the_quantized : forall (q : Z -> Z -> Z) xs c, In c xs ->
  firstn (length xs) (map (fun x => q x (maxl (xs ++ [c]))) (xs ++ [c]))
  = map (fun x => q x (maxl xs)) xs.
Proof.
  intros q xs c H. rewrite (a_copy_keeps_the_max xs c H).
  apply (rowwise_keeps_the_rows (fun x => q x (maxl xs))).
Qed.

(* ------------------------------------------------------------------ *)
(** ** Integer sums over the rows                                      *)
(* ------------------------------------------------------------------ *)

(** sum_t x_t * g_t, the adapters' gradient over a piece's tokens. *)
Fixpoint dot (xs gs : list Z) : Z :=
  match xs, gs with
  | x :: xs', g :: gs' => x * g + dot xs' gs'
  | _, _ => 0
  end.

Lemma dot_app : forall xs gs ys hs, length xs = length gs ->
  dot (xs ++ ys) (gs ++ hs) = dot xs gs + dot ys hs.
Proof.
  induction xs as [|x xs IH]; intros gs ys hs L; destruct gs as [|g gs]; cbn in L; try lia.
  - reflexivity.
  - cbn [app dot]. rewrite IH by lia. lia.
Qed.

Theorem a_zero_row_keeps_the_sum : forall xs gs c, length xs = length gs ->
  dot (xs ++ [c]) (gs ++ [0]) = dot xs gs.
Proof. intros xs gs c L. rewrite dot_app by exact L. cbn [dot]. lia. Qed.

(* ------------------------------------------------------------------ *)
(** ** Refutations                                                     *)
(* ------------------------------------------------------------------ *)

(** `exact_sum`'s rounding of a term x: x on the grid 2^s with
    s = e - 61 + ceil_log2 n (a right shift, floor division here). *)
Definition length_scaled (e n x : Z) : Z := x / 2 ^ (e - 61 + Z.log2_up n).

Theorem a_length_scale_does_not_survive :
  length_scaled 61 1 1 <> length_scaled 61 2 1.
Proof. unfold length_scaled. cbv. discriminate. Qed.

Theorem a_different_copy_moves_the_max :
  exists xs c, maxl (xs ++ [c]) <> maxl xs.
Proof. exists [1], 5. cbv. discriminate. Qed.

Print Assumptions rowwise_keeps_the_rows.
Print Assumptions a_copy_keeps_the_max.
Print Assumptions a_copy_keeps_the_quantized.
Print Assumptions a_zero_row_keeps_the_sum.
Print Assumptions a_length_scale_does_not_survive.
Print Assumptions a_different_copy_moves_the_max.
