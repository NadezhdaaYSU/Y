(** * GRPO's pieces of a batch (`python/y_lang/exact/rl.py`).

    `rl.pieces(full, chunk_tokens)` cuts a batch's rows into pieces of
    consecutive rows for `GRPO.step`, each piece at most `chunk_tokens`
    tokens counted as its rows times its longest row (`full[i]`: row i's
    prompt plus completion). It walks the rows once: a row starts a new piece
    when the current one is not empty and would pass the bound with it.

    A piece's gradients are added to the step's in order, so a row dropped or
    run twice would change the update silently, and a piece past the bound
    is the one a GPU's memory was sized against. This file proves, for every
    batch and bound:

    [the_pieces_are_the_rows_in_order]: the pieces, joined, are the rows
    0 .. n - 1 in order, so every row is in exactly one piece and each piece
    is consecutive rows. [no_piece_is_empty]: for n >= 1. [every_piece_of_two_or_more_rows_is_within_the_bound]:
    a piece of two or more rows is within `chunk_tokens`; only a single row
    longer than the bound by itself makes a piece past it.
    [no_bound_is_one_piece]: with `chunk_tokens` None, one piece.

    ** Refutation.

    [a_single_long_row_passes_the_bound]: one row of 5 tokens with a bound of
    4 is a piece of its own past the bound: the bound is on pieces of two or
    more rows, which is all the code promises.

    ** Not proved here.

    That `GRPO.step` runs every piece once in this order (it iterates over
    the returned list), and the pieces' training itself.

    Checked with Rocq 9.1:  coqc -Q . "" ExactPieces.v  *)

From Stdlib Require Import ZArith Lia List Bool.
Import ListNotations.
Open Scope Z_scope.

Section Pieces.
  Variable full : nat -> Z.               (** each row's length *)
  Variable bound : option Z.              (** chunk_tokens, None for no bound *)

  Definition maxl (xs : list Z) : Z := fold_right Z.max 0 xs.

  (** rows times the longest row *)
  Definition cost (piece : list nat) : Z := Z.of_nat (length piece) * maxl (map full piece).

  Definition splits (cur : list nat) (i : nat) : bool :=
    match cur, bound with
    | [], _ => false
    | _, None => false
    | _, Some c => c <? cost (cur ++ [i])
    end.

  (** The loop from the current piece `cur` over the remaining rows. *)
  Fixpoint walk (rows : list nat) (cur : list nat) : list (list nat) :=
    match rows with
    | [] => [cur]
    | i :: t => if splits cur i then cur :: walk t [i] else walk t (cur ++ [i])
    end.

  Definition pieces (n : nat) : list (list nat) := walk (seq 0 n) [].

  Lemma walk_concat : forall rows cur, concat (walk rows cur) = cur ++ rows.
  Proof.
    induction rows as [|i t IH]; intro cur; cbn [walk].
    - cbn. rewrite !app_nil_r. reflexivity.
    - destruct (splits cur i); cbn [concat]; rewrite IH; [reflexivity | rewrite <- app_assoc; reflexivity].
  Qed.

  Theorem the_pieces_are_the_rows_in_order : forall n, concat (pieces n) = seq 0 n.
  Proof. intro n. unfold pieces. rewrite walk_concat. reflexivity. Qed.

  Lemma walk_nonempty : forall rows cur, (cur <> [] \/ rows <> []) ->
    Forall (fun p => p <> []) (walk rows cur).
  Proof.
    induction rows as [|i t IH]; intros cur H; cbn [walk].
    - constructor; [destruct H as [H|H]; [exact H | congruence] | constructor].
    - destruct (splits cur i) eqn:S.
      + constructor.
        * destruct cur; [discriminate | congruence].
        * apply IH. left. discriminate.
      + apply IH. left. destruct cur; discriminate.
  Qed.

  Theorem no_piece_is_empty : forall n, (1 <= n)%nat -> Forall (fun p => p <> []) (pieces n).
  Proof.
    intros n Hn. unfold pieces. apply walk_nonempty. right.
    destruct n as [|m]; [lia | discriminate].
  Qed.

  Definition within (p : list nat) : Prop :=
    (2 <= length p)%nat -> forall c, bound = Some c -> cost p <= c.

  Lemma walk_within : forall rows cur, within cur -> Forall within (walk rows cur).
  Proof.
    induction rows as [|i t IH]; intros cur Hcur; cbn [walk].
    - constructor; [exact Hcur | constructor].
    - destruct (splits cur i) eqn:S.
      + constructor; [exact Hcur|]. apply IH. intros L. cbn in L. lia.
      + apply IH. intros L c Hc. unfold splits in S. destruct cur as [|x xs].
        * cbn in L. lia.
        * rewrite Hc in S. apply Z.ltb_ge in S. exact S.
  Qed.

  Theorem every_piece_of_two_or_more_rows_is_within_the_bound : forall n,
    Forall within (pieces n).
  Proof. intro n. unfold pieces. apply walk_within. intro L. cbn in L. lia. Qed.

  Lemma walk_unbounded : bound = None -> forall rows cur, walk rows cur = [cur ++ rows].
  Proof.
    intro HN. induction rows as [|i t IH]; intro cur; cbn [walk].
    - rewrite app_nil_r. reflexivity.
    - assert (splits cur i = false) as -> by (unfold splits; destruct cur; [reflexivity | rewrite HN; reflexivity]).
      rewrite IH, <- app_assoc. reflexivity.
  Qed.

  Theorem no_bound_is_one_piece : bound = None -> forall n, pieces n = [seq 0 n].
  Proof. intros HN n. unfold pieces. rewrite (walk_unbounded HN). reflexivity. Qed.
End Pieces.

Definition five (i : nat) : Z := 5.

Theorem a_single_long_row_passes_the_bound :
  pieces five (Some 4) 1 = [[0%nat]] /\ cost five [0%nat] > 4.
Proof. split; vm_compute; reflexivity. Qed.

Print Assumptions the_pieces_are_the_rows_in_order.
Print Assumptions no_piece_is_empty.
Print Assumptions every_piece_of_two_or_more_rows_is_within_the_bound.
Print Assumptions no_bound_is_one_piece.
Print Assumptions a_single_long_row_passes_the_bound.
