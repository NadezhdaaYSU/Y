(** * Padding and splitting an exact product (`python/y_lang/exact/lora.py`).

    `lora.mm_y_any` runs `a @ b^T` for integer a [M][K], b [N][K] (entries
    in [-127, 127]) on Y's int8 GEMM, which takes only some shapes: it pads K
    to a multiple of 32 and N to a multiple of 8 with zeros, pads a's rows to
    the kernel's row bucket when compiled, slices the result back to
    [M][N], and for K past `Y_MAX_K` = 131,040 runs pieces of K whose int32
    results it adds in int64. `ExactLinear` pads its rows to a multiple of
    32 for `torch._int_mm` the same way, and since 2026-11-25 compiled code
    pads every length, a pad of 0 included. The code's reasons are comments:
    "a zero adds nothing", "rows of a matmul are independent". This file
    proves them.

    [zero_padding_k_changes_no_sum]: entries past K that are 0 in either
    operand add nothing, however many there are.
    [a_padded_row_leaves_the_real_rows] and [a_padded_column_leaves_the_real_columns]:
    each output (m, n) reads row m of a and row n of b only, so extra rows
    of either change no output the slice keeps.
    [the_pieces_of_k_are_the_whole_sum]: pieces of `Y_MAX_K` (the last one
    short, masked) add up to the sum over K, for every K.
    [each_piece_fits_int32]: a piece's sum fits int32, and
    [the_int64_total_holds]: the total fits int64 for any K below 2^43.
    Refutation [a_piece_one_term_longer_can_overflow]: 133,145 terms of
    127 * 127 leave int32, so `Y_MAX_K` is load-bearing.

    ** Not proved here.

    The GEMM itself (`Int8GemmExact.v` for Y's kernel, cuBLAS taken as an
    exact int32 product); that the code's pads are zeros (`torch.nn.functional.pad`
    pads with 0) and its slices the leading [M][N].

    Checked with Rocq 9.1:  coqc -Q . "" ExactMmPadding.v  *)

From Stdlib Require Import ZArith Lia.
Require ExactMlpKernel.
Open Scope Z_scope.

Module MK := ExactMlpKernel.

(** The K-sum of row m of a and row n of b. *)
Definition dot (K : nat) (a b : nat -> nat -> Z) (m n : nat) : Z :=
  MK.sumto K (fun k => a m k * b n k).

Theorem zero_padding_k_changes_no_sum : forall K e a b a' b' m n,
  (forall k, (k < K)%nat -> a' m k = a m k /\ b' n k = b n k) ->
  (forall k, (K <= k)%nat -> a' m k = 0 \/ b' n k = 0) ->
  dot (K + e) a' b' m n = dot K a b m n.
Proof.
  intros K e a b a' b' m n Hin Hout. unfold dot. rewrite MK.sumto_add.
  assert (Z0 : MK.sumto e (fun j => a' m (K + j)%nat * b' n (K + j)%nat) = 0).
  { induction e as [|e IH]; cbn [MK.sumto]; [reflexivity|].
    rewrite IH. destruct (Hout (K + e)%nat ltac:(lia)) as [->| ->]; ring. }
  rewrite Z0, Z.add_0_r. apply MK.sumto_ext. intros k Hk. destruct (Hin k Hk) as [-> ->]. reflexivity.
Qed.

Theorem a_padded_row_leaves_the_real_rows : forall K a a' b m n,
  (forall k, a' m k = a m k) -> dot K a' b m n = dot K a b m n.
Proof. intros K a a' b m n H. unfold dot. apply MK.sumto_ext. intros k _. rewrite H. reflexivity. Qed.

Theorem a_padded_column_leaves_the_real_columns : forall K a b b' m n,
  (forall k, b' n k = b n k) -> dot K a b' m n = dot K a b m n.
Proof. intros K a b b' m n H. unfold dot. apply MK.sumto_ext. intros k _. rewrite H. reflexivity. Qed.

Definition Y_MAX_K : nat := 131040.

(** The code's loop `for k0 in range(0, K, Y_MAX_K)` over pieces, each
    piece's terms past K masked to 0. *)
Theorem the_pieces_of_k_are_the_whole_sum : forall K (f : nat -> Z) (Y : nat), (0 < Y)%nat ->
  MK.sumto ((K + Y - 1) / Y) (fun c => MK.sumto Y (fun x => MK.masked K f (c * Y + x)%nat)) = MK.sumto K f.
Proof. intros K f Y H. apply MK.the_blocked_loop_is_the_dot, H. Qed.

Definition in_int32 (z : Z) : Prop := - 2 ^ 31 <= z <= 2 ^ 31 - 1.
Definition in_int64 (z : Z) : Prop := - 2 ^ 63 <= z <= 2 ^ 63 - 1.

Lemma sum_of_bounded : forall n f, (forall i, (i < n)%nat -> -16129 <= f i <= 16129) ->
  - (Z.of_nat n * 16129) <= MK.sumto n f <= Z.of_nat n * 16129.
Proof.
  induction n as [|n IH]; intros f H; cbn [MK.sumto]; [lia|].
  specialize (IH f (fun i Hi => H i ltac:(lia))). specialize (H n ltac:(lia)). rewrite Nat2Z.inj_succ. lia.
Qed.

Theorem each_piece_fits_int32 : forall f c,
  (forall i, -16129 <= f i <= 16129) ->
  in_int32 (MK.sumto Y_MAX_K (fun x => f (c * Y_MAX_K + x)%nat)).
Proof.
  intros f c H. pose proof (sum_of_bounded Y_MAX_K (fun x => f (c * Y_MAX_K + x)%nat) (fun i _ => H _)) as B.
  unfold in_int32. change (Z.of_nat Y_MAX_K) with 131040 in B. change (2 ^ 31) with 2147483648. lia.
Qed.

Theorem the_int64_total_holds : forall K f,
  Z.of_nat K < 2 ^ 43 -> (forall i, -16129 <= f i <= 16129) -> in_int64 (MK.sumto K f).
Proof.
  intros K f HK H. pose proof (sum_of_bounded K f (fun i _ => H i)) as B. unfold in_int64.
  change (2 ^ 63) with 9223372036854775808. change (2 ^ 43) with 8796093022208 in HK. nia.
Qed.

Theorem a_piece_one_term_longer_can_overflow : ~ in_int32 (133145 * (127 * 127)).
Proof. unfold in_int32. change (2 ^ 31) with 2147483648. lia. Qed.

Print Assumptions zero_padding_k_changes_no_sum.
Print Assumptions a_padded_row_leaves_the_real_rows.
Print Assumptions a_padded_column_leaves_the_real_columns.
Print Assumptions the_pieces_of_k_are_the_whole_sum.
Print Assumptions each_piece_fits_int32.
Print Assumptions the_int64_total_holds.
Print Assumptions a_piece_one_term_longer_can_overflow.
