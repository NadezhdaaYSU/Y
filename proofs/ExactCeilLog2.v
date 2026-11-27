(** * `_ceil_log2`'s two definitions agree (`python/y_lang/exact/train_lm.py`).

    `exact_sum` puts n terms on the grid 2^s with s = e - 61 + ceil_log2(n),
    so its bits depend on ceil_log2. `_ceil_log2(n)` computes it two ways:
    eagerly `(n - 1).bit_length()`, and under torch.compile, where n is
    symbolic and `bit_length` would fix it to one value, by comparisons:
    `k = 0; while (1 << k) < n: k += 1`. A compiled step and an eager one (and
    the CPU) must get the same s, so the two must be the same function. This
    file proves they are, for every n >= 1, and that both are the least k with
    2^k >= n (`Z.log2_up`). (An n <= 0 takes the eager form even when
    compiling, so only n >= 1 reaches the loop.)

    [the_loop_is_the_least_power] and [bit_length_is_the_least_power]: each
    definition is `Z.log2_up n`. [the_two_definitions_agree]: hence equal.
    [the_loop_stops_within_n_steps]: the loop takes at most n steps, so the
    fuel here (n) is enough and the code's loop ends.

    ** Refutation.

    [floor_log2_is_not_it]: floor(log2 n), the other natural reading of
    "log2", differs at n = 3 (1 against 2), and a scale one too small lets
    the int64 sum overflow (`ExactTrainArith.past_the_int64_cap_the_sum_wraps`).

    ** Not proved here.

    That Python's `int.bit_length` is the binary length (its specification)
    and that torch.compile traces the loop's comparisons faithfully (the
    compiled-against-eager tests compare every call of a step).

    Checked with Rocq 9.1:  coqc -Q . "" ExactCeilLog2.v  *)

From Stdlib Require Import ZArith Lia.
Open Scope Z_scope.

(** `while (1 << k) < n: k += 1`, from k, with fuel. *)
Fixpoint loop (fuel : nat) (n k : Z) : Z :=
  match fuel with
  | O => k
  | S f => if 2 ^ k <? n then loop f n (k + 1) else k
  end.

(** Python's `int.bit_length` of m >= 0: 0 for 0, else floor(log2 m) + 1. *)
Definition bit_length (m : Z) : Z := if m =? 0 then 0 else Z.log2 m + 1.

(** From k at most the answer, with fuel for the distance, the loop stops at
    the least k with 2^k >= n. *)
Lemma loop_spec : forall fuel n k, 0 < n -> k <= Z.log2_up n ->
  Z.log2_up n - k <= Z.of_nat fuel -> loop fuel n k = Z.log2_up n.
Proof.
  induction fuel as [|f IH]; intros n k Hn Hk Hf; cbn [loop].
  - lia.
  - destruct (2 ^ k <? n) eqn:E.
    + apply Z.ltb_lt in E. apply (Z.log2_up_lt_pow2 n k Hn) in E.
      apply IH; [exact Hn | lia | lia].
    + apply Z.ltb_ge in E. apply (Z.log2_up_le_pow2 n k Hn) in E. lia.
Qed.

Theorem the_loop_stops_within_n_steps : forall n, 1 <= n -> Z.log2_up n <= n.
Proof.
  intros n Hn. apply Z.log2_up_le_pow2; [lia|].
  apply Z.lt_le_incl, Z.pow_gt_lin_r; lia.
Qed.

Theorem the_loop_is_the_least_power : forall n, 1 <= n -> loop (Z.to_nat n) n 0 = Z.log2_up n.
Proof.
  intros n Hn. apply loop_spec; [lia | apply Z.log2_up_nonneg |].
  pose proof (the_loop_stops_within_n_steps n Hn). rewrite Z2Nat.id by lia. lia.
Qed.

Theorem bit_length_is_the_least_power : forall n, 1 <= n -> bit_length (n - 1) = Z.log2_up n.
Proof.
  intros n Hn. unfold bit_length. destruct (Z.eq_dec n 1) as [->|Hne].
  - reflexivity.
  - destruct (n - 1 =? 0) eqn:E; [apply Z.eqb_eq in E; lia|].
    rewrite Z.log2_up_eqn by lia. rewrite <- Z.sub_1_r. lia.
Qed.

Theorem the_two_definitions_agree : forall n, 1 <= n -> loop (Z.to_nat n) n 0 = bit_length (n - 1).
Proof.
  intros n Hn. rewrite the_loop_is_the_least_power, bit_length_is_the_least_power by exact Hn. reflexivity.
Qed.

Theorem floor_log2_is_not_it : Z.log2 3 <> Z.log2_up 3.
Proof. vm_compute. discriminate. Qed.

Print Assumptions the_loop_stops_within_n_steps.
Print Assumptions the_loop_is_the_least_power.
Print Assumptions bit_length_is_the_least_power.
Print Assumptions the_two_definitions_agree.
Print Assumptions floor_log2_is_not_it.
