(** * The attention kernel's row indices (`python/y_lang/exact/attn_weights_kernel.py`).

    `weights` flattens the scores [b, h, q, T] to rows r = (b NH + h) Q + q,
    the key scales [b, h, 1, T] to rows b NH + h and the mask [b, 1, q, T] to
    rows b Q + q, each row T long. `_attn_p_kernel` reads, for score row r,
    the key-scale row r / Q and the mask row (r / (NH Q)) Q + r mod Q, at
    element offsets row * T + c, with r and these rows in int32 and the
    offsets in int64.

    [the_key_scale_row_is_the_rows_head]: r / Q = b NH + h.
    [the_mask_row_is_the_rows_query]: (r / (NH Q)) Q + r mod Q = b Q + q.
    [every_row_is_one_query]: every r in 0 .. B NH Q - 1 is (b NH + h) Q + q
    for exactly one b < B, h < NH, q < Q, so the kernel's R = B NH Q rows are
    the queries once each.
    [the_offsets_stay_in_the_tensors]: for r < B NH Q and c < T the three
    offsets are below the tensors' sizes B NH Q T, B NH T and B Q T.
    [the_int32_rows_stay_below_r]: both row indices are at most r, so with
    R below 2^31 no int32 index wraps.

    ** Refutations.

    [the_key_scale_formula_reads_the_wrong_mask_row]: the mask row is not
    r / Q (at NH = 2, Q = 3 the query (0, 1, 0) reads mask row 1, not 0),
    nor r mod Q (at batch 1 it reads batch 0's row).

    ** Not proved here.

    That Triton's `//` and `%` on nonnegative int32 are Z.div and Z.modulo,
    and that `torch.reshape` of a contiguous [b, h, q, T] tensor is this
    row order.

    Checked with Rocq 9.1:  coqc -Q . "" ExactAttnRows.v  *)

From Stdlib Require Import ZArith Lia.
Open Scope Z_scope.

Definition row (NH Q b h q : Z) : Z := (b * NH + h) * Q + q.
Definition sk_row (Q r : Z) : Z := r / Q.
Definition mask_row (NH Q r : Z) : Z := (r / (NH * Q)) * Q + r mod Q.

Section Rows.
  Variables B NH Q T : Z.
  Hypothesis HNH : 0 < NH.
  Hypothesis HQ : 0 < Q.

  Lemma row_div_q : forall b h q, 0 <= h < NH -> 0 <= q < Q ->
    row NH Q b h q / Q = b * NH + h /\ row NH Q b h q mod Q = q.
  Proof.
    intros b h q Hh Hq. unfold row. split.
    - rewrite Z.div_add_l by lia. rewrite Z.div_small by lia. lia.
    - rewrite Z.add_comm, Z.mod_add by lia. apply Z.mod_small. lia.
  Qed.

  Theorem the_key_scale_row_is_the_rows_head : forall b h q,
    0 <= h < NH -> 0 <= q < Q -> sk_row Q (row NH Q b h q) = b * NH + h.
  Proof. intros. apply row_div_q; assumption. Qed.

  Theorem the_mask_row_is_the_rows_query : forall b h q,
    0 <= h < NH -> 0 <= q < Q -> mask_row NH Q (row NH Q b h q) = b * Q + q.
  Proof.
    intros b h q Hh Hq. unfold mask_row.
    destruct (row_div_q b h q Hh Hq) as [_ Hm]. rewrite Hm.
    replace (row NH Q b h q) with (b * (NH * Q) + (h * Q + q)) by (unfold row; ring).
    rewrite Z.div_add_l by lia. rewrite (Z.div_small (h * Q + q)) by nia. lia.
  Qed.

  Theorem every_row_is_one_query : forall r, 0 <= r < B * NH * Q ->
    (exists b h q, 0 <= b < B /\ 0 <= h < NH /\ 0 <= q < Q /\ r = row NH Q b h q) /\
    (forall b h q b' h' q', 0 <= h < NH -> 0 <= q < Q -> 0 <= h' < NH -> 0 <= q' < Q ->
       row NH Q b h q = row NH Q b' h' q' -> b = b' /\ h = h' /\ q = q').
  Proof.
    intros r Hr. split.
    - exists (r / (NH * Q)), ((r / Q) mod NH), (r mod Q).
      assert (Hdq : r / Q / NH = r / (NH * Q)) by (rewrite Z.div_div by lia; f_equal; lia).
      pose proof (Z.mod_pos_bound (r / Q) NH HNH). pose proof (Z.mod_pos_bound r Q HQ).
      split; [split|split; [lia|split; [lia|]]].
      + apply Z.div_pos; nia.
      + apply Z.div_lt_upper_bound; nia.
      + unfold row. rewrite <- Hdq.
        rewrite (Z.mul_comm (r / Q / NH) NH), <- Z.div_mod by lia.
        rewrite Z.mul_comm, <- Z.div_mod by lia. reflexivity.
    - intros b h q b' h' q' Hh Hq Hh' Hq' E.
      destruct (row_div_q b h q Hh Hq) as [D M]. destruct (row_div_q b' h' q' Hh' Hq') as [D' M'].
      rewrite E in D, M. rewrite D in D'. rewrite M in M'.
      assert (Eb : (b * NH + h) / NH = (b' * NH + h') / NH) by (rewrite D'; reflexivity).
      rewrite !Z.div_add_l in Eb by lia. rewrite !Z.div_small in Eb by lia.
      assert (b = b') by lia. subst b'. split; [reflexivity|split; lia].
  Qed.

  Theorem the_offsets_stay_in_the_tensors : forall r c,
    0 <= r < B * NH * Q -> 0 <= c < T ->
    0 <= r * T + c < B * NH * Q * T /\
    0 <= sk_row Q r * T + c < B * NH * T /\
    0 <= mask_row NH Q r * T + c < B * Q * T.
  Proof.
    intros r c Hr Hc.
    destruct (proj1 (every_row_is_one_query r Hr)) as [b [h [q [Hb [Hh [Hq ->]]]]]].
    rewrite the_key_scale_row_is_the_rows_head, the_mask_row_is_the_rows_query by assumption.
    unfold row in *.
    assert (L : forall x n y m, 0 <= x < n -> 0 <= y < m -> 0 <= x * m + y < n * m) by (intros; nia).
    assert (H1 : 0 <= b * NH + h < B * NH) by (apply L; lia).
    assert (H2 : 0 <= b * Q + q < B * Q) by (apply L; lia).
    split; [|split]; apply L; try lia; apply L; lia.
  Qed.

  Theorem the_int32_rows_stay_below_r : forall r, 0 <= r ->
    0 <= sk_row Q r <= r /\ 0 <= mask_row NH Q r <= r.
  Proof.
    intros r Hr. unfold sk_row, mask_row. split.
    - split; [apply Z.div_pos; lia|apply Z.div_le_upper_bound; nia].
    - set (D := r / (NH * Q)). set (E := r mod (NH * Q)).
      assert (Hr' : r = NH * Q * D + E) by (apply Z.div_mod; lia).
      assert (HE : 0 <= E < NH * Q) by (apply Z.mod_pos_bound; nia).
      assert (HD : 0 <= D) by (apply Z.div_pos; nia).
      assert (Hm : r mod Q = E mod Q).
      { rewrite Hr'. replace (NH * Q * D + E) with (E + NH * D * Q) by ring.
        apply Z.mod_add. lia. }
      rewrite Hm. pose proof (Z.mod_le E Q (proj1 HE) HQ). pose proof (Z.mod_pos_bound E Q HQ).
      assert (HQD : 0 <= Q * D) by nia.
      assert (Hle : D * Q <= NH * Q * D).
      { replace (NH * Q * D) with (NH * (Q * D)) by ring.
        replace (D * Q) with (1 * (Q * D)) by ring.
        apply Z.mul_le_mono_nonneg_r; lia. }
      split; [nia|lia].
  Qed.
End Rows.

(** The refutation: the other two natural formulas read other rows. *)
Theorem the_key_scale_formula_reads_the_wrong_mask_row :
  row 2 3 0 1 0 / 3 = 1 /\ mask_row 2 3 (row 2 3 0 1 0) = 0 /\
  row 2 3 1 0 2 mod 3 = 2 /\ mask_row 2 3 (row 2 3 1 0 2) = 5.
Proof. vm_compute. repeat split. Qed.

Print Assumptions the_key_scale_row_is_the_rows_head.
Print Assumptions the_mask_row_is_the_rows_query.
Print Assumptions every_row_is_one_query.
Print Assumptions the_offsets_stay_in_the_tensors.
Print Assumptions the_int32_rows_stay_below_r.
Print Assumptions the_key_scale_formula_reads_the_wrong_mask_row.
