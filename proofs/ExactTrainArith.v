(** * Exact training's arithmetic (`python/y_lang/exact/train_lm.py`).

    Exact training runs every sum as a sum of integers, so that no order of
    addition can change it (`ExactReduce`). Four pieces of integer arithmetic
    stand between the Python and that fact, and this file proves them.

    ** The digit split ([digits], [ndigits]).

    A 12-bit gradient meets an int8 GEMM as int8 digits: q = d0 + 128 d1 + ...
    with every d in [-64, 63], each digit the remainder of a floor division
    (`hi = (r + 64) >> 7; d = r - (hi << 7)`, [carry], [digit]).
    [digits_rebuild]: the digits and what is left after them give back q.
    [nothing_left_iff_in_reach]: nothing is left exactly when q lies in the
    range n digits reach, [-reach n 64, reach n 63], which is the condition the
    code refuses the rest of. [ndigits_hold] and [ndigits_is_the_fewest]:
    `ndigits` gives enough digits for every value of 9 to 64 bits, and no
    fewer would do. [the_old_count_was_one_short]: ceil((bits - 1) / 7),
    the count until 2026-11-21, leaves 1 over at 14 bits.

    ** The wide product ([product_of_values], [the_shifted_blocks_are_the_dot]).

    The stacked GEMM computes, for every digit pair (i, j), the K-term sum
    P i j = sum_k A k i * B k j of digit products, and the code adds
    128^(i+j) P i j over the pairs. That is the dot product of the values
    sum_k qa_k * qb_k. [the_shifted_blocks_are_the_dot] proves it;
    [every_order_of_the_blocks_is_exact] adds that, with the bound the code
    checks before anything runs (`K * capacity(a) * capacity(b) < 2^53`, here
    the carrier's cap), any order and association of those weighted blocks
    in int64 or float64 gives it exactly.

    ** The rounding shift ([shift_round]).

    `_round_shift` rounds mant / 2^r to the nearest integer, ties to even,
    with a shift, a mask and a comparison. [shift_round_is_nearest] and
    [nearest_even_unique]: it returns the one nearest-even integer.
    [a_clamped_shift_rounds_the_same]: the code clamps the shift (to 62 in
    int64), and for any mant below 2^53 a longer shift gives the same 0.
    [rounding_commutes_with_sign]: the code rounds |x| and puts the sign
    back, which is the nearest-even rounding of x.

    ** The fixed-point sums and the scales ([fixed_point_sums_are_exact],
    [product_scales_are_normal]).

    `Ops.sum` puts n <= 2^c terms on the grid 2^s with s = e - 61 + c, where
    every |x| < 2^e, so every rounded term is at most 2^(61-c) in magnitude
    ([a_rounded_term_is_bounded]) and any int64 order of their sum is exact.
    q8 refuses a scale past 2^511, so a product's scale 2^(sa + sb) is a
    normal float64.

    ** Refutations, the bounds being load-bearing.

    [the_old_count_was_one_short]; [round_half_up_differs_on_a_tie];
    [past_the_int64_cap_the_sum_wraps]; [a_scale_of_512_could_leave_float64].

    ** Not proved here.

    That the floats' bit fields are what [shift_round] is given, the
    elementwise IEEE operations, and the int8 GEMM itself
    (`Int8GemmExact.v`). That float32's `round(x * 2^a * 2^b)` is
    [shift_round] is in `ExactScaledRound.v` (checked also on 6.5M values
    and ties by python/tests/test_exact_train_lm.py).

    Checked with Rocq 9.1:  coqc -Q . "" ExactTrainArith.v  *)

From Stdlib Require Import ZArith Lia List Bool.
Require ExactReduce ExactPvExact.
Import ListNotations.
Open Scope Z_scope.

Module ER := ExactReduce.
Module EP := ExactPvExact.

(* ------------------------------------------------------------------ *)
(** ** The digit split                                                 *)
(* ------------------------------------------------------------------ *)

(** `hi = (r + 64) >> 7`: an arithmetic shift is a floor division. *)
Definition carry (r : Z) : Z := (r + 64) / 128.
(** `d = r - (hi << 7)`. *)
Definition digit (r : Z) : Z := r - 128 * carry r.

Lemma carry_spec : forall r, r + 64 = 128 * carry r + (r + 64) mod 128 /\
                             0 <= (r + 64) mod 128 < 128.
Proof.
  intro r. unfold carry. split.
  - apply Z.div_mod. lia.
  - apply Z.mod_pos_bound. lia.
Qed.

Theorem digit_range : forall r, -64 <= digit r <= 63.
Proof. intro r. unfold digit. destruct (carry_spec r). lia. Qed.

Fixpoint digits (n : nat) (r : Z) : list Z :=
  match n with
  | O => []
  | S m => digit r :: digits m (carry r)
  end.

(** What is left after [n] digits: the code refuses a value unless it is 0. *)
Fixpoint rest (n : nat) (r : Z) : Z :=
  match n with
  | O => r
  | S m => rest m (carry r)
  end.

(** sum_i d_i * 128^i *)
Fixpoint value (ds : list Z) : Z :=
  match ds with
  | [] => 0
  | d :: t => d + 128 * value t
  end.

Theorem digits_rebuild : forall n r,
  value (digits n r) + 128 ^ Z.of_nat n * rest n r = r.
Proof.
  induction n as [|n IH]; intro r.
  - cbn [digits rest value]. change (128 ^ Z.of_nat 0) with 1. lia.
  - cbn [digits rest value]. rewrite Nat2Z.inj_succ, Z.pow_succ_r by lia.
    specialize (IH (carry r)).
    set (v := value (digits n (carry r))) in *.
    set (P := 128 ^ Z.of_nat n) in *.
    set (R := rest n (carry r)) in *.
    assert (Ev : v = carry r - P * R) by lia.
    rewrite Ev. unfold digit. ring.
Qed.

(** [reach n d] = sum_{i<n} d * 128^i: with d = 63 the largest value n
    digits reach, with d = 64 the magnitude of the smallest. *)
Fixpoint reach (n : nat) (d : Z) : Z :=
  match n with
  | O => 0
  | S m => d + 128 * reach m d
  end.

Lemma reach_nonneg : forall n d, 0 <= d -> 0 <= reach n d.
Proof. induction n; intros d Hd; cbn [reach]; [lia|]. specialize (IHn d Hd). lia. Qed.

Lemma value_in_reach : forall n r,
  - reach n 64 <= value (digits n r) <= reach n 63.
Proof.
  induction n as [|n IH]; intro r; cbn [digits value reach]; [lia|].
  specialize (IH (carry r)). pose proof (digit_range r). lia.
Qed.

Lemma rest_zero_in_reach : forall n r,
  - reach n 64 <= r <= reach n 63 -> rest n r = 0.
Proof.
  induction n as [|n IH]; intros r H; cbn [rest reach] in *; [lia|].
  apply IH. destruct (carry_spec r). lia.
Qed.

(** Nothing is left exactly when the value is in reach of the digits: the
    digits hold every value in [-reach n 64, reach n 63] and the code's
    `r != 0` check refuses exactly the others. *)
Theorem nothing_left_iff_in_reach : forall n r,
  rest n r = 0 <-> - reach n 64 <= r <= reach n 63.
Proof.
  intros n r. split.
  - intro H0. pose proof (digits_rebuild n r) as E. rewrite H0, Z.mul_0_r, Z.add_0_r in E.
    rewrite <- E. apply value_in_reach.
  - apply rest_zero_in_reach.
Qed.

Corollary in_reach_the_digits_are_the_value : forall n r,
  - reach n 64 <= r <= reach n 63 -> value (digits n r) = r.
Proof.
  intros n r H. pose proof (digits_rebuild n r) as E.
  rewrite (rest_zero_in_reach n r H), Z.mul_0_r, Z.add_0_r in E. exact E.
Qed.

Theorem every_digit_is_int8 : forall n r, Forall (fun d => -64 <= d <= 63) (digits n r).
Proof.
  induction n as [|n IH]; intro r; cbn [digits]; constructor; [apply digit_range | apply IH].
Qed.

(** `ndigits`: one digit (q itself, at most 127) for 8 bits or fewer, else
    the fewest n with 63 * (128^n - 1) / 127 = reach n 63 >= 2^(bits-1) - 1. *)
Definition top (bits : Z) : Z := 2 ^ (bits - 1) - 1.

Fixpoint ndigits_from (fuel n : nat) (t : Z) : nat :=
  match fuel with
  | O => n
  | S f => if reach n 63 <? t then ndigits_from f (S n) t else n
  end.

Definition ndigits (bits : Z) : nat :=
  if bits <=? 8 then 1%nat else ndigits_from 12 1 (top bits).

Definition bits_9_to_64 : list Z := map (fun i => 9 + Z.of_nat i) (seq 0 56).

Definition holds_and_fewest (b : Z) : bool :=
  (top b <=? reach (ndigits b) 63)
  && (reach (Nat.pred (ndigits b)) 63 <? top b).

Lemma holds_and_fewest_9_to_64 : forallb holds_and_fewest bits_9_to_64 = true.
Proof. vm_compute. reflexivity. Qed.

Lemma in_bits_9_to_64 : forall b, 9 <= b <= 64 -> In b bits_9_to_64.
Proof.
  intros b Hb. unfold bits_9_to_64. apply in_map_iff.
  exists (Z.to_nat (b - 9)). split; [lia|]. apply in_seq. lia.
Qed.

(** Every value of 9 to 64 bits, |q| <= 2^(bits-1) - 1, is held exactly by
    `ndigits(bits)` digits, so the check after them never fires on a
    clamped quantizer output (`digits(..., bounded=True)`). *)
Theorem ndigits_hold : forall bits q,
  9 <= bits <= 64 -> Z.abs q <= top bits ->
  rest (ndigits bits) q = 0 /\ value (digits (ndigits bits) q) = q.
Proof.
  intros bits q Hb Hq.
  pose proof (proj1 (forallb_forall _ _) holds_and_fewest_9_to_64 bits (in_bits_9_to_64 bits Hb)) as H.
  unfold holds_and_fewest in H. apply andb_prop in H. destruct H as [H1 _]. apply Z.leb_le in H1.
  assert (Hr : - reach (ndigits bits) 64 <= q <= reach (ndigits bits) 63).
  { assert (reach (ndigits bits) 63 <= reach (ndigits bits) 64).
    { clear. induction (ndigits bits); cbn [reach]; lia. }
    lia. }
  split; [apply rest_zero_in_reach | apply in_reach_the_digits_are_the_value]; exact Hr.
Qed.

(** One digit fewer would not hold the largest value. *)
Theorem ndigits_is_the_fewest : forall bits,
  9 <= bits <= 64 -> reach (Nat.pred (ndigits bits)) 63 < top bits.
Proof.
  intros bits Hb.
  pose proof (proj1 (forallb_forall _ _) holds_and_fewest_9_to_64 bits (in_bits_9_to_64 bits Hb)) as H.
  unfold holds_and_fewest in H. apply andb_prop in H. destruct H as [_ H2]. apply Z.ltb_lt in H2. exact H2.
Qed.

(** ceil((bits - 1) / 7), the count until 2026-11-21, gives 14-bit values
    two digits, and 8191 = 2^13 - 1 leaves 1 over: two digits reach 8127. *)
Definition old_count (bits : Z) : nat := Z.to_nat ((bits - 1 + 6) / 7).

Theorem the_old_count_was_one_short :
  old_count 14 = 2%nat /\ reach 2 63 = 8127 /\ rest (old_count 14) (top 14) = 1
  /\ ndigits 14 = 3%nat.
Proof. vm_compute. repeat split. Qed.

(** The transcription, against the Python: `train_lm.ndigits` and
    `train_lm.capacity` return these (python/tests/test_exact_train_lm.py,
    `test_ndigits_is_the_proofs`). *)
Example ndigits_table :
  map ndigits [8; 9; 12; 13; 14; 15; 16; 20; 21; 22; 28; 29; 64]
  = [1; 2; 2; 2; 3; 3; 3; 3; 4; 4; 5; 5; 10]%nat
  /\ map (fun b => reach (ndigits b) 64) [12; 16; 22] = [8256; 1056832; 135274560].
Proof. vm_compute. split; reflexivity. Qed.

Print Assumptions digits_rebuild.
Print Assumptions nothing_left_iff_in_reach.
Print Assumptions every_digit_is_int8.
Print Assumptions ndigits_hold.
Print Assumptions ndigits_is_the_fewest.
Print Assumptions the_old_count_was_one_short.
Print Assumptions ndigits_table.

(* ------------------------------------------------------------------ *)
(** ** The wide product                                                *)
(* ------------------------------------------------------------------ *)

(** sum_{i<n} f i *)
Fixpoint srange (n : nat) (f : nat -> Z) : Z :=
  match n with
  | O => 0
  | S m => srange m f + f m
  end.

Lemma srange_S : forall n f, srange (S n) f = srange n f + f n.
Proof. reflexivity. Qed.

Lemma srange_ext : forall n f g,
  (forall i, (i < n)%nat -> f i = g i) -> srange n f = srange n g.
Proof.
  induction n as [|n IH]; intros f g H; [reflexivity|]. rewrite !srange_S.
  rewrite (IH f g) by (intros; apply H; lia). rewrite (H n) by lia. reflexivity.
Qed.

Lemma srange_add : forall n f g, srange n (fun i => f i + g i) = srange n f + srange n g.
Proof. induction n as [|n IH]; intros; [reflexivity|]. rewrite !srange_S, IH. ring. Qed.

Lemma srange_scale : forall n c f, srange n (fun i => c * f i) = c * srange n f.
Proof. induction n as [|n IH]; intros; [cbn; ring|]. rewrite !srange_S, IH. ring. Qed.

Lemma srange_zero : forall n, srange n (fun _ => 0) = 0.
Proof. induction n as [|n IH]; [reflexivity|]. rewrite srange_S, IH. reflexivity. Qed.

Lemma srange_swap : forall n m (f : nat -> nat -> Z),
  srange n (fun i => srange m (fun j => f i j)) = srange m (fun j => srange n (fun i => f i j)).
Proof.
  induction n as [|n IH]; intros m f.
  - cbn [srange]. symmetry. apply srange_zero.
  - rewrite srange_S, IH.
    rewrite <- (srange_add m (fun j => srange n (fun i => f i j)) (fun j => f n j)).
    apply srange_ext. intros j _. rewrite srange_S. reflexivity.
Qed.

Lemma srange_mono : forall n f g,
  (forall i, (i < n)%nat -> f i <= g i) -> srange n f <= srange n g.
Proof.
  induction n as [|n IH]; intros f g H; [cbn; lia|]. rewrite !srange_S.
  specialize (IH f g (fun i Hi => H i ltac:(lia))). specialize (H n ltac:(lia)). lia.
Qed.

Lemma srange_abs : forall n f, Z.abs (srange n f) <= srange n (fun i => Z.abs (f i)).
Proof.
  induction n as [|n IH]; intros f; [cbn; lia|]. rewrite !srange_S.
  specialize (IH f). pose proof (Z.abs_triangle (srange n f) (f n)). lia.
Qed.

(** sum_{i<n} 128^i f i: the value of digits given by position. *)
Definition wval (n : nat) (f : nat -> Z) : Z := srange n (fun i => 128 ^ Z.of_nat i * f i).

Lemma srange_shift : forall n f, srange (S n) f = f O + srange n (fun i => f (S i)).
Proof.
  induction n as [|n IH]; intros f; [cbn; ring|].
  rewrite srange_S, IH, srange_S. ring.
Qed.

Lemma wval_S : forall n f, wval (S n) f = f O + 128 * wval n (fun i => f (S i)).
Proof.
  intros n f. unfold wval. rewrite srange_shift, <- srange_scale.
  change (128 ^ Z.of_nat 0) with 1. rewrite Z.mul_1_l. f_equal.
  apply srange_ext. intros i _. rewrite Nat2Z.inj_succ, Z.pow_succ_r by lia. ring.
Qed.

(** The list form [value] is the positional form. *)
Lemma value_is_wval : forall ds, value ds = wval (length ds) (fun i => nth i ds 0).
Proof.
  induction ds as [|d t IH]; [reflexivity|].
  cbn [value length]. rewrite wval_S, IH. reflexivity.
Qed.

Lemma reach_is_wval : forall n d, reach n d = wval n (fun _ => d).
Proof.
  induction n as [|n IH]; intros d; [reflexivity|].
  cbn [reach]. rewrite wval_S, IH. reflexivity.
Qed.

(** The product of two digit values is the sum over digit pairs of
    128^(i+j) times the digits' product. *)
Theorem product_of_values : forall na nb a b,
  wval na a * wval nb b
  = srange na (fun i => srange nb (fun j => 128 ^ Z.of_nat (i + j) * (a i * b j))).
Proof.
  intros na nb a b. unfold wval.
  rewrite Z.mul_comm, <- srange_scale.
  apply srange_ext. intros i _.
  rewrite Z.mul_comm, <- srange_scale.
  apply srange_ext. intros j _.
  rewrite Nat2Z.inj_add, Z.pow_add_r by lia. ring.
Qed.

(** The stacked GEMM: block (i, j) is P i j = sum_k A k i * B k j, digit i
    of a against digit j of b over the K summed positions. Adding the blocks
    shifted by 128^(i+j) gives the dot product of the values. *)
Theorem the_shifted_blocks_are_the_dot : forall K na nb (A B : nat -> nat -> Z),
  srange na (fun i => srange nb (fun j => 128 ^ Z.of_nat (i + j) * srange K (fun k => A k i * B k j)))
  = srange K (fun k => wval na (A k) * wval nb (B k)).
Proof.
  intros K na nb A B.
  rewrite (srange_ext K (fun k => wval na (A k) * wval nb (B k))
             (fun k => srange na (fun i => srange nb (fun j => 128 ^ Z.of_nat (i + j) * (A k i * B k j)))))
    by (intros; apply product_of_values).
  rewrite (srange_swap K na). apply srange_ext. intros i _.
  rewrite (srange_swap K nb). apply srange_ext. intros j _.
  symmetry. apply srange_scale.
Qed.

Lemma srange_bound : forall n f c,
  (forall i, (i < n)%nat -> Z.abs (f i) <= c) -> Z.abs (srange n f) <= Z.of_nat n * c.
Proof.
  induction n as [|n IH]; intros f c H; [cbn; lia|]. rewrite srange_S, Nat2Z.inj_succ.
  specialize (IH f c (fun i Hi => H i ltac:(lia))). specialize (H n ltac:(lia)).
  pose proof (Z.abs_triangle (srange n f) (f n)). lia.
Qed.

(** With every digit of a at most [da] and of b at most [db] in magnitude
    (64 for several digits, 127 for one, `capacity`), the shifted blocks'
    magnitudes add up to at most K * reach na da * reach nb db: the bound
    `K * capacity(a) * capacity(b)` the code checks before anything runs. *)
Theorem the_blocks_are_bounded : forall K na nb (A B : nat -> nat -> Z) da db,
  0 <= da -> 0 <= db ->
  (forall k i, (k < K)%nat -> (i < na)%nat -> Z.abs (A k i) <= da) ->
  (forall k j, (k < K)%nat -> (j < nb)%nat -> Z.abs (B k j) <= db) ->
  srange na (fun i => srange nb (fun j => Z.abs (128 ^ Z.of_nat (i + j) * srange K (fun k => A k i * B k j))))
  <= Z.of_nat K * reach na da * reach nb db.
Proof.
  intros K na nb A B da db Hda Hdb HA HB.
  rewrite !reach_is_wval.
  assert (E : srange na (fun i => srange nb (fun j => 128 ^ Z.of_nat (i + j) * (Z.of_nat K * (da * db))))
              = Z.of_nat K * (wval na (fun _ => da) * wval nb (fun _ => db))).
  { rewrite (product_of_values na nb (fun _ => da) (fun _ => db)).
    rewrite <- srange_scale. apply srange_ext. intros i _.
    rewrite <- srange_scale. apply srange_ext. intros j _. ring. }
  rewrite <- Z.mul_assoc, <- E. apply srange_mono. intros i Hi. apply srange_mono. intros j Hj.
  assert (Hp : 0 < 128 ^ Z.of_nat (i + j)) by (apply Z.pow_pos_nonneg; lia).
  rewrite Z.abs_mul, (Z.abs_eq (128 ^ _)) by lia.
  apply Z.mul_le_mono_nonneg_l; [lia|].
  apply srange_bound. intros k Hk.
  rewrite Z.abs_mul. apply Z.mul_le_mono_nonneg; try apply Z.abs_nonneg; [apply HA | apply HB]; lia.
Qed.

(* Lists of the shifted blocks, for ExactReduce's schedules. *)

Lemma zsum_map_seq : forall n f, ER.zsum (map f (seq 0 n)) = srange n f.
Proof.
  induction n as [|n IH]; intros f; [reflexivity|].
  rewrite seq_S, map_app, ER.zsum_app, IH, Nat.add_0_l. cbn [map ER.zsum]. rewrite srange_S. ring.
Qed.

Lemma zsum_flat_map_seq : forall n (g : nat -> list Z),
  ER.zsum (flat_map g (seq 0 n)) = srange n (fun i => ER.zsum (g i)).
Proof.
  induction n as [|n IH]; intros g; [reflexivity|].
  rewrite seq_S, flat_map_app, ER.zsum_app, IH, Nat.add_0_l. cbn [flat_map]. rewrite app_nil_r, srange_S. reflexivity.
Qed.

Lemma map_flat_map_comm : forall (f : Z -> Z) (g : nat -> list Z) l,
  map f (flat_map g l) = flat_map (fun x => map f (g x)) l.
Proof. induction l as [|x l IH]; [reflexivity|]. cbn [flat_map]. rewrite map_app, IH. reflexivity. Qed.

(** The na * nb shifted blocks, in the code's order (i, then j). *)
Definition blocks (K na nb : nat) (A B : nat -> nat -> Z) : list Z :=
  flat_map (fun i => map (fun j => 128 ^ Z.of_nat (i + j) * srange K (fun k => A k i * B k j)) (seq 0 nb))
           (seq 0 na).

(** Every order and association of the shifted blocks, in a carrier whose
    cap the code's bound respects (int64; float64, whose cap 2^53 is the
    bound `K * capacity(a) * capacity(b) < 2^53`), gives the dot product of
    the values. *)
Theorem every_order_of_the_blocks_is_exact : forall c K na nb A B da db t,
  0 <= da -> 0 <= db ->
  (forall k i, (k < K)%nat -> (i < na)%nat -> Z.abs (A k i) <= da) ->
  (forall k j, (k < K)%nat -> (j < nb)%nat -> Z.abs (B k j) <= db) ->
  Z.of_nat K * reach na da * reach nb db <= ER.cap c ->
  ER.covers t (length (blocks K na nb A B)) ->
  ER.teval (ER.cadd c) (blocks K na nb A B) t = srange K (fun k => wval na (A k) * wval nb (B k)).
Proof.
  intros c K na nb A B da db t Hda Hdb HA HB Hcap Ht.
  rewrite (ER.any_schedule_gives_the_sum (ER.cap c)); [| apply ER.every_carrier_is_exact_within_its_cap | exact Ht |].
  - unfold blocks. rewrite zsum_flat_map_seq.
    rewrite <- (the_shifted_blocks_are_the_dot K na nb A B).
    apply srange_ext. intros i _. apply zsum_map_seq.
  - unfold ER.abs_sum, blocks. rewrite map_flat_map_comm, zsum_flat_map_seq.
    eapply Z.le_trans; [| exact Hcap].
    eapply Z.le_trans; [| apply (the_blocks_are_bounded K na nb A B da db Hda Hdb HA HB)].
    apply Z.eq_le_incl. apply srange_ext. intros i _. rewrite map_map. apply zsum_map_seq.
Qed.

(** The premise is satisfiable: 12-bit gradients (two digits, at most 64)
    against int8 values (one, at most 127) over 1,024 tokens, T2's widest
    product, inside float64's 2^53. *)
Example t2_widest_product_is_inside_float64 :
  Z.of_nat 1024 * reach 2 64 * reach 1 127 <= ER.cap ER.F64.
Proof. vm_compute. discriminate. Qed.

Print Assumptions product_of_values.
Print Assumptions the_shifted_blocks_are_the_dot.
Print Assumptions the_blocks_are_bounded.
Print Assumptions every_order_of_the_blocks_is_exact.
Print Assumptions t2_widest_product_is_inside_float64.

(* ------------------------------------------------------------------ *)
(** ** The rounding shift                                              *)
(* ------------------------------------------------------------------ *)

(** `_round_shift` for mant >= 0 and a shift r >= 1:
    `q0 = mant >> r; rem = mant & ((1 << r) - 1); half = 1 << (r - 1);
     q = q0 + ((rem > half) | ((rem == half) & ((q0 & 1) == 1)))`. *)
Definition shift_round (m r : Z) : Z :=
  let q0 := m / 2 ^ r in
  let rem := m mod 2 ^ r in
  let half := 2 ^ (r - 1) in
  q0 + (if (half <? rem) || ((rem =? half) && Z.odd q0) then 1 else 0).

(** The code's bit operations are those divisions. *)
Lemma the_bit_operations_are_the_divisions : forall m r, 0 <= m -> 0 <= r ->
  Z.shiftr m r = m / 2 ^ r /\ Z.land m (Z.ones r) = m mod 2 ^ r
  /\ Z.shiftl 1 r = 2 ^ r /\ Z.ones r = 2 ^ r - 1.
Proof.
  intros m r Hm Hr. split; [apply Z.shiftr_div_pow2; lia|].
  split; [apply Z.land_ones; lia|].
  split; [rewrite Z.shiftl_1_l; reflexivity|].
  unfold Z.ones. rewrite Z.shiftl_1_l. lia.
Qed.

(** q is the integer nearest m / 2^r, a tie going to the even one. *)
Definition nearest_even (m r q : Z) : Prop :=
  Z.abs (m - q * 2 ^ r) <= 2 ^ (r - 1)
  /\ (Z.abs (m - q * 2 ^ r) = 2 ^ (r - 1) -> Z.even q = true).

Lemma pow_split : forall r, 1 <= r -> 2 ^ r = 2 * 2 ^ (r - 1) /\ 0 < 2 ^ (r - 1).
Proof.
  intros r Hr. split.
  - rewrite <- Z.pow_succ_r by lia. f_equal. lia.
  - apply Z.pow_pos_nonneg; lia.
Qed.

Theorem shift_round_is_nearest : forall m r, 0 <= m -> 1 <= r ->
  nearest_even m r (shift_round m r).
Proof.
  intros m r Hm Hr. destruct (pow_split r Hr) as [E Hh].
  set (H := 2 ^ (r - 1)) in *. set (P := 2 ^ r) in *.
  assert (HP : 0 < P) by lia.
  pose proof (Z.div_mod m P ltac:(lia)) as DM.
  pose proof (Z.mod_pos_bound m P HP) as MB.
  unfold shift_round, nearest_even. fold P H.
  set (q0 := m / P) in *. set (rem := m mod P) in *.
  destruct (Z.ltb_spec H rem) as [Hgt|Hle]; cbn [orb].
  - split; [|intro; exfalso]; nia.
  - destruct (Z.eqb_spec rem H) as [Heq|Hne]; cbn [andb orb].
    + destruct (Z.odd q0) eqn:Ho.
      * split; [nia|]. intros _. rewrite Z.even_add. rewrite <- Z.negb_odd, Ho. reflexivity.
      * split; [nia|]. intros _. rewrite Z.add_0_r, <- Z.negb_odd, Ho. reflexivity.
    + split; [nia|]. intro. exfalso. nia.
Qed.

Theorem nearest_even_unique : forall m r q1 q2, 1 <= r ->
  nearest_even m r q1 -> nearest_even m r q2 -> q1 = q2.
Proof.
  intros m r q1 q2 Hr [A1 T1] [A2 T2]. destruct (pow_split r Hr) as [E Hh].
  set (H := 2 ^ (r - 1)) in *. set (P := 2 ^ r) in *.
  assert (Hd : Z.abs (q2 - q1) <= 1) by nia.
  destruct (Z.eq_dec q1 q2) as [|Hne]; [assumption|]. exfalso.
  assert (Hq : q2 = q1 + 1 \/ q1 = q2 + 1) by lia.
  destruct Hq as [Hq|Hq]; subst.
  - assert (E1 : Z.abs (m - q1 * P) = H) by nia. assert (E2 : Z.abs (m - (q1 + 1) * P) = H) by nia.
    specialize (T1 E1). specialize (T2 E2). rewrite Z.even_add in T2. rewrite T1 in T2. discriminate.
  - assert (E1 : Z.abs (m - (q2 + 1) * P) = H) by nia. assert (E2 : Z.abs (m - q2 * P) = H) by nia.
    specialize (T1 E1). specialize (T2 E2). rewrite Z.even_add in T1. rewrite T2 in T1. discriminate.
Qed.

(** The code rounds |x| and puts the sign back: that is the nearest-even
    rounding of x. *)
Theorem rounding_commutes_with_sign : forall m r q,
  nearest_even m r q -> nearest_even (- m) r (- q).
Proof.
  intros m r q [A T]. unfold nearest_even.
  replace (- m - - q * 2 ^ r) with (- (m - q * 2 ^ r)) by ring. rewrite Z.abs_opp.
  split; [exact A|]. intro E. rewrite Z.even_opp. apply T, E.
Qed.

(** The shift is clamped (to 62 in int64; 30 in int32, whose values are
    below 2^24): for any shift at or past the clamp the nearest integer is
    0, which the clamped shift also returns. *)
Theorem a_clamped_shift_rounds_the_same : forall m r,
  0 <= m < 2 ^ 53 -> 62 <= r -> shift_round m 62 = 0 /\ nearest_even m r 0.
Proof.
  intros m r Hm Hr.
  assert (H53 : 2 ^ 53 <= 2 ^ (r - 1)) by (apply Z.pow_le_mono_r; lia).
  split.
  - unfold shift_round. rewrite Z.div_small, Z.mod_small by lia. cbn [Z.odd]. change (2 ^ (62 - 1)) with 2305843009213693952.
    destruct (Z.ltb_spec 2305843009213693952 m); [lia|].
    destruct (Z.eqb_spec m 2305843009213693952); [lia|]. reflexivity.
  - unfold nearest_even. rewrite Z.mul_0_l, Z.sub_0_r, Z.abs_eq by lia. split; [lia|]. intro; lia.
Qed.

Theorem the_int32_clamp_rounds_the_same : forall m r,
  0 <= m < 2 ^ 24 -> 30 <= r -> shift_round m 30 = 0 /\ nearest_even m r 0.
Proof.
  intros m r Hm Hr.
  assert (H24 : 2 ^ 24 <= 2 ^ (r - 1)) by (apply Z.pow_le_mono_r; lia).
  split.
  - unfold shift_round. rewrite Z.div_small, Z.mod_small by lia. cbn [Z.odd]. change (2 ^ (30 - 1)) with 536870912.
    destruct (Z.ltb_spec 536870912 m); [lia|].
    destruct (Z.eqb_spec m 536870912); [lia|]. reflexivity.
  - unfold nearest_even. rewrite Z.mul_0_l, Z.sub_0_r, Z.abs_eq by lia. split; [lia|]. intro; lia.
Qed.

(** Round half up gives 1 for 1 / 2, which is no nearer than 0: the tie rule
    is what makes the rounding the one IEEE `round` gives. *)
Definition round_half_up (m r : Z) : Z := (m + 2 ^ (r - 1)) / 2 ^ r.

Theorem round_half_up_differs_on_a_tie :
  round_half_up 1 1 = 1 /\ shift_round 1 1 = 0 /\ nearest_even 1 1 0.
Proof. unfold nearest_even. vm_compute. repeat split; intros; congruence. Qed.

(* ------------------------------------------------------------------ *)
(** ** The fixed-point sums and the scales                             *)
(* ------------------------------------------------------------------ *)

(** A value below 2^p, rounded, is at most 2^p: |x| < 2^e and s = e - 61 + c
    give |x * 2^-s| < 2^(61-c), so every rounded term is at most 2^(61-c). *)
Theorem a_rounded_term_is_bounded : forall m r p, 1 <= r -> 0 <= p ->
  0 <= m < 2 ^ (r + p) -> 0 <= shift_round m r <= 2 ^ p.
Proof.
  intros m r p Hr Hp Hm.
  assert (HP : 0 < 2 ^ r) by (apply Z.pow_pos_nonneg; lia).
  assert (Hq : m / 2 ^ r < 2 ^ p).
  { apply Z.div_lt_upper_bound; [lia|]. rewrite <- Z.pow_add_r by lia. lia. }
  assert (Hq0 : 0 <= m / 2 ^ r) by (apply Z.div_pos; lia).
  unfold shift_round. destruct (_ || _); lia.
Qed.

(** `Ops.sum` and `scatter_rows`: n <= 2^c terms, each at most 2^(61-c), so
    their absolute values sum to at most 2^61, inside int64's cap: every
    order and association of the int64 sum gives the integer sum. *)
Theorem fixed_point_sums_are_exact : forall c terms t,
  0 <= c <= 61 -> Z.of_nat (length terms) <= 2 ^ c ->
  Forall (fun q => Z.abs q <= 2 ^ (61 - c)) terms ->
  ER.covers t (length terms) ->
  ER.teval (ER.cadd ER.I64) terms t = ER.zsum terms.
Proof.
  intros c terms t Hc Hn Hq Ht.
  apply (ER.any_schedule_gives_the_sum (ER.cap ER.I64)); [apply ER.every_carrier_is_exact_within_its_cap | exact Ht |].
  assert (Hp : 0 <= 2 ^ (61 - c)) by (apply Z.pow_nonneg; lia).
  pose proof (ER.zsum_bounded (2 ^ (61 - c)) terms Hp Hq) as B.
  assert (E : 2 ^ c * 2 ^ (61 - c) = 2 ^ 61) by (rewrite <- Z.pow_add_r by lia; f_equal; lia).
  cbn [ER.cap]. unfold EP.I64MAX. change (2 ^ 61) with 2305843009213693952 in E. nia.
Qed.

(** Past int64's cap two terms of 2^62 wrap. *)
Theorem past_the_int64_cap_the_sum_wraps :
  ER.cadd ER.I64 (2 ^ 62) (2 ^ 62) = - 2 ^ 63.
Proof. vm_compute. reflexivity. Qed.

(** q8 refuses |s| > 511, so a product's scale sa + sb is a normal float64
    exponent, in [-1022, 1023]. *)
Theorem product_scales_are_normal : forall sa sb,
  Z.abs sa <= 511 -> Z.abs sb <= 511 -> -1022 <= sa + sb <= 1023.
Proof. intros. lia. Qed.

Theorem a_scale_of_512_could_leave_float64 : 512 + 512 > 1023.
Proof. lia. Qed.

(** `Ops.sum` refuses -s outside [-1022, 1023] and s = -1023, so both 2^s
    and 2^-s are normal float64 values. *)
Theorem sum_scales_are_normal : forall s,
  -1022 <= - s <= 1023 -> s <> -1023 -> -1022 <= s <= 1023.
Proof. intros. lia. Qed.

Print Assumptions the_bit_operations_are_the_divisions.
Print Assumptions shift_round_is_nearest.
Print Assumptions nearest_even_unique.
Print Assumptions rounding_commutes_with_sign.
Print Assumptions a_clamped_shift_rounds_the_same.
Print Assumptions the_int32_clamp_rounds_the_same.
Print Assumptions round_half_up_differs_on_a_tie.
Print Assumptions a_rounded_term_is_bounded.
Print Assumptions fixed_point_sums_are_exact.
Print Assumptions past_the_int64_cap_the_sum_wraps.
Print Assumptions product_scales_are_normal.
Print Assumptions a_scale_of_512_could_leave_float64.
Print Assumptions sum_scales_are_normal.
