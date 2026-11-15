(** * One decode step of the exact path: every valid execution, the same output.

    This is the capstone of the exact-inference proofs. It models one decode
    step of `python/y_lang/exact/` on a Qwen2 model (every layer's two norms,
    q/k/v, RoPE, attention over the cache, o_proj, the MLP, the residuals,
    the final norm, the LM head and the argmax) and proves that the step's
    output does not depend on how the hardware reduces.

    ** Executions are machines.

    Every place where the code combines more than two values is a SITE: the
    sums inside each linear layer, each attention score, the softmax
    denominator, each `p.V` output, the norm's limb sums, and the maxima. A
    [machine] says how each site is computed: a reducer for each sum (any
    association, any order, any carrier), a max for each maximum, and the
    argmax. A tile shape, a K split, a thread count, a batch composition, a
    device and a kernel choice (Triton, cuBLASLt, oneDNN, Y's own) each give
    a machine. A site can be computed differently at every layer, row and
    column.

    A machine is [valid] when each sum reducer is exact on every input whose
    absolute values sum to at most that site's bound [need], each max returns
    a maximal element, and the argmax returns the first maximal index.

    [ExactReduce.v] proves that every schedule in a carrier whose cap covers
    the bound is exact ([sched_dot_exact], [sched_sum_exact]), and
    [the_code's_carriers_meet_every_need] proves that the carriers the code
    uses cover the bounds, for every shape the code admits.

    ** The theorem.

    [every_valid_execution_takes_the_same_decode_step]: for any two valid
    machines, the same weights, cache and token give the same next token and
    the same new cache rows. Its premises are upper bounds: weight rows hold
    int8 values and are no longer than their layer's K, cache rows are fewer
    than T, the hidden size is the model's, T * 2^28 * 127 <= 2^53.

    [code_executions_agree] composes it with the carrier bounds: any two
    executions that put each sum in a carrier the code uses there, with any
    schedules, agree. [qwen25_shapes_are_accepted] shows the shape premises
    hold for Qwen2.5 0.5B, 1.5B, 3B and 7B at activation widths 127 and 511
    and every T up to 32768.

    ** The trust boundary: what the theorem takes as given.

    The floating-point operations that are not reductions are Section
    variables: the quantisers' scale and rounding divides, the linear
    epilogue, the norm's finish (divide, square root, multiply), RoPE, the
    logit, the exp argument, the V-scale fold, the output divide, SiLU, the
    residual adds and the embedding lookup. The theorem says the step is a
    function of them; it holds whatever functions they are, PROVIDED each is
    the same function on every device. That is an assumption about IEEE 754
    hardware and about the compilers, listed with what each relies on in
    `docs/exact_decode_theorem.md`. Four hypotheses constrain them:

    - [frope_length]: RoPE maps a head to a head of the same length;
    - [fge_ival]: the float comparison is the order [ival] (IEEE order on
      non-NaN values, -0 equal to +0);
    - [targ_ival], [fold_ival]: the exp argument and the V-scale fold read a
      maximum only through its IEEE value, so a max that returned -0 where
      another returned +0 changes nothing.

    [the_trusted_interface_is_satisfiable] exhibits functions meeting them.

    ** What is NOT modelled.

    - Rows holding an infinity or a NaN (the norm then takes a float path).
    - Ragged batches: left padding adds masked keys, which the model does not
      have. Equal-length batches of any size are machines.
    - The prefill, except as one decode step per prompt token.
    - Triton's two-digit kernel, which accumulates `128 * hi` and `lo` in
      one interleaved int32 sum; the separate form (`torch._int_mm`, Y's
      GEMM, float64) is modelled.
    - That a kernel's sums ARE some association of the site's terms: for
      Y's kernels the per-kernel proofs show it; for others it is assumed.

    Checked with Rocq 9.1:  coqc -Q . "" ExactDecodeStep.v  *)

From Stdlib Require Import ZArith Arith Lia List Bool Permutation.
Require ExactReduce ExactDecodeConstants ExactDecodeWidths NormGridSum ExactGemmMicro ExactPvExact.
Import ListNotations.
Open Scope Z_scope.

Module ER := ExactReduce.
Module C := ExactDecodeConstants.
Module W := ExactDecodeWidths.
Module NG := NormGridSum.

(* ------------------------------------------------------------------ *)
(** ** Integer helpers                                                 *)
(* ------------------------------------------------------------------ *)

Definition clampz (lv x : Z) : Z := Z.max (- lv) (Z.min lv x).
Definition clamp0 (hi x : Z) : Z := Z.max 0 (Z.min hi x).

Lemma clampz_range : forall lv x, 0 <= lv -> Z.abs (clampz lv x) <= lv.
Proof. intros. unfold clampz. lia. Qed.

Lemma clamp0_range : forall hi x, 0 <= hi -> 0 <= clamp0 hi x <= hi.
Proof. intros. unfold clamp0. lia. Qed.

Definition prods (l : list (Z * Z)) : list Z := map (fun xy => fst xy * snd xy) l.
Definition pdot (l : list (Z * Z)) : Z := ER.zsum (prods l).

Lemma abs_sum_prods : forall ma mb l, 0 <= ma -> 0 <= mb ->
  Forall (fun xy => Z.abs (fst xy) <= ma /\ Z.abs (snd xy) <= mb) l ->
  ER.abs_sum (prods l) <= Z.of_nat (length l) * (ma * mb).
Proof.
  intros ma mb l Ha Hb H.
  induction H as [|[x y] l [Hx Hy] Hl IH]; [cbn; lia|].
  unfold ER.abs_sum, prods in *. cbn [map ER.zsum length fst snd] in *.
  rewrite Nat2Z.inj_succ, Z.abs_mul. nia.
Qed.

Lemma Forall_combine : forall (A B : Type) (P : A -> Prop) (Q : B -> Prop) a b,
  Forall P a -> Forall Q b -> Forall (fun xy => P (fst xy) /\ Q (snd xy)) (combine a b).
Proof.
  intros A B P Q a b Ha. revert b.
  induction Ha as [|x a Hx Ha IH]; intros [|y b] Hb; cbn; try constructor.
  - inversion Hb; subst. split; assumption.
  - inversion Hb; subst. apply IH. assumption.
Qed.

Lemma combine_len_l : forall (A B : Type) (a : list A) (b : list B),
  (length (combine a b) <= length a)%nat.
Proof. intros. rewrite length_combine. lia. Qed.

Lemma combine_len_r : forall (A B : Type) (a : list A) (b : list B),
  (length (combine a b) <= length b)%nat.
Proof. intros. rewrite length_combine. lia. Qed.

Definition enum {A : Type} (l : list A) : list (nat * A) := combine (seq 0 (length l)) l.

Lemma map_snd_enum_from : forall (A B : Type) (f : A -> B) (l : list A) k,
  map (fun tr => f (snd tr)) (combine (seq k (length l)) l) = map f l.
Proof.
  intros A B f l. induction l as [|x l IH]; intros k; [reflexivity|].
  cbn [length seq combine map snd]. f_equal. apply IH.
Qed.

Lemma in_enum : forall (A : Type) (l : list A) n x, In (n, x) (enum l) -> In x l.
Proof. intros A l n x H. eapply in_combine_r. exact H. Qed.

Definition zipw {A B C : Type} (f : A -> B -> C) (a : list A) (b : list B) : list C :=
  map (fun xy => f (fst xy) (snd xy)) (combine a b).

Lemma zipw_len : forall (A B C : Type) (f : A -> B -> C) a b,
  (length (zipw f a b) <= length a)%nat.
Proof. intros. unfold zipw. rewrite length_map. apply combine_len_l. Qed.

Lemma flat_map_ext_in : forall (A B : Type) (f g : A -> list B) l,
  (forall x, In x l -> f x = g x) -> flat_map f l = flat_map g l.
Proof.
  intros A B f g l. induction l as [|x l IH]; intros H; [reflexivity|].
  cbn. rewrite H by (left; reflexivity). f_equal. apply IH. intros y Hy. apply H. right. exact Hy.
Qed.

(* ------------------------------------------------------------------ *)
(** ** Base-2^d digits, as the Python p.V digit split takes them        *)
(* ------------------------------------------------------------------ *)

(** `(p >> shift) & ((1 << dbits) - 1)` at `shift = j * d`. *)
Definition digit (d : Z) (j : nat) (x : Z) : Z := Z.land (Z.shiftr x (Z.of_nat j * d)) (Z.ones d).

(** The loop `shift = 0; while shift < P_BITS + 1: ...; shift += dbits` runs
    once for each j with `j * d < 29`. *)
Definition ndig (d : Z) : nat := Z.to_nat ((28 + d) / d).

Lemma digit_mod : forall d j x, 0 <= d ->
  digit d j x = (x / 2 ^ (Z.of_nat j * d)) mod 2 ^ d.
Proof.
  intros d j x Hd. unfold digit. rewrite Z.land_ones by lia.
  rewrite Z.shiftr_div_pow2 by nia. reflexivity.
Qed.

Lemma digit_range : forall d j x, 0 <= d -> 0 <= digit d j x <= 2 ^ d - 1.
Proof.
  intros d j x Hd. rewrite digit_mod by lia.
  pose proof (Z.mod_pos_bound (x / 2 ^ (Z.of_nat j * d)) (2 ^ d) ltac:(apply Z.pow_pos_nonneg; lia)).
  lia.
Qed.

Lemma mod_mul_r : forall a b c, 0 < b -> 0 < c ->
  a mod (b * c) = a mod b + b * ((a / b) mod c).
Proof.
  intros a b c Hb Hc. symmetry. apply Z.mod_unique with (q := (a / b) / c).
  - left. pose proof (Z.mod_pos_bound a b Hb). pose proof (Z.mod_pos_bound (a / b) c Hc). nia.
  - pose proof (Z.div_mod a b ltac:(lia)). pose proof (Z.div_mod (a / b) c ltac:(lia)). nia.
Qed.

Lemma digits_sum_mod : forall d x n, 1 <= d ->
  ER.zsum (map (fun j => 2 ^ (Z.of_nat j * d) * digit d j x) (seq 0 n))
  = x mod 2 ^ (Z.of_nat n * d).
Proof.
  intros d x n Hd. induction n as [|n IH].
  - cbn. rewrite Z.mod_1_r. reflexivity.
  - rewrite seq_S, map_app, ER.zsum_app, IH. cbn [map ER.zsum Nat.add].
    rewrite digit_mod by lia.
    assert (E : 2 ^ (Z.of_nat (S n) * d) = 2 ^ (Z.of_nat n * d) * 2 ^ d).
    { rewrite <- Z.pow_add_r by nia. f_equal. lia. }
    rewrite E, mod_mul_r by (apply Z.pow_pos_nonneg; nia). ring.
Qed.

Theorem the_digits_recombine : forall d x, 1 <= d -> 0 <= x <= 2 ^ 28 ->
  ER.zsum (map (fun j => 2 ^ (Z.of_nat j * d) * digit d j x) (seq 0 (ndig d))) = x.
Proof.
  intros d x Hd Hx. rewrite digits_sum_mod by lia. apply Z.mod_small.
  split; [lia|]. unfold ndig.
  assert (Hq : 29 <= (28 + d) / d * d).
  { pose proof (Z.div_mod (28 + d) d ltac:(lia)).
    pose proof (Z.mod_pos_bound (28 + d) d ltac:(lia)). nia. }
  rewrite Z2Nat.id by (apply Z.div_pos; lia).
  apply Z.le_lt_trans with (2 ^ 28); [lia|].
  apply Z.pow_lt_mono_r; lia.
Qed.

(** Two sums swapped: sum over j of c_j * (sum over terms of g_j(w) * h(v)). *)
Lemma zsum_map_zero : forall (js : list nat) (f : nat -> Z),
  (forall j, f j = 0) -> ER.zsum (map f js) = 0.
Proof. intros js f H. induction js as [|j js IH]; cbn; [reflexivity|]. rewrite H, IH. reflexivity. Qed.

Lemma zsum_map_add : forall (A : Type) (js : list A) f g,
  ER.zsum (map (fun j => f j + g j) js) = ER.zsum (map f js) + ER.zsum (map g js).
Proof. intros A js f g. induction js as [|j js IH]; cbn; [reflexivity|]. rewrite IH. ring. Qed.

Lemma zsum_map_mul_r : forall (A : Type) (js : list A) f k,
  ER.zsum (map (fun j => f j * k) js) = ER.zsum (map f js) * k.
Proof. intros A js f k. induction js as [|j js IH]; cbn; [ring|]. rewrite IH. ring. Qed.

Lemma swap_sums : forall (l : list (Z * Z)) (js : list nat) (c : nat -> Z) (g : nat -> Z -> Z) (h : Z -> Z),
  ER.zsum (map (fun j => c j * ER.zsum (map (fun wv => g j (fst wv) * h (snd wv)) l)) js)
  = ER.zsum (map (fun wv => ER.zsum (map (fun j => c j * g j (fst wv)) js) * h (snd wv)) l).
Proof.
  intros l js c g h. induction l as [|[w v] l IH].
  - cbn. apply zsum_map_zero. intros. ring.
  - cbn [map ER.zsum fst snd]. rewrite <- IH.
    rewrite <- zsum_map_mul_r, <- zsum_map_add. f_equal. apply map_ext. intros j. ring.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The model's shapes                                              *)
(* ------------------------------------------------------------------ *)

Record dims : Type := {
  dH : Z;        (* hidden size *)
  dI : Z;        (* MLP width *)
  dNH : Z;       (* query heads *)
  dNKV : Z;      (* key/value heads *)
  dD : Z;        (* head dimension *)
  dReq : Z       (* requested activation width, `Y_EXACT_ACT_LEVELS` *)
}.

Inductive lin : Type := QKV | OProj | Gate | Up | Down | Head.

(** The contraction length of each linear layer. *)
Definition K_of (dm : dims) (w : lin) : Z :=
  match w with
  | Down => dI dm
  | OProj => dNH dm * dD dm
  | _ => dH dm
  end.

(** The widths the code derives: `ExactLinear.a_levels` and `k_levels_for`. *)
Definition a_of (dm : dims) (w : lin) : Z := W.act_levels_for (K_of dm w) C.W_LEVELS (dReq dm).
Definition kl_of (dm : dims) : Z := W.k_levels_for (dD dm) C.Q_LEVELS.

Lemma a_of_props : forall dm w, 0 <= dReq dm ->
  127 <= a_of dm w <= C.DIGIT_MAX /\ (a_of dm w + 1) mod 128 = 0.
Proof.
  intros dm w Hr. unfold a_of, W.act_levels_for.
  set (cap := Z.min (Z.min (C.ACC_BUDGET / Z.max (K_of dm w * C.W_LEVELS) 1 - 128) C.DIGIT_MAX) (dReq dm)).
  assert (Hc : -128 <= cap).
  { unfold cap, C.ACC_BUDGET, C.DIGIT_MAX.
    assert (0 <= 2147483648 / Z.max (K_of dm w * C.W_LEVELS) 1) by (apply Z.div_pos; lia). lia. }
  pose proof (W.floor_pow2m1_le cap Hc). pose proof (W.floor_pow2m1_ge cap).
  pose proof (W.floor_pow2m1_shape cap).
  assert (cap <= C.DIGIT_MAX) by (unfold cap; lia).
  unfold C.DIGIT_MAX in *. repeat split; lia.
Qed.

Lemma kl_of_ge : forall dm, 127 <= kl_of dm.
Proof. intros. apply W.floor_pow2m1_ge. Qed.

(* ------------------------------------------------------------------ *)
(** ** Sites, machines and their validity                              *)
(* ------------------------------------------------------------------ *)

Inductive lpart : Type := Plain | Hi | Lo | Comb.

Inductive stage : Type :=
| NormExp | NormHi | NormLo | QuantMax
| Lin (w : lin) (p : lpart)
| Score | LogitMax | Denom | FoldMax
| PvDirect | PvDigit (j : nat) | PvRecomb.

Definition site : Type := (stage * list nat)%type.

(** The bound each sum site needs, from the operands' widths and the
    reduction's length. [d] is the digit width a `p.V` site uses. *)
Definition need_st (dm : dims) (T d : Z) (st : stage) : Z :=
  match st with
  | NormHi | NormLo => dH dm * 2 ^ 40
  | Lin w Plain => K_of dm w * 127 * 127
  | Lin w Hi => K_of dm w * 128 * 127
  | Lin w Lo => K_of dm w * 127 * 127
  | Lin w Comb => K_of dm w * (a_of dm w + 128) * 127
  | Score => dD dm * C.Q_LEVELS * kl_of dm
  | Denom => T * 2 ^ 28
  | PvDirect | PvRecomb => T * 2 ^ 28 * C.V_LEVELS
  | PvDigit _ => T * (2 ^ d - 1) * C.V_LEVELS
  | _ => 0
  end.

(* ------------------------------------------------------------------ *)
(** ** The carriers the code uses meet every bound                     *)
(* ------------------------------------------------------------------ *)

(** Where the code puts each sum (see `docs/exact_decode_theorem.md`):
    - linear layers: int32 (`torch._int_mm`, Triton `tl.dot`, Y's GEMM), or
      float64 (the CPU fallback), or float32 when `fp32_exact` holds;
    - scores: float32 (`torch.matmul`) or int32 (Y's `dp2a` kernel);
    - the norm's limbs and the denominator: int64;
    - `p.V` in one sum: int64 (Y's kernels) or float64 (no digit fits);
      in digits: float32 at `digit_width(T, V_LEVELS)`, recombined in float64. *)
Definition code_uses (dm : dims) (T d : Z) (st : stage) (c : ER.carrier) : Prop :=
  match st, c with
  | Lin w Plain, (ER.I32 | ER.F64) => a_of dm w <= 127
  | Lin w Plain, ER.F32 => a_of dm w <= 127 /\ K_of dm w * 127 * 127 < 2 ^ 24
  | Lin w (Hi | Lo | Comb), (ER.I32 | ER.F64) => 127 < a_of dm w
  | Score, (ER.F32 | ER.I32) => True
  | (NormHi | NormLo | Denom), ER.I64 => True
  | PvDirect, (ER.I64 | ER.F64) => True
  | PvDigit _, ER.F32 => d = W.digit_width T C.V_LEVELS /\ 1 <= d
  | PvRecomb, ER.F64 => True
  | (NormExp | QuantMax | LogitMax | FoldMax), _ => True
  | _, _ => False
  end.

(** The shapes the code admits: `ExactLinear` refuses int8 past the int32
    budget, `assert_score_budget` a head dimension past float32, the norm
    rows past 8192 values, and `exact_attention_core` a context past 2^53. *)
Definition code_shape_ok (dm : dims) (T : Z) : Prop :=
  0 <= dReq dm
  /\ (forall w, 0 < K_of dm w /\ K_of dm w * 127 * 127 < C.ACC_BUDGET)
  /\ 0 < dD dm /\ dD dm * C.Q_LEVELS * 127 <= C.SCORE_BUDGET
  /\ 0 <= dH dm <= C.SUMSQ_MAX_H
  /\ 1 <= T /\ T * 2 ^ 28 * C.V_LEVELS <= 2 ^ 53.

Theorem the_code's_carriers_meet_every_need : forall dm T d st c,
  code_shape_ok dm T -> code_uses dm T d st c -> need_st dm T d st <= ER.cap c.
Proof.
  intros dm T d st c [Hr [HK [HD0 [HD [HH [HT H53]]]]]] Hu.
  unfold C.ACC_BUDGET, C.Q_LEVELS, C.SCORE_BUDGET, C.SUMSQ_MAX_H, C.V_LEVELS in *.
  assert (Hc : forall c, 2 ^ 24 <= ER.cap c).
  { intros []; cbn; unfold ExactGemmMicro.I32MAX, ExactPvExact.I64MAX; lia. }
  assert (P28 : 2 ^ 28 = 268435456) by reflexivity.
  assert (P40 : 2 ^ 40 = 1099511627776) by reflexivity.
  assert (P53 : 2 ^ 53 = 9007199254740992) by reflexivity.
  destruct st; cbn [need_st]; try (pose proof (Hc c); lia).
  - destruct c; cbn in Hu |- *; try contradiction; unfold ExactPvExact.I64MAX, C.V_LEVELS in *; rewrite ?P28, ?P40, ?P53 in *; nia.
  - destruct c; cbn in Hu |- *; try contradiction; unfold ExactPvExact.I64MAX, C.V_LEVELS in *; rewrite ?P28, ?P40, ?P53 in *; nia.
  - destruct (HK w) as [HK0 HK1].
    destruct (W.the_derived_activation_width_fits_the_accumulator (K_of dm w) (dReq dm) HK0 Hr
                ltac:(unfold C.ACC_BUDGET; lia)) as [[Ha1 Ha2] [_ Hwide]].
    fold (a_of dm w) in Ha1, Ha2, Hwide. unfold C.ACC_BUDGET in Hwide.
    destruct p, c; cbn in Hu |- *; try contradiction; unfold ExactGemmMicro.I32MAX;
      try lia; try (specialize (Hwide Hu); nia).
  - destruct (W.the_derived_key_width_fits_float32 (dD dm) C.Q_LEVELS HD0 ltac:(unfold C.Q_LEVELS; lia)
                ltac:(unfold C.Q_LEVELS, C.SCORE_BUDGET; lia)) as [_ Hk].
    fold (kl_of dm) in Hk. unfold C.Q_LEVELS, C.SCORE_BUDGET in Hk.
    destruct c; cbn in Hu; try contradiction; cbn [ER.cap]; unfold ExactGemmMicro.I32MAX, C.Q_LEVELS; lia.
  - destruct c; cbn in Hu |- *; try contradiction; unfold ExactPvExact.I64MAX, C.V_LEVELS in *; rewrite ?P28, ?P40, ?P53 in *; nia.
  - destruct c; cbn in Hu |- *; try contradiction; unfold ExactPvExact.I64MAX, C.V_LEVELS in *; rewrite ?P28, ?P40, ?P53 in *; nia.
  - destruct c; cbn in Hu |- *; try contradiction. destruct Hu as [Hd Hd1].
    destruct (W.the_digit_width_fits_float32 T C.V_LEVELS) as [_ Hfit].
    rewrite <- Hd in Hfit. specialize (Hfit ltac:(lia)). unfold C.V_LEVELS, C.SCORE_BUDGET in *. change (2 ^ 24) with 16777216 in *. lia.
  - destruct c; cbn in Hu |- *; try contradiction. unfold C.V_LEVELS in *. rewrite ?P28, ?P53 in *. lia.
Qed.


Section Step.

(** *** The trusted elementwise operations *)

Variable R : Type.                    (* a float32 or float64 value *)
Variable rdef : R.
Variable bits : R -> Z.               (* a float32's bit pattern *)
Variable absbits : R -> Z.            (* the bits of |x| *)
Variable ival : R -> Z.               (* the IEEE order, -0 = +0 *)
Variable fge : R -> R -> bool.        (* `a >= b` *)
Hypothesis fge_ival : forall a b, fge a b = (ival b <=? ival a).

Variable fscale : Z -> Z -> R.                 (* the row max's bits, width -> amax / lv, 1 where 0 *)
Variable fround : R -> R -> Z.                 (* round(x / s) *)
Variable fepi : Z -> R -> R -> option R -> R.  (* acc * sx * sw (+ bias) *)
Variable fnorm : Z -> Z -> Z -> R -> R -> R.   (* big, small, e, x, weight -> normed x * weight *)
Variable frope : Z -> list R -> list R.        (* position, one head *)
Hypothesis frope_length : forall pos v, length (frope pos v) = length v.
Variable fqa : R -> R.                         (* sq * scaling * log2(e) * 65536 *)
Variable flogit : Z -> R -> R -> R.            (* score * a * sk *)
Variable targ : R -> R -> Z.                   (* round(m - logit) *)
Hypothesis targ_ival : forall m m' x, ival m = ival m' -> targ m x = targ m' x.
Variable fw : Z -> R -> R.                     (* p * sv *)
Variable fold_round : R -> R -> Z.             (* round(w * (2^28 / safe(wmax))) *)
Variable fold_scale : R -> R.                  (* safe(wmax) / 2^28 *)
Hypothesis fold_ival : forall m m' x, ival m = ival m' ->
  fold_round x m = fold_round x m' /\ fold_scale m = fold_scale m'.
Variable fout : Z -> R -> Z -> R.              (* (acc * wscale) / max(l, 1) *)
Variable NEG : R.                              (* a masked key's logit: finfo(float64).min *)
Variable fadd : R -> R -> R.                   (* the residual add *)
Variable fsilu_mul : R -> R -> R.              (* silu(gate) * up *)
Variable embed : nat -> list R.

(** *** Machines *)

Record machine : Type := {
  m_dot : site -> list (Z * Z) -> Z;
  m_sum : site -> list Z -> Z;
  m_comb : site -> Z -> Z -> Z;        (* `128 * hi_acc + lo_acc` *)
  m_imax : site -> list Z -> Z;
  m_fmax : site -> list R -> R;
  m_argmax : list R -> nat;
  m_digits : list nat -> Z             (* 0: `p.V` in one sum; d: in base-2^d digits *)
}.

Definition need (dm : dims) (T : Z) (M : machine) (s : site) : Z :=
  need_st dm T (m_digits M (snd s)) (fst s).

Definition exact_dot (B : Z) (f : list (Z * Z) -> Z) : Prop :=
  forall l, ER.abs_sum (prods l) <= B -> f l = pdot l.
Definition exact_sum (B : Z) (f : list Z -> Z) : Prop :=
  forall l, ER.abs_sum l <= B -> f l = ER.zsum l.
Definition exact_comb (B : Z) (f : Z -> Z -> Z) : Prop :=
  forall h l, Z.abs (128 * h) + Z.abs l <= B -> f h l = 128 * h + l.

Definition imax_ok (f : list Z -> Z) : Prop :=
  forall l, (l = [] -> f l = 0) /\ (l <> [] -> ER.is_max_of Z (fun z => z) l (f l)).
Definition fmax_ok (f : list R -> R) : Prop :=
  forall l, (l = [] -> f l = rdef) /\ (l <> [] -> ER.is_max_of R ival l (f l)).
Definition argmax_ok (f : list R -> nat) : Prop :=
  forall l, (l = [] -> f l = 0%nat) /\ (l <> [] -> ER.is_first_argmax R ival rdef l (f l)).

Record valid (dm : dims) (T : Z) (M : machine) : Prop := {
  v_imax : forall s, imax_ok (m_imax M s);
  v_fmax : forall s, fmax_ok (m_fmax M s);
  v_argmax : argmax_ok (m_argmax M);
  v_dot : forall s, exact_dot (need dm T M s) (m_dot M s);
  v_sum : forall s, exact_sum (need dm T M s) (m_sum M s);
  v_comb : forall s, exact_comb (need dm T M s) (m_comb M s);
  v_digits : forall p, 0 <= m_digits M p
}.

(** *** The reference machine *)

Fixpoint ltree (n : nat) : ER.tree :=
  match n with
  | O => ER.Zero
  | S m => match m with O => ER.Leaf 0 | _ => ER.Node (ltree m) (ER.Leaf m) end
  end.

Lemma ltree_leaves : forall n, ER.leaves (ltree n) = seq 0 n.
Proof.
  induction n as [|n IH]; [reflexivity|].
  destruct n as [|n]; [reflexivity|].
  change (ER.leaves (ltree (S (S n)))) with (ER.leaves (ltree (S n)) ++ [S n]).
  rewrite IH. rewrite (seq_S (S n) 0). reflexivity.
Qed.

Lemma ltree_covers : forall n, ER.covers (ltree n) n.
Proof. intros n. unfold ER.covers. rewrite ltree_leaves. apply Permutation_refl. Qed.

Lemma ltree_nozero : forall n, n <> O -> ER.nozero (ltree n).
Proof.
  induction n as [|n IH]; intros H; [contradiction|].
  destruct n as [|n]; cbn; [exact I|]. split; [apply IH; discriminate | exact I].
Qed.

Definition zmx (a b : Z) : Z := if b <=? a then a else b.
Lemma zmx_picks : forall a b, (zmx a b = a \/ zmx a b = b) /\ zmx a b = Z.max a b.
Proof. intros a b. unfold zmx. destruct (Z.leb_spec b a); split; auto; lia. Qed.

Definition zmax (l : list Z) : Z :=
  match l with [] => 0 | _ => ER.xeval Z zmx 0 l (ltree (length l)) end.

Lemma zmax_ok : imax_ok zmax.
Proof.
  intros l. split; [intros ->; reflexivity|]. intros H. destruct l as [|x r]; [contradiction|].
  apply (ER.any_max_tree_is_a_max Z (fun z => z) zmx zmx_picks).
  - apply ltree_nozero. discriminate.
  - apply ltree_covers.
Qed.

Definition fmx (a b : R) : R := if fge a b then a else b.
Lemma fmx_picks : forall a b, (fmx a b = a \/ fmx a b = b) /\ ival (fmx a b) = Z.max (ival a) (ival b).
Proof.
  intros a b. unfold fmx. rewrite fge_ival.
  destruct (Z.leb_spec (ival b) (ival a)); split; auto; lia.
Qed.

Definition fmaxr (l : list R) : R :=
  match l with [] => rdef | _ => ER.xeval R fmx rdef l (ltree (length l)) end.

Lemma fmaxr_ok : fmax_ok fmaxr.
Proof.
  intros l. split; [intros ->; reflexivity|]. intros H. destruct l as [|x r]; [contradiction|].
  apply (ER.any_max_tree_is_a_max R ival fmx fmx_picks).
  - apply ltree_nozero. discriminate.
  - apply ltree_covers.
Qed.

Definition argmax_ref (l : list R) : nat :=
  match l with [] => O | _ => ER.argmax_scan R ival rdef l end.

Lemma argmax_ref_ok : argmax_ok argmax_ref.
Proof.
  intros l. split; [intros ->; reflexivity|]. intros H. destruct l as [|x r]; [contradiction|].
  apply ER.the_scan_is_a_first_argmax. discriminate.
Qed.

(** Sums in the integers, the first maximal element, digits never. *)
Definition Mref : machine := {|
  m_dot := fun _ l => pdot l;
  m_sum := fun _ l => ER.zsum l;
  m_comb := fun _ h l => 128 * h + l;
  m_imax := fun _ l => zmax l;
  m_fmax := fun _ l => fmaxr l;
  m_argmax := argmax_ref;
  m_digits := fun _ => 0
|}.

Theorem the_reference_machine_is_valid : forall dm T, valid dm T Mref.
Proof.
  intros dm T. constructor; cbn.
  - intros. apply zmax_ok.
  - intros. apply fmaxr_ok.
  - apply argmax_ref_ok.
  - intros s l _. reflexivity.
  - intros s l _. reflexivity.
  - intros s h l _. reflexivity.
  - intros. lia.
Qed.

(** What validity gives, call by call. *)
Lemma imax_is_zmax : forall f l, imax_ok f -> f l = zmax l.
Proof.
  intros f l H. destruct l as [|x r].
  - rewrite (proj1 (H [])) by reflexivity. reflexivity.
  - apply (ER.integer_max_is_unique (x :: r)).
    + apply (proj2 (H (x :: r))). discriminate.
    + apply (proj2 (zmax_ok (x :: r))). discriminate.
Qed.

Lemma fmax_ival : forall f l, fmax_ok f -> ival (f l) = ival (fmaxr l).
Proof.
  intros f l H. destruct l as [|x r].
  - rewrite (proj1 (H [])) by reflexivity. reflexivity.
  - apply (ER.max_key_unique R ival (x :: r)).
    + apply (proj2 (H (x :: r))). discriminate.
    + apply (proj2 (fmaxr_ok (x :: r))). discriminate.
Qed.

Lemma argmax_is_ref : forall f l, argmax_ok f -> f l = argmax_ref l.
Proof.
  intros f l H. destruct l as [|x r].
  - rewrite (proj1 (H [])) by reflexivity. reflexivity.
  - apply (ER.first_argmax_unique R ival rdef (x :: r)).
    + apply (proj2 (H (x :: r))). discriminate.
    + apply (proj2 (argmax_ref_ok (x :: r))). discriminate.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The step                                                        *)
(* ------------------------------------------------------------------ *)

(** *** Quantisation: `_q_per_row`, `quantize_rows`, `q8_per_tensor` *)

Definition quant (M : machine) (path : list nat) (lv : Z) (xs : list R) : list Z * R :=
  (map (fun x => clampz lv (fround x (fscale (m_imax M (QuantMax, path) (map absbits xs)) lv))) xs,
   fscale (m_imax M (QuantMax, path) (map absbits xs)) lv).

Definition quant_ref (lv : Z) (xs : list R) : list Z * R :=
  (map (fun x => clampz lv (fround x (fscale (zmax (map absbits xs)) lv))) xs,
   fscale (zmax (map absbits xs)) lv).

Lemma quant_value : forall dm T M path lv xs, valid dm T M -> quant M path lv xs = quant_ref lv xs.
Proof.
  intros dm T M path lv xs HM. unfold quant, quant_ref.
  rewrite (imax_is_zmax _ _ (v_imax _ _ _ HM (QuantMax, path))). reflexivity.
Qed.

Lemma quant_ref_range : forall lv xs, 0 <= lv -> Forall (fun q => Z.abs q <= lv) (fst (quant_ref lv xs)).
Proof.
  intros lv xs H. apply Forall_forall. intros q Hq. cbn [fst quant_ref] in Hq.
  apply in_map_iff in Hq. destruct Hq as [x [<- _]]. apply clampz_range. exact H.
Qed.

Lemma quant_ref_length : forall lv xs, length (fst (quant_ref lv xs)) = length xs.
Proof. intros. cbn. apply length_map. Qed.

Definition quant_rows (M : machine) (path : list nat) (tag : nat) (lv : Z) (rows : list (list R))
  : list (list Z * R) :=
  map (fun tr => quant M (fst tr :: tag :: path) lv (snd tr)) (enum rows).

Lemma quant_rows_value : forall dm T M path tag lv rows, valid dm T M ->
  quant_rows M path tag lv rows = map (quant_ref lv) rows.
Proof.
  intros. unfold quant_rows, enum.
  rewrite <- (map_snd_enum_from _ _ (quant_ref lv) rows 0).
  apply map_ext. intros. eapply quant_value. eassumption.
Qed.

Lemma quant_rows_range : forall lv rows, 0 <= lv ->
  Forall (fun k => Forall (fun q => Z.abs q <= lv) (fst k)) (map (quant_ref lv) rows).
Proof.
  intros lv rows H. apply Forall_forall. intros k Hk. apply in_map_iff in Hk.
  destruct Hk as [r [<- _]]. apply quant_ref_range, H.
Qed.

(** *** Linear layers: `ExactLinear.forward` and `_forward_split` *)

Record lin_w : Type := { w8 : list (list Z); wsc : list R; wb : list (option R) }.

Definition lin_acc (M : machine) (dm : dims) (w : lin) (path : list nat) (x8 : list Z) (n : nat)
  (row : list Z) : Z :=
  if a_of dm w <=? 127 then m_dot M (Lin w Plain, n :: path) (combine x8 row)
  else m_comb M (Lin w Comb, n :: path)
         (m_dot M (Lin w Hi, n :: path) (combine (map W.split_hi x8) row))
         (m_dot M (Lin w Lo, n :: path) (combine (map W.split_lo x8) row)).

Definition lin_tag (w : lin) : nat :=
  match w with QKV => 0 | OProj => 1 | Gate => 2 | Up => 3 | Down => 4 | Head => 5 end.

Definition linear (M : machine) (dm : dims) (w : lin) (path : list nat) (lw : lin_w) (xs : list R)
  : list R :=
  let q := quant M (lin_tag w :: path) (a_of dm w) xs in
  map (fun nr => fepi (lin_acc M dm w path (fst q) (fst nr) (snd nr)) (snd q)
                      (nth (fst nr) (wsc lw) rdef) (nth (fst nr) (wb lw) None))
      (enum (w8 lw)).

Definition lin_ok (dm : dims) (w : lin) (lw : lin_w) : Prop :=
  Forall (fun row => Z.of_nat (length row) <= K_of dm w /\ Forall (fun x => Z.abs x <= 127) row) (w8 lw).

Lemma split_dot : forall x8 row,
  128 * pdot (combine (map W.split_hi x8) row) + pdot (combine (map W.split_lo x8) row)
  = pdot (combine x8 row).
Proof.
  unfold pdot, prods. induction x8 as [|x x8 IH]; intros [|y row]; cbn [combine map ER.zsum fst snd]; try ring.
  rewrite <- IH. destruct (W.the_two_digits_recombine x) as [E _].
  replace (x * y) with ((W.split_hi x * 128 + W.split_lo x) * y) by (rewrite E; reflexivity). ring.
Qed.

Lemma split_abs : forall a x8 row, 0 <= a -> (a + 1) mod 128 = 0 ->
  Forall (fun x => Z.abs x <= a) x8 -> Forall (fun y => Z.abs y <= 127) row ->
  Z.abs (128 * pdot (combine (map W.split_hi x8) row)) + Z.abs (pdot (combine (map W.split_lo x8) row))
  <= Z.of_nat (length (combine x8 row)) * ((a + 128) * 127).
Proof.
  unfold pdot, prods. intros a x8. induction x8 as [|x x8 IH]; intros [|y row] Ha Hm Hx Hy;
    cbn [combine map ER.zsum fst snd length]; try lia.
  inversion Hx as [|? ? Hx0 Hx']; inversion Hy as [|? ? Hy0 Hy']; subst.
  specialize (IH row Ha Hm Hx' Hy').
  pose proof (W.the_split_terms_fit_the_accumulator a 127 x y Ha Hm Hx0 Hy0) as T1.
  rewrite Nat2Z.inj_succ.
  assert (E : 128 * (W.split_hi x * y + ER.zsum (map (fun xy => fst xy * snd xy) (combine (map W.split_hi x8) row)))
              = 128 * W.split_hi x * y + 128 * ER.zsum (map (fun xy => fst xy * snd xy) (combine (map W.split_hi x8) row)))
    by ring.
  rewrite E. lia.
Qed.

Lemma lin_acc_value : forall dm T M w path x8 n row, valid dm T M -> 0 <= dReq dm ->
  Forall (fun x => Z.abs x <= a_of dm w) x8 ->
  Z.of_nat (length row) <= K_of dm w -> Forall (fun y => Z.abs y <= 127) row ->
  lin_acc M dm w path x8 n row = pdot (combine x8 row).
Proof.
  intros dm T M w path x8 n row HM Hr Hx Hlen Hrow.
  destruct (a_of_props dm w Hr) as [[Ha1 Ha2] Hm]. unfold C.DIGIT_MAX in Ha2.
  pose proof (combine_len_r _ _ x8 row) as Lc.
  unfold lin_acc. destruct (Z.leb_spec (a_of dm w) 127) as [Hp|Hs].
  - apply (v_dot _ _ _ HM). unfold need. cbn [fst need_st].
    assert (Hx' : Forall (fun x => Z.abs x <= 127) x8)
      by (eapply Forall_impl; [|exact Hx]; intros x H; cbn in H; lia).
    pose proof (abs_sum_prods 127 127 (combine x8 row) ltac:(lia) ltac:(lia)
      (Forall_combine _ _ _ _ _ _ Hx' Hrow)). nia.
  - assert (Hhi : Forall (fun h => Z.abs h <= 128) (map W.split_hi x8)).
    { apply Forall_map. eapply Forall_impl; [|exact Hx]. intros x Hxa.
      pose proof (W.the_high_digit_is_bounded (a_of dm w) x ltac:(lia) Hm Hxa). lia. }
    assert (Hlo : Forall (fun h => Z.abs h <= 127) (map W.split_lo x8)).
    { apply Forall_map. apply Forall_forall. intros x _.
      destruct (W.the_two_digits_recombine x) as [_ B]. lia. }
    rewrite (v_dot _ _ _ HM (Lin w Hi, n :: path)).
    2:{ unfold need. cbn [fst need_st].
        pose proof (combine_len_r _ _ (map W.split_hi x8) row).
        pose proof (abs_sum_prods 128 127 _ ltac:(lia) ltac:(lia) (Forall_combine _ _ _ _ _ _ Hhi Hrow)). nia. }
    rewrite (v_dot _ _ _ HM (Lin w Lo, n :: path)).
    2:{ unfold need. cbn [fst need_st].
        pose proof (combine_len_r _ _ (map W.split_lo x8) row).
        pose proof (abs_sum_prods 127 127 _ ltac:(lia) ltac:(lia) (Forall_combine _ _ _ _ _ _ Hlo Hrow)). nia. }
    rewrite (v_comb _ _ _ HM (Lin w Comb, n :: path)).
    + apply split_dot.
    + unfold need. cbn [fst need_st].
      pose proof (split_abs (a_of dm w) x8 row ltac:(lia) Hm Hx Hrow). nia.
Qed.

Lemma linear_agree : forall dm T M w path lw xs, valid dm T M -> 0 <= dReq dm -> lin_ok dm w lw ->
  linear M dm w path lw xs = linear Mref dm w path lw xs.
Proof.
  intros dm T M w path lw xs HM Hr Hok. unfold linear.
  rewrite (quant_value dm T M), (quant_value dm T Mref) by (assumption || apply the_reference_machine_is_valid).
  apply map_ext_in. intros [n row] Hin. cbn [fst snd]. f_equal.
  assert (Hrow : Z.of_nat (length row) <= K_of dm w /\ Forall (fun x => Z.abs x <= 127) row).
  { unfold lin_ok in Hok. rewrite Forall_forall in Hok. apply Hok. eapply in_enum. exact Hin. }
  assert (Hx : Forall (fun x => Z.abs x <= a_of dm w) (fst (quant_ref (a_of dm w) xs))).
  { apply quant_ref_range. destruct (a_of_props dm w Hr). lia. }
  rewrite (lin_acc_value dm T M), (lin_acc_value dm T Mref);
    try tauto; try assumption; apply the_reference_machine_is_valid.
Qed.

(** *** RMSNorm: `sum_of_squares` and `ExactRMSNorm.forward` *)

Definition norm_e (M : machine) (path : list nat) (xs : list R) : Z :=
  m_imax M (NormExp, path) (map (fun x => snd (W.f32_square (bits x))) xs).

Definition norm_tail (e big0 small0 : Z) (xs ws : list R) : list R :=
  map (fun xw => fnorm (big0 + small0 / 2 ^ 40) (small0 mod 2 ^ 40) e (fst xw) (snd xw)) (combine xs ws).

Definition norm (M : machine) (path : list nat) (xs ws : list R) : list R :=
  norm_tail (norm_e M path xs)
    (m_sum M (NormHi, path) (map (fun x => W.norm_hi (norm_e M path xs) (bits x)) xs))
    (m_sum M (NormLo, path) (map (fun x => W.norm_lo (norm_e M path xs) (bits x)) xs))
    xs ws.

Lemma limb_sum_bound : forall (f : R -> Z) (xs : list R) H,
  (forall x, 0 <= f x < 2 ^ 40) -> Z.of_nat (length xs) <= H ->
  ER.abs_sum (map f xs) <= H * 2 ^ 40.
Proof.
  intros f xs H Hf Hl.
  assert (F : Forall (fun z => Z.abs z <= 2 ^ 40) (map f xs)).
  { apply Forall_map, Forall_forall. intros x _. specialize (Hf x). lia. }
  pose proof (ER.zsum_bounded (2 ^ 40) (map f xs) ltac:(lia) F). rewrite length_map in H0. nia.
Qed.

Lemma norm_agree : forall dm T M path xs ws, valid dm T M ->
  Z.of_nat (length xs) <= dH dm -> norm M path xs ws = norm Mref path xs ws.
Proof.
  intros dm T M path xs ws HM Hl. unfold norm, norm_e.
  rewrite (imax_is_zmax _ _ (v_imax _ _ _ HM (NormExp, path))). cbn [m_imax m_sum Mref].
  set (e := zmax (map (fun x => snd (W.f32_square (bits x))) xs)).
  rewrite (v_sum _ _ _ HM (NormHi, path)).
  2:{ unfold need. cbn [fst need_st]. apply limb_sum_bound; [|exact Hl].
      intros x. apply W.the_norm_limbs_are_in_range. }
  rewrite (v_sum _ _ _ HM (NormLo, path)).
  2:{ unfold need. cbn [fst need_st]. apply limb_sum_bound with (f := fun x => W.norm_lo e (bits x)); [|exact Hl].
      intros x. apply W.the_norm_limbs_are_in_range. }
  reflexivity.
Qed.

Lemma norm_length : forall M path xs ws, (length (norm M path xs ws) <= length xs)%nat.
Proof. intros. unfold norm, norm_tail. rewrite length_map. apply combine_len_l. Qed.

(** *** Attention: `exact_attention_core` for one query position *)

Definition scores (M : machine) (path : list nat) (qi : list Z) (kq : list (list Z * R)) : list Z :=
  map (fun tk => m_dot M (Score, fst tk :: path) (combine qi (fst (snd tk)))) (enum kq).

Definition logits (a : R) (s : list Z) (kq : list (list Z * R)) : list R :=
  map (fun sk => flogit (fst sk) a (snd (snd sk))) (combine s kq).

(** The logits the max reads: `torch.where(blocked, NEG, logit)`. *)
Definition mask_logits (lg : list R) (bl : list bool) : list R :=
  map (fun xb : R * bool => if snd xb then NEG else fst xb) (combine lg bl).

(** `p = exp2(clamp(round(m - logit), 0, 2^30))`, and 0 at a masked key
    (the code sets the argument to 2^30 there, where `exp2` is 0, then sets
    p to 0). *)
Definition probs (m : R) (lg : list R) (bl : list bool) : list Z :=
  map (fun xb : R * bool => if snd xb then 0 else W.exp2_neg_q16_16 (clamp0 (2 ^ 30) (targ m (fst xb))))
      (combine lg bl).

Definition folded (p : list Z) (vq : list (list Z * R)) : list R :=
  map (fun pv => fw (fst pv) (snd (snd pv))) (combine p vq).

Definition wint (wmax : R) (w : list R) : list Z :=
  map (fun x => clamp0 (2 ^ 28) (fold_round x wmax)) w.

(** Attention's p.V: one sum, or the digit split recombined in float64. *)
Definition pv (M : machine) (path : list nat) (Wt vcol : list Z) : Z :=
  let d := m_digits M path in
  if d <=? 0 then m_dot M (PvDirect, path) (combine Wt vcol)
  else m_sum M (PvRecomb, path)
         (map (fun j => ER.rne 53 (2 ^ (Z.of_nat j * d)
                         * m_dot M (PvDigit j, path) (combine (map (digit d j) Wt) vcol)))
              (seq 0 (ndig d))).

Definition attend_out (M : machine) (dm : dims) (path : list nat) (Wt : list Z)
  (vq : list (list Z * R)) (ws : R) (l : Z) : list R :=
  map (fun d => fout (pv M (d :: path) Wt (map (fun v => nth d (fst v) 0) vq)) ws l)
      (seq 0 (Z.to_nat (dD dm))).

Definition attend5 (M : machine) (dm : dims) (path : list nat) (w : list R) (vq : list (list Z * R)) (l : Z) :=
  attend_out M dm path (wint (m_fmax M (FoldMax, path) w) w) vq (fold_scale (m_fmax M (FoldMax, path) w)) l.

Definition attend4 (M : machine) (dm : dims) (path : list nat) (p : list Z) (vq : list (list Z * R)) :=
  attend5 M dm path (folded p vq) vq (m_sum M (Denom, path) p).

Definition attend3 (M : machine) (dm : dims) (path : list nat) (lg : list R) (bl : list bool)
  (vq : list (list Z * R)) :=
  attend4 M dm path (probs (m_fmax M (LogitMax, path) (mask_logits lg bl)) lg bl) vq.

Definition attend2 (M : machine) (dm : dims) (path : list nat) (qq : list Z * R)
  (kq vq : list (list Z * R)) (bl : list bool) :=
  attend3 M dm path (logits (fqa (snd qq)) (scores M path (fst qq) kq) kq) bl vq.

Definition attend (M : machine) (dm : dims) (path : list nat) (q : list R) (ks vs : list (list R))
  (bl : list bool) :=
  attend2 M dm path (quant M (0%nat :: path) C.Q_LEVELS q)
    (quant_rows M path 1 (kl_of dm) ks) (quant_rows M path 2 C.V_LEVELS vs) bl.

(** The `p.V` value, in either mode. *)
Lemma pv_value : forall dm T M path Wt vcol, valid dm T M ->
  Forall (fun x => 0 <= x <= 2 ^ 28) Wt -> Forall (fun v => Z.abs v <= C.V_LEVELS) vcol ->
  Z.of_nat (length Wt) <= T -> T * 2 ^ 28 * C.V_LEVELS <= 2 ^ 53 ->
  pv M path Wt vcol = pdot (combine Wt vcol).
Proof.
  intros dm T M path Wt vcol HM HW Hv HT H53. unfold C.V_LEVELS in *.
  pose proof (combine_len_l _ _ Wt vcol) as Lc.
  assert (HlT : Z.of_nat (length (combine Wt vcol)) <= T) by lia.
  set (l := combine Wt vcol) in *.
  assert (Fl : Forall (fun xy => 0 <= fst xy <= 2 ^ 28 /\ Z.abs (snd xy) <= 127) l)
    by (exact (Forall_combine _ _ (fun x => 0 <= x <= 2 ^ 28) (fun v => Z.abs v <= 127) Wt vcol HW Hv)).
  unfold pv. set (d := m_digits M path).
  destruct (Z.leb_spec d 0) as [Hd|Hd].
  - apply (v_dot _ _ _ HM). unfold need. cbn [fst snd need_st]. unfold C.V_LEVELS.
    assert (Fl' : Forall (fun xy => Z.abs (fst xy) <= 2 ^ 28 /\ Z.abs (snd xy) <= 127) l)
      by (eapply Forall_impl; [|exact Fl]; intros xy H; cbn in H; lia).
    pose proof (abs_sum_prods (2 ^ 28) 127 l ltac:(lia) ltac:(lia) Fl').
    change (2 ^ 28) with 268435456 in *. nia.
  - (* each digit sum is exact *)
    assert (Ed : forall j, m_dot M (PvDigit j, path) (combine (map (digit d j) Wt) vcol)
                         = ER.zsum (map (fun wv => digit d j (fst wv) * snd wv) l)).
    { intros j. rewrite (v_dot _ _ _ HM (PvDigit j, path)).
      - unfold pdot, prods, l. f_equal. clear. revert vcol.
        induction Wt as [|x Wt IH]; intros [|v vcol]; cbn; try reflexivity. f_equal. apply IH.
      - unfold need. cbn [fst snd need_st]. unfold C.V_LEVELS. fold d.
        assert (Fd : Forall (fun xy => Z.abs (fst xy) <= 2 ^ d - 1 /\ Z.abs (snd xy) <= 127)
                       (combine (map (digit d j) Wt) vcol)).
        { apply (Forall_combine _ _ (fun x => Z.abs x <= 2 ^ d - 1) (fun v => Z.abs v <= 127)); [|exact Hv].
          apply Forall_map, Forall_forall. intros x _.
          pose proof (digit_range d j x ltac:(lia)). lia. }
        assert (0 <= 2 ^ d - 1) by (pose proof (Z.pow_pos_nonneg 2 d ltac:(lia) ltac:(lia)); lia).
        pose proof (abs_sum_prods (2 ^ d - 1) 127 _ ltac:(lia) ltac:(lia) Fd) as Hab.
        pose proof (combine_len_l _ _ (map (digit d j) Wt) vcol) as Lj.
        rewrite length_map in Lj.
        assert (Hj : Z.of_nat (length (combine (map (digit d j) Wt) vcol)) <= T) by lia.
        set (P := 2 ^ d - 1) in *. nia. }
    (* the bound on each recombined term *)
    set (A := fun j => ER.zsum (map (fun wv => 2 ^ (Z.of_nat j * d) * digit d j (fst wv) * Z.abs (snd wv)) l)).
    assert (Hterm : forall j, Z.abs (2 ^ (Z.of_nat j * d) * ER.zsum (map (fun wv => digit d j (fst wv) * snd wv) l)) <= A j).
    { intros j. unfold A. clear -Hd Fl. induction l as [|[w v] l IH]; cbn; [lia|].
      inversion Fl as [|? ? [Hw Hv] Fl']; subst. specialize (IH Fl').
      pose proof (digit_range d j w ltac:(lia)).
      assert (0 <= 2 ^ (Z.of_nat j * d)) by (apply Z.pow_nonneg; lia).
      rewrite Z.mul_add_distr_l.
      assert (Z.abs (2 ^ (Z.of_nat j * d) * (digit d j w * v)) = 2 ^ (Z.of_nat j * d) * digit d j w * Z.abs v)
        by (rewrite !Z.abs_mul; rewrite (Z.abs_eq (2 ^ _)) by lia; rewrite (Z.abs_eq (digit _ _ _)) by lia; ring).
      lia. }
    assert (HA0 : forall j, 0 <= A j).
    { intros j. unfold A. clear -Hd. induction l as [|[w v] l IH]; cbn; [lia|].
      pose proof (digit_range d j w ltac:(lia)).
      assert (0 <= 2 ^ (Z.of_nat j * d)) by (apply Z.pow_nonneg; lia). nia. }
    assert (HAsum : ER.zsum (map A (seq 0 (ndig d))) = ER.zsum (map (fun wv => fst wv * Z.abs (snd wv)) l)).
    { unfold A.
      transitivity (ER.zsum (map (fun j => 2 ^ (Z.of_nat j * d)
                       * ER.zsum (map (fun wv => digit d j (fst wv) * Z.abs (snd wv)) l)) (seq 0 (ndig d)))).
      { f_equal. apply map_ext. intros j. clear. induction l as [|wv l IH]; cbn; [ring|]. rewrite IH. ring. }
      rewrite (swap_sums l (seq 0 (ndig d)) (fun j => 2 ^ (Z.of_nat j * d)) (digit d) Z.abs).
      f_equal. apply map_ext_in. intros [w v] Hin. cbn [fst snd].
      rewrite Forall_forall in Fl. specialize (Fl _ Hin). cbn in Fl.
      rewrite the_digits_recombine by lia. reflexivity. }
    assert (HAtot : ER.zsum (map (fun wv => fst wv * Z.abs (snd wv)) l) <= T * 2 ^ 28 * 127).
    { assert (ER.zsum (map (fun wv => fst wv * Z.abs (snd wv)) l) <= Z.of_nat (length l) * (2 ^ 28 * 127)).
      { clear -Fl. induction Fl as [|[w v] l [Hw Hv] Fl IH]; cbn [map ER.zsum length fst snd] in *; [lia|].
        rewrite Nat2Z.inj_succ. nia. }
      nia. }
    assert (Hle : forall j, In j (seq 0 (ndig d)) -> A j <= ER.zsum (map A (seq 0 (ndig d)))).
    { clear -HA0. induction (seq 0 (ndig d)) as [|k ks IH]; intros j Hj; [destruct Hj|].
      cbn. destruct Hj as [<-|Hj].
      - assert (0 <= ER.zsum (map A ks)).
        { clear IH. induction ks as [|k' ks IH']; cbn; [lia|]. specialize (HA0 k'). lia. }
        lia.
      - specialize (IH j Hj). specialize (HA0 k). lia. }
    (* the float64 power-of-two scaling is exact, and so is the recombination *)
    assert (Eterms : map (fun j => ER.rne 53 (2 ^ (Z.of_nat j * d)
                       * m_dot M (PvDigit j, path) (combine (map (digit d j) Wt) vcol))) (seq 0 (ndig d))
                     = map (fun j => 2 ^ (Z.of_nat j * d) * ER.zsum (map (fun wv => digit d j (fst wv) * snd wv) l))
                           (seq 0 (ndig d))).
    { apply map_ext_in. intros j Hj. rewrite Ed. apply ER.rne_exact; [lia|].
      specialize (Hterm j). specialize (Hle j Hj). lia. }
    rewrite Eterms.
    rewrite (v_sum _ _ _ HM (PvRecomb, path)).
    + rewrite (swap_sums l (seq 0 (ndig d)) (fun j => 2 ^ (Z.of_nat j * d)) (digit d) (fun v => v)).
      unfold pdot, prods. fold l. f_equal. apply map_ext_in. intros [w v] Hin. cbn [fst snd].
      rewrite Forall_forall in Fl. specialize (Fl _ Hin). cbn in Fl.
      rewrite the_digits_recombine by lia. reflexivity.
    + unfold need. cbn [fst snd need_st]. unfold C.V_LEVELS.
      assert (ER.abs_sum (map (fun j => 2 ^ (Z.of_nat j * d)
                * ER.zsum (map (fun wv => digit d j (fst wv) * snd wv) l)) (seq 0 (ndig d)))
              <= ER.zsum (map A (seq 0 (ndig d)))).
      { unfold ER.abs_sum. rewrite map_map. clear -Hterm.
        induction (seq 0 (ndig d)) as [|k ks IH]; cbn; [lia|]. specialize (Hterm k). lia. }
      lia.
Qed.

Lemma pv_agree : forall dm T M path Wt vcol, valid dm T M ->
  Forall (fun x => 0 <= x <= 2 ^ 28) Wt -> Forall (fun v => Z.abs v <= C.V_LEVELS) vcol ->
  Z.of_nat (length Wt) <= T -> T * 2 ^ 28 * C.V_LEVELS <= 2 ^ 53 ->
  pv M path Wt vcol = pv Mref path Wt vcol.
Proof.
  intros. rewrite (pv_value dm T M), (pv_value dm T Mref);
    try assumption; try reflexivity; apply the_reference_machine_is_valid.
Qed.

Definition vq_ok (vq : list (list Z * R)) : Prop :=
  Forall (fun k => Forall (fun q => Z.abs q <= C.V_LEVELS) (fst k)) vq.

Lemma attend5_agree : forall dm T M path w vq l, valid dm T M ->
  vq_ok vq -> Z.of_nat (length w) <= T -> T * 2 ^ 28 * C.V_LEVELS <= 2 ^ 53 ->
  attend5 M dm path w vq l = attend5 Mref dm path w vq l.
Proof.
  intros dm T M path w vq l HM Hv Hl H53. unfold attend5. cbn [m_fmax Mref].
  pose proof (fmax_ival _ w (v_fmax _ _ _ HM (FoldMax, path))) as Iv.
  destruct (fold_ival _ _ rdef Iv) as [_ Es]. rewrite Es.
  assert (Ew : wint (m_fmax M (FoldMax, path) w) w = wint (fmaxr w) w).
  { unfold wint. apply map_ext. intros x. f_equal. apply (fold_ival _ _ x Iv). }
  rewrite Ew. unfold attend_out. apply map_ext. intros d. f_equal.
  apply (pv_agree dm T M); try assumption.
  - apply Forall_forall. intros x Hx. unfold wint in Hx. apply in_map_iff in Hx.
    destruct Hx as [y [<- _]]. apply clamp0_range. lia.
  - apply Forall_forall. intros v Hin. apply in_map_iff in Hin. destruct Hin as [k [<- Hk]].
    unfold vq_ok in Hv. rewrite Forall_forall in Hv. specialize (Hv k Hk).
    destruct (Nat.lt_ge_cases d (length (fst k))) as [Hd|Hd].
    + rewrite Forall_forall in Hv. apply Hv, nth_In, Hd.
    + rewrite nth_overflow by exact Hd. unfold C.V_LEVELS. lia.
  - unfold wint. rewrite length_map. exact Hl.
Qed.

Lemma attend4_agree : forall dm T M path p vq, valid dm T M ->
  vq_ok vq -> Forall (fun x => 0 <= x <= 2 ^ 28) p -> Z.of_nat (length p) <= T ->
  T * 2 ^ 28 * C.V_LEVELS <= 2 ^ 53 ->
  attend4 M dm path p vq = attend4 Mref dm path p vq.
Proof.
  intros dm T M path p vq HM Hv Hp Hl H53. unfold attend4. cbn [m_sum Mref].
  rewrite (v_sum _ _ _ HM (Denom, path)).
  - apply (attend5_agree dm T M); try assumption.
    unfold folded. rewrite length_map. pose proof (combine_len_l _ _ p vq). lia.
  - unfold need. cbn [fst need_st].
    assert (Hp' : Forall (fun x => Z.abs x <= 2 ^ 28) p)
      by (eapply Forall_impl; [|exact Hp]; intros x H; cbn in H; lia).
    pose proof (ER.zsum_bounded (2 ^ 28) p ltac:(lia) Hp'). nia.
Qed.

Lemma probs_range : forall m lg bl, Forall (fun x => 0 <= x <= 2 ^ 28) (probs m lg bl).
Proof.
  intros. apply Forall_forall. intros x Hx. unfold probs in Hx. apply in_map_iff in Hx.
  destruct Hx as [y [<- _]]. destruct (snd y); [lia|]. apply W.the_integer_exp_is_a_q28_weight.
Qed.

Lemma probs_length : forall m lg bl, (length (probs m lg bl) <= length lg)%nat.
Proof. intros. unfold probs. rewrite length_map. apply combine_len_l. Qed.

Lemma attend3_agree : forall dm T M path lg bl vq, valid dm T M ->
  vq_ok vq -> Z.of_nat (length lg) <= T -> T * 2 ^ 28 * C.V_LEVELS <= 2 ^ 53 ->
  attend3 M dm path lg bl vq = attend3 Mref dm path lg bl vq.
Proof.
  intros dm T M path lg bl vq HM Hv Hl H53. unfold attend3. cbn [m_fmax Mref].
  assert (Ep : probs (m_fmax M (LogitMax, path) (mask_logits lg bl)) lg bl
               = probs (fmaxr (mask_logits lg bl)) lg bl).
  { unfold probs. apply map_ext. intros xb. destruct (snd xb); [reflexivity|].
    f_equal. f_equal. apply targ_ival.
    apply (fmax_ival _ _ (v_fmax _ _ _ HM (LogitMax, path))). }
  rewrite Ep. apply (attend4_agree dm T M); try assumption.
  - apply probs_range.
  - pose proof (probs_length (fmaxr (mask_logits lg bl)) lg bl). lia.
Qed.

Lemma attend2_agree : forall dm T M path qq kq vq bl, valid dm T M ->
  Forall (fun q => Z.abs q <= C.Q_LEVELS) (fst qq) -> Z.of_nat (length (fst qq)) <= dD dm ->
  Forall (fun k => Forall (fun q => Z.abs q <= kl_of dm) (fst k)) kq -> Z.of_nat (length kq) <= T ->
  vq_ok vq -> T * 2 ^ 28 * C.V_LEVELS <= 2 ^ 53 ->
  attend2 M dm path qq kq vq bl = attend2 Mref dm path qq kq vq bl.
Proof.
  intros dm T M path qq kq vq bl HM Hq Hql Hk Hkl Hv H53. unfold attend2.
  assert (Es : scores M path (fst qq) kq = scores Mref path (fst qq) kq).
  { unfold scores. apply map_ext_in. intros [t k] Hin. cbn [fst snd m_dot Mref].
    apply (v_dot _ _ _ HM). unfold need. cbn [fst need_st].
    assert (Hk' : Forall (fun q => Z.abs q <= kl_of dm) (fst k)).
    { rewrite Forall_forall in Hk. apply Hk. eapply in_enum. exact Hin. }
    pose proof (kl_of_ge dm). unfold C.Q_LEVELS in *.
    pose proof (abs_sum_prods 127 (kl_of dm) _ ltac:(lia) ltac:(lia) (Forall_combine _ _ _ _ _ _ Hq Hk')).
    pose proof (combine_len_l _ _ (fst qq) (fst k)). nia. }
  rewrite Es. apply (attend3_agree dm T M); try assumption.
  unfold logits. rewrite length_map. pose proof (combine_len_r _ _ (scores Mref path (fst qq) kq) kq). lia.
Qed.

Lemma attend_agree : forall dm T M path q ks vs bl, valid dm T M ->
  Z.of_nat (length q) <= dD dm -> Z.of_nat (length ks) <= T -> T * 2 ^ 28 * C.V_LEVELS <= 2 ^ 53 ->
  attend M dm path q ks vs bl = attend Mref dm path q ks vs bl.
Proof.
  intros dm T M path q ks vs bl HM Hq Hk H53. pose proof (the_reference_machine_is_valid dm T) as HR.
  unfold attend.
  rewrite (quant_value dm T M), (quant_value dm T Mref), (quant_rows_value dm T M),
    (quant_rows_value dm T Mref), (quant_rows_value dm T M path 2), (quant_rows_value dm T Mref path 2)
    by assumption.
  apply (attend2_agree dm T M); try assumption.
  - apply quant_ref_range. unfold C.Q_LEVELS. lia.
  - rewrite quant_ref_length. exact Hq.
  - apply quant_rows_range. pose proof (kl_of_ge dm). lia.
  - rewrite length_map. exact Hk.
  - apply quant_rows_range. unfold C.V_LEVELS. lia.
Qed.

(** *** A decoder layer *)

Definition heads (d n : nat) (l : list R) : list (list R) :=
  map (fun i => firstn d (skipn (i * d) l)) (seq 0 n).

Definition dn (dm : dims) : nat := Z.to_nat (dD dm).
Definition nh (dm : dims) : nat := Z.to_nat (dNH dm).
Definition nkv (dm : dims) : nat := Z.to_nat (dNKV dm).
Definition rep (dm : dims) : nat := Nat.max 1 (nh dm / nkv dm).

Definition qs_of (dm : dims) (pos : Z) (qkv : list R) : list (list R) :=
  map (frope pos) (heads (dn dm) (nh dm) qkv).
Definition ks_of (dm : dims) (pos : Z) (qkv : list R) : list (list R) :=
  map (frope pos) (heads (dn dm) (nkv dm) (skipn (nh dm * dn dm) qkv)).
Definition vs_of (dm : dims) (qkv : list R) : list (list R) :=
  heads (dn dm) (nkv dm) (skipn ((nh dm + nkv dm) * dn dm) qkv).

(** The cache of each KV head with this step's row appended. *)
Definition grow (c : list (list (list R))) (new : list (list R)) (n : nat) : list (list (list R)) :=
  map (fun g => nth g c [] ++ [nth g new []]) (seq 0 n).

Definition attn_all (M : machine) (dm : dims) (li : nat) (qs : list (list R))
  (ck cv : list (list (list R))) (bl : list bool) : list R :=
  flat_map (fun h => attend M dm [h; li] (nth h qs []) (nth (h / rep dm) ck []) (nth (h / rep dm) cv []) bl)
           (seq 0 (nh dm)).

Record layer_w : Type := {
  ln1 : list R; ln2 : list R;
  lqkv : lin_w; low : lin_w; lgate : lin_w; lup : lin_w; ldown : lin_w
}.

Definition mlp (M : machine) (dm : dims) (li : nat) (lw : layer_w) (x2 : list R) : list R :=
  zipw fadd x2
    (linear M dm Down [li] (ldown lw)
       (zipw fsilu_mul (linear M dm Gate [li] (lgate lw) (norm M [1%nat; li] x2 (ln2 lw)))
                       (linear M dm Up [li] (lup lw) (norm M [1%nat; li] x2 (ln2 lw))))).

Definition layer_qkv (M : machine) (dm : dims) (li : nat) (pos : Z) (lw : layer_w)
  (ck cv : list (list (list R))) (bl : list bool) (x qkv : list R)
  : list R * (list (list R) * list (list R)) :=
  (mlp M dm li lw
     (zipw fadd x (linear M dm OProj [li] (low lw)
        (attn_all M dm li (qs_of dm pos qkv) (grow ck (ks_of dm pos qkv) (nkv dm))
                  (grow cv (vs_of dm qkv) (nkv dm)) (bl ++ [false])))),
   (ks_of dm pos qkv, vs_of dm qkv)).

Definition layer (M : machine) (dm : dims) (li : nat) (pos : Z) (lw : layer_w)
  (ck cv : list (list (list R))) (bl : list bool) (x : list R)
  : list R * (list (list R) * list (list R)) :=
  layer_qkv M dm li pos lw ck cv bl x (linear M dm QKV [li] (lqkv lw) (norm M [0%nat; li] x (ln1 lw))).

Definition layer_ok (dm : dims) (lw : layer_w) : Prop :=
  lin_ok dm QKV (lqkv lw) /\ lin_ok dm OProj (low lw) /\ lin_ok dm Gate (lgate lw)
  /\ lin_ok dm Up (lup lw) /\ lin_ok dm Down (ldown lw).

(** Every cached key row of a layer: fewer than T of them per KV head. *)
Definition cache_ok (T : Z) (ck : list (list (list R))) : Prop :=
  Forall (fun c => Z.of_nat (length c) + 1 <= T) ck.

Lemma grow_length : forall T c new n g, 1 <= T -> cache_ok T c ->
  Z.of_nat (length (nth g (grow c new n) [])) <= T.
Proof.
  intros T c new n g HT Hc.
  destruct (Nat.lt_ge_cases g (length (grow c new n))) as [Hg|Hg].
  - pose proof (nth_In (grow c new n) [] Hg) as Hin. unfold grow in Hin |- *.
    apply in_map_iff in Hin. destruct Hin as [g' [E _]]. rewrite <- E.
    rewrite length_app. cbn [length].
    destruct (Nat.lt_ge_cases g' (length c)) as [Hc'|Hc'].
    + unfold cache_ok in Hc. rewrite Forall_forall in Hc. specialize (Hc _ (nth_In c [] Hc')). lia.
    + rewrite nth_overflow by exact Hc'. cbn. lia.
  - rewrite nth_overflow by exact Hg. cbn. lia.
Qed.

Lemma in_heads_length : forall d n l x, In x (heads d n l) -> (length x <= d)%nat.
Proof.
  intros d n l x H. unfold heads in H. apply in_map_iff in H. destruct H as [i [<- _]].
  rewrite length_firstn. lia.
Qed.

Lemma qs_length : forall dm pos qkv h, 0 <= dD dm ->
  Z.of_nat (length (nth h (qs_of dm pos qkv) [])) <= dD dm.
Proof.
  intros dm pos qkv h HD. unfold qs_of.
  destruct (Nat.lt_ge_cases h (length (map (frope pos) (heads (dn dm) (nh dm) qkv)))) as [Hh|Hh].
  - pose proof (nth_In _ [] Hh) as Hin. apply in_map_iff in Hin. destruct Hin as [x [E Hx]].
    rewrite <- E, frope_length. pose proof (in_heads_length _ _ _ _ Hx). unfold dn in *. lia.
  - rewrite nth_overflow by exact Hh. cbn. lia.
Qed.

Lemma attn_all_agree : forall dm T M li qs ck cv bl, valid dm T M -> 1 <= T ->
  T * 2 ^ 28 * C.V_LEVELS <= 2 ^ 53 ->
  (forall h, Z.of_nat (length (nth h qs [])) <= dD dm) ->
  (forall g, Z.of_nat (length (nth g ck [])) <= T) ->
  attn_all M dm li qs ck cv bl = attn_all Mref dm li qs ck cv bl.
Proof.
  intros dm T M li qs ck cv bl HM HT H53 Hq Hk. unfold attn_all. apply flat_map_ext_in.
  intros h _. apply (attend_agree dm T M); auto.
Qed.

Lemma layer_agree : forall dm T M li pos lw ck cv bl x, valid dm T M ->
  0 <= dReq dm -> 0 <= dD dm -> 1 <= T -> T * 2 ^ 28 * C.V_LEVELS <= 2 ^ 53 ->
  layer_ok dm lw -> cache_ok T ck -> Z.of_nat (length x) <= dH dm ->
  layer M dm li pos lw ck cv bl x = layer Mref dm li pos lw ck cv bl x.
Proof.
  intros dm T M li pos lw ck cv bl x HM Hr HD HT H53 [Hqkv [Ho [Hg [Hu Hd]]]] Hc Hx.
  unfold layer. rewrite (norm_agree dm T M) by assumption.
  rewrite (linear_agree dm T M) by assumption.
  unfold layer_qkv. rewrite (attn_all_agree dm T M); try assumption.
  2:{ intros h. apply qs_length, HD. }
  2:{ intros g. apply grow_length; assumption. }
  rewrite (linear_agree dm T M) by assumption. unfold mlp.
  assert (Hx2 : forall o, Z.of_nat (length (zipw fadd x o)) <= dH dm)
    by (intros o; pose proof (zipw_len _ _ _ fadd x o); lia).
  rewrite (norm_agree dm T M) by (assumption || apply Hx2).
  rewrite (linear_agree dm T M Gate), (linear_agree dm T M Up), (linear_agree dm T M Down)
    by assumption.
  reflexivity.
Qed.

Lemma layer_length : forall M dm li pos lw ck cv bl x,
  (length (fst (layer M dm li pos lw ck cv bl x)) <= length x)%nat.
Proof.
  intros. unfold layer, layer_qkv, mlp. cbn [fst].
  eapply Nat.le_trans; [apply zipw_len | apply zipw_len].
Qed.

(** *** The step: all layers, the final norm, the LM head, the argmax *)

Definition cache : Type := (list (list (list R)) * list (list (list R)))%type.

Fixpoint run_layers (M : machine) (dm : dims) (pos : Z) (bl : list bool) (li : nat)
  (ls : list (layer_w * cache))
  (x : list R) : list R * list (list (list R) * list (list R)) :=
  match ls with
  | [] => (x, [])
  | lc :: rest =>
      (fst (run_layers M dm pos bl (S li) rest (fst (layer M dm li pos (fst lc) (fst (snd lc)) (snd (snd lc)) bl x))),
       snd (layer M dm li pos (fst lc) (fst (snd lc)) (snd (snd lc)) bl x)
       :: snd (run_layers M dm pos bl (S li) rest (fst (layer M dm li pos (fst lc) (fst (snd lc)) (snd (snd lc)) bl x))))
  end.

Record model_w : Type := { mlayers : list layer_w; mnorm : list R; mhead : lin_w }.

Definition step (M : machine) (dm : dims) (mw : model_w) (caches : list cache) (bl : list bool)
  (tok : nat) (pos : Z)
  : nat * list (list (list R) * list (list R)) :=
  (m_argmax M (linear M dm Head [0%nat] (mhead mw)
     (norm M [2%nat] (fst (run_layers M dm pos bl 0 (combine (mlayers mw) caches) (embed tok))) (mnorm mw))),
   snd (run_layers M dm pos bl 0 (combine (mlayers mw) caches) (embed tok))).

Definition step_premises (dm : dims) (T : Z) (mw : model_w) (caches : list cache) (tok : nat) : Prop :=
  0 <= dReq dm /\ 0 <= dD dm /\ 1 <= T /\ T * 2 ^ 28 * C.V_LEVELS <= 2 ^ 53
  /\ Forall (layer_ok dm) (mlayers mw) /\ lin_ok dm Head (mhead mw)
  /\ Forall (fun c => cache_ok T (fst c)) caches
  /\ Z.of_nat (length (embed tok)) <= dH dm.

Lemma run_layers_agree : forall dm T M pos bl ls li x, valid dm T M ->
  0 <= dReq dm -> 0 <= dD dm -> 1 <= T -> T * 2 ^ 28 * C.V_LEVELS <= 2 ^ 53 ->
  Forall (fun lc => layer_ok dm (fst lc) /\ cache_ok T (fst (snd lc))) ls ->
  Z.of_nat (length x) <= dH dm ->
  run_layers M dm pos bl li ls x = run_layers Mref dm pos bl li ls x
  /\ (length (fst (run_layers M dm pos bl li ls x)) <= length x)%nat.
Proof.
  intros dm T M pos bl ls. induction ls as [|lc ls IH]; intros li x HM Hr HD HT H53 Hls Hx.
  - split; reflexivity.
  - inversion Hls as [|? ? [Hl Hc] Hls']; subst. cbn [run_layers].
    rewrite (layer_agree dm T M) by assumption.
    pose proof (layer_length Mref dm li pos (fst lc) (fst (snd lc)) (snd (snd lc)) bl x) as Ll.
    destruct (IH (S li) (fst (layer Mref dm li pos (fst lc) (fst (snd lc)) (snd (snd lc)) bl x))
                 HM Hr HD HT H53 Hls' ltac:(lia)) as [E L].
    split.
    + f_equal; [f_equal; exact E | f_equal; f_equal; exact E].
    + cbn [fst]. eapply Nat.le_trans; [exact L | exact Ll].
Qed.

(** ** THE THEOREM *)

Theorem every_valid_execution_takes_the_same_decode_step :
  forall dm T M1 M2 mw caches bl tok pos,
    step_premises dm T mw caches tok ->
    valid dm T M1 -> valid dm T M2 ->
    step M1 dm mw caches bl tok pos = step M2 dm mw caches bl tok pos.
Proof.
  assert (Key : forall dm T M mw caches bl tok pos, step_premises dm T mw caches tok -> valid dm T M ->
            step M dm mw caches bl tok pos = step Mref dm mw caches bl tok pos).
  { intros dm T M mw caches bl tok pos [Hr [HD [HT [H53 [Hl [Hh [Hc Hx]]]]]]] HM.
    assert (Hls : Forall (fun lc => layer_ok dm (fst lc) /\ cache_ok T (fst (snd lc)))
                         (combine (mlayers mw) caches)).
    { apply Forall_forall. intros [lw c] Hin. cbn [fst snd]. split.
      - rewrite Forall_forall in Hl. apply Hl. eapply in_combine_l. exact Hin.
      - rewrite Forall_forall in Hc. apply (Hc c). eapply in_combine_r. exact Hin. }
    destruct (run_layers_agree dm T M pos bl _ 0 (embed tok) HM Hr HD HT H53 Hls Hx) as [E L].
    unfold step. rewrite E. rewrite E in L.
    rewrite (norm_agree dm T M) by (assumption || lia).
    rewrite (linear_agree dm T M) by assumption.
    rewrite (argmax_is_ref _ _ (v_argmax _ _ _ HM)). reflexivity. }
  intros dm T M1 M2 mw caches bl tok pos HP H1 H2.
  rewrite (Key dm T M1), (Key dm T M2) by assumption. reflexivity.
Qed.

(** The step is the reference machine's: sums in the integers. *)
Corollary every_valid_execution_takes_the_reference_step :
  forall dm T M mw caches bl tok pos,
    step_premises dm T mw caches tok -> valid dm T M ->
    step M dm mw caches bl tok pos = step Mref dm mw caches bl tok pos.
Proof.
  intros. apply (every_valid_execution_takes_the_same_decode_step dm T); try assumption.
  apply the_reference_machine_is_valid.
Qed.

(* ------------------------------------------------------------------ *)
(** ** Schedules make machines: the link to [ExactReduce.v]            *)
(* ------------------------------------------------------------------ *)

(** A dot product by schedule [t], each product formed in carrier [c]
    (rounded, or fused into the add), the sums in [c]. *)
Definition sched_dot (c : ER.carrier) (fused : bool) (t : nat -> ER.tree) (l : list (Z * Z)) : Z :=
  ER.teval (ER.cadd c) (map (fun xy => ER.cmul c fused (fst xy) (snd xy)) l) (t (length l)).

Definition sched_sum (c : ER.carrier) (t : nat -> ER.tree) (l : list Z) : Z :=
  ER.teval (ER.cadd c) l (t (length l)).

Definition sched_comb (c : ER.carrier) (h l : Z) : Z := ER.cadd c (ER.cmul c false 128 h) l.

Theorem sched_dot_exact : forall c fused t B,
  B <= ER.cap c -> (forall n, ER.covers (t n) n) -> exact_dot B (sched_dot c fused t).
Proof.
  intros c fused t B Hc Ht l Hl. unfold sched_dot.
  assert (Em : map (fun xy => ER.cmul c fused (fst xy) (snd xy)) l = prods l).
  { apply map_ext_in. intros [x y] Hin. cbn [fst snd]. apply ER.cmul_exact.
    assert (Z.abs (x * y) <= ER.abs_sum (prods l)).
    { unfold ER.abs_sum, prods. clear -Hin. induction l as [|a l IH]; [destruct Hin|].
      cbn. destruct Hin as [->|Hin]; cbn.
      - pose proof (ER.abs_sum_nonneg (map (fun xy => fst xy * snd xy) l)). unfold ER.abs_sum in H. lia.
      - specialize (IH Hin). lia. }
    lia. }
  rewrite Em. replace (length l) with (length (prods l)) by (unfold prods; apply length_map).
  apply (ER.any_schedule_gives_the_sum (ER.cap c)).
  - apply ER.every_carrier_is_exact_within_its_cap.
  - apply Ht.
  - lia.
Qed.

Theorem sched_sum_exact : forall c t B,
  B <= ER.cap c -> (forall n, ER.covers (t n) n) -> exact_sum B (sched_sum c t).
Proof.
  intros c t B Hc Ht l Hl. unfold sched_sum.
  apply (ER.any_schedule_gives_the_sum (ER.cap c)).
  - apply ER.every_carrier_is_exact_within_its_cap.
  - apply Ht.
  - lia.
Qed.

Theorem sched_comb_exact : forall c B, B <= ER.cap c -> exact_comb B (sched_comb c).
Proof.
  intros c B Hc h l Hl. unfold sched_comb.
  rewrite ER.cmul_exact by lia.
  apply ER.every_carrier_is_exact_within_its_cap. lia.
Qed.

(** A machine from a carrier, a product mode and a schedule at every site. *)
Definition sched_machine (car : site -> ER.carrier) (fused : site -> bool) (t : site -> nat -> ER.tree)
  (imx : site -> list Z -> Z) (fmx' : site -> list R -> R) (amx : list R -> nat) (digits : list nat -> Z)
  : machine := {|
  m_dot := fun s => sched_dot (car s) (fused s) (t s);
  m_sum := fun s => sched_sum (car s) (t s);
  m_comb := fun s => sched_comb (car s);
  m_imax := imx; m_fmax := fmx'; m_argmax := amx; m_digits := digits
|}.

Theorem schedules_in_fitting_carriers_are_valid :
  forall dm T car fused t imx fmx' amx digits,
    (forall s, need_st dm T (digits (snd s)) (fst s) <= ER.cap (car s)) ->
    (forall s n, ER.covers (t s n) n) ->
    (forall s, imax_ok (imx s)) -> (forall s, fmax_ok (fmx' s)) -> argmax_ok amx ->
    (forall p, 0 <= digits p) ->
    valid dm T (sched_machine car fused t imx fmx' amx digits).
Proof.
  intros dm T car fused t imx fmx' amx digits Hfit Ht Hi Hf Ha Hd.
  constructor; cbn; try assumption.
  - intros s. apply sched_dot_exact; [apply Hfit | apply Ht].
  - intros s. apply sched_sum_exact; [apply Hfit | apply Ht].
  - intros s. apply sched_comb_exact. apply Hfit.
Qed.

(** Any comparison-based max, by any tree, meets the max specifications. *)
Theorem max_trees_are_valid_maxima : forall (t : nat -> ER.tree),
  (forall n, n <> O -> ER.nozero (t n)) -> (forall n, ER.covers (t n) n) ->
  imax_ok (fun l => match l with [] => 0 | _ => ER.xeval Z zmx 0 l (t (length l)) end)
  /\ fmax_ok (fun l => match l with [] => rdef | _ => ER.xeval R fmx rdef l (t (length l)) end).
Proof.
  intros t Hz Hc. split; intros l; (split; [intros ->; reflexivity|]); intros H;
    destruct l as [|x r]; try contradiction.
  - apply (ER.any_max_tree_is_a_max Z (fun z => z) zmx zmx_picks); [apply Hz; discriminate | apply Hc].
  - apply (ER.any_max_tree_is_a_max R ival fmx fmx_picks); [apply Hz; discriminate | apply Hc].
Qed.

(** Any two executions that put every sum in a carrier the code uses there,
    with any schedules and any maxima, decode the same step. *)
Corollary code_executions_agree :
  forall dm T mw caches bl tok pos car1 car2 fu1 fu2 t1 t2 im1 im2 fm1 fm2 am1 am2 dg1 dg2,
  code_shape_ok dm T -> step_premises dm T mw caches tok ->
  (forall s, code_uses dm T (dg1 (snd s)) (fst s) (car1 s)) ->
  (forall s, code_uses dm T (dg2 (snd s)) (fst s) (car2 s)) ->
  (forall s n, ER.covers (t1 s n) n) -> (forall s n, ER.covers (t2 s n) n) ->
  (forall s, imax_ok (im1 s)) -> (forall s, imax_ok (im2 s)) ->
  (forall s, fmax_ok (fm1 s)) -> (forall s, fmax_ok (fm2 s)) ->
  argmax_ok am1 -> argmax_ok am2 ->
  (forall p, 0 <= dg1 p) -> (forall p, 0 <= dg2 p) ->
  step (sched_machine car1 fu1 t1 im1 fm1 am1 dg1) dm mw caches bl tok pos
  = step (sched_machine car2 fu2 t2 im2 fm2 am2 dg2) dm mw caches bl tok pos.
Proof.
  intros dm T mw caches bl tok pos car1 car2 fu1 fu2 t1 t2 im1 im2 fm1 fm2 am1 am2 dg1 dg2
    Hs Hp Hu1 Hu2 Ht1 Ht2 Hi1 Hi2 Hf1 Hf2 Ha1 Ha2 Hd1 Hd2.
  apply (every_valid_execution_takes_the_same_decode_step dm T); [exact Hp| |];
    apply schedules_in_fitting_carriers_are_valid; try assumption;
    intros s; apply the_code's_carriers_meet_every_need; auto.
Qed.

(* ------------------------------------------------------------------ *)
(** ** Ragged batches: left padding changes nothing                     *)
(* ------------------------------------------------------------------ *)

(** A ragged batch left-pads its shorter prompts, so a row's cache holds P
    padding rows, masked, before its R real ones. [left_padding_changes_nothing]:
    any valid execution of the padded row gives the step of the same prompt
    with no padding.

    It needs three facts about the run, [attend_pad_ok], at every attention
    call: the mask constant is at most the largest real logit, a padding
    key's folded weight `0 * sv` is at most the largest real one, and the
    fold rounds it to 0. All three hold whenever the logits and scales are
    finite (`NEG` is the most negative finite float64; a real key at the max
    has weight `2^28 * sv > 0`; `round(0 * x) = 0` for finite `x`). With an
    infinity or a NaN in a row they can fail, and then the code's padded and
    unpadded rows can differ. *)

Lemma combine_app : forall (A B : Type) (a1 a2 : list A) (b1 b2 : list B),
  length a1 = length b1 -> combine (a1 ++ a2) (b1 ++ b2) = combine a1 b1 ++ combine a2 b2.
Proof.
  intros A B a1. induction a1 as [|x a1 IH]; intros a2 [|y b1] b2 H; cbn in *; try discriminate; [reflexivity|].
  f_equal. apply IH. lia.
Qed.

Lemma combine_repeat_r : forall (A B : Type) (l : list A) (b : B),
  combine l (repeat b (length l)) = map (fun a => (a, b)) l.
Proof. intros A B l b. induction l as [|x l IH]; cbn; [reflexivity|]. rewrite IH. reflexivity. Qed.

Lemma combine_repeat_l : forall (A B : Type) (a : A) (l : list B),
  combine (repeat a (length l)) l = map (fun b => (a, b)) l.
Proof. intros A B a l. induction l as [|x l IH]; cbn; [reflexivity|]. rewrite IH. reflexivity. Qed.

Lemma zsum_repeat0 : forall n, ER.zsum (repeat 0 n) = 0.
Proof. induction n as [|n IH]; cbn; [reflexivity|]. rewrite IH. reflexivity. Qed.

Lemma repeat_snoc : forall (A : Type) (a : A) n, repeat a n ++ [a] = repeat a (S n).
Proof. intros A a n. induction n as [|n IH]; cbn; [reflexivity|]. rewrite IH. reflexivity. Qed.

Lemma pdot_app : forall a1 a2 b1 b2, length a1 = length b1 ->
  pdot (combine (a1 ++ a2) (b1 ++ b2)) = pdot (combine a1 b1) + pdot (combine a2 b2).
Proof.
  intros. rewrite combine_app by assumption. unfold pdot, prods. rewrite map_app. apply ER.zsum_app.
Qed.

Lemma pdot_zeros : forall b, pdot (combine (repeat 0 (length b)) b) = 0.
Proof.
  intros b. induction b as [|y b IH]; [reflexivity|].
  cbn [length repeat combine]. unfold pdot, prods in *. cbn [map ER.zsum fst snd]. rewrite IH. ring.
Qed.

Lemma fmaxr_app_ival : forall pre l, l <> [] ->
  (forall x, In x pre -> ival x <= ival (fmaxr l)) ->
  ival (fmaxr (pre ++ l)) = ival (fmaxr l).
Proof.
  intros pre l Hl Hpre.
  assert (Hne : pre ++ l <> []) by (destruct l; [contradiction | destruct pre; discriminate]).
  destruct (proj2 (fmaxr_ok (pre ++ l)) Hne) as [Iin Imax].
  destruct (proj2 (fmaxr_ok l) Hl) as [Jin Jmax].
  apply Z.le_antisymm.
  - apply in_app_or in Iin. destruct Iin as [I|I]; [apply Hpre, I | apply Jmax, I].
  - apply Imax. apply in_or_app. right. exact Jin.
Qed.

Lemma logits_length : forall a s kq, length s = length kq -> length (logits a s kq) = length kq.
Proof. intros. unfold logits. rewrite length_map, length_combine. lia. Qed.

Lemma scores_ref : forall path qi kq,
  scores Mref path qi kq = map (fun k => pdot (combine qi (fst k))) kq.
Proof.
  intros. unfold scores, enum. cbn [m_dot Mref].
  exact (map_snd_enum_from _ _ (fun k => pdot (combine qi (fst k))) kq 0).
Qed.

(** The real keys' logits and folded weights, as the unpadded reference run
    computes them. *)
Definition lg_real (dm : dims) (q : list R) (kr : list (list R)) : list R :=
  logits (fqa (snd (quant_ref C.Q_LEVELS q)))
    (map (fun k => pdot (combine (fst (quant_ref C.Q_LEVELS q)) (fst k))) (map (quant_ref (kl_of dm)) kr))
    (map (quant_ref (kl_of dm)) kr).

Definition w_real (dm : dims) (q : list R) (kr vr : list (list R)) : list R :=
  folded (probs (fmaxr (lg_real dm q kr)) (lg_real dm q kr) (repeat false (length kr)))
         (map (quant_ref C.V_LEVELS) vr).

Definition attend_pad_ok (dm : dims) (q : list R) (kp kr vp vr : list (list R)) : Prop :=
  kr <> [] /\ length vp = length kp /\ length vr = length kr
  /\ ival NEG <= ival (fmaxr (lg_real dm q kr))
  /\ Forall (fun v => ival (fw 0 (snd (quant_ref C.V_LEVELS v))) <= ival (fmaxr (w_real dm q kr vr))
                      /\ fold_round (fw 0 (snd (quant_ref C.V_LEVELS v))) (fmaxr (w_real dm q kr vr)) = 0) vp.

Lemma map_fold_zero : forall (l : list (list Z * R)) m mr,
  ival m = ival mr ->
  Forall (fun v => fold_round (fw 0 (snd v)) mr = 0) l ->
  map (fun x => clamp0 (2 ^ 28) (fold_round (fw 0 (snd x)) m)) l = repeat 0 (length l).
Proof.
  intros l m mr Im H. induction H as [|v l H0 Hl IH]; [reflexivity|].
  cbn [map repeat length]. rewrite IH, (proj1 (fold_ival _ _ (fw 0 (snd v)) Im)), H0. reflexivity.
Qed.

Lemma attend3_padding : forall dm path lgp lgr vqp vqr,
  length lgp = length vqp -> length lgr = length vqr -> lgr <> [] ->
  ival NEG <= ival (fmaxr lgr) ->
  Forall (fun v => ival (fw 0 (snd v)) <= ival (fmaxr (folded (probs (fmaxr lgr) lgr (repeat false (length lgr))) vqr))
                   /\ fold_round (fw 0 (snd v)) (fmaxr (folded (probs (fmaxr lgr) lgr (repeat false (length lgr))) vqr)) = 0) vqp ->
  attend3 Mref dm path (lgp ++ lgr) (repeat true (length lgp) ++ repeat false (length lgr)) (vqp ++ vqr)
  = attend3 Mref dm path lgr (repeat false (length lgr)) vqr.
Proof.
  intros dm path lgp lgr vqp vqr Hp Hr Hne Hneg Hpad.
  assert (H1 : (1 <= length lgr)%nat) by (destruct lgr; [contradiction | cbn; lia]).
  unfold attend3, attend4, attend5. cbn [m_fmax m_sum Mref].
  (* the masked logits *)
  assert (Em : mask_logits (lgp ++ lgr) (repeat true (length lgp) ++ repeat false (length lgr))
               = repeat NEG (length lgp) ++ lgr).
  { unfold mask_logits. rewrite combine_app by (rewrite repeat_length; reflexivity).
    rewrite map_app, !combine_repeat_r, !map_map. cbn [fst snd]. f_equal.
    - clear. induction lgp; cbn; [reflexivity|]. rewrite IHlgp. reflexivity.
    - apply map_id. }
  assert (Em0 : mask_logits lgr (repeat false (length lgr)) = lgr).
  { unfold mask_logits. rewrite combine_repeat_r, map_map. apply map_id. }
  rewrite Em, Em0.
  assert (Im : ival (fmaxr (repeat NEG (length lgp) ++ lgr)) = ival (fmaxr lgr)).
  { apply fmaxr_app_ival; [exact Hne|]. intros x Hx. apply repeat_spec in Hx. subst x. exact Hneg. }
  (* the integer weights *)
  set (m := fmaxr (repeat NEG (length lgp) ++ lgr)) in *.
  assert (Ep : probs m (lgp ++ lgr) (repeat true (length lgp) ++ repeat false (length lgr))
               = repeat 0 (length lgp) ++ probs (fmaxr lgr) lgr (repeat false (length lgr))).
  { unfold probs. rewrite combine_app by (rewrite repeat_length; reflexivity).
    rewrite map_app, !combine_repeat_r, !map_map. cbn [fst snd]. f_equal.
    - clear. induction lgp; cbn; [reflexivity|]. rewrite IHlgp. reflexivity.
    - apply map_ext. intros x. f_equal. f_equal. apply targ_ival. exact Im. }
  rewrite Ep.
  set (pr := probs (fmaxr lgr) lgr (repeat false (length lgr))) in *.
  assert (Lpr : length pr = length lgr).
  { unfold pr, probs. rewrite length_map, combine_repeat_r, length_map. reflexivity. }
  rewrite ER.zsum_app, zsum_repeat0, Z.add_0_l.
  (* the folded weights *)
  assert (Ef : folded (repeat 0 (length lgp) ++ pr) (vqp ++ vqr)
               = map (fun v => fw 0 (snd v)) vqp ++ folded pr vqr).
  { unfold folded. rewrite combine_app by (rewrite repeat_length; exact Hp).
    rewrite map_app. f_equal. rewrite Hp, combine_repeat_l, map_map. reflexivity. }
  rewrite Ef.
  set (wr := folded pr vqr) in *.
  set (wp := map (fun v => fw 0 (snd v)) vqp).
  assert (Hwr : wr <> []).
  { intros E. assert (length wr = 0%nat) by (rewrite E; reflexivity).
    unfold wr, folded in H. rewrite length_map, length_combine in H. lia. }
  assert (Iw : ival (fmaxr (wp ++ wr)) = ival (fmaxr wr)).
  { apply fmaxr_app_ival; [exact Hwr|]. intros x Hx. unfold wp in Hx. apply in_map_iff in Hx.
    destruct Hx as [v [<- Hv]]. rewrite Forall_forall in Hpad. apply (Hpad v Hv). }
  destruct (fold_ival _ _ rdef Iw) as [_ Es]. rewrite Es.
  assert (Ew : wint (fmaxr (wp ++ wr)) (wp ++ wr) = repeat 0 (length vqp) ++ wint (fmaxr wr) wr).
  { unfold wint. rewrite map_app. f_equal.
    - unfold wp. rewrite map_map. apply (map_fold_zero _ _ (fmaxr wr) Iw).
      eapply Forall_impl; [|exact Hpad]. intros v [_ H0]. exact H0.
    - apply map_ext. intros x. f_equal. apply (fold_ival _ _ x Iw). }
  rewrite Ew. unfold attend_out. apply map_ext. intros d. f_equal.
  unfold pv. cbn [m_digits m_dot Mref]. cbn [Z.leb Z.compare].
  rewrite map_app, pdot_app by (rewrite repeat_length, length_map; reflexivity).
  replace (length vqp) with (length (map (fun v => nth d (fst v) 0) vqp)) by apply length_map.
  rewrite pdot_zeros. reflexivity.
Qed.

Lemma attend_padding : forall dm path q kp kr vp vr,
  attend_pad_ok dm q kp kr vp vr ->
  attend Mref dm path q (kp ++ kr) (vp ++ vr) (repeat true (length kp) ++ repeat false (length kr))
  = attend Mref dm path q kr vr (repeat false (length kr)).
Proof.
  intros dm path q kp kr vp vr [Hne [Hvp [Hvr [Hneg Hpad]]]].
  pose proof (the_reference_machine_is_valid dm 1) as HR.
  unfold attend, attend2.
  rewrite !(quant_value dm 1 Mref), !(quant_rows_value dm 1 Mref) by exact HR.
  rewrite !scores_ref, !map_app.
  unfold logits at 1. rewrite combine_app by (rewrite !length_map; reflexivity). rewrite map_app.
  fold (logits (fqa (snd (quant_ref C.Q_LEVELS q)))
          (map (fun k => pdot (combine (fst (quant_ref C.Q_LEVELS q)) (fst k))) (map (quant_ref (kl_of dm)) kp))
          (map (quant_ref (kl_of dm)) kp)).
  fold (logits (fqa (snd (quant_ref C.Q_LEVELS q)))
          (map (fun k => pdot (combine (fst (quant_ref C.Q_LEVELS q)) (fst k))) (map (quant_ref (kl_of dm)) kr))
          (map (quant_ref (kl_of dm)) kr)).
  fold (lg_real dm q kr).
  assert (Lp : length (logits (fqa (snd (quant_ref C.Q_LEVELS q)))
            (map (fun k => pdot (combine (fst (quant_ref C.Q_LEVELS q)) (fst k))) (map (quant_ref (kl_of dm)) kp))
            (map (quant_ref (kl_of dm)) kp)) = length kp)
    by (rewrite logits_length by (rewrite !length_map; reflexivity); apply length_map).
  assert (Lr : length (lg_real dm q kr) = length kr)
    by (unfold lg_real; rewrite logits_length by (rewrite !length_map; reflexivity); apply length_map).
  rewrite <- Lp at 1. rewrite <- Lr at 1 2.
  apply attend3_padding.
  - rewrite Lp, length_map. lia.
  - rewrite Lr, length_map. lia.
  - intros E. rewrite E in Lr. destruct kr; [contradiction | discriminate].
  - exact Hneg.
  - rewrite Lr. apply Forall_map. exact Hpad.
Qed.

(** Padding rows ahead of each KV head's cache. *)
Definition padc (pd c : list (list (list R))) : list (list (list R)) := zipw (@app (list R)) pd c.

Lemma nth_map_gen : forall (A B : Type) (f : A -> B) l g d d', (g < length l)%nat ->
  nth g (map f l) d = f (nth g l d').
Proof.
  intros A B f l g d d' Hg. rewrite (nth_indep _ d (f d')) by (rewrite length_map; exact Hg).
  apply map_nth.
Qed.

Lemma nth_grow : forall c new n g, (g < n)%nat ->
  nth g (grow c new n) [] = nth g c [] ++ [nth g new []].
Proof.
  intros c new n g Hg. unfold grow.
  rewrite (nth_map_gen _ _ _ _ g [] 0%nat) by (rewrite length_seq; exact Hg).
  rewrite seq_nth by exact Hg. reflexivity.
Qed.

Lemma nth_padc : forall pd c g, length pd = length c -> (g < length c)%nat ->
  nth g (padc pd c) [] = nth g pd [] ++ nth g c [].
Proof.
  intros pd c g Hl Hg. unfold padc, zipw.
  rewrite (nth_map_gen _ _ _ _ g [] ([], [])) by (rewrite length_combine; lia).
  rewrite combine_nth by exact Hl. reflexivity.
Qed.

Definition layer_pad_ok (dm : dims) (li : nat) (pos : Z) (lw : layer_w)
  (ck cv kp vp : list (list (list R))) (x : list R) (P R0 : nat) : Prop :=
  length kp = nkv dm /\ length ck = nkv dm /\ length vp = nkv dm /\ length cv = nkv dm
  /\ (forall g, (g < nkv dm)%nat -> length (nth g kp []) = P /\ length (nth g ck []) = R0)
  /\ forall h, (h < nh dm)%nat -> (h / rep dm < nkv dm)%nat
     /\ attend_pad_ok dm
          (nth h (qs_of dm pos (linear Mref dm QKV [li] (lqkv lw) (norm Mref [0%nat; li] x (ln1 lw)))) [])
          (nth (h / rep dm) kp [])
          (nth (h / rep dm) (grow ck (ks_of dm pos (linear Mref dm QKV [li] (lqkv lw) (norm Mref [0%nat; li] x (ln1 lw)))) (nkv dm)) [])
          (nth (h / rep dm) vp [])
          (nth (h / rep dm) (grow cv (vs_of dm (linear Mref dm QKV [li] (lqkv lw) (norm Mref [0%nat; li] x (ln1 lw)))) (nkv dm)) []).

Lemma layer_padding : forall dm li pos lw ck cv kp vp x P R0,
  layer_pad_ok dm li pos lw ck cv kp vp x P R0 ->
  layer Mref dm li pos lw (padc kp ck) (padc vp cv) (repeat true P ++ repeat false R0) x
  = layer Mref dm li pos lw ck cv (repeat false R0) x.
Proof.
  intros dm li pos lw ck cv kp vp x P R0 [Lk [Lc [Lv [Lcv [Hlen Hh]]]]].
  unfold layer, layer_qkv.
  set (qkv := linear Mref dm QKV [li] (lqkv lw) (norm Mref [0%nat; li] x (ln1 lw))) in *.
  assert (Ea : attn_all Mref dm li (qs_of dm pos qkv) (grow (padc kp ck) (ks_of dm pos qkv) (nkv dm))
                 (grow (padc vp cv) (vs_of dm qkv) (nkv dm)) ((repeat true P ++ repeat false R0) ++ [false])
               = attn_all Mref dm li (qs_of dm pos qkv) (grow ck (ks_of dm pos qkv) (nkv dm))
                 (grow cv (vs_of dm qkv) (nkv dm)) (repeat false R0 ++ [false])).
  { unfold attn_all. apply flat_map_ext_in. intros h Hin. apply in_seq in Hin.
    destruct (Hh h ltac:(lia)) as [Hg Hok].
    set (g := (h / rep dm)%nat) in *.
    destruct (Hlen g Hg) as [Lkg Lcg].
    rewrite !nth_grow by exact Hg. rewrite !nth_grow in Hok by exact Hg.
    rewrite !nth_padc by lia.
    rewrite <- !app_assoc.
    rewrite repeat_snoc.
    replace (S R0) with (length (nth g ck [] ++ [nth g (ks_of dm pos qkv) []]))
      by (rewrite length_app, Lcg; cbn; lia).
    rewrite <- Lkg.
    apply attend_padding. exact Hok. }
  rewrite Ea. reflexivity.
Qed.

(** A model's layers with their real caches and padding rows. *)
Definition padded_ls (ls : list (layer_w * cache * cache)) : list (layer_w * cache) :=
  map (fun t => (fst (fst t), (padc (fst (snd t)) (fst (snd (fst t))), padc (snd (snd t)) (snd (snd (fst t)))))) ls.

Definition real_ls (ls : list (layer_w * cache * cache)) : list (layer_w * cache) := map fst ls.

Fixpoint run_pad_ok (dm : dims) (pos : Z) (P R0 : nat) (li : nat) (ls : list (layer_w * cache * cache))
  (x : list R) : Prop :=
  match ls with
  | [] => True
  | t :: rest =>
      layer_pad_ok dm li pos (fst (fst t)) (fst (snd (fst t))) (snd (snd (fst t))) (fst (snd t)) (snd (snd t)) x P R0
      /\ run_pad_ok dm pos P R0 (S li) rest
           (fst (layer Mref dm li pos (fst (fst t)) (fst (snd (fst t))) (snd (snd (fst t))) (repeat false R0) x))
  end.

Lemma run_padding : forall dm pos P R0 ls li x,
  run_pad_ok dm pos P R0 li ls x ->
  run_layers Mref dm pos (repeat true P ++ repeat false R0) li (padded_ls ls) x
  = run_layers Mref dm pos (repeat false R0) li (real_ls ls) x.
Proof.
  intros dm pos P R0 ls. induction ls as [|t ls IH]; intros li x H; [reflexivity|].
  destruct H as [Hl Hr]. cbn [run_layers padded_ls real_ls map fst snd] in *.
  destruct t as [[lw c] pd]. cbn [fst snd] in *.
  rewrite (layer_padding dm li pos lw (fst c) (snd c) (fst pd) (snd pd) x P R0 Hl).
  specialize (IH (S li) _ Hr).
  f_equal; [f_equal; exact IH | f_equal; f_equal; exact IH].
Qed.

Lemma combine_map_map : forall (A B C : Type) (f : A -> B) (g : A -> C) l,
  combine (map f l) (map g l) = map (fun t => (f t, g t)) l.
Proof. intros. induction l as [|a l IH]; cbn; [reflexivity|]. rewrite IH. reflexivity. Qed.

(** ** Ragged batches, the theorem *)
Theorem left_padding_changes_nothing : forall dm T M mw ls P R0 tok pos,
  step_premises dm T mw (map snd (padded_ls ls)) tok -> valid dm T M ->
  mlayers mw = map (fun t => fst (fst t)) ls ->
  run_pad_ok dm pos P R0 0 ls (embed tok) ->
  step M dm mw (map snd (padded_ls ls)) (repeat true P ++ repeat false R0) tok pos
  = step Mref dm mw (map snd (real_ls ls)) (repeat false R0) tok pos.
Proof.
  intros dm T M mw ls P R0 tok pos Hs HM Hl Hp.
  rewrite (every_valid_execution_takes_the_reference_step dm T M) by assumption.
  unfold step. rewrite Hl.
  assert (E1 : combine (map (fun t => fst (fst t)) ls) (map snd (padded_ls ls)) = padded_ls ls).
  { unfold padded_ls. rewrite map_map, combine_map_map. apply map_ext. intros t. reflexivity. }
  assert (E2 : combine (map (fun t => fst (fst t)) ls) (map snd (real_ls ls)) = real_ls ls).
  { unfold real_ls. rewrite map_map, combine_map_map. apply map_ext. intros [[a b] c]. reflexivity. }
  rewrite E1, E2, (run_padding dm pos P R0 ls 0 (embed tok) Hp). reflexivity.
Qed.

End Step.

(* ------------------------------------------------------------------ *)
(** ** The premises are met by real models                             *)
(* ------------------------------------------------------------------ *)

Definition qwen25_0_5b (req : Z) : dims := {| dH := 896; dI := 4864; dNH := 14; dNKV := 2; dD := 64; dReq := req |}.
Definition qwen25_1_5b (req : Z) : dims := {| dH := 1536; dI := 8960; dNH := 12; dNKV := 2; dD := 128; dReq := req |}.
Definition qwen25_3b (req : Z) : dims := {| dH := 2048; dI := 11008; dNH := 16; dNKV := 2; dD := 128; dReq := req |}.
Definition qwen25_7b (req : Z) : dims := {| dH := 3584; dI := 18944; dNH := 28; dNKV := 4; dD := 128; dReq := req |}.

Theorem qwen25_shapes_are_accepted : forall mk req T,
  In mk [qwen25_0_5b; qwen25_1_5b; qwen25_3b; qwen25_7b] -> In req [127; 511] -> 1 <= T <= 32768 ->
  code_shape_ok (mk req) T.
Proof.
  intros mk req T Hm Hreq HT.
  unfold code_shape_ok, C.ACC_BUDGET, C.Q_LEVELS, C.SCORE_BUDGET, C.SUMSQ_MAX_H, C.V_LEVELS.
  destruct Hreq as [<-|[<-|[]]];
  destruct Hm as [<-|[<-|[<-|[<-|[]]]]]; cbn [dReq dD dH qwen25_0_5b qwen25_1_5b qwen25_3b qwen25_7b];
  repeat split; try (match goal with w : lin |- _ => destruct w end; cbn; lia);
  try (change (2 ^ 28) with 268435456); try (change (2 ^ 53) with 9007199254740992); lia.
Qed.

(** The widths the code derives at those shapes: the activation widths
    (with `Y_EXACT_ACT_LEVELS` at 511 every layer of these models gets 511),
    and K at 1023 for head dimensions 64 and 128. *)
Theorem the_derived_widths_at_qwen25 :
  (forall w, a_of (qwen25_0_5b 511) w = 511) /\ (forall w, a_of (qwen25_7b 511) w = 511)
  /\ (forall w, a_of (qwen25_7b 127) w = 127)
  /\ kl_of (qwen25_0_5b 127) = 1023 /\ kl_of (qwen25_7b 127) = 1023
  /\ W.digit_width 517 C.V_LEVELS = 8 /\ W.digit_width 32768 C.V_LEVELS = 2.
Proof. repeat split; try (intros []); vm_compute; reflexivity. Qed.

(** The whole premise set is satisfiable at Qwen2.5-0.5B: the shapes are
    accepted, and the reference machine is valid. *)
Theorem the_premises_are_satisfiable_at_qwen25_0_5b :
  forall R rdef ival fge (fge_ival : forall a b, fge a b = (ival b <=? ival a)),
  code_shape_ok (qwen25_0_5b 511) 32768
  /\ valid R rdef ival (qwen25_0_5b 511) 32768 (Mref R rdef ival fge).
Proof.
  intros. split.
  - apply qwen25_shapes_are_accepted; cbn; auto; lia.
  - apply the_reference_machine_is_valid. exact fge_ival.
Qed.

(** The trusted interface's hypotheses hold of some functions. *)
Theorem the_trusted_interface_is_satisfiable :
  (forall a b : Z, (b <=? a) = (b <=? a))
  /\ (forall (pos : Z) (v : list Z), length ((fun _ v => v) pos v) = length v)
  /\ (forall m m' x : Z, m = m' -> m - x = m' - x)
  /\ (forall m m' x : Z, m = m' -> x + m = x + m' /\ m = m').
Proof. repeat split; intros; subst; reflexivity. Qed.

(* ------------------------------------------------------------------ *)
(** ** The refutations                                                 *)
(* ------------------------------------------------------------------ *)

(** A float32 accumulation at the softmax denominator is not a valid
    reducer: two keys at T = 2 already lose a unit. *)
Theorem a_float32_denominator_is_not_exact :
  ~ exact_sum (2 * 2 ^ 28) (sched_sum ER.F32 ltree).
Proof.
  intros H. specialize (H [2 ^ 28; 1] ltac:(vm_compute; discriminate)).
  vm_compute in H. discriminate.
Qed.

(** ... and the code never puts it there. *)
Theorem the_code_does_not_put_the_denominator_in_float32 : forall dm T d,
  ~ code_uses dm T d Denom ER.F32.
Proof. intros dm T d H. exact H. Qed.

(** A head dimension past the score budget needs more than float32 holds:
    one bit more of K at head dimension 128 is refused. *)
Theorem a_wider_key_does_not_fit_float32 :
  128 * C.Q_LEVELS * 2047 > ER.cap ER.F32.
Proof. vm_compute. reflexivity. Qed.

Print Assumptions every_valid_execution_takes_the_same_decode_step.
Print Assumptions every_valid_execution_takes_the_reference_step.
Print Assumptions the_reference_machine_is_valid.
Print Assumptions the_digits_recombine.
Print Assumptions sched_dot_exact.
Print Assumptions schedules_in_fitting_carriers_are_valid.
Print Assumptions max_trees_are_valid_maxima.
Print Assumptions the_code's_carriers_meet_every_need.
Print Assumptions code_executions_agree.
Print Assumptions qwen25_shapes_are_accepted.
Print Assumptions the_derived_widths_at_qwen25.
Print Assumptions the_premises_are_satisfiable_at_qwen25_0_5b.
Print Assumptions the_trusted_interface_is_satisfiable.
Print Assumptions a_float32_denominator_is_not_exact.
Print Assumptions the_code_does_not_put_the_denominator_in_float32.
Print Assumptions a_wider_key_does_not_fit_float32.
Print Assumptions attend_padding.
Print Assumptions left_padding_changes_nothing.
