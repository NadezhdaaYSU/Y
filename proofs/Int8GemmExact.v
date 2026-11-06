(** * The int8 tensor-core GEMM computes the source dot products

    [proofs/Int8GemmSchedule.v] proves this kernel's SCHEDULE: which lane owns
    which element of C, that the split-K classes tile the contraction, and that
    the atomic reduction is order-independent.  It says nothing whatever about
    the VALUE that lands at [C[r][c]].  This file closes that, and it is the
    GPU twin of [ExactGemmWhole.the_threaded_gemm_holds_the_source_dot_products]
    - the same claim for the CPU's exact GEMM.

    It is available for THIS kernel and no other GEMM in the repository: 950 of
    the 952 [mma.sync] instructions Y emits are floating point, and an f16
    tensor-core GEMM is simply not equal to the naive nest.  The int8
    instruction accumulates into int32, integer addition is associative, and
    the relationship is an EQUALITY.

    ** THE LICENCE, AND THE DEFECT THAT WRITING THIS FOUND

    The theorem below is FALSE without a hypothesis, and the compiler was not
    checking it.  [mma...s32.s8.s8.s32] accumulates into int32 and this kernel
    has NO FLUSH - the CPU's exact GEMM widens to int64 every [Fl] k-pairs, and
    there is no equivalent here because the OUTPUT is int32 too.  So the bound
    is on the whole contraction:

    <<  | sum over k < K of A[r][k] * B[c][k] |  <=  K * m^2  <=  i32::MAX  >>

    for operand magnitude [m].  The emitter refused only on M % 16, N % 8 and
    K % 32; nothing bounded K, and nothing anywhere in [proofs/] or [tests/]
    mentioned the accumulator's range.

    **Measured on the device before any of this was written**, M=16 N=8, every
    element of A and B set to 127, one warp, grid (1,1,1):

    | K       | exact        | device        |          |
    |---------|--------------|---------------|----------|
    | 133 088 | 2 146 576 352| 2 146 576 352 | ok       |
    | 133 120 | 2 147 092 480| 2 147 092 480 | ok       |
    | 133 152 | 2 147 608 608| -2 147 358 688| WRAPPED  |
    | 133 184 | 2 148 124 736| -2 146 842 560| WRAPPED  |

    One K-step wide.  [the_measured_overflow_is_two_s_complement] reproduces
    that third row from [wrap32] alone, so the model is refereed against the
    silicon rather than asserted to describe it.

    Latent rather than live: the largest K in the corpus is 16 384, four
    orders below the bound.  That is the reason to fix it now - this
    repository's own rule is to find these while the path is still dead.

    ** WHAT IS PROVED

    - [the_lanes_cover_the_a_fragment] / [_b_fragment]: the 32 lanes' register
      bytes are a BIJECTION onto the 16x32 and 32x8 fragments.  Every element
      is loaded, once, so no fragment position keeps a value from the previous
      K step.  Both are [MixedRadix] - the eighth and ninth consumers.
    - [the_ldmatrix_a_fragment_is_the_source_element] / [_b_] (since
      2026-11-06, when the kernel began staging K tiles in shared memory):
      the [cp.async] copies put chunk [c] of a stage row at chunk
      [c XOR swz row], and the lane addresses [ldmatrix] is given undo it, so
      register [q], byte [b] of lane [l] is exactly the source element that
      bijection names.  [the_stage_slots_are_disjoint] says no copy overwrites
      another; [the_copy_lands_at_its_swizzled_place] that a thread's one
      swizzle serves all its rows.  They replace the address theorems of the
      kernel this one replaced, which loaded each register straight from
      global memory.
    - [the_k_loop_is_the_contraction]: 32 products per step over [K/32] steps
      re-index to the flat [sum over k < K].
    - [bounded_products_accumulate_exactly]: under the licence the int32
      accumulator ([ExactGemmMicro.wsum], wrapping at every step) equals the
      [Z] sum.  This is what makes the licence load-bearing rather than
      paperwork.
    - [the_owned_split_is_exact_in_int32]: the two wrapping folds - a CTA's
      register, the atomic combine - for ANY owner map, so for the K TILES
      the staged kernel stripes over [z] as well as the K steps the old one
      did ([wclass_is_wpart]).
    - [the_emitted_int8_gemm_holds_the_source_dot_products]: the capstone -
      in [Z] over the split's classes, in int32 over both folds, and for the
      unsplit launch's store.

    ** WHAT THIS DOES NOT CLAIM

    - [ldmatrix.sync.aligned.m8n8.x4.shared.b16]'s distribution of a matrix's
      bytes to lanes is a DEFINITION too ([ldmatrix_x4]), and so is the
      ordering [cp.async.wait_group] + [bar.sync] gives the stage: that the
      copies have landed when the fragments are read is [stage_holds], a
      hypothesis.  The device tests run every tile against an exact oracle.
    - [mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32]'s own semantics are a
      DEFINITION here, not a theorem - the trusted base, exactly as
      [vpdpwssd]'s semantics are on the CPU side.  They are pinned empirically
      by [tests/ptx_int8_mma_layout.rs], which runs the instruction on the
      device against a plain integer matmul.  A proof over [Z] cannot supply
      an ISA.
    - The ISA does not specify the order in which one [mma] sums its 32
      products, so nothing here depends on it: the bound is on the sum of
      ABSOLUTE values, which dominates every partial sum in every order.
    - Nothing about int8 QUANTIZATION.  The claim is that the kernel computes
      the integer matrix product its source names.
    - CLOSED 2026-09-06.  This used to say the int32 conjunct was stated for
      the FLAT accumulation only, and that the [nz > 1] case needed a wrapping
      twin of [GridStrideSplit.combine].  It has one:
      [the_split_k_accumulation_is_exact_in_int32] covers BOTH wrapping folds
      the kernel performs - a CTA's own int32 register accumulator ([wclass])
      and [red.global.add.s32] combining the CTAs' partials in memory
      ([wcombine]) - at every split factor and every landing order.  One
      licence hypothesis serves both, because it bounds the sum of ABSOLUTE
      values and every partial of either fold is a sum over a SUBSET of the
      products.  Measured at the licensed maximum K on the device at
      nz = 1, 2, 3, 8, 17 and 64: exact at every one.
    - **What writing that theorem forced into the open is that the licence is
      about a ZEROED destination**, which is stated nowhere in the compiler.
      See [each_launch_is_licensed_and_three_of_them_wrap] and
      [the_combine_needs_a_zeroed_destination] below: a caller who splits K
      across LAUNCHES into the same int32 [C] - the obvious thing to do, since
      this kernel accumulates into [C] - has every launch licensed and the
      accumulation not.
    - The tie is transcription-plus-gate, as in [Int8GemmSchedule]:
      [ptx_emitter] does not go through the [Ix] extraction layer, so this is a
      model checked against emitted text and against the device by
      [tests/int8_gemm_exactness.rs], not the byte-identity tie the CPU chain
      has.
    - [ExactGemmMicro] is required for its int32 model ([I32MAX], [wrap32],
      [wsum]) and for nothing else.  That model is generic - there is no
      [vpdpwssd] in it - and a second copy of "what an int32 is" is precisely
      the drift this directory exists to prevent.  Extracting it to a shared
      [Int32.v] is recorded and not done; it would touch three CPU proofs and
      muddy this increment.

    Build: coqc -R . Y Int8GemmExact.v   (Rocq 9.1)
*)

Require Import Coq.Arith.Arith.
Require Import Coq.micromega.Lia.
Require Import Coq.ZArith.ZArith.
From Stdlib Require Import List Permutation.

Require MixedRadix.
Require Decomposition.
Require GridStrideSplit.
Require Int8GemmSchedule.
Require ExactGemmMicro.

Import ListNotations.

Module MR  := MixedRadix.
Module D   := Decomposition.
Module GS  := GridStrideSplit.
Module SCH := Int8GemmSchedule.
Module MC  := ExactGemmMicro.

Open Scope Z_scope.

(* ------------------------------------------------------------------ *)
(** ** The range fold                                                  *)
(* ------------------------------------------------------------------ *)

(** Taken from [Decomposition] rather than restated - the fourth file to reuse
    it, and the range fold is still not a property of any decomposition. *)
Definition sum_k (f : nat -> Z) (n : nat) : Z := D.acc_range Z.add f 0 n.

(** Pointwise-equal functions fold alike.  Used instead of
    [functional_extensionality], which is an AXIOM - every [Print Assumptions]
    in this directory must report "Closed under the global context". *)
Lemma acc_range_ext : forall f h lo len,
  (forall i, f i = h i) ->
  D.acc_range Z.add f lo len = D.acc_range Z.add h lo len.
Proof.
  intros f h lo len H. induction len as [| len IH]; cbn [D.acc_range].
  - reflexivity.
  - rewrite IH, H. reflexivity.
Qed.

Lemma acc_range_shift : forall f lo len,
  D.acc_range Z.add f lo len = D.acc_range Z.add (fun p => f (lo + p)%nat) 0 len.
Proof.
  intros f lo len. induction len as [| len IH]; cbn [D.acc_range].
  - reflexivity.
  - rewrite IH. replace (lo + (0 + len))%nat with (lo + len)%nat by lia. reflexivity.
Qed.

(** Peel one step. *)
Lemma acc_range_succ : forall f lo len,
  D.acc_range Z.add f lo (Datatypes.S len)
  = D.acc_range Z.add f lo len + f (lo + len)%nat.
Proof. reflexivity. Qed.

(** Blocks of [B] terms re-index to the flat range.

    **Stated for an ABSTRACT [B], and that is not generality for its own sake -
    it is the difference between 0.27 s and not terminating.**  Proved directly
    at 32 the tactics all succeed and the goal closes to something
    SYNTACTICALLY IDENTICAL on both sides, and then [Qed] does not return: a
    [nat] literal is unary, so a proof term carrying `32` carries thirty-two
    nested [S] through every conversion check, inside a fold that is itself 32
    deep.  With [B] a variable nothing can unfold at all, and the literal
    enters only at the [apply] below, as a single unification.

    Same landmine as [SoftmaxErrorBound]'s [ring] on a large power, wearing a
    different hat: keep every literal out of anything that normalises. *)
Lemma acc_range_blocks : forall (B : nat) (g : nat -> Z) (S : nat),
  D.acc_range Z.add (fun s => D.acc_range Z.add (fun p => g (B * s + p)%nat) 0 B) 0 S
  = D.acc_range Z.add g 0 (B * S)%nat.
Proof.
  intros B g S. induction S as [| S IH].
  - rewrite Nat.mul_0_r. reflexivity.
  - rewrite acc_range_succ. cbv beta.
    replace (0 + S)%nat with S by lia.
    replace (B * Datatypes.S S)%nat with (B * S + B)%nat by lia.
    rewrite D.sum_range_split.
    replace (0 + B * S)%nat with (B * S)%nat by lia.
    rewrite IH.
    rewrite (acc_range_shift g (B * S)%nat B).
    reflexivity.
Qed.

(** 32 products per [mma], [K/32] steps, and the flat contraction. *)
Lemma sum_k_blocks : forall g S,
  sum_k (fun s => sum_k (fun p => g (32 * s + p)%nat) 32) S = sum_k g (32 * S)%nat.
Proof. intros g S. unfold sum_k. apply acc_range_blocks. Qed.

(* ------------------------------------------------------------------ *)
(** ** The fragment maps, and that the lanes cover the fragments       *)
(* ------------------------------------------------------------------ *)

(** With [g = laneid >> 2] in 0..7 and [t = laneid & 3] in 0..3, lane [(g,t)]
    holds four [.b32] registers of A and two of B.  Register [r], byte [b] of
    the A group is fragment element [(a_row g r, a_col t r b)] of the 16x32
    tile; register [r], byte [b] of the B group is [(b_col g, b_k t r b)] of
    the 32x8 tile, which is column-major so its lane holds four CONTIGUOUS
    k-values.

    Derived and validated in [tests/ptx_int8_mma_layout.rs]. *)
Definition a_row (g r : nat) : nat := (g + 8 * (r mod 2))%nat.
Definition a_col (t r b : nat) : nat := (4 * t + 16 * (r / 2) + b)%nat.
Definition b_col (g : nat) : nat := g.
Definition b_k   (t r b : nat) : nat := (4 * t + 16 * r + b)%nat.

(** The accumulator: [d0,d1] at row [g], columns [2t] and [2t+1]; [d2,d3] at
    row [g+8], same columns.  This is the map [Int8GemmSchedule] already
    partitions; it is restated here only to compose with it. *)
Definition d_row (g j : nat) : nat := (g + 8 * (j / 2))%nat.
Definition d_col (t j : nat) : nat := (2 * t + j mod 2)%nat.

Lemma a_in_range : forall g t r b,
  (g < 8)%nat -> (t < 4)%nat -> (r < 4)%nat -> (b < 4)%nat ->
  (a_row g r < 16)%nat /\ (a_col t r b < 32)%nat.
Proof.
  intros g t r b Hg Ht Hr Hb. unfold a_row, a_col.
  pose proof (Nat.mod_upper_bound r 2 ltac:(lia)) as Hm.
  assert (r / 2 < 2)%nat by (apply Nat.Div0.div_lt_upper_bound; lia).
  lia.
Qed.

Lemma b_in_range : forall g t r b,
  (g < 8)%nat -> (t < 4)%nat -> (r < 2)%nat -> (b < 4)%nat ->
  (b_col g < 8)%nat /\ (b_k t r b < 32)%nat.
Proof. intros g t r b Hg Ht Hr Hb. unfold b_col, b_k. lia. Qed.

(** **Injective**: no two lane/register/byte triples name one fragment element,
    so nothing is loaded twice and no two source elements collide.  Both digits
    fall straight out of [MixedRadix] - [a_row] is a two-digit index in radix 8
    and [a_col] a three-digit one in radices 4, 4, 2. *)
Lemma a_frag_injective : forall g1 t1 r1 b1 g2 t2 r2 b2,
  (g1 < 8)%nat -> (t1 < 4)%nat -> (r1 < 4)%nat -> (b1 < 4)%nat ->
  (g2 < 8)%nat -> (t2 < 4)%nat -> (r2 < 4)%nat -> (b2 < 4)%nat ->
  a_row g1 r1 = a_row g2 r2 -> a_col t1 r1 b1 = a_col t2 r2 b2 ->
  g1 = g2 /\ t1 = t2 /\ r1 = r2 /\ b1 = b2.
Proof.
  intros g1 t1 r1 b1 g2 t2 r2 b2 Hg1 Ht1 Hr1 Hb1 Hg2 Ht2 Hr2 Hb2 Hrow Hcol.
  unfold a_row, a_col in *.
  pose proof (Nat.mod_upper_bound r1 2 ltac:(lia)) as M1.
  pose proof (Nat.mod_upper_bound r2 2 ltac:(lia)) as M2.
  assert (Q1 : (r1 / 2 < 2)%nat) by (apply Nat.Div0.div_lt_upper_bound; lia).
  assert (Q2 : (r2 / 2 < 2)%nat) by (apply Nat.Div0.div_lt_upper_bound; lia).
  (* the row: radix 8, high digit `r mod 2`, low digit `g` *)
  assert (Hr : (r1 mod 2)%nat = (r2 mod 2)%nat /\ g1 = g2).
  { apply (MR.quot_rem_unique 8); [ lia | lia | lia | lia ]. }
  (* the column: b + 4*(t + 4*(r/2)) *)
  assert (Hc : (r1 / 2)%nat = (r2 / 2)%nat /\ t1 = t2 /\ b1 = b2).
  { apply (MR.two_digit_unique 4 4); [ lia | lia | lia | lia | lia | lia | lia ]. }
  destruct Hr as [Hrm Hg]. destruct Hc as [Hrd [Ht Hb]].
  pose proof (Nat.div_mod_eq r1 2) as E1.
  pose proof (Nat.div_mod_eq r2 2) as E2.
  repeat split; try assumption. lia.
Qed.

Lemma b_frag_injective : forall g1 t1 r1 b1 g2 t2 r2 b2,
  (g1 < 8)%nat -> (t1 < 4)%nat -> (r1 < 2)%nat -> (b1 < 4)%nat ->
  (g2 < 8)%nat -> (t2 < 4)%nat -> (r2 < 2)%nat -> (b2 < 4)%nat ->
  b_col g1 = b_col g2 -> b_k t1 r1 b1 = b_k t2 r2 b2 ->
  g1 = g2 /\ t1 = t2 /\ r1 = r2 /\ b1 = b2.
Proof.
  intros g1 t1 r1 b1 g2 t2 r2 b2 Hg1 Ht1 Hr1 Hb1 Hg2 Ht2 Hr2 Hb2 Hcol Hk.
  unfold b_col, b_k in *.
  assert (H : r1 = r2 /\ t1 = t2 /\ b1 = b2).
  { apply (MR.two_digit_unique 4 4); [ lia | lia | lia | lia | lia | lia | lia ]. }
  destruct H as [H1 [H2 H3]]. repeat split; assumption.
Qed.

(** **Onto**: every one of the 16x32 A positions and 32x8 B positions is held
    by some lane.  This is the leg that says the fragment the [mma] reads is
    fully populated by the emitted loads - without it the theorem below would
    describe a tile with holes in it. *)
Theorem the_lanes_cover_the_a_fragment : forall i p,
  (i < 16)%nat -> (p < 32)%nat ->
  exists g t r b,
    (g < 8)%nat /\ (t < 4)%nat /\ (r < 4)%nat /\ (b < 4)%nat /\
    a_row g r = i /\ a_col t r b = p.
Proof.
  intros i p Hi Hp.
  destruct (MR.pack_onto 8 2 i ltac:(lia) ltac:(lia)) as [qi [ri [Hqi [Hri Hi']]]].
  destruct (MR.pack_onto 16 2 p ltac:(lia) ltac:(lia)) as [qp [rp [Hqp [Hrp Hp']]]].
  destruct (MR.pack_onto 4 4 rp ltac:(lia) ltac:(lia)) as [qr [rr [Hqr [Hrr Hp'']]]].
  unfold MR.pack in *.
  exists ri, qr, (qi + 2 * qp)%nat, rr.
  assert (Hm : ((qi + 2 * qp) mod 2)%nat = qi).
  { replace (qi + 2 * qp)%nat with (qp * 2 + qi)%nat by lia.
    destruct (MR.pack_unpack 2 qp qi ltac:(lia) ltac:(lia)) as [_ Hlo].
    unfold MR.lo, MR.pack in Hlo. exact Hlo. }
  assert (Hd : ((qi + 2 * qp) / 2)%nat = qp).
  { replace (qi + 2 * qp)%nat with (qp * 2 + qi)%nat by lia.
    destruct (MR.pack_unpack 2 qp qi ltac:(lia) ltac:(lia)) as [Hhi _].
    unfold MR.hi, MR.pack in Hhi. exact Hhi. }
  unfold a_row, a_col. rewrite Hm, Hd.
  repeat split; try lia.
Qed.

Theorem the_lanes_cover_the_b_fragment : forall c p,
  (c < 8)%nat -> (p < 32)%nat ->
  exists g t r b,
    (g < 8)%nat /\ (t < 4)%nat /\ (r < 2)%nat /\ (b < 4)%nat /\
    b_col g = c /\ b_k t r b = p.
Proof.
  intros c p Hc Hp.
  destruct (MR.pack_onto 16 2 p ltac:(lia) ltac:(lia)) as [qp [rp [Hqp [Hrp Hp']]]].
  destruct (MR.pack_onto 4 4 rp ltac:(lia) ltac:(lia)) as [qr [rr [Hqr [Hrr Hp'']]]].
  unfold MR.pack in *.
  exists c, qr, qp, rr. unfold b_col, b_k. repeat split; lia.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The staged path: cp.async, a swizzled stage, ldmatrix           *)
(* ------------------------------------------------------------------ *)

(** Since 2026-11-06 the fragments do not come straight from global memory
    ([src/ptx_emitter/int8_staged_gemm.rs]).  A K tile of A - [cta_m] rows of
    [cta_k = 16 C] bytes - is copied by [cp.async] in 16-byte chunks into a
    shared stage, chunk [c] of stage row [row] landing at chunk
    [c XOR swz row], so the eight rows one [ldmatrix] matrix reads fall in
    eight different bank groups; then [ldmatrix.x4] hands each lane its four
    fragment registers.  B likewise, its rows being B's [N]-rows.

    This section proves that what a lane receives is the element
    [the_lanes_cover_the_a_fragment] names: the copy and the swizzle are
    bookkeeping that cancels.  It replaces the address theorems of the
    kernel this one replaced, which loaded each register from
    [(brow + 16 mi + g) * K + 4 t] and [+8K], [+16], [+8K+16] in global
    memory.  [ldmatrix]'s distribution is, like [mma]'s semantics below, a
    DEFINITION - the ISA, pinned on the device by the int8 GEMM's tests. *)

Open Scope nat_scope.

(** Small facts, checked by enumeration rather than derived from the bit
    library: every value involved is below 32. *)
Lemma check_below : forall (P : nat -> bool) n,
  forallb P (seq 0 n) = true -> forall a, a < n -> P a = true.
Proof.
  intros P n H a Ha. rewrite forallb_forall in H. apply H. apply in_seq. lia.
Qed.

Lemma check_below2 : forall (P : nat -> nat -> bool) n1 n2,
  forallb (fun a => forallb (fun b => P a b) (seq 0 n2)) (seq 0 n1) = true ->
  forall a b, a < n1 -> b < n2 -> P a b = true.
Proof.
  intros P n1 n2 H a b Ha Hb.
  pose proof (check_below _ n1 H a Ha) as H1. cbv beta in H1.
  exact (check_below _ n2 H1 b Hb).
Qed.

(** The lane arithmetic the emitter writes with masks and shifts. *)
Lemma lane_bits : forall L, L < 32 ->
  Nat.land L 15 = L mod 16 /\ Nat.land L 7 = L mod 8 /\ Nat.shiftr L 4 = L / 16
  /\ Nat.shiftr (Nat.land L 16) 1 = 8 * (L / 16)
  /\ Nat.land (Nat.shiftr L 3) 1 = (L / 8) mod 2.
Proof.
  intros L HL.
  pose proof (check_below
    (fun L => (Nat.land L 15 =? L mod 16) && (Nat.land L 7 =? L mod 8)
              && (Nat.shiftr L 4 =? L / 16) && (Nat.shiftr (Nat.land L 16) 1 =? 8 * (L / 16))
              && (Nat.land (Nat.shiftr L 3) 1 =? (L / 8) mod 2))%bool
    32 ltac:(vm_compute; reflexivity) L HL) as H.
  cbv beta in H. repeat rewrite Bool.andb_true_iff in H. rewrite !Nat.eqb_eq in H.
  tauto.
Qed.

(** XOR on chunk indices (all below 8). *)
Lemma xor_shift : forall h s x, h < 2 -> s < 8 -> x < 4 ->
  Nat.lxor (Nat.lxor h s) (2 * x) = Nat.lxor (h + 2 * x) s.
Proof.
  intros h s x Hh Hs Hx.
  assert (H : forall h s, h < 2 -> s < 8 ->
            forallb (fun x => Nat.lxor (Nat.lxor h s) (2 * x) =? Nat.lxor (h + 2 * x) s) (seq 0 4) = true).
  { intros h' s' Hh' Hs'.
    exact (check_below2 (fun h s => forallb (fun x => Nat.lxor (Nat.lxor h s) (2 * x) =? Nat.lxor (h + 2 * x) s) (seq 0 4))
             2 8 ltac:(vm_compute; reflexivity) h' s' Hh' Hs'). }
  pose proof (check_below _ 4 (H h s Hh Hs) x Hx) as E. cbv beta in E.
  apply Nat.eqb_eq. exact E.
Qed.

Lemma xor_bound : forall C c s, (C = 2 \/ C = 4 \/ C = 8) -> c < C -> s < C -> Nat.lxor c s < C.
Proof.
  intros C c s HC Hc Hs.
  destruct HC as [-> | [-> | ->]];
    [ pose proof (check_below2 (fun c s => Nat.lxor c s <? 2) 2 2 ltac:(vm_compute; reflexivity) c s Hc Hs)
    | pose proof (check_below2 (fun c s => Nat.lxor c s <? 4) 4 4 ltac:(vm_compute; reflexivity) c s Hc Hs)
    | pose proof (check_below2 (fun c s => Nat.lxor c s <? 8) 8 8 ltac:(vm_compute; reflexivity) c s Hc Hs) ];
    cbv beta in *; apply Nat.ltb_lt; assumption.
Qed.

Lemma xor_cancel : forall c1 c2 s, Nat.lxor c1 s = Nat.lxor c2 s -> c1 = c2.
Proof.
  intros c1 c2 s H.
  rewrite <- (Nat.lxor_0_r c1), <- (Nat.lxor_0_r c2), <- (Nat.lxor_nilpotent s).
  rewrite <- !Nat.lxor_assoc, H. reflexivity.
Qed.

(** The swizzle: [shr.u32 %s, %row, log2 R; and.b32 %s, %s, C-1], with
    [R = 128 / cta_k] rows to a 128-byte line and [C = cta_k / 16] chunks to
    a row ([emit_fp8_e4m3_swizzle]).  [R * C = 8] at every [cta_k] the
    emitter accepts: 32, 64 and 128. *)
Definition swz (R C row : nat) : nat := (row / R) mod C.

Lemma swz_repeats_every_8_rows : forall R C row x,
  0 < R -> 0 < C -> R * C = 8 -> swz R C (row + 8 * x) = swz R C row.
Proof.
  intros R C row x HR HC H8. unfold swz.
  replace (row + 8 * x) with (row + (x * C) * R) by nia.
  rewrite Nat.div_add by lia. apply Nat.Div0.mod_add.
Qed.

Lemma swz_bound : forall R C row, 0 < C -> swz R C row < C.
Proof. intros. unfold swz. apply Nat.mod_upper_bound. lia. Qed.

(** Byte [b] of logical chunk [c] of stage row [row]: where the copy put it. *)
Definition soff (R C row c b : nat) : nat := row * (16 * C) + 16 * Nat.lxor c (swz R C row) + b.

(** **The slots are disjoint**: no two (row, chunk, byte) share a byte of the
    stage, so no copy overwrites another's. *)
Theorem the_stage_slots_are_disjoint : forall R C r1 c1 b1 r2 c2 b2,
  0 < R -> (C = 2 \/ C = 4 \/ C = 8) -> c1 < C -> c2 < C -> b1 < 16 -> b2 < 16 ->
  soff R C r1 c1 b1 = soff R C r2 c2 b2 -> r1 = r2 /\ c1 = c2 /\ b1 = b2.
Proof.
  intros R C r1 c1 b1 r2 c2 b2 HR HC Hc1 Hc2 Hb1 Hb2 H.
  assert (HC0 : 0 < C) by lia.
  pose proof (xor_bound C c1 (swz R C r1) HC Hc1 (swz_bound R C r1 HC0)) as X1.
  pose proof (xor_bound C c2 (swz R C r2) HC Hc2 (swz_bound R C r2 HC0)) as X2.
  unfold soff in H.
  destruct (MR.quot_rem_unique (16 * C) r1 (16 * Nat.lxor c1 (swz R C r1) + b1)
              r2 (16 * Nat.lxor c2 (swz R C r2) + b2) ltac:(lia) ltac:(lia) ltac:(lia) ltac:(lia)) as [Er Ex].
  subst r2.
  destruct (MR.quot_rem_unique 16 (Nat.lxor c1 (swz R C r1)) b1 (Nat.lxor c2 (swz R C r1)) b2
              ltac:(lia) Hb1 Hb2 ltac:(lia)) as [Ec Eb].
  repeat split; [ | exact Eb ]. exact (xor_cancel c1 c2 _ Ec).
Qed.

(** The copy.  Copy-role thread [(lr, lc)] (row [tid / cpr], chunk
    [tid mod cpr]) copies its [u]-th chunk to its shared base
    [lr * cta_k + 16 * (lc XOR swz lr)] plus [u * rpp * cta_k], from global
    [&A[row0 + lr + u*rpp][k0 + 16 lc]].  It computes the swizzle ONCE, from
    [lr]; that is right because [rpp] is a multiple of 8 (the emitter refuses
    a tile where it is not) and the swizzle repeats every 8 rows. *)
Definition copy_dst (R C rpp lr lc u : nat) : nat :=
  lr * (16 * C) + 16 * Nat.lxor lc (swz R C lr) + u * rpp * (16 * C).

Theorem the_copy_lands_at_its_swizzled_place : forall R C rpp lr lc u,
  0 < R -> 0 < C -> R * C = 8 -> rpp mod 8 = 0 ->
  copy_dst R C rpp lr lc u = soff R C (lr + u * rpp) lc 0.
Proof.
  intros R C rpp lr lc u HR HC H8 Hrpp. unfold copy_dst, soff.
  pose proof (Nat.div_mod_eq rpp 8) as E. rewrite Hrpp, Nat.add_0_r in E.
  replace (lr + u * rpp) with (lr + 8 * (u * (rpp / 8))) by nia.
  rewrite swz_repeats_every_8_rows by assumption. nia.
Qed.

(** What the copies leave in a stage, once every thread's group has landed
    and the barrier has published them: stage row [row]'s logical chunk [c],
    byte [b], holds [G row (16 c + b)] - for A, [A[row0 + row][k0 + 16c + b]]
    (zero past M or K: a copy past the matrix is zero-sized).  That every
    (row, chunk) is copied, by one thread, is
    [Int8GemmSchedule.the_copies_cover_the_stage_once]; that the copies land
    in disjoint slots is [the_stage_slots_are_disjoint]. *)
Definition stage_holds (R C rows : nat) (G : nat -> nat -> Z) (mem : nat -> Z) : Prop :=
  forall row c b, row < rows -> c < C -> b < 16 -> mem (soff R C row c b) = G row (16 * c + b).

(** **[ldmatrix.sync.aligned.m8n8.x4.shared.b16], as a DEFINITION (ISA).**
    Matrix [q]'s eight 16-byte rows are at the addresses lanes [8q .. 8q+7]
    supply; lane [l] receives, in register [q], the two b16 elements at row
    [l / 4], columns [2 (l mod 4)] and [+1] - bytes [4 (l mod 4) .. +3] of
    that row. *)
Definition ldmatrix_x4 (mem : nat -> Z) (addr : nat -> nat) (l q b : nat) : Z :=
  mem (addr (8 * q + l / 4) + 4 * (l mod 4) + b).

(** The address lane [L] supplies for A's m-tile [i] at K step [ks] of the
    stage, as [emit_fp8_e4m3_ldmatrix_offsets] computes it: row
    [(L & 15) + warp_row0], K half [L >> 4], chunk
    [(half XOR swz (L & 7)) XOR 2ks], and [i * 16 * cta_k] as an immediate. *)
Definition a_lane_addr (R C wr0 i ks L : nat) : nat :=
  (Nat.land L 15 + wr0) * (16 * C)
  + 16 * Nat.lxor (Nat.lxor (Nat.shiftr L 4) (swz R C (Nat.land L 7))) (2 * ks)
  + i * 16 * (16 * C).

(** B's: row [((L & 16) >> 1) + (L & 7) + warp_col0], K half [(L >> 3) & 1],
    and [jp * 16 * cta_k] for the pair of n8 tiles [jp]. *)
Definition b_lane_addr (R C wc0 jp ks L : nat) : nat :=
  (Nat.shiftr (Nat.land L 16) 1 + Nat.land L 7 + wc0) * (16 * C)
  + 16 * Nat.lxor (Nat.lxor (Nat.land (Nat.shiftr L 3) 1) (swz R C (Nat.land L 7))) (2 * ks)
  + jp * 16 * (16 * C).

(** **A's fragment.**  Register [q], byte [b] of lane [l] is the A element
    [(a_row g q, 32 ks + a_col t q b)] of the warp's m-tile [i], with
    [g = l / 4] and [t = l mod 4] - exactly what the [mma] reads. *)
Theorem the_ldmatrix_a_fragment_is_the_source_element :
  forall R C rows G mem wr0 i ks l q b,
    0 < R -> R * C = 8 -> (C = 2 \/ C = 4 \/ C = 8) ->
    stage_holds R C rows G mem -> wr0 mod 8 = 0 -> wr0 + 16 * i + 16 <= rows ->
    2 * ks < C -> l < 32 -> q < 4 -> b < 4 ->
    ldmatrix_x4 mem (a_lane_addr R C wr0 i ks) l q b
    = G (wr0 + 16 * i + a_row (l / 4) q) (32 * ks + a_col (l mod 4) q b).
Proof.
  intros R C rows G mem wr0 i ks l q b HR H8 HC Hst Hw Hrows Hks Hl Hq Hb.
  unfold ldmatrix_x4, a_lane_addr.
  set (g := l / 4). set (t := l mod 4).
  assert (Hg : g < 8) by (subst g; apply Nat.Div0.div_lt_upper_bound; lia).
  assert (Ht : t < 4) by (subst t; apply Nat.mod_upper_bound; lia).
  destruct (lane_bits (8 * q + g) ltac:(lia)) as [B15 [B7 [B4 _]]].
  rewrite B15, B7, B4.
  assert (M16 : (8 * q + g) mod 16 = 8 * (q mod 2) + g).
  { pose proof (Nat.div_mod_eq q 2). pose proof (Nat.mod_upper_bound q 2 ltac:(lia)).
    replace (8 * q + g) with ((8 * (q mod 2) + g) + (q / 2) * 16) by lia.
    rewrite Nat.Div0.mod_add. apply Nat.mod_small. lia. }
  assert (M8 : (8 * q + g) mod 8 = g).
  { replace (8 * q + g) with (g + q * 8) by lia. rewrite Nat.Div0.mod_add. apply Nat.mod_small. lia. }
  assert (D16 : (8 * q + g) / 16 = q / 2).
  { pose proof (Nat.div_mod_eq q 2). pose proof (Nat.mod_upper_bound q 2 ltac:(lia)).
    replace (8 * q + g) with ((8 * (q mod 2) + g) + (q / 2) * 16) by lia.
    rewrite Nat.div_add by lia. rewrite Nat.div_small by lia. lia. }
  rewrite M16, M8, D16.
  set (row := wr0 + 16 * i + a_row g q).
  assert (HC0 : 0 < C) by lia.
  assert (Hsw : swz R C g = swz R C row).
  { subst row. unfold a_row. pose proof (Nat.div_mod_eq wr0 8) as E. rewrite Hw, Nat.add_0_r in E.
    replace (wr0 + 16 * i + (g + 8 * (q mod 2))) with (g + 8 * (wr0 / 8 + 2 * i + q mod 2)) by lia.
    symmetry. apply swz_repeats_every_8_rows; lia. }
  assert (Hq2 : q / 2 < 2) by (apply Nat.Div0.div_lt_upper_bound; lia).
  assert (Hsw8 : swz R C g < 8) by (pose proof (swz_bound R C g HC0); lia).
  rewrite (xor_shift (q / 2) (swz R C g) ks Hq2 Hsw8 ltac:(lia)), Hsw.
  assert (Hc : q / 2 + 2 * ks < C).
  { destruct HC as [-> | [-> | ->]]; lia. }
  assert (Haddr : (8 * (q mod 2) + g + wr0) * (16 * C) + 16 * Nat.lxor (q / 2 + 2 * ks) (swz R C row)
                  + i * 16 * (16 * C) + 4 * t + b
                  = soff R C row (q / 2 + 2 * ks) (4 * t + b)).
  { unfold soff. subst row. unfold a_row. nia. }
  rewrite Haddr, Hst.
  - f_equal. unfold a_col. pose proof (Nat.div_mod_eq q 2). lia.
  - subst row. unfold a_row. pose proof (Nat.mod_upper_bound q 2 ltac:(lia)). lia.
  - exact Hc.
  - lia.
Qed.

(** **B's fragment.**  Register [q] of lane [l] is register [q mod 2] of the
    fragment of n8 tile [2 jp + q / 2] - the emitter pushes [{r0,r1}] and
    [{r2,r3}] as the fragments of tiles [2jp] and [2jp+1] - at
    [(b_col g, 32 ks + b_k t (q mod 2) b)]. *)
Theorem the_ldmatrix_b_fragment_is_the_source_element :
  forall R C rows G mem wc0 jp ks l q b,
    0 < R -> R * C = 8 -> (C = 2 \/ C = 4 \/ C = 8) ->
    stage_holds R C rows G mem -> wc0 mod 8 = 0 -> wc0 + 16 * jp + 16 <= rows ->
    2 * ks < C -> l < 32 -> q < 4 -> b < 4 ->
    ldmatrix_x4 mem (b_lane_addr R C wc0 jp ks) l q b
    = G (wc0 + 8 * (2 * jp + q / 2) + b_col (l / 4)) (32 * ks + b_k (l mod 4) (q mod 2) b).
Proof.
  intros R C rows G mem wc0 jp ks l q b HR H8 HC Hst Hw Hrows Hks Hl Hq Hb.
  unfold ldmatrix_x4, b_lane_addr.
  set (g := l / 4). set (t := l mod 4).
  assert (Hg : g < 8) by (subst g; apply Nat.Div0.div_lt_upper_bound; lia).
  assert (Ht : t < 4) by (subst t; apply Nat.mod_upper_bound; lia).
  destruct (lane_bits (8 * q + g) ltac:(lia)) as [_ [B7 [_ [B16 B3]]]].
  rewrite B7, B16, B3.
  assert (M8 : (8 * q + g) mod 8 = g).
  { replace (8 * q + g) with (g + q * 8) by lia. rewrite Nat.Div0.mod_add. apply Nat.mod_small. lia. }
  assert (D8 : (8 * q + g) / 8 = q).
  { replace (8 * q + g) with (g + q * 8) by lia. rewrite Nat.div_add by lia. rewrite Nat.div_small by lia. lia. }
  assert (D16 : (8 * q + g) / 16 = q / 2).
  { pose proof (Nat.div_mod_eq q 2). pose proof (Nat.mod_upper_bound q 2 ltac:(lia)).
    replace (8 * q + g) with ((8 * (q mod 2) + g) + (q / 2) * 16) by lia.
    rewrite Nat.div_add by lia. rewrite Nat.div_small by lia. lia. }
  rewrite M8, D8, D16.
  set (row := wc0 + 8 * (2 * jp + q / 2) + b_col g).
  assert (HC0 : 0 < C) by lia.
  assert (Hsw : swz R C g = swz R C row).
  { subst row. unfold b_col. pose proof (Nat.div_mod_eq wc0 8) as E. rewrite Hw, Nat.add_0_r in E.
    replace (wc0 + 8 * (2 * jp + q / 2) + g) with (g + 8 * (wc0 / 8 + 2 * jp + q / 2)) by lia.
    symmetry. apply swz_repeats_every_8_rows; lia. }
  assert (Hq2 : q mod 2 < 2) by (apply Nat.mod_upper_bound; lia).
  assert (Hsw8 : swz R C g < 8) by (pose proof (swz_bound R C g HC0); lia).
  rewrite (xor_shift (q mod 2) (swz R C g) ks Hq2 Hsw8 ltac:(lia)), Hsw.
  assert (Hc : q mod 2 + 2 * ks < C).
  { destruct HC as [-> | [-> | ->]]; lia. }
  assert (Haddr : (8 * (q / 2) + g + wc0) * (16 * C) + 16 * Nat.lxor (q mod 2 + 2 * ks) (swz R C row)
                  + jp * 16 * (16 * C) + 4 * t + b
                  = soff R C row (q mod 2 + 2 * ks) (4 * t + b)).
  { unfold soff. subst row. unfold b_col. nia. }
  rewrite Haddr, Hst.
  - f_equal. unfold b_k. lia.
  - subst row. unfold b_col. assert (q / 2 < 2) by (apply Nat.Div0.div_lt_upper_bound; lia). lia.
  - exact Hc.
  - lia.
Qed.

Open Scope Z_scope.

(* ------------------------------------------------------------------ *)
(** ** The instruction, as a DEFINITION (the trusted base)             *)
(* ------------------------------------------------------------------ *)

(** [mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32] computes, for every
    element [(i, jj)] of the 16x8 accumulator,

      D[i][jj] = C[i][jj] + sum over p < 32 of Afrag[i][p] * Bfrag[p][jj]

    **This is an ASSUMPTION about the ISA, not a theorem.**  A proof over [Z]
    cannot supply an instruction set.  It sits in the trusted base exactly
    where [vpdpwssd]'s semantics sit for the CPU chain, and it is pinned
    empirically by [tests/ptx_int8_mma_layout.rs], which runs the real
    instruction on the device against a plain integer matmul.

    Note what is deliberately NOT assumed: the ORDER in which one [mma] sums
    its 32 products.  The ISA does not specify it, so nothing below depends on
    it - the licence bounds the sum of ABSOLUTE values, which dominates every
    partial sum in every order and at every granularity. *)
Definition mma (Afrag Bfrag : nat -> nat -> Z) (i jj : nat) : Z :=
  sum_k (fun p => Afrag i p * Bfrag p jj) 32.

(** The fragments the emitted loads populate at K-offset [kk].  That they hold
    these values is [the_emitted_*_address_is_its_fragment_element]; that they
    hold ALL of them is [the_lanes_cover_the_*_fragment]. *)
Definition a_frag (A : nat -> nat -> Z) (brow mi kk : nat) : nat -> nat -> Z :=
  fun i p => A (brow + 16 * mi + i)%nat (kk + p)%nat.
Definition b_frag (B : nat -> nat -> Z) (bcol ni kk : nat) : nat -> nat -> Z :=
  fun p jj => B (bcol + 8 * ni + jj)%nat (kk + p)%nat.

(* ------------------------------------------------------------------ *)
(** ** The K loop, and every split factor                              *)
(* ------------------------------------------------------------------ *)

(** One K step of the emitted loop: the pointers advance by [32 * nz] bytes per
    iteration from a base of [32 * z], so CTA [z] visits step indices congruent
    to [z] modulo [nz] - the residue classes [GridStrideSplit] partitions. *)
Definition step (A B : nat -> nat -> Z) (brow bcol mi ni i jj s : nat) : Z :=
  mma (a_frag A brow mi (32 * s)) (b_frag B bcol ni (32 * s)) i jj.

Definition prod (A B : nat -> nat -> Z) (row col k : nat) : Z :=
  A row k * B col k.

Theorem the_k_loop_is_the_contraction :
  forall A B brow bcol mi ni i jj S,
    sum_k (step A B brow bcol mi ni i jj) S
    = sum_k (prod A B (brow + 16 * mi + i)%nat (bcol + 8 * ni + jj)%nat) (32 * S)%nat.
Proof.
  intros. unfold step, mma, a_frag, b_frag, prod.
  apply (sum_k_blocks
           (fun k => A (brow + 16 * mi + i)%nat k * B (bcol + 8 * ni + jj)%nat k) S).
Qed.

Lemma sum_upto_is_sum_k : forall f S, GS.sum_upto Z.add f S = sum_k f S.
Proof. reflexivity. Qed.

(** **Every split factor gives the whole contraction.**  [GridStrideSplit]
    supplies this with no new reasoning - the residue classes are the same
    decomposition the attention kernel and this kernel's own output tiling use.
    What is new is only that the CLASS TERMS are [mma] steps. *)
Theorem every_split_factor_gives_the_contraction :
  forall A B brow bcol mi ni i jj S nz,
    (0 < nz)%nat ->
    GS.combine Z.add (step A B brow bcol mi ni i jj) nz S nz
    = sum_k (prod A B (brow + 16 * mi + i)%nat (bcol + 8 * ni + jj)%nat) (32 * S)%nat.
Proof.
  intros A B brow bcol mi ni i jj S nz Hnz.
  rewrite GS.grid_stride_exact by assumption.
  rewrite sum_upto_is_sum_k.
  apply the_k_loop_is_the_contraction.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The licence                                                     *)
(* ------------------------------------------------------------------ *)

(** The largest contraction an int32 accumulator holds exactly, for full-range
    int8 operands: [|sum| <= K * 128^2] must fit [i32::MAX].
    Signed int8 includes -128; using 127 misses the largest positive product.

    The emitter refuses at K-STEP granularity, because [K mod 32 = 0] is
    already its shape precondition and a bound that is not a multiple of 32 is
    not expressible as a refusal it can state. *)
Definition MAX_EXACT_K : Z := MC.I32MAX / (128 * 128).
Definition MAX_EXACT_K_STEPS : Z := MAX_EXACT_K / 32 * 32.

Theorem the_bound_is_one_k_step_wide :
  MAX_EXACT_K = 131071 /\ MAX_EXACT_K_STEPS = 131040.
Proof. split; vm_compute; reflexivity. Qed.

(** The two sides of it.  [MAX_EXACT_K_STEPS] is admissible and the next K step
    is not - one step wide, which is what makes an off-by-one in the emitter's
    constant visible rather than absorbed. *)
Theorem the_licence_admits_131040_and_refuses_131072 :
  131040 * (128 * 128) <= MC.I32MAX /\ MC.I32MAX < 131072 * (128 * 128).
Proof. unfold MC.I32MAX. split; lia. Qed.

(** **The refutation, refereed against the silicon.**  The device returned
    -2147358688 at K = 133152 with every operand 127.  That number is
    reproduced here from [wrap32] alone - the model was not fitted to it, and a
    model that merely said "it overflows" would agree with any wrong answer. *)
Theorem the_measured_overflow_is_two_s_complement :
  133152 * (127 * 127) = 2147608608
  /\ MC.wrap32 2147608608 = -2147358688.
Proof. split; vm_compute; reflexivity. Qed.

(** The old 127-based licence admitted this full-I8 input. The device
    returned -2147483648 for A = B = -128 at K = 131072, even with C zeroed. *)
Theorem the_signed_minimum_exposes_the_old_licence :
  131072 * ((-128) * (-128)) = 2147483648
  /\ MC.wrap32 2147483648 = -2147483648.
Proof. split; vm_compute; reflexivity. Qed.

(** Every partial sum of a bounded sequence is bounded by its length times the
    bound - which dominates the accumulator's value at every granularity and in
    every summation order, so nothing below depends on the [mma]'s unspecified
    internal order. *)
Lemma acc_range_abs_bound : forall f lo len c,
  0 <= c -> (forall i, Z.abs (f i) <= c) ->
  Z.abs (D.acc_range Z.add f lo len) <= Z.of_nat len * c.
Proof.
  intros f lo len c Hc H. induction len as [| len IH]; cbn [D.acc_range].
  - simpl. lia.
  - rewrite Nat2Z.inj_succ.
    pose proof (H (lo + len)%nat) as Hi.
    pose proof (Z.abs_triangle (D.acc_range Z.add f lo len) (f (lo + len)%nat)) as Ht.
    lia.
Qed.

(** **The licence is load-bearing, and this is where.**  [ExactGemmMicro.wsum]
    wraps at EVERY step; under the licence it equals the [Z] sum. *)
Theorem bounded_products_accumulate_exactly : forall f n c,
  0 <= c -> (forall i, Z.abs (f i) <= c) -> Z.of_nat n * c <= MC.I32MAX ->
  MC.wsum f 0 n = sum_k f n.
Proof.
  intros f n c Hc H Hlic. unfold sum_k.
  rewrite <- MC.sum_from_is_acc_range.
  apply MC.flush_exact_in_int32.
  intros i Hi.
  pose proof (acc_range_abs_bound f 0 i c Hc H) as Hb.
  rewrite MC.sum_from_is_acc_range.
  assert (Z.of_nat i * c <= Z.of_nat n * c) by (apply Z.mul_le_mono_nonneg_r; [ lia | apply Nat2Z.inj_le; lia ]).
  unfold MC.in_i32, MC.I32MIN, MC.I32MAX in *. lia.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The capstone                                                    *)

(* ------------------------------------------------------------------ *)
(** ** The SPLIT-K accumulation, in int32 rather than in [Z]           *)
(* ------------------------------------------------------------------ *)

(** [bounded_products_accumulate_exactly] above is the int32 statement for the
    FLAT accumulation - one class, the products visited in index order, which
    is what the kernel does at [gridDim.z = 1] and is the case the device
    measurement was taken in.  At [gridDim.z > 1] there are two wrapping folds
    and not one:

      - each CTA accumulates ITS OWN residue class in an int32 register, one
        [mma] per K step, in ascending visit order;
      - [red.global.add.s32] then combines the CTAs' partials in int32 IN
        MEMORY, in whatever order they land.

    Everything below is that pair.  Nothing new is needed about the
    PARTITION - [GridStrideSplit] already has it, and this file already
    instantiates it in [every_split_factor_gives_the_contraction] - what is
    new is that neither fold wraps.

    **The licence is what makes both true, and it is stronger than it looks.**
    It bounds the sum of ABSOLUTE values of all K products, and every partial
    of either fold is a sum over a SUBSET of those products.  So one hypothesis
    covers both, at every split factor and every landing order. *)

Definition afun (f : nat -> Z) : nat -> Z := fun i => Z.abs (f i).

(** A class's own int32 accumulator: [GS.class_sum] with a wrap at every
    step it takes. *)
Fixpoint wclass (f : nat -> Z) (w n S : nat) : Z :=
  match S with
  | O => 0
  | Datatypes.S k =>
      if Nat.eqb (k mod n) w then MC.wrap32 (wclass f w n k + f k)
      else wclass f w n k
  end.

(** The atomic combine: a wrapping fold of the partials, in list order, from
    the destination's initial value.  That initial value is [C] before the
    launch, and it is not decoration - see the refutation at the foot of this
    section. *)
Fixpoint wcombine (c0 : Z) (l : list Z) : Z :=
  match l with
  | [] => c0
  | x :: r => wcombine (MC.wrap32 (c0 + x)) r
  end.

(** A single class's budget is at most the whole - summing a NON-NEGATIVE
    function over a subset of the range cannot exceed summing it over all of
    it.  This is what lets one licence hypothesis serve both folds. *)
Lemma class_abs_le_total : forall f w n S,
  GS.class_sum Z.add (afun f) w n S <= sum_k (afun f) S.
Proof.
  intros f w n S. induction S as [| S IH].
  - cbn. lia.
  - cbn [GS.class_sum]. unfold sum_k in *. rewrite acc_range_succ.
    assert (H0 : 0 <= afun f (0 + S)%nat) by (unfold afun; apply Z.abs_nonneg).
    replace (0 + S)%nat with S in * by lia.
    destruct (Nat.eqb (S mod n) w); lia.
Qed.

Lemma class_abs_bounds : forall f w n S,
  Z.abs (GS.class_sum Z.add f w n S) <= GS.class_sum Z.add (afun f) w n S.
Proof.
  intros f w n S. induction S as [| S IH]; cbn [GS.class_sum]; [ simpl; lia | ].
  destruct (Nat.eqb (S mod n) w); [ | exact IH ].
  pose proof (Z.abs_triangle (GS.class_sum Z.add f w n S) (f S)).
  unfold afun in *. lia.
Qed.

(** **A CTA's own int32 accumulator does not wrap.** *)
Lemma wclass_exact : forall f w n S,
  GS.class_sum Z.add (afun f) w n S <= MC.I32MAX ->
  wclass f w n S = GS.class_sum Z.add f w n S.
Proof.
  intros f w n S. induction S as [| S IH]; intros Hb.
  - reflexivity.
  - pose proof (class_abs_bounds f w n (Datatypes.S S)) as Hab.
    cbn [wclass GS.class_sum] in Hab, Hb |- *.
    (* [unfold] leaves a beta-redex that [lia] cannot see through; this is the
       one place [afun] has to be normalised by hand. *)
    assert (Ha : afun f S = Z.abs (f S)) by reflexivity.
    rewrite Ha in Hab, Hb.
    (* [remember] rather than [destruct ... eqn:], because the scrutinee occurs
       in the hypotheses as well as the goal and only [remember] abstracts all
       three at once. *)
    remember (Nat.eqb (S mod n) w) as bb eqn:E; destruct bb.
    + assert (Hs : GS.class_sum Z.add (afun f) w n S <= MC.I32MAX).
      { pose proof (Z.abs_nonneg (f S)). lia. }
      rewrite (IH Hs).
      apply MC.wrap32_id.
      assert (HX : Z.abs (GS.class_sum Z.add f w n S + f S) <= MC.I32MAX) by lia.
      apply Z.abs_le in HX.
      unfold MC.in_i32, MC.I32MIN, MC.I32MAX in *. lia.
    + exact (IH Hb).
Qed.

Lemma abs_fold_nonneg : forall l, 0 <= fold_right Z.add 0 (map Z.abs l).
Proof.
  induction l as [| x r IH]; cbn [map fold_right]; [ lia | ].
  pose proof (Z.abs_nonneg x). lia.
Qed.

(** **The atomic combine does not wrap either**, given the same budget - and
    the statement carries [c0], so what the budget has to cover is the
    destination's initial value as well as the partials. *)
Lemma wcombine_exact : forall l c0,
  Z.abs c0 + fold_right Z.add 0 (map Z.abs l) <= MC.I32MAX ->
  wcombine c0 l = c0 + fold_right Z.add 0 l.
Proof.
  induction l as [| x r IH]; intros c0 H; cbn [wcombine map fold_right] in *.
  - lia.
  - pose proof (Z.abs_triangle c0 x) as Ht.
    pose proof (abs_fold_nonneg r) as Hr.
    pose proof (Z.abs_nonneg x) as Hx.
    rewrite MC.wrap32_id by (unfold MC.in_i32, MC.I32MIN, MC.I32MAX in *; lia).
    rewrite IH by lia. lia.
Qed.

Lemma fold_map_le : forall (l : list nat) (g h : nat -> Z),
  (forall w, g w <= h w) ->
  fold_right Z.add 0 (map g l) <= fold_right Z.add 0 (map h l).
Proof.
  induction l as [| a r IH]; intros g h H; cbn [map fold_right]; [ lia | ].
  pose proof (H a). pose proof (IH g h H). lia.
Qed.

(** **THE THEOREM.**  Both folds, at every split factor and every landing
    order, from a zeroed destination: the int32 result is the [Z] result. *)
Theorem the_split_k_accumulation_is_exact_in_int32 :
  forall f n S order,
    (0 < n)%nat ->
    Permutation order (seq 0 n) ->
    GS.sum_upto Z.add (afun f) S <= MC.I32MAX ->
    wcombine 0 (map (fun w => wclass f w n S) order) = GS.sum_upto Z.add f S.
Proof.
  intros f n S order Hn Hperm Hb.
  (* Each class's own budget is at most the whole. *)
  assert (Hc : forall w, GS.class_sum Z.add (afun f) w n S <= MC.I32MAX).
  { intros w. pose proof (class_abs_le_total f w n S) as Hw.
    rewrite sum_upto_is_sum_k in Hb. lia. }
  (* Rewrite every wrapping class partial to its Z value. *)
  assert (Hmap : map (fun w => wclass f w n S) order
                 = map (fun w => GS.class_sum Z.add f w n S) order).
  { apply map_ext. intros w. apply wclass_exact, Hc. }
  rewrite Hmap.
  (* The combine's own budget: sum of |partials| <= sum of class abs sums. *)
  assert (Hbud : fold_right Z.add 0 (map Z.abs (map (fun w => GS.class_sum Z.add f w n S) order))
                 <= MC.I32MAX).
  { rewrite map_map.
    eapply Z.le_trans.
    - apply (fold_map_le order (fun w => Z.abs (GS.class_sum Z.add f w n S))
                               (fun w => GS.class_sum Z.add (afun f) w n S)).
      intros w. apply class_abs_bounds.
    - rewrite (GS.fold_right_add_permutation _ (map (fun w => GS.class_sum Z.add (afun f) w n S) (seq 0 n)))
        by (apply Permutation_map; exact Hperm).
      rewrite <- GS.combine_is_fold.
      rewrite (GS.grid_stride_exact (afun f) n S Hn). exact Hb. }
  rewrite wcombine_exact by (simpl Z.abs; lia).
  rewrite Z.add_0_l.
  apply GS.atomics_may_land_in_any_order; assumption.
Qed.

(** The capstone's int32 conjunct, at every split factor and every landing
    order, from a zeroed destination.  Together with
    [every_split_factor_gives_the_contraction] - which is the same statement in
    [Z] - this is the whole of the split-K claim. *)
Corollary the_split_k_combine_is_exact_in_int32 :
  forall A B m K row col nz order,
    0 <= m ->
    (forall r k, Z.abs (A r k) <= m) ->
    (forall c k, Z.abs (B c k) <= m) ->
    Z.of_nat K * (m * m) <= MC.I32MAX ->
    (0 < nz)%nat ->
    Permutation order (seq 0 nz) ->
    wcombine 0 (map (fun w => wclass (prod A B row col) w nz K) order)
    = sum_k (prod A B row col) K.
Proof.
  intros A B m K row col nz order Hm HA HB Hlic Hnz Hperm.
  rewrite <- sum_upto_is_sum_k.
  apply the_split_k_accumulation_is_exact_in_int32; try assumption.
  rewrite sum_upto_is_sum_k. unfold sum_k.
  (* the sum of |products| is bounded by its own absolute value, so the
     absolute bound below serves.  Proved inline: the stdlib name for this has
     moved between releases and a proof should not depend on which one. *)
  assert (Hself : forall z : Z, z <= Z.abs z)
    by (intros z; destruct (Z.abs_spec z) as [[? E] | [? E]]; lia).
  eapply Z.le_trans; [ apply Hself | ].
  eapply Z.le_trans; [ | exact Hlic ].
  apply acc_range_abs_bound; [ nia | ].
  intros i. unfold afun, prod. rewrite Z.abs_idemp, Z.abs_mul.
  apply Z.mul_le_mono_nonneg;
    auto using Z.abs_nonneg.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The licence is about a ZEROED destination                       *)
(* ------------------------------------------------------------------ *)

(** Writing [wcombine 0] above forced a hypothesis to be stated that exists
    nowhere in the compiler: the licence [K * 128^2 <= i32::MAX] is sufficient
    only when [C] starts at zero.  The emitter's own comment says a caller must
    zero it, every test does, and NOTHING says the licence depends on it.

    That is not academic, because this kernel ACCUMULATES into [C] - which is
    exactly what lets [gridDim.z] split the contraction - so a caller who
    splits K across LAUNCHES instead, into the same int32 [C], is doing the
    obvious thing.  Every launch is individually licensed and the accumulation
    is not.

    Measured on the device, M=16 N=8, every operand -128, K = 65,504 per launch
    (half the licensed maximum, rounded down to a whole K step),
    C zeroed once before the first:

      launch 1:  C[0] =  1,073,217,536   exact
      launch 2:  C[0] =  2,146,435,072   exact
      launch 3:  C[0] = -1,075,314,688   WRAPPED   (exact: 3,219,652,608)

    The three theorems below reproduce that from [MC.wrap32] alone - the CPU
    chain's int32 model, written months earlier for a different instruction on
    a different architecture - which is what makes them a refutation rather
    than a transcription of what the card said. *)

Definition LICENSED_HALF : Z := (MAX_EXACT_K_STEPS / 64 * 32) * (128 * 128).

Theorem each_launch_is_licensed_and_three_of_them_wrap :
  LICENSED_HALF = 1073217536
  /\ LICENSED_HALF <= MC.I32MAX
  /\ 2 * LICENSED_HALF <= MC.I32MAX
  /\ MC.I32MAX < 3 * LICENSED_HALF.
Proof.
  unfold LICENSED_HALF, MAX_EXACT_K_STEPS, MAX_EXACT_K, MC.I32MAX.
  repeat split; vm_compute; try reflexivity; discriminate.
Qed.

(** The wrapped value the device returned, from the model. *)
Theorem the_third_launch_wraps_to_the_measured_value :
  MC.wrap32 (3 * LICENSED_HALF) = -1075314688.
Proof. vm_compute. reflexivity. Qed.

(** And the statement of the hypothesis: with a destination that is not zero,
    the combine is not the sum, at magnitudes the licence admits. *)
Theorem the_combine_needs_a_zeroed_destination :
  wcombine (2 * LICENSED_HALF) [LICENSED_HALF]
  <> 2 * LICENSED_HALF + LICENSED_HALF.
Proof. vm_compute. discriminate. Qed.

(** The control that stops the previous theorem being read as "the combine is
    broken": from zero, the same partial is exact. *)
Theorem from_zero_the_same_partial_is_exact :
  wcombine 0 [LICENSED_HALF; LICENSED_HALF] = 2 * LICENSED_HALF.
Proof. vm_compute. reflexivity. Qed.

(* ------------------------------------------------------------------ *)

(* ------------------------------------------------------------------ *)
(** ** K TILES striped over [z], in int32                              *)
(* ------------------------------------------------------------------ *)

(** The staged kernel splits K by TILES of [cta_k = 32 T]: CTA [z] of [nz]
    takes K tiles [z], [z + nz], ..., so product [k] is CTA
    [(k / cta_k) mod nz]'s - [Int8GemmSchedule.k_tile_owner].  The theorem
    above is about the residue classes [k mod n] of the kernel this one
    replaced; this is the same pair of wrapping folds for ANY owner map, so it
    covers both ([wclass_is_wpart]). *)
Fixpoint wpart (owner : nat -> nat) (f : nat -> Z) (w S : nat) : Z :=
  match S with
  | O => 0
  | Datatypes.S k =>
      if Nat.eqb (owner k) w then MC.wrap32 (wpart owner f w k + f k)
      else wpart owner f w k
  end.

Lemma wclass_is_wpart : forall f w n S, wclass f w n S = wpart (fun k => k mod n)%nat f w S.
Proof.
  intros f w n S. induction S as [| S IH]; cbn [wclass wpart]; [ reflexivity | now rewrite IH ].
Qed.

Lemma part_abs_le_total : forall owner f w S,
  D.part Z.add owner (afun f) w S <= sum_k (afun f) S.
Proof.
  intros owner f w S. induction S as [| S IH].
  - cbn. lia.
  - cbn [D.part]. unfold sum_k in *. rewrite acc_range_succ.
    assert (H0 : 0 <= afun f (0 + S)%nat) by (unfold afun; apply Z.abs_nonneg).
    replace (0 + S)%nat with S in * by lia.
    destruct (Nat.eqb (owner S) w); lia.
Qed.

Lemma part_abs_bounds : forall owner f w S,
  Z.abs (D.part Z.add owner f w S) <= D.part Z.add owner (afun f) w S.
Proof.
  intros owner f w S. induction S as [| S IH]; cbn [D.part]; [ simpl; lia | ].
  destruct (Nat.eqb (owner S) w); [ | exact IH ].
  pose proof (Z.abs_triangle (D.part Z.add owner f w S) (f S)).
  unfold afun in *. lia.
Qed.

(** A CTA's own int32 register does not wrap. *)
Lemma wpart_exact : forall owner f w S,
  D.part Z.add owner (afun f) w S <= MC.I32MAX ->
  wpart owner f w S = D.part Z.add owner f w S.
Proof.
  intros owner f w S. induction S as [| S IH]; intros Hb.
  - reflexivity.
  - pose proof (part_abs_bounds owner f w (Datatypes.S S)) as Hab.
    cbn [wpart D.part] in Hab, Hb |- *.
    assert (Ha : afun f S = Z.abs (f S)) by reflexivity.
    rewrite Ha in Hab, Hb.
    remember (Nat.eqb (owner S) w) as bb eqn:E; destruct bb.
    + assert (Hs : D.part Z.add owner (afun f) w S <= MC.I32MAX).
      { pose proof (Z.abs_nonneg (f S)). lia. }
      rewrite (IH Hs).
      apply MC.wrap32_id.
      assert (HX : Z.abs (D.part Z.add owner f w S + f S) <= MC.I32MAX) by lia.
      apply Z.abs_le in HX.
      unfold MC.in_i32, MC.I32MIN, MC.I32MAX in *. lia.
    + exact (IH Hb).
Qed.

(** **Both folds, any owner map, any landing order, from a zeroed C.** *)
Theorem the_owned_split_is_exact_in_int32 :
  forall owner f n S order,
    (0 < n)%nat ->
    (forall s, (s < S)%nat -> (owner s < n)%nat) ->
    Permutation order (seq 0 n) ->
    sum_k (afun f) S <= MC.I32MAX ->
    wcombine 0 (map (fun w => wpart owner f w S) order) = sum_k f S.
Proof.
  intros owner f n S order Hn Hown Hperm Hb.
  assert (Hc : forall w, D.part Z.add owner (afun f) w S <= MC.I32MAX).
  { intros w. pose proof (part_abs_le_total owner f w S). lia. }
  assert (Hmap : map (fun w => wpart owner f w S) order
                 = map (fun w => D.part Z.add owner f w S) order).
  { apply map_ext. intros w. apply wpart_exact, Hc. }
  rewrite Hmap.
  assert (Hbud : fold_right Z.add 0 (map Z.abs (map (fun w => D.part Z.add owner f w S) order))
                 <= MC.I32MAX).
  { rewrite map_map.
    eapply Z.le_trans.
    - apply (fold_map_le order (fun w => Z.abs (D.part Z.add owner f w S))
                               (fun w => D.part Z.add owner (afun f) w S)).
      intros w. apply part_abs_bounds.
    - rewrite (GS.fold_right_add_permutation _ (map (fun w => D.part Z.add owner (afun f) w S) (seq 0 n)))
        by (apply Permutation_map; exact Hperm).
      rewrite <- SCH.combine_is_fold.
      rewrite (D.decomposition_exact owner (afun f) S n Hown). exact Hb. }
  rewrite wcombine_exact by (simpl Z.abs; lia).
  rewrite Z.add_0_l.
  rewrite (GS.fold_right_add_permutation _ (map (fun w => D.part Z.add owner f w S) (seq 0 n)))
    by (apply Permutation_map; exact Hperm).
  rewrite <- SCH.combine_is_fold.
  apply (D.decomposition_exact owner f S n Hown).
Qed.

(* ------------------------------------------------------------------ *)
(** ** The capstone                                                    *)
(* ------------------------------------------------------------------ *)

(** The element the epilogue writes for register [j] of lane [(g, t)] of mma
    [(mi, nj')] of warp [(wm, wn)] of tile [(tm, tn)] IS the schedule's
    [out_row] / [out_col], so every theorem in [Int8GemmSchedule] about which
    thread owns it applies to the element this file gives a value to. *)
Lemma the_written_row_is_the_schedules_row : forall wms ni tm wm mi g j,
  (tm * (wms * (ni * 16)) + wm * (ni * 16) + 16 * mi + d_row g j)%nat
  = SCH.out_row wms ni tm wm mi g (j / 2).
Proof. intros. unfold SCH.out_row, SCH.MMA_M, d_row. ring. Qed.

Lemma the_written_col_is_the_schedules_col : forall wns nj tn wn nj' t j,
  (tn * (wns * (nj * 8)) + wn * (nj * 8) + 8 * nj' + d_col t j)%nat
  = SCH.out_col wns nj tn wn nj' t (j mod 2).
Proof. intros. unfold SCH.out_col, SCH.MMA_N, d_col. ring. Qed.

(** The staged registers ARE [a_frag] / [b_frag] at the K step
    [s = T * kt + ks]: the stage holds K tile [kt] (byte offset
    [cta_k * kt = 32 T kt]) of the CTA's rows from [row0]. *)
Corollary the_staged_a_register_is_the_mma_fragment :
  forall R C T rows (A : nat -> nat -> Z) mem row0 kt wr0 i ks l q b,
    (0 < R)%nat -> (R * C = 8)%nat -> (C = 2 \/ C = 4 \/ C = 8)%nat -> C = (2 * T)%nat ->
    stage_holds R C rows (fun row k => A (row0 + row)%nat (32 * T * kt + k)%nat) mem ->
    (wr0 mod 8 = 0)%nat -> (wr0 + 16 * i + 16 <= rows)%nat -> (ks < T)%nat ->
    (l < 32)%nat -> (q < 4)%nat -> (b < 4)%nat ->
    ldmatrix_x4 mem (a_lane_addr R C wr0 i ks) l q b
    = a_frag A (row0 + wr0) i (32 * (T * kt + ks)) (a_row (l / 4) q) (a_col (l mod 4) q b).
Proof.
  intros. rewrite (the_ldmatrix_a_fragment_is_the_source_element R C rows
             (fun row k => A (row0 + row)%nat (32 * T * kt + k)%nat) mem wr0 i ks l q b);
    try assumption; try lia.
  unfold a_frag. f_equal; lia.
Qed.

Corollary the_staged_b_register_is_the_mma_fragment :
  forall R C T rows (B : nat -> nat -> Z) mem col0 kt wc0 jp ks l q b,
    (0 < R)%nat -> (R * C = 8)%nat -> (C = 2 \/ C = 4 \/ C = 8)%nat -> C = (2 * T)%nat ->
    stage_holds R C rows (fun row k => B (col0 + row)%nat (32 * T * kt + k)%nat) mem ->
    (wc0 mod 8 = 0)%nat -> (wc0 + 16 * jp + 16 <= rows)%nat -> (ks < T)%nat ->
    (l < 32)%nat -> (q < 4)%nat -> (b < 4)%nat ->
    ldmatrix_x4 mem (b_lane_addr R C wc0 jp ks) l q b
    = b_frag B (col0 + wc0) (2 * jp + q / 2) (32 * (T * kt + ks)) (b_k (l mod 4) (q mod 2) b) (b_col (l / 4)).
Proof.
  intros. rewrite (the_ldmatrix_b_fragment_is_the_source_element R C rows
             (fun row k => B (col0 + row)%nat (32 * T * kt + k)%nat) mem wc0 jp ks l q b);
    try assumption; try lia.
  unfold b_frag. f_equal; lia.
Qed.

Lemma acc_range_ext_below : forall f h lo len,
  (forall i, (i < len)%nat -> f (lo + i)%nat = h (lo + i)%nat) ->
  D.acc_range Z.add f lo len = D.acc_range Z.add h lo len.
Proof.
  intros f h lo len H. induction len as [| len IH]; cbn [D.acc_range]; [ reflexivity | ].
  rewrite IH by (intros; apply H; lia). rewrite H by lia. reflexivity.
Qed.

Lemma acc_range_zero : forall f lo len,
  (forall i, (i < len)%nat -> f (lo + i)%nat = 0) -> D.acc_range Z.add f lo len = 0.
Proof.
  intros f lo len H. induction len as [| len IH]; cbn [D.acc_range]; [ reflexivity | ].
  rewrite IH by (intros; apply H; lia). rewrite H by lia. reflexivity.
Qed.

(** The licence bounds the sum of |products|. *)
Lemma the_licence_bounds_the_absolute_sum : forall A B m K row col,
  0 <= m -> (forall r k, Z.abs (A r k) <= m) -> (forall c k, Z.abs (B c k) <= m) ->
  Z.of_nat K * (m * m) <= MC.I32MAX ->
  sum_k (afun (prod A B row col)) K <= MC.I32MAX.
Proof.
  intros A B m K row col Hm HA HB Hlic.
  assert (Hself : forall z : Z, z <= Z.abs z)
    by (intros z; destruct (Z.abs_spec z) as [[? E] | [? E]]; lia).
  eapply Z.le_trans; [ apply Hself | ]. eapply Z.le_trans; [ | exact Hlic ].
  apply acc_range_abs_bound; [ nia | ].
  intros i. unfold afun, prod. rewrite Z.abs_idemp, Z.abs_mul.
  apply Z.mul_le_mono_nonneg; auto using Z.abs_nonneg.
Qed.

(** **The theorem this file exists for.**

    For every launch geometry - any split factor [nz], any order the CTAs'
    partials land in - and every lane of every mma of every warp of every
    tile, the emitted kernel holds exactly

      sum over k < K of A[row][k] * B[col][k]

    at the [(row, col)] the schedule proof says that lane owns.  Three
    conjuncts, the three things the kernel does:

    - in [Z], the K tiles' mma steps - the zero-filled tail of the last K tile
      included ([A'], [B'] are A and B with the zeros the copies write past
      K) - summed over the split's classes;
    - in int32, each CTA's own register fold and the atomic combine, which the
      LICENCE keeps from wrapping;
    - the unsplit launch's STORE, which is its one class.

    The fragments those steps multiply are what the staged path delivers:
    [the_staged_a_register_is_the_mma_fragment] / [_b_]. *)
Theorem the_emitted_int8_gemm_holds_the_source_dot_products :
  forall (A B : nat -> nat -> Z) m K S T KT nz order wms wns ni nj tm tn wm wn mi nj' g t j,
    0 <= m ->
    (forall r k, Z.abs (A r k) <= m) ->
    (forall c k, Z.abs (B c k) <= m) ->
    Z.of_nat K * (m * m) <= MC.I32MAX ->
    (K = 32 * S)%nat -> (0 < T)%nat -> (S <= KT * T)%nat -> (0 < nz)%nat ->
    Permutation order (seq 0 nz) ->
    let brow := (tm * (wms * (ni * 16)) + wm * (ni * 16))%nat in
    let bcol := (tn * (wns * (nj * 8)) + wn * (nj * 8))%nat in
    let row := SCH.out_row wms ni tm wm mi g (j / 2) in
    let col := SCH.out_col wns nj tn wn nj' t (j mod 2) in
    let A' := fun r k => if (k <? K)%nat then A r k else 0 in
    let B' := fun c k => if (k <? K)%nat then B c k else 0 in
    D.combine Z.add (SCH.k_tile_owner T nz) (step A' B' brow bcol mi nj' (d_row g j) (d_col t j)) (KT * T) nz
      = sum_k (prod A B row col) K
    /\ wcombine 0 (map (fun w => wpart (SCH.k_tile_owner (32 * T) nz) (prod A B row col) w K) order)
      = sum_k (prod A B row col) K
    /\ wpart (SCH.k_tile_owner (32 * T) 1) (prod A B row col) 0 K = sum_k (prod A B row col) K.
Proof.
  intros A B m K S T KT nz order wms wns ni nj tm tn wm wn mi nj' g t j
         Hm HA HB Hlic HK HT HKT Hnz Hperm brow bcol row col A' B'.
  pose proof (the_licence_bounds_the_absolute_sum A B m K row col Hm HA HB Hlic) as Hbud.
  split; [ | split ].
  - rewrite SCH.the_split_k_sums_to_the_whole_contraction by exact Hnz.
    replace (KT * T)%nat with (S + (KT * T - S))%nat by lia.
    rewrite SCH.the_zero_filled_tail_adds_nothing.
    + fold (sum_k (step A' B' brow bcol mi nj' (d_row g j) (d_col t j)) S).
      rewrite the_k_loop_is_the_contraction.
      assert (Er : (brow + 16 * mi + d_row g j)%nat = row)
        by (unfold brow, row; apply the_written_row_is_the_schedules_row).
      assert (Ec : (bcol + 8 * nj' + d_col t j)%nat = col)
        by (unfold bcol, col; apply the_written_col_is_the_schedules_col).
      rewrite Er, Ec, <- HK. unfold sum_k.
      apply acc_range_ext_below. intros i Hi. unfold prod, A', B'.
      replace ((0 + i <? K)%nat) with true by (symmetry; apply Nat.ltb_lt; lia).
      reflexivity.
    + intros s Hs. unfold step, mma, sum_k.
      apply acc_range_zero. intros p Hp. unfold a_frag, A'.
      replace ((32 * s + (0 + p) <? K)%nat) with false by (symmetry; apply Nat.ltb_ge; lia).
      ring.
  - apply (the_owned_split_is_exact_in_int32 _ _ nz); try assumption.
    intros s _. apply SCH.every_k_step_has_exactly_one_owner. exact Hnz.
  - rewrite wpart_exact.
    + destruct (SCH.the_unsplit_launch_stores_the_whole_contraction (32 * T) (prod A B row col) K) as [H _].
      exact H.
    + pose proof (part_abs_le_total (SCH.k_tile_owner (32 * T) 1) (prod A B row col) 0 K). lia.
Qed.

(* ------------------------------------------------------------------ *)
(** ** No axioms                                                       *)
(* ------------------------------------------------------------------ *)

Print Assumptions acc_range_ext.
Print Assumptions acc_range_shift.
Print Assumptions acc_range_blocks.
Print Assumptions sum_k_blocks.
Print Assumptions a_in_range.
Print Assumptions b_in_range.
Print Assumptions a_frag_injective.
Print Assumptions b_frag_injective.
Print Assumptions the_lanes_cover_the_a_fragment.
Print Assumptions the_lanes_cover_the_b_fragment.
Print Assumptions the_k_loop_is_the_contraction.
Print Assumptions every_split_factor_gives_the_contraction.
Print Assumptions the_bound_is_one_k_step_wide.
Print Assumptions the_licence_admits_131040_and_refuses_131072.
Print Assumptions the_measured_overflow_is_two_s_complement.
Print Assumptions the_signed_minimum_exposes_the_old_licence.
Print Assumptions acc_range_abs_bound.
Print Assumptions bounded_products_accumulate_exactly.
Print Assumptions the_split_k_accumulation_is_exact_in_int32.
Print Assumptions the_split_k_combine_is_exact_in_int32.
Print Assumptions each_launch_is_licensed_and_three_of_them_wrap.
Print Assumptions the_third_launch_wraps_to_the_measured_value.
Print Assumptions the_combine_needs_a_zeroed_destination.
Print Assumptions from_zero_the_same_partial_is_exact.
Print Assumptions lane_bits.
Print Assumptions xor_shift.
Print Assumptions xor_bound.
Print Assumptions xor_cancel.
Print Assumptions swz_repeats_every_8_rows.
Print Assumptions the_stage_slots_are_disjoint.
Print Assumptions the_copy_lands_at_its_swizzled_place.
Print Assumptions the_ldmatrix_a_fragment_is_the_source_element.
Print Assumptions the_ldmatrix_b_fragment_is_the_source_element.
Print Assumptions wclass_is_wpart.
Print Assumptions wpart_exact.
Print Assumptions the_owned_split_is_exact_in_int32.
Print Assumptions the_written_row_is_the_schedules_row.
Print Assumptions the_written_col_is_the_schedules_col.
Print Assumptions the_staged_a_register_is_the_mma_fragment.
Print Assumptions the_staged_b_register_is_the_mma_fragment.
Print Assumptions the_licence_bounds_the_absolute_sum.
Print Assumptions the_emitted_int8_gemm_holds_the_source_dot_products.
