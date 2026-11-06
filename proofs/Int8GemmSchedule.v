(** * The int8 tensor-core GEMM's schedule, and the launch contract it states.

    This is the only GPU GEMM in this repository whose EXACTNESS is available
    to prove: [mma.sync...s32.s8.s8.s32] accumulates into an int32 EXACTLY,
    where every floating-point [mma] rounds.  [Int8GemmExact.v] proves the
    VALUE each output receives; this file proves the SCHEDULE - which thread
    owns which element of C, that every output tile is computed exactly once
    on any grid, that the K split partitions the contraction, and which
    epilogue may store.

    ** The kernel, since 2026-11-06

    [src/ptx_emitter/int8_staged_gemm.rs].  A CTA of [warps_m x warps_n]
    warps computes a [cta_m x cta_n] tile, each warp a [num_i x num_j] grid
    of 16x8 [mma] tiles; K is staged through shared memory in tiles of
    [cta_k = 32 T] bytes.  Measured on an RTX 4070 Ti SUPER against cuBLASLt's
    [torch._int_mm]: 1.02-1.10x at 2048^3-4096^3, 0.97-1.00x at 8192^3
    (docs/benchmarks/int8_gemm_2026-11-06/).
    The schedule:

      - THE BLOCK: exactly [32 warps_m warps_n] threads, else [trap].  Every
        thread has a copy role and a warp tile, so neither a smaller block
        (rows of a stage uncopied) nor a larger one (a warp outside the tile)
        can be made correct by idling threads.
        [the_trap_is_what_confines_a_warp_to_its_own_tile] and
        [without_the_trap_a_fifth_warp_lands_in_the_next_tile];
        [the_copies_cover_the_stage_once] and
        [without_the_trap_a_short_block_leaves_rows_uncopied].
      - THE GRID: any.  A CTA's linear id [ctaid.y * nctaid.x + ctaid.x]
        strides over the tile ids, and a tile id names its tile in grouped
        raster order - a bijection, [grouped_is_a_bijection].
        [every_output_tile_is_computed_exactly_once].
      - THE K SPLIT: CTA [z] of [nctaid.z] takes K tiles [z], [z + nctaid.z],
        ...; a K tile is [T] steps of 32, so step [s] belongs to CTA
        [(s / T) mod nz] - [Decomposition]'s owner map, no new partition
        reasoning.  [the_split_k_sums_to_the_whole_contraction].
      - THE EPILOGUE: a launch with [nctaid.z = 1] STORES (its one class is the
        whole contraction); one that splits K REDUCES into a zeroed C; the
        scaled form ignores [z] and stores.
        [the_unsplit_launch_stores_the_whole_contraction],
        [a_split_cta_holds_only_part_of_the_contraction].

    ** What it replaced, and the defect the first version of this file was for

    Until 2026-11-06 the kernel gave each WARP one output tile and read A and
    B straight from global memory - 0.40x cuBLASLt at 4096^3, register-tiled
    from 0.09x.  It had one predicate, the K-loop bound, and never mentioned
    [%ntid.x]: at M=64 N=32 K=128 with A = 3 and B = 5 (every output must be
    1920), a 64-thread block returned 1344 of 2048 elements wrong,
    [C[256] = 3840] - EXACTLY DOUBLE, because warp 1's [g = tid/4] fell in
    8..15 and its second A-row read landed in the NEXT tile's rows, which
    [red.global.add.s32] summed in.  It was fixed with a warp-uniform guard
    ([tid < 32]) and proved here.  The staged kernel cannot idle a warp - its
    copies need every thread - so it traps instead, and the theorems below
    are the trap's.  The old kernel's measurements and its guard are in
    [tests/int8_gemm_launch_contract.rs] and the README.

    The int8 [mma] ceiling on this card is 179,761 G MAC/s (102% of
    [66 SM x 2.61 GHz x 1024 MAC/SM/cycle]); at 4096^3 cuBLASLt reaches ~86% of
    it and the staged kernel ~88%.  An
    earlier ceiling of 374,027 stored one accumulator set of eight, so ptxas
    deleted seven chains - a constant dead fraction that a doubling control
    divides out.

    ** What this does NOT claim

    - Nothing HERE about the VALUE at [C[r][c]]: [Int8GemmExact.v] proves it,
      and reuses [out_row] / [out_col] from here so the two describe the same
      element.
    - Nothing about [mma]'s or [ldmatrix]'s own semantics: ISA facts, pinned by
      [tests/ptx_int8_mma_layout.rs] and the device tests.
    - Nothing about the ORDER in which CTAs run.  The schedule is correct for
      every order: stores write disjoint tiles, and the reductions commute.
    - The tie is transcription-plus-gate: [ptx_emitter] does not go through
      the [Ix] extraction layer, so this is a model checked against emitted
      text by [tests/int8_gemm_launch_contract.rs].

    Build: coqc -Q . "" Int8GemmSchedule.v   (Rocq 9.1)
*)

Require Import Coq.Arith.Arith.
Require Import Coq.micromega.Lia.
Require Import Coq.ZArith.ZArith.
Require Import Coq.Lists.List.
Require Import Coq.Sorting.Permutation.
Import ListNotations.

Require MixedRadix.
Require Decomposition.
Require GridStrideSplit.

Module MR := MixedRadix.
Module D := Decomposition.
Module GS := GridStrideSplit.

Open Scope nat_scope.

(* ------------------------------------------------------------------ *)
(** ** The mma, and a warp's lanes                                     *)
(* ------------------------------------------------------------------ *)

(** [m16n8k32]: a warp's [mma] computes a 16x8 tile of C from 32 k-values,
    so the emitter refuses unless [m mod 16 = 0 /\ n mod 8 = 0 /\ k mod 32 = 0]. *)
Definition MMA_M : nat := 16.
Definition MMA_N : nat := 8.
Definition MMA_K : nat := 32.

(** [shr.u32 %g, %lane, 2] and [and.b32 %t, %lane, 3], validated against the
    ISA in [tests/ptx_int8_mma_layout.rs]. *)
Definition lane_g (lane : nat) : nat := lane / 4.
Definition lane_t (lane : nat) : nat := lane mod 4.

Theorem lane_decomposition_is_injective :
  forall l1 l2, lane_g l1 = lane_g l2 -> lane_t l1 = lane_t l2 -> l1 = l2.
Proof.
  intros l1 l2 Hg Ht. unfold lane_g, lane_t in *.
  pose proof (Nat.div_mod_eq l1 4). pose proof (Nat.div_mod_eq l2 4). lia.
Qed.

Theorem a_warps_lanes_fill_the_tile :
  forall lane, lane < 32 -> lane_g lane < 8 /\ lane_t lane < 4.
Proof.
  intros lane H. unfold lane_g, lane_t. split.
  - apply Nat.Div0.div_lt_upper_bound. lia.
  - apply Nat.mod_upper_bound. lia.
Qed.

(* ------------------------------------------------------------------ *)
(** ** A CTA's threads                                                 *)
(* ------------------------------------------------------------------ *)

(** [shr.u32 %w, %tid, 5], [and.b32 %lane, %tid, 31], then
    [rem.u32 %wm, %w, warps_m] and [div.u32 %wn, %w, warps_m]. *)
Definition warp_id (tid : nat) : nat := tid / 32.
Definition lane_id (tid : nat) : nat := tid mod 32.
Definition warp_m (warps_m w : nat) : nat := w mod warps_m.
Definition warp_n (warps_m w : nat) : nat := w / warps_m.

(** Accumulator register [2h + e] of [mma] [(i, j)] in lane [(g, t)] of warp
    [(wm, wn)] of tile [(tm, tn)] holds C at this row and column.  The
    emitter's epilogue: [row0 = tile_m*cta_m + warp_m*per_warp_m + g], plus
    [i*16 + 8h]; [col0 = tile_n*cta_n + warp_n*per_warp_n + 2t], plus
    [j*8 + e] - with [per_warp_m = num_i*16], [cta_m = warps_m*per_warp_m],
    and likewise in N. *)
Definition out_row (warps_m num_i tm wm i g h : nat) : nat :=
  tm * (warps_m * (num_i * MMA_M)) + wm * (num_i * MMA_M) + i * MMA_M + 8 * h + g.
Definition out_col (warps_n num_j tn wn j t e : nat) : nat :=
  tn * (warps_n * (num_j * MMA_N)) + wn * (num_j * MMA_N) + j * MMA_N + 2 * t + e.

(** An output row is a five-digit index - tile, warp, mma, half, lane group -
    and [MixedRadix.quot_rem_unique] peels it one digit at a time. *)
Theorem out_row_injective :
  forall wms ni tm1 wm1 i1 g1 h1 tm2 wm2 i2 g2 h2,
    0 < wms -> 0 < ni ->
    wm1 < wms -> wm2 < wms -> i1 < ni -> i2 < ni ->
    g1 < 8 -> g2 < 8 -> h1 < 2 -> h2 < 2 ->
    out_row wms ni tm1 wm1 i1 g1 h1 = out_row wms ni tm2 wm2 i2 g2 h2 ->
    tm1 = tm2 /\ wm1 = wm2 /\ i1 = i2 /\ g1 = g2 /\ h1 = h2.
Proof.
  intros wms ni tm1 wm1 i1 g1 h1 tm2 wm2 i2 g2 h2 Hw Hn Hwm1 Hwm2 Hi1 Hi2 Hg1 Hg2 Hh1 Hh2 H.
  unfold out_row, MMA_M in H.
  assert (E : ((tm1 * wms + wm1) * ni + i1) * 16 + (h1 * 8 + g1)
              = ((tm2 * wms + wm2) * ni + i2) * 16 + (h2 * 8 + g2)) by nia.
  destruct (MR.quot_rem_unique 16 ((tm1 * wms + wm1) * ni + i1) (h1 * 8 + g1)
              ((tm2 * wms + wm2) * ni + i2) (h2 * 8 + g2) ltac:(lia) ltac:(lia) ltac:(lia) E) as [E1 E2].
  destruct (MR.quot_rem_unique 8 _ _ _ _ ltac:(lia) Hg1 Hg2 E2) as [Eh Eg].
  destruct (MR.quot_rem_unique ni _ _ _ _ Hn Hi1 Hi2 E1) as [E3 Ei].
  destruct (MR.quot_rem_unique wms _ _ _ _ Hw Hwm1 Hwm2 E3) as [Et Ew].
  repeat split; assumption.
Qed.

Theorem out_col_injective :
  forall wns nj tn1 wn1 j1 t1 e1 tn2 wn2 j2 t2 e2,
    0 < wns -> 0 < nj ->
    wn1 < wns -> wn2 < wns -> j1 < nj -> j2 < nj ->
    t1 < 4 -> t2 < 4 -> e1 < 2 -> e2 < 2 ->
    out_col wns nj tn1 wn1 j1 t1 e1 = out_col wns nj tn2 wn2 j2 t2 e2 ->
    tn1 = tn2 /\ wn1 = wn2 /\ j1 = j2 /\ t1 = t2 /\ e1 = e2.
Proof.
  intros wns nj tn1 wn1 j1 t1 e1 tn2 wn2 j2 t2 e2 Hw Hn Hwn1 Hwn2 Hj1 Hj2 Ht1 Ht2 He1 He2 H.
  unfold out_col, MMA_N in H.
  assert (E : ((tn1 * wns + wn1) * nj + j1) * 8 + (t1 * 2 + e1)
              = ((tn2 * wns + wn2) * nj + j2) * 8 + (t2 * 2 + e2)) by nia.
  destruct (MR.quot_rem_unique 8 ((tn1 * wns + wn1) * nj + j1) (t1 * 2 + e1)
              ((tn2 * wns + wn2) * nj + j2) (t2 * 2 + e2) ltac:(lia) ltac:(lia) ltac:(lia) E) as [E1 E2].
  destruct (MR.quot_rem_unique 2 _ _ _ _ ltac:(lia) He1 He2 E2) as [Et Ee].
  destruct (MR.quot_rem_unique nj _ _ _ _ Hn Hj1 Hj2 E1) as [E3 Ej].
  destruct (MR.quot_rem_unique wns _ _ _ _ Hw Hwn1 Hwn2 E3) as [Etn Ewn].
  repeat split; assumption.
Qed.

Theorem out_row_onto :
  forall wms ni r, 0 < wms -> 0 < ni ->
    exists tm wm i g h, wm < wms /\ i < ni /\ g < 8 /\ h < 2
                        /\ out_row wms ni tm wm i g h = r.
Proof.
  intros wms ni r Hw Hn.
  exists (r / 16 / ni / wms), (r / 16 / ni mod wms), (r / 16 mod ni), (r mod 16 mod 8), (r mod 16 / 8).
  repeat split.
  - apply Nat.mod_upper_bound. lia.
  - apply Nat.mod_upper_bound. lia.
  - apply Nat.mod_upper_bound. lia.
  - apply Nat.Div0.div_lt_upper_bound. pose proof (Nat.mod_upper_bound r 16 ltac:(lia)). lia.
  - unfold out_row, MMA_M.
    pose proof (Nat.div_mod_eq r 16) as A1.
    pose proof (Nat.div_mod_eq (r / 16) ni) as A2.
    pose proof (Nat.div_mod_eq (r / 16 / ni) wms) as A3.
    pose proof (Nat.div_mod_eq (r mod 16) 8) as A4.
    nia.
Qed.

Theorem out_col_onto :
  forall wns nj c, 0 < wns -> 0 < nj ->
    exists tn wn j t e, wn < wns /\ j < nj /\ t < 4 /\ e < 2
                        /\ out_col wns nj tn wn j t e = c.
Proof.
  intros wns nj c Hw Hn.
  exists (c / 8 / nj / wns), (c / 8 / nj mod wns), (c / 8 mod nj), (c mod 8 / 2), (c mod 8 mod 2).
  repeat split.
  - apply Nat.mod_upper_bound. lia.
  - apply Nat.mod_upper_bound. lia.
  - apply Nat.Div0.div_lt_upper_bound. pose proof (Nat.mod_upper_bound c 8 ltac:(lia)). lia.
  - apply Nat.mod_upper_bound. lia.
  - unfold out_col, MMA_N.
    pose proof (Nat.div_mod_eq c 8) as A1.
    pose proof (Nat.div_mod_eq (c / 8) nj) as A2.
    pose proof (Nat.div_mod_eq (c / 8 / nj) wns) as A3.
    pose proof (Nat.div_mod_eq (c mod 8) 2) as A4.
    nia.
Qed.

(** **Every element of C has exactly one owner**: some (tile, warp, mma,
    lane, register) writes it, and no two do.  The GPU twin of the CPU
    chain's [c_written_exactly_once]. *)
Theorem c_element_has_exactly_one_owner :
  forall wms wns ni nj r c, 0 < wms -> 0 < wns -> 0 < ni -> 0 < nj ->
    (exists tm wm i g h tn wn j t e,
       wm < wms /\ i < ni /\ g < 8 /\ h < 2 /\ wn < wns /\ j < nj /\ t < 4 /\ e < 2
       /\ out_row wms ni tm wm i g h = r /\ out_col wns nj tn wn j t e = c)
    /\ (forall tm1 wm1 i1 g1 h1 tm2 wm2 i2 g2 h2,
          wm1 < wms -> wm2 < wms -> i1 < ni -> i2 < ni ->
          g1 < 8 -> g2 < 8 -> h1 < 2 -> h2 < 2 ->
          out_row wms ni tm1 wm1 i1 g1 h1 = r -> out_row wms ni tm2 wm2 i2 g2 h2 = r ->
          tm1 = tm2 /\ wm1 = wm2 /\ i1 = i2 /\ g1 = g2 /\ h1 = h2)
    /\ (forall tn1 wn1 j1 t1 e1 tn2 wn2 j2 t2 e2,
          wn1 < wns -> wn2 < wns -> j1 < nj -> j2 < nj ->
          t1 < 4 -> t2 < 4 -> e1 < 2 -> e2 < 2 ->
          out_col wns nj tn1 wn1 j1 t1 e1 = c -> out_col wns nj tn2 wn2 j2 t2 e2 = c ->
          tn1 = tn2 /\ wn1 = wn2 /\ j1 = j2 /\ t1 = t2 /\ e1 = e2).
Proof.
  intros wms wns ni nj r c Hwm Hwn Hni Hnj. split; [ | split ].
  - destruct (out_row_onto wms ni r Hwm Hni) as [tm [wm [i [g [h [? [? [? [? Hr]]]]]]]]].
    destruct (out_col_onto wns nj c Hwn Hnj) as [tn [wn [j [t [e [? [? [? [? Hc]]]]]]]]].
    exists tm, wm, i, g, h, tn, wn, j, t, e. repeat split; assumption.
  - intros. apply (out_row_injective wms ni); try assumption. congruence.
  - intros. apply (out_col_injective wns nj); try assumption. congruence.
Qed.

(** A thread is its warp and its lane, and a warp its place in the tile:
    [tid = (wn * warps_m + wm) * 32 + 4g + t] is a bijection, so no two threads
    hold one (warp, lane) and every (warp, lane) of the tile is some thread. *)
Theorem a_thread_is_its_warp_and_lane :
  forall wms tid, 0 < wms ->
    tid = (warp_n wms (warp_id tid) * wms + warp_m wms (warp_id tid)) * 32
          + 4 * lane_g (lane_id tid) + lane_t (lane_id tid).
Proof.
  intros wms tid Hw. unfold warp_n, warp_m, warp_id, lane_id, lane_g, lane_t.
  pose proof (Nat.div_mod_eq tid 32) as A1.
  pose proof (Nat.div_mod_eq (tid / 32) wms) as A2.
  pose proof (Nat.div_mod_eq (tid mod 32) 4) as A3.
  nia.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The block-size trap                                             *)
(* ------------------------------------------------------------------ *)

(** The kernel opens with

      mov.u32 %r, %ntid.x;  setp.ne.u32 %p, %r, <32 warps_m warps_n>;  @%p trap;

    and the same for [%ntid.y] and [%ntid.z] against 1.  What that buys: a
    thread's warp column [wn = (tid/32) / warps_m] is below [warps_n], so the
    columns it writes lie inside its own tile.  The ROW digit needs no trap -
    it is a remainder, below [warps_m] for any [tid] - which is why the
    statement is about columns, and why its upper bound is the whole content. *)
Theorem the_trap_is_what_confines_a_warp_to_its_own_tile :
  forall wms wns nj tn tid j t e,
    0 < wms -> tid < 32 * (wms * wns) -> j < nj -> t < 4 -> e < 2 ->
    tn * (wns * (nj * MMA_N)) <= out_col wns nj tn (warp_n wms (warp_id tid)) j t e
    /\ out_col wns nj tn (warp_n wms (warp_id tid)) j t e
       < tn * (wns * (nj * MMA_N)) + wns * (nj * MMA_N).
Proof.
  intros wms wns nj tn tid j t e Hw Htid Hj Ht He.
  assert (Hwn : warp_n wms (warp_id tid) < wns).
  { unfold warp_n, warp_id. apply Nat.Div0.div_lt_upper_bound.
    apply Nat.Div0.div_lt_upper_bound. lia. }
  unfold out_col, MMA_N in *. split; [ lia | ].
  remember (warp_n wms (warp_id tid)) as wn.
  assert (wn * (nj * 8) + j * 8 + 2 * t + e < wns * (nj * 8)) by nia.
  lia.
Qed.

(** **The refutation, at the tile the rule picks for small shapes** (64x64 on
    2x2 warps of 32x32, so [num_j = 4]).  A 160-thread block's fifth warp,
    thread 128, has warp column 2 - and its first column is 64, the first
    column of the NEXT tile, which another CTA owns. *)
Theorem without_the_trap_a_fifth_warp_lands_in_the_next_tile :
  warp_n 2 (warp_id 128) = 2
  /\ out_col 2 4 0 (warp_n 2 (warp_id 128)) 0 0 0 = 64
  /\ ~ (out_col 2 4 0 (warp_n 2 (warp_id 128)) 0 0 0 < 0 * (2 * (4 * MMA_N)) + 2 * (4 * MMA_N)).
Proof. unfold warp_n, warp_id, out_col, MMA_N. cbn. repeat split; lia. Qed.

(** The copies.  Thread [tid] copies chunk [tid mod cpr] of rows
    [tid / cpr + u * rpp], [u < chunks], where [rpp = threads / cpr] and
    [chunks = rows / rpp] - so the trap's [threads] is in the copy role too. *)
Definition copy_row (cpr rpp tid u : nat) : nat := tid / cpr + u * rpp.
Definition copy_chunk (cpr tid : nat) : nat := tid mod cpr.

(** With the block the kernel was built for, the copies cover a stage's
    [chunks * rpp] rows of [cpr] chunks exactly once. *)
Theorem the_copies_cover_the_stage_once :
  forall cpr rpp chunks, 0 < cpr -> 0 < rpp ->
    (forall row c, row < chunks * rpp -> c < cpr ->
       exists tid u, tid < rpp * cpr /\ u < chunks
                     /\ copy_row cpr rpp tid u = row /\ copy_chunk cpr tid = c)
    /\ (forall tid1 u1 tid2 u2, tid1 < rpp * cpr -> tid2 < rpp * cpr ->
          copy_row cpr rpp tid1 u1 = copy_row cpr rpp tid2 u2 ->
          copy_chunk cpr tid1 = copy_chunk cpr tid2 ->
          tid1 = tid2 /\ u1 = u2).
Proof.
  intros cpr rpp chunks Hc Hr. split.
  - intros row c Hrow Hcc.
    exists ((row mod rpp) * cpr + c), (row / rpp). unfold copy_row, copy_chunk.
    pose proof (Nat.mod_upper_bound row rpp ltac:(lia)) as Hm.
    assert (Ed : ((row mod rpp) * cpr + c) / cpr = row mod rpp).
    { rewrite Nat.div_add_l by lia. rewrite Nat.div_small by lia. lia. }
    assert (Em : ((row mod rpp) * cpr + c) mod cpr = c).
    { rewrite Nat.Div0.add_mod, Nat.Div0.mod_mul, Nat.add_0_l, Nat.Div0.mod_mod.
      apply Nat.mod_small. exact Hcc. }
    rewrite Ed, Em. pose proof (Nat.div_mod_eq row rpp).
    repeat split; nia.
  - intros tid1 u1 tid2 u2 H1 H2 Hrow Hch. unfold copy_row, copy_chunk in *.
    assert (Hd1 : tid1 / cpr < rpp) by (apply Nat.Div0.div_lt_upper_bound; lia).
    assert (Hd2 : tid2 / cpr < rpp) by (apply Nat.Div0.div_lt_upper_bound; lia).
    assert (E : u1 * rpp + tid1 / cpr = u2 * rpp + tid2 / cpr) by lia.
    destruct (MR.quot_rem_unique rpp _ _ _ _ Hr Hd1 Hd2 E) as [Eu Ed].
    pose proof (Nat.div_mod_eq tid1 cpr). pose proof (Nat.div_mod_eq tid2 cpr).
    split; [ nia | exact Eu ].
Qed.

(** **The refutation: a short block leaves rows uncopied.**  The 64x64 tile
    copies 4 chunks a row with 128 threads (32 rows a pass); with 64 threads
    no thread ever copies row 16, so the [mma] would read a stage row that
    still holds the previous tile's bytes. *)
Theorem without_the_trap_a_short_block_leaves_rows_uncopied :
  ~ (exists tid u, tid < 64 /\ u < 2 /\ copy_row 4 32 tid u = 16).
Proof.
  intros [tid [u [Ht [Hu H]]]]. unfold copy_row in H.
  assert (tid / 4 < 16) by (apply Nat.Div0.div_lt_upper_bound; lia).
  destruct u as [| [| u]]; lia.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The grid: tile ids, grid-strided, in grouped raster order       *)
(* ------------------------------------------------------------------ *)

(** A CTA's linear id, [mad.lo.u32 %pid, %ctaid.y, %nctaid.x, %ctaid.x], is a
    bijection from the grid's x*y plane onto [0, nctaid.x * nctaid.y). *)
Definition cta_pid (ncx cx cy : nat) : nat := cy * ncx + cx.

Theorem cta_pid_is_unique :
  forall ncx cx1 cy1 cx2 cy2, cx1 < ncx -> cx2 < ncx ->
    cta_pid ncx cx1 cy1 = cta_pid ncx cx2 cy2 -> cx1 = cx2 /\ cy1 = cy2.
Proof.
  intros ncx cx1 cy1 cx2 cy2 H1 H2 H. unfold cta_pid in H.
  destruct (MR.quot_rem_unique ncx _ _ _ _ ltac:(lia) H1 H2 H). auto.
Qed.

(** The kernel traps when [nctaid.x * nctaid.y > 2^32 - 1]: the CTA count
    and the tile ids are 32-bit, and the count would wrap.  The smallest x*y
    grid past it, (65538, 65535), wraps to a stride of 65534 - every CTA would
    then step over tiles another CTA also visits.  Stated in [Z]: these
    numbers in unary [nat] are four billion constructors, and [vm_compute] on
    the first draft of this theorem ran the machine out of memory. *)
Theorem without_the_grid_trap_the_stride_wraps :
  (4294967295 < 65538 * 65535)%Z /\ ((65538 * 65535) mod 4294967296 = 65534)%Z.
Proof. split; [ lia | vm_compute; reflexivity ]. Qed.

(** The tile loop's own wrap: [tile + n_ctas] past [W = 2^32] ends the loop
    ([setp.lt.u32] of the sum against the id) rather than revisiting a small
    id - which a reduction would add twice.  The test is exact, for any
    modulus [W] (so no literal enters the proof). *)
Theorem the_tile_increment_detects_its_wrap :
  forall W t p, t < W -> p < W -> ((t + p) mod W < t <-> W <= t + p).
Proof.
  intros W t p Ht Hp.
  destruct (Nat.lt_ge_cases (t + p) W) as [L | L].
  - rewrite Nat.mod_small by exact L. lia.
  - assert (E : (t + p) mod W = t + p - W).
    { replace (t + p) with ((t + p - W) + 1 * W) at 1 by lia.
      rewrite Nat.Div0.mod_add. apply Nat.mod_small. lia. }
    rewrite E. lia.
Qed.

(** Grid-strided tile ids: CTA [pid] of [P] takes ids [pid], [pid + P], ...,
    the residue class [GridStrideSplit] partitions. *)
Definition tile_owner (P t : nat) : nat := t mod P.

Theorem every_output_tile_has_exactly_one_owner :
  forall P t, 0 < P -> tile_owner P t < P /\ forall i, i * P + tile_owner P t = t -> i = t / P.
Proof.
  intros P t HP. unfold tile_owner. split.
  - apply Nat.mod_upper_bound. lia.
  - intros i Hi. pose proof (Nat.div_mod_eq t P).
    destruct (MR.quot_rem_unique P i (t mod P) (t / P) (t mod P) HP
                (Nat.mod_upper_bound t P ltac:(lia)) (Nat.mod_upper_bound t P ltac:(lia))
                ltac:(lia)). assumption.
Qed.

(** The grouped raster order (the emitter's [emit_int8_grouped_tile]):
    consecutive ids cover [G] M tiles of one N tile, then the next N tile;
    the last group holds the [TM - first] M tiles left. *)
Definition g_size (TM G grp : nat) : nat := Nat.min (TM - grp * G) G.
Definition grouped (TM TN G t : nat) : nat * nat :=
  let grp := t / (G * TN) in
  let gs := g_size TM G grp in
  let r := t mod (G * TN) in
  (grp * G + r mod gs, r / gs).

(** Its inverse. *)
Definition ungrouped (TM TN G tm tn : nat) : nat :=
  let grp := tm / G in
  grp * (G * TN) + tn * g_size TM G grp + (tm - grp * G).

Lemma ungrouped_spec :
  forall TM TN G tm tn, 0 < G -> tm < TM -> tn < TN ->
    let grp := tm / G in
    let gs := g_size TM G grp in
    tm - grp * G < gs /\ tn * gs + (tm - grp * G) < G * TN
    /\ ungrouped TM TN G tm tn < TM * TN.
Proof.
  intros TM TN G tm tn HG Htm Htn grp gs.
  pose proof (Nat.div_mod_eq tm G) as Edm.
  pose proof (Nat.mod_upper_bound tm G ltac:(lia)) as Hmod.
  assert (Hlt : tm - grp * G < gs).
  { subst gs grp. unfold g_size. apply Nat.min_glb_lt; lia. }
  assert (HgsG : gs <= G) by (subst gs; unfold g_size; lia).
  assert (Hgs : gs <= TM - grp * G) by (subst gs; unfold g_size; lia).
  split; [ exact Hlt | split ].
  - nia.
  - unfold ungrouped. fold grp. fold gs.
    assert (grp * G <= tm) by lia.
    assert (tn * gs + (tm - grp * G) < TN * gs) by nia.
    assert (grp * (G * TN) + TN * gs <= TM * TN) by nia.
    lia.
Qed.

(** **[grouped] is a bijection** from [0, TM*TN) onto [0,TM) x [0,TN): every
    tile id names a tile in range, and [ungrouped] inverts it both ways. *)
Theorem grouped_is_a_bijection :
  forall TM TN G, 0 < G -> 0 < TN ->
    (forall t, t < TM * TN ->
       fst (grouped TM TN G t) < TM /\ snd (grouped TM TN G t) < TN
       /\ ungrouped TM TN G (fst (grouped TM TN G t)) (snd (grouped TM TN G t)) = t)
    /\ (forall tm tn, tm < TM -> tn < TN ->
          ungrouped TM TN G tm tn < TM * TN /\ grouped TM TN G (ungrouped TM TN G tm tn) = (tm, tn)).
Proof.
  intros TM TN G HG HTN. split.
  - intros t Ht. unfold grouped.
    set (P := G * TN). set (grp := t / P). set (gs := g_size TM G grp). set (r := t mod P).
    assert (HP : 0 < P) by (subst P; nia).
    pose proof (Nat.div_mod_eq t P) as Et. fold grp r in Et.
    pose proof (Nat.mod_upper_bound t P ltac:(lia)) as Hr. fold r in Hr.
    (* the group starts inside the matrix *)
    assert (Hfirst : grp * G < TM).
    { assert (P * grp <= t) by lia. subst P. nia. }
    assert (Hgs0 : 0 < gs) by (subst gs; unfold g_size; apply Nat.min_glb_lt; lia).
    assert (HgsG : gs <= G) by (subst gs; unfold g_size; lia).
    assert (HgsT : gs <= TM - grp * G) by (subst gs; unfold g_size; lia).
    (* r < gs * TN: a full group has gs = G; the last has gs = TM - first *)
    assert (Hrg : r < gs * TN).
    { subst gs. unfold g_size.
      destruct (Nat.le_ge_cases (TM - grp * G) G) as [L | L].
      - rewrite Nat.min_l by exact L. subst P. nia.
      - rewrite Nat.min_r by exact L. subst P. nia. }
    pose proof (Nat.div_mod_eq r gs) as Er.
    pose proof (Nat.mod_upper_bound r gs ltac:(lia)) as Hrm.
    assert (Hq : r / gs < TN) by (apply Nat.Div0.div_lt_upper_bound; lia).
    cbn [fst snd]. repeat split.
    + lia.
    + exact Hq.
    + unfold ungrouped.
      assert (Hgrp : (grp * G + r mod gs) / G = grp).
      { rewrite Nat.div_add_l by lia. rewrite Nat.div_small by lia. lia. }
      rewrite Hgrp. fold gs.
      replace (grp * G + r mod gs - grp * G) with (r mod gs) by lia.
      subst P. nia.
  - intros tm tn Htm Htn.
    destruct (ungrouped_spec TM TN G tm tn HG Htm Htn) as [Hlt [Hin Hrange]].
    split; [ exact Hrange | ].
    unfold ungrouped, grouped.
    set (grp := tm / G). set (gs := g_size TM G grp).
    fold grp gs in Hlt, Hin.
    assert (E1 : (grp * (G * TN) + tn * gs + (tm - grp * G)) / (G * TN) = grp).
    { replace (grp * (G * TN) + tn * gs + (tm - grp * G))
        with ((tn * gs + (tm - grp * G)) + grp * (G * TN)) by lia.
      rewrite Nat.div_add by nia. rewrite Nat.div_small by exact Hin. lia. }
    assert (E2 : (grp * (G * TN) + tn * gs + (tm - grp * G)) mod (G * TN)
                 = tn * gs + (tm - grp * G)).
    { replace (grp * (G * TN) + tn * gs + (tm - grp * G))
        with ((tn * gs + (tm - grp * G)) + grp * (G * TN)) by lia.
      rewrite Nat.Div0.mod_add. apply Nat.mod_small. exact Hin. }
    rewrite E1. fold gs. rewrite E2.
    assert (E3 : (tn * gs + (tm - grp * G)) mod gs = tm - grp * G).
    { rewrite Nat.Div0.add_mod, Nat.Div0.mod_mul, Nat.add_0_l, Nat.Div0.mod_mod.
      apply Nat.mod_small. exact Hlt. }
    assert (E4 : (tn * gs + (tm - grp * G)) / gs = tn).
    { rewrite Nat.div_add_l by lia. rewrite Nat.div_small by exact Hlt. lia. }
    rewrite E3, E4. f_equal.
    pose proof (Nat.div_mod_eq tm G). subst grp. nia.
Qed.

(** The emitter's two shortcuts are the general formula at a single tile row
    or column, so the bijection covers them too. *)
Theorem grouped_one_tile_row :
  forall TN G t, 0 < G -> t < TN -> grouped 1 TN G t = (0, t).
Proof.
  intros TN G t HG Ht. unfold grouped, g_size.
  assert (E : t / (G * TN) = 0) by (apply Nat.div_small; nia).
  rewrite E. rewrite (Nat.mod_small t (G * TN)) by nia.
  replace (Nat.min (1 - 0 * G) G) with 1 by lia.
  rewrite Nat.mod_1_r, Nat.div_1_r. reflexivity.
Qed.

Theorem grouped_one_tile_column :
  forall TM G t, 0 < G -> t < TM -> grouped TM 1 G t = (t, 0).
Proof.
  intros TM G t HG Ht. unfold grouped, g_size.
  rewrite Nat.mul_1_r.
  pose proof (Nat.div_mod_eq t G) as E.
  pose proof (Nat.mod_upper_bound t G ltac:(lia)) as Hm.
  assert (Hlt : t mod G < Nat.min (TM - t / G * G) G).
  { apply Nat.min_glb_lt; lia. }
  rewrite (Nat.mod_small (t mod G)) by exact Hlt.
  rewrite (Nat.div_small (t mod G)) by exact Hlt.
  f_equal. lia.
Qed.

(** **Every output tile is computed exactly once, on any grid.**  Its id has
    one owning CTA ([tile_owner]), and the grouped order is a bijection. *)
Theorem every_output_tile_is_computed_exactly_once :
  forall TM TN G P tm tn, 0 < G -> 0 < TN -> 0 < P -> tm < TM -> tn < TN ->
    exists t, t < TM * TN /\ grouped TM TN G t = (tm, tn) /\ tile_owner P t < P
    /\ forall t', t' < TM * TN -> grouped TM TN G t' = (tm, tn) -> t' = t.
Proof.
  intros TM TN G P tm tn HG HTN HP Htm Htn.
  destruct (grouped_is_a_bijection TM TN G HG HTN) as [Hfwd Hbwd].
  destruct (Hbwd tm tn Htm Htn) as [Hin Hg].
  exists (ungrouped TM TN G tm tn). repeat split; try assumption.
  - apply Nat.mod_upper_bound. lia.
  - intros t' Ht' Hg'. destruct (Hfwd t' Ht') as [_ [_ Hinv]].
    rewrite Hg' in Hinv. cbn [fst snd] in Hinv. symmetry. exact Hinv.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The K split: K tiles striped over [%ctaid.z]                    *)
(* ------------------------------------------------------------------ *)

(** A K tile is [T = cta_k / 32] steps of 32; step [s] is in K tile [s / T],
    which belongs to CTA [(s / T) mod nz]. *)
Definition k_tile_owner (T nz s : nat) : nat := (s / T) mod nz.

Theorem every_k_step_has_exactly_one_owner :
  forall T nz s, 0 < nz -> k_tile_owner T nz s < nz.
Proof. intros. unfold k_tile_owner. apply Nat.mod_upper_bound. lia. Qed.

(** **The theorem.**  The CTAs' partials sum to the whole contraction, for
    any split factor and any K tile width - [Decomposition]'s schema at this
    owner map, with no divisibility precondition. *)
Theorem the_split_k_sums_to_the_whole_contraction :
  forall T nz f S, 0 < nz ->
    D.combine Z.add (k_tile_owner T nz) f S nz = D.acc_range Z.add f 0 S.
Proof.
  intros T nz f S Hnz. apply D.decomposition_exact.
  intros i _. apply every_k_step_has_exactly_one_owner. exact Hnz.
Qed.

Corollary any_split_factor_gives_the_same_answer :
  forall T f S nz1 nz2, 0 < nz1 -> 0 < nz2 ->
    D.combine Z.add (k_tile_owner T nz1) f S nz1 = D.combine Z.add (k_tile_owner T nz2) f S nz2.
Proof.
  intros. rewrite !the_split_k_sums_to_the_whole_contraction by assumption. reflexivity.
Qed.

(** [D.combine] is the sum of the partials as a list. *)
Lemma combine_is_fold :
  forall owner f S t,
    D.combine Z.add owner f S t
    = fold_right Z.add 0%Z (map (fun w => D.part Z.add owner f w S) (seq 0 t)).
Proof.
  intros owner f S t. induction t as [| t IH].
  - reflexivity.
  - cbn [D.combine]. rewrite IH, seq_S, map_app, fold_right_app.
    cbn [map fold_right]. rewrite Nat.add_0_l.
    symmetry. rewrite GS.fold_right_add_shift. ring.
Qed.

(** **[red.global.add.s32] may land the partials in any order at all.** *)
Theorem the_atomic_reduction_is_order_independent :
  forall T nz f S order, 0 < nz -> Permutation order (seq 0 nz) ->
    fold_right Z.add 0%Z (map (fun w => D.part Z.add (k_tile_owner T nz) f w S) order)
    = D.acc_range Z.add f 0 S.
Proof.
  intros T nz f S order Hnz Hperm.
  rewrite (GS.fold_right_add_permutation _ (map (fun w => D.part Z.add (k_tile_owner T nz) f w S) (seq 0 nz)))
    by (apply Permutation_map; exact Hperm).
  rewrite <- combine_is_fold. apply the_split_k_sums_to_the_whole_contraction. exact Hnz.
Qed.

(** The refutations, reused from [GridStrideSplit] so the kernels' are
    comparable: a rounding accumulate would break the split, and - the
    failure an atomic can show that an in-order split cannot - the order in
    which the partials land. *)
Theorem a_rounding_accumulate_would_break_the_split :
  GS.combine GS.KS.fadd GS.spike 3 30 3 <> GS.sum_upto GS.KS.fadd GS.spike 30.
Proof. exact GS.rounding_breaks_the_stride_split. Qed.

Theorem a_rounding_accumulate_would_break_the_landing_order :
  fold_left GS.KS.fadd
    (map (fun w => GS.class_sum GS.KS.fadd GS.tail w 3 3) (seq 0 3)) 0%Z
  <> fold_left GS.KS.fadd
    (map (fun w => GS.class_sum GS.KS.fadd GS.tail w 3 3) (rev (seq 0 3))) 0%Z.
Proof. exact GS.rounding_is_order_dependent. Qed.

(** A CTA's stripe.  The emitter computes
    [n_my = (KT - z + nz - 1) / nz] (0 when [z >= KT]) and visits K tiles
    [z + i * nz] for [i < n_my].  Those are exactly its class's tiles below
    [KT]: none past it, and none missed. *)
Definition n_my (KT z nz : nat) : nat := if z <? KT then (KT - z + nz - 1) / nz else 0.

Theorem the_stripe_stays_below_the_k_tiles :
  forall KT z nz i, 0 < nz -> i < n_my KT z nz -> z + i * nz < KT.
Proof.
  intros KT z nz i Hnz Hi. unfold n_my in Hi.
  destruct (Nat.ltb_spec z KT) as [Hz | Hz]; [ | lia ].
  assert (Hlt : i * nz < KT - z + nz - 1 - (nz - 1)).
  { pose proof (Nat.div_mod_eq (KT - z + nz - 1) nz) as E.
    pose proof (Nat.mod_upper_bound (KT - z + nz - 1) nz ltac:(lia)) as M.
    assert (i + 1 <= (KT - z + nz - 1) / nz) by lia.
    nia. }
  lia.
Qed.

Theorem the_stripe_reaches_every_tile_of_its_class :
  forall KT z nz kt, 0 < nz -> z < nz -> kt < KT -> kt mod nz = z ->
    exists i, i < n_my KT z nz /\ kt = z + i * nz.
Proof.
  intros KT z nz kt Hnz Hz Hkt Hcls.
  exists (kt / nz). pose proof (Nat.div_mod_eq kt nz) as E. split; [ | lia ].
  unfold n_my. destruct (Nat.ltb_spec z KT) as [HzK | HzK]; [ | lia ].
  apply Nat.div_le_lower_bound; [ lia | ].
  (* nz * (kt/nz + 1) <= KT - z + nz - 1  since  nz * (kt/nz) = kt - z < KT - z *)
  assert (nz * (kt / nz) = kt - z) by lia.
  nia.
Qed.

(** Steps past [K/32] read zeros - the copies of K chunks past K are
    zero-sized - so the zero-filled tail of the last K tile adds nothing. *)
Theorem the_zero_filled_tail_adds_nothing :
  forall f S X, (forall s, S <= s -> f s = 0%Z) ->
    D.acc_range Z.add f 0 (S + X) = D.acc_range Z.add f 0 S.
Proof.
  intros f S X H. induction X as [| X IH].
  - rewrite Nat.add_0_r. reflexivity.
  - rewrite Nat.add_succ_r. cbn [D.acc_range]. rewrite IH, H by lia. ring.
Qed.

(* ------------------------------------------------------------------ *)
(** ** Which epilogue may store                                        *)
(* ------------------------------------------------------------------ *)

(** The K loop's class and worker count as the emitter chooses them.  The
    scaled form ([stores]) moves the constants 0 and 1; the plain form reads
    [%ctaid.z] / [%nctaid.z]. *)
Definition emitted_class (stores : bool) (cz : nat) : nat := if stores then 0 else cz.
Definition emitted_workers (stores : bool) (nz : nat) : nat := if stores then 1 else nz.

(** A storing CTA walks the WHOLE contraction, so what it writes is a final
    value: at one worker every step's owner is CTA 0. *)
Theorem a_storing_cta_computes_the_whole_contraction :
  forall T f S cz nz,
    D.part Z.add (k_tile_owner T (emitted_workers true nz)) f (emitted_class true cz) S
    = D.acc_range Z.add f 0 S.
Proof.
  intros T f S cz nz. cbn [emitted_class emitted_workers].
  induction S as [| S IH]; cbn [D.part D.acc_range]; [ reflexivity | ].
  assert (Hown : k_tile_owner T 1 S = 0) by (unfold k_tile_owner; apply Nat.mod_1_r).
  rewrite Hown. cbn [Nat.eqb]. rewrite IH. reflexivity.
Qed.

(** And [cz] and [nz] are not free in that answer: every z-CTA of the scaled
    form computes the same value and writes identical bytes. *)
Theorem every_storing_cta_writes_the_same_value :
  forall T f S cz1 nz1 cz2 nz2,
    D.part Z.add (k_tile_owner T (emitted_workers true nz1)) f (emitted_class true cz1) S
    = D.part Z.add (k_tile_owner T (emitted_workers true nz2)) f (emitted_class true cz2) S.
Proof. intros. rewrite !a_storing_cta_computes_the_whole_contraction. reflexivity. Qed.

(** **The plain form stores exactly when it is not split.**  It branches on
    [setp.ne.u32 %p, %nz, 1]: at [nz = 1] its one class is the whole
    contraction - the value a reduction into a zeroed C would leave - so the
    store is the reduction, without the 16.7M [red.global.add.s32] that cost
    a quarter of the kernel's time at 4096^3. *)
Theorem the_unsplit_launch_stores_the_whole_contraction :
  forall T f S,
    D.part Z.add (k_tile_owner T 1) f 0 S = D.acc_range Z.add f 0 S
    /\ (0 + D.part Z.add (k_tile_owner T 1) f 0 S)%Z = D.acc_range Z.add f 0 S.
Proof.
  intros T f S.
  assert (H : D.part Z.add (k_tile_owner T 1) f 0 S = D.acc_range Z.add f 0 S).
  { apply (a_storing_cta_computes_the_whole_contraction T f S 0 1). }
  split; [ exact H | rewrite H; ring ].
Qed.

(** **The refutation**: split, a CTA holds only part of the contraction - two
    K tiles of two steps, two CTAs - so a store would write part of the
    answer.  This is the defect measured on the scaled form before it stopped
    reading [%ctaid.z]: 2048 of 2048 elements wrong at z = 2. *)
Theorem a_split_cta_holds_only_part_of_the_contraction :
  D.part Z.add (k_tile_owner 2 2) (fun _ => 1%Z) 0 4 <> D.acc_range Z.add (fun _ => 1%Z) 0 4.
Proof. vm_compute. discriminate. Qed.

(** And the partials DISAGREE, so with stores the last writer would decide -
    why the measured answer moved between launches. *)
Theorem two_split_ctas_would_store_different_values :
  D.part Z.add (k_tile_owner 2 2) (fun j => Z.of_nat j) 0 4
  <> D.part Z.add (k_tile_owner 2 2) (fun j => Z.of_nat j) 1 4.
Proof. vm_compute. discriminate. Qed.

(** The reducing epilogue keeps the stripe - the constants are the scaled
    form's ALONE.  Without this, "walk the whole contraction everywhere" would
    satisfy every theorem above and delete the split. *)
Theorem the_reducing_epilogue_still_splits :
  forall cz nz, emitted_class false cz = cz /\ emitted_workers false nz = nz.
Proof. intros. split; reflexivity. Qed.

(* ------------------------------------------------------------------ *)
(** ** The shape refusal is what the predicates need                   *)
(* ------------------------------------------------------------------ *)

(** [emit_int8_gemm_kernel] refuses unless [M mod 16 = 0], [N mod 8 = 0] and
    [K mod 32 = 0].  The kernel copies rows past M and N as zeros and
    predicates its stores per m16 and n8 block, which is exact because a
    block that starts inside the matrix ends inside it. *)
Theorem whole_blocks_are_in_or_out :
  forall m b w, 0 < w -> m mod w = 0 -> b mod w = 0 -> b < m -> b + (w - 1) < m.
Proof.
  intros m b w Hw Hm Hb Hlt.
  pose proof (Nat.div_mod_eq m w) as Em. pose proof (Nat.div_mod_eq b w) as Eb.
  rewrite Hm in Em. rewrite Hb in Eb.
  assert (b / w < m / w) by nia.
  nia.
Qed.

(** And it bites: a ragged M would put an m16 block across the edge. *)
Theorem an_uncovered_shape_is_refused :
  63 mod MMA_M <> 0 /\ 48 < 63 /\ ~ (48 + (MMA_M - 1) < 63).
Proof. unfold MMA_M. cbn. repeat split; lia. Qed.

Print Assumptions lane_decomposition_is_injective.
Print Assumptions a_warps_lanes_fill_the_tile.
Print Assumptions out_row_injective.
Print Assumptions out_col_injective.
Print Assumptions out_row_onto.
Print Assumptions out_col_onto.
Print Assumptions c_element_has_exactly_one_owner.
Print Assumptions a_thread_is_its_warp_and_lane.
Print Assumptions the_trap_is_what_confines_a_warp_to_its_own_tile.
Print Assumptions without_the_trap_a_fifth_warp_lands_in_the_next_tile.
Print Assumptions the_copies_cover_the_stage_once.
Print Assumptions without_the_trap_a_short_block_leaves_rows_uncopied.
Print Assumptions cta_pid_is_unique.
Print Assumptions without_the_grid_trap_the_stride_wraps.
Print Assumptions the_tile_increment_detects_its_wrap.
Print Assumptions every_output_tile_has_exactly_one_owner.
Print Assumptions ungrouped_spec.
Print Assumptions grouped_is_a_bijection.
Print Assumptions grouped_one_tile_row.
Print Assumptions grouped_one_tile_column.
Print Assumptions every_output_tile_is_computed_exactly_once.
Print Assumptions every_k_step_has_exactly_one_owner.
Print Assumptions the_split_k_sums_to_the_whole_contraction.
Print Assumptions any_split_factor_gives_the_same_answer.
Print Assumptions combine_is_fold.
Print Assumptions the_atomic_reduction_is_order_independent.
Print Assumptions a_rounding_accumulate_would_break_the_split.
Print Assumptions a_rounding_accumulate_would_break_the_landing_order.
Print Assumptions the_stripe_stays_below_the_k_tiles.
Print Assumptions the_stripe_reaches_every_tile_of_its_class.
Print Assumptions the_zero_filled_tail_adds_nothing.
Print Assumptions a_storing_cta_computes_the_whole_contraction.
Print Assumptions every_storing_cta_writes_the_same_value.
Print Assumptions the_unsplit_launch_stores_the_whole_contraction.
Print Assumptions a_split_cta_holds_only_part_of_the_contraction.
Print Assumptions two_split_ctas_would_store_different_values.
Print Assumptions the_reducing_epilogue_still_splits.
Print Assumptions whole_blocks_are_in_or_out.
Print Assumptions an_uncovered_shape_is_refused.
