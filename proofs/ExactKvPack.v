(** * The packed KV cache's bytes (`python/y_lang/exact/kv.py`).

    With `Y_EXACT_KV_PACK=inplace` each cached row is one int8 tensor: K's
    int16 values as their bytes then its float32 scale's four bytes
    (`pack_k`), V's int8 values then its scale's bytes (`pack_v`), built from
    `_signed_byte(x) = ((x & 0xFF) ^ 0x80) - 0x80` and arithmetic shifts, and
    read back by viewing the bytes as int16 (K) or reading them as int8 (V)
    and rebuilding the scale's 32 bits (`_scale_from`). A wrong byte would
    change a cached key, value or scale silently. This file proves the round
    trip exact for every value the cache holds.

    [signed_byte_is_the_low_byte]: `_signed_byte x` is the integer in
    [-128, 127] congruent to x mod 256 (the `^ 0x80` on a byte checked on
    all 256 bytes). [a_byte_is_kept]: on a value already in [-128, 127] it is
    the identity, so `.to(torch.int8)` of every byte the row holds is exact.

    [k_bytes_rebuild_the_value]: for every int16 value r, the bytes `pack_k`
    writes (low: `_signed_byte r`; high: `r >> 8`) are int8, and the int16
    read back from them (little-endian: high * 256 + the low byte unsigned)
    is r.

    [v_scale_bytes_rebuild_the_bits]: for every int32 b (a float32's bits),
    the four bytes `_signed_byte (b >> 8 i)` rebuild b as `_scale_from`
    does with 8-bit pieces: the top piece signed, the others masked.
    [k_scale_halves_rebuild_the_bits]: the same with K's two int16 pieces,
    each read from two of those bytes, rebuilt with 16-bit pieces.

    ** Refutations.

    [an_unsigned_top_piece_breaks_a_negative_scale]: masking the top piece
    too (unsigned) rebuilds -1's bits as 2^32 - 1, not -1.
    [a_value_past_int16_loses_its_high_byte]: r = 32768, past int16: its high
    byte 128 is not an int8, which is why `pack_k` refuses K_LEVELS past
    32767.

    ** Not proved here.

    That a float32's `.view(int32)` and back is the identity on bits, that
    PyTorch's int8 storage is little-endian when viewed as int16 (x86 and
    CUDA), and that `quantize_rows` keeps |K| <= K_LEVELS (its clamp).

    Checked with Rocq 9.1:  coqc -Q . "" ExactKvPack.v  *)

From Stdlib Require Import ZArith Lia List Bool.
Import ListNotations.
Open Scope Z_scope.

Definition signed_byte (x : Z) : Z := Z.lxor (Z.land x 255) 128 - 128.

(** `^ 0x80` on a byte, all 256 of them. *)
Lemma xor_128_table : forallb (fun y => Z.lxor y 128 =? (y + 128) mod 256) (map Z.of_nat (seq 0 256)) = true.
Proof. vm_compute. reflexivity. Qed.

Lemma xor_128 : forall y, 0 <= y < 256 -> Z.lxor y 128 = (y + 128) mod 256.
Proof.
  intros y Hy. pose proof xor_128_table as T. rewrite forallb_forall in T.
  apply Z.eqb_eq, T. apply in_map_iff. exists (Z.to_nat y). split; [lia|]. apply in_seq. lia.
Qed.

Lemma land_255 : forall x, Z.land x 255 = x mod 256.
Proof. intro x. change 255 with (Z.ones 8). rewrite Z.land_ones by lia. reflexivity. Qed.

Theorem signed_byte_is_the_low_byte : forall x,
  -128 <= signed_byte x <= 127 /\ (signed_byte x - x) mod 256 = 0.
Proof.
  intro x. unfold signed_byte. rewrite land_255.
  pose proof (Z.mod_pos_bound x 256 ltac:(lia)). rewrite xor_128 by lia.
  pose proof (Z.mod_pos_bound (x mod 256 + 128) 256 ltac:(lia)).
  split; [lia|]. rewrite Zplus_mod_idemp_l.
  replace ((x + 128) mod 256 - 128 - x) with (((x + 128) mod 256 - (x + 128)) + 0) by ring.
  rewrite Z.add_0_r. rewrite Zminus_mod, Z.mod_mod, Z.sub_diag by lia. reflexivity.
Qed.

Theorem a_byte_is_kept : forall y, -128 <= y <= 127 -> signed_byte y = y.
Proof.
  intros y Hy. unfold signed_byte. rewrite land_255.
  pose proof (Z.mod_pos_bound y 256 ltac:(lia)). rewrite xor_128 by lia.
  destruct (Z.le_gt_cases 0 y).
  - rewrite (Z.mod_small y 256) by lia. rewrite Z.mod_small by lia. lia.
  - replace (y mod 256) with (y + 256) by (rewrite <- (Z.mod_small (y + 256) 256) by lia;
      rewrite Z.add_mod, Z.mod_same, Z.add_0_r, Z.mod_mod by lia; reflexivity).
    replace (y + 256 + 128) with ((y + 128) + 1 * 256) by ring. rewrite Z.mod_add by lia.
    rewrite Z.mod_small by lia. lia.
Qed.

(** The low byte unsigned, as a little-endian read sees it. *)
Definition ubyte (s : Z) : Z := s mod 256.

Lemma signed_byte_ubyte : forall x, ubyte (signed_byte x) = x mod 256.
Proof.
  intro x. unfold ubyte. destruct (signed_byte_is_the_low_byte x) as [_ C].
  replace (signed_byte x) with ((signed_byte x - x) + x) by ring.
  rewrite Zplus_mod, C, Z.add_0_l, Z.mod_mod by lia. reflexivity.
Qed.

Theorem k_bytes_rebuild_the_value : forall r, -32768 <= r <= 32767 ->
  (-128 <= Z.shiftr r 8 <= 127) /\ (-128 <= signed_byte r <= 127) /\
  Z.shiftr r 8 * 256 + ubyte (signed_byte r) = r.
Proof.
  intros r Hr. rewrite Z.shiftr_div_pow2 by lia. change (2 ^ 8) with 256.
  assert (r / 256 < 128) by (apply Z.div_lt_upper_bound; lia).
  assert (-128 <= r / 256) by (apply Z.div_le_lower_bound; lia).
  split; [lia|].
  split; [apply signed_byte_is_the_low_byte|].
  rewrite signed_byte_ubyte. pose proof (Z.div_mod r 256 ltac:(lia)). lia.
Qed.

(** `_scale_from` with n pieces of p bits: the top piece signed, the rest
    masked, each weighted by 2^(p i). For 4 pieces of 8 bits: *)
Definition rebuild8 (s0 s1 s2 s3 : Z) : Z :=
  s3 * 2 ^ 24 + ubyte s0 + ubyte s1 * 2 ^ 8 + ubyte s2 * 2 ^ 16.

Lemma shr_byte : forall b i, 0 <= i -> signed_byte (Z.shiftr b (8 * i)) mod 256 = (b / 2 ^ (8 * i)) mod 256.
Proof.
  intros b i Hi. change (signed_byte (Z.shiftr b (8 * i)) mod 256) with (ubyte (signed_byte (Z.shiftr b (8 * i)))).
  rewrite signed_byte_ubyte, Z.shiftr_div_pow2 by lia. reflexivity.
Qed.

Lemma base256 : forall b,
  b = (b / 2 ^ 24) * 2 ^ 24 + (b / 2 ^ 16) mod 256 * 2 ^ 16 + (b / 2 ^ 8) mod 256 * 2 ^ 8 + b mod 256.
Proof.
  intro b.
  pose proof (Z.div_mod b 256 ltac:(lia)) as E0.
  pose proof (Z.div_mod (b / 256) 256 ltac:(lia)) as E1.
  pose proof (Z.div_mod (b / 256 / 256) 256 ltac:(lia)) as E2.
  rewrite !Z.div_div in E1, E2 by lia. change (256 * 256) with 65536 in *. change (65536 * 256) with 16777216 in *.
  change (2 ^ 24) with 16777216. change (2 ^ 16) with 65536. change (2 ^ 8) with 256.
  rewrite Z.div_div in E2 by lia. change (65536 * 256) with 16777216 in E2. lia.
Qed.

Theorem v_scale_bytes_rebuild_the_bits : forall b, -2 ^ 31 <= b <= 2 ^ 31 - 1 ->
  rebuild8 (signed_byte b) (signed_byte (Z.shiftr b 8)) (signed_byte (Z.shiftr b 16))
           (signed_byte (Z.shiftr b 24)) = b.
Proof.
  intros b Hb. unfold rebuild8.
  rewrite !signed_byte_ubyte, !Z.shiftr_div_pow2 by lia.
  rewrite a_byte_is_kept.
  - pose proof (base256 b). lia.
  - change (2 ^ 24) with 16777216. change (2 ^ 31) with 2147483648 in Hb.
    assert (b / 16777216 < 128) by (apply Z.div_lt_upper_bound; lia).
    assert (-128 <= b / 16777216) by (apply Z.div_le_lower_bound; lia).
    lia.
Qed.

(** K: the scale's bytes as two int16 pieces (bytes 0, 1 and 2, 3, read
    little-endian), rebuilt with 16-bit pieces. *)
Definition int16_of (lo hi : Z) : Z := hi * 256 + ubyte lo.
Definition rebuild16 (p0 p1 : Z) : Z := p1 * 2 ^ 16 + p0 mod 2 ^ 16.

Theorem k_scale_halves_rebuild_the_bits : forall b, -2 ^ 31 <= b <= 2 ^ 31 - 1 ->
  rebuild16 (int16_of (signed_byte b) (signed_byte (Z.shiftr b 8)))
            (int16_of (signed_byte (Z.shiftr b 16)) (signed_byte (Z.shiftr b 24))) = b.
Proof.
  intros b Hb. pose proof (v_scale_bytes_rebuild_the_bits b Hb) as V. unfold rebuild8 in V.
  unfold rebuild16, int16_of.
  set (s0 := signed_byte b) in *. set (s1 := signed_byte (Z.shiftr b 8)) in *.
  set (s2 := signed_byte (Z.shiftr b 16)) in *. set (s3 := signed_byte (Z.shiftr b 24)) in *.
  assert (B1 : -128 <= s1 <= 127) by apply signed_byte_is_the_low_byte.
  assert (U0 : 0 <= ubyte s0 < 256) by (unfold ubyte; apply Z.mod_pos_bound; lia).
  assert (U2 : 0 <= ubyte s2 < 256) by (unfold ubyte; apply Z.mod_pos_bound; lia).
  (* the low int16 piece mod 2^16 is its unsigned 16 bits: ubyte s1 * 256 + ubyte s0 *)
  assert (L : (s1 * 256 + ubyte s0) mod 2 ^ 16 = ubyte s1 * 256 + ubyte s0).
  { unfold ubyte in *. change (2 ^ 16) with 65536.
    pose proof (Z.mod_pos_bound s1 256 ltac:(lia)). pose proof (Z.div_mod s1 256 ltac:(lia)).
    symmetry. apply Z.mod_unique_pos with (q := s1 / 256); lia. }
  rewrite L. change (2 ^ 16) with 65536. change (2 ^ 24) with 16777216 in V. change (2 ^ 8) with 256 in V.
  change (2 ^ 16) with 65536 in V. lia.
Qed.

Theorem an_unsigned_top_piece_breaks_a_negative_scale :
  ubyte (signed_byte (-1)) + ubyte (signed_byte (Z.shiftr (-1) 8)) * 2 ^ 8
  + ubyte (signed_byte (Z.shiftr (-1) 16)) * 2 ^ 16 + ubyte (signed_byte (Z.shiftr (-1) 24)) * 2 ^ 24 = 2 ^ 32 - 1.
Proof. vm_compute. reflexivity. Qed.

Theorem a_value_past_int16_loses_its_high_byte : Z.shiftr 32768 8 = 128.
Proof. vm_compute. reflexivity. Qed.

Print Assumptions signed_byte_is_the_low_byte.
Print Assumptions a_byte_is_kept.
Print Assumptions k_bytes_rebuild_the_value.
Print Assumptions v_scale_bytes_rebuild_the_bits.
Print Assumptions k_scale_halves_rebuild_the_bits.
Print Assumptions an_unsigned_top_piece_breaks_a_negative_scale.
Print Assumptions a_value_past_int16_loses_its_high_byte.
