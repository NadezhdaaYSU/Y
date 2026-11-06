//! The int8 tensor-core GEMM: `@tile(M, N, K) kernel k(A, B: GlobalMemory<I8>,
//! C: GlobalMemory<I32>)`, and the scaled form `(A, B: GlobalMemory<I8>,
//! scale_a, scale_b, bias, C: GlobalMemory<F32>)`. A `[M][K]` and B `[N][K]`
//! (K-contiguous, what `mma.sync.m16n8k32.row.col` reads), every output the
//! exact int32 sum. Since 2026-11-06 it replaces a kernel that gave each warp
//! its own output tile and read A and B straight from global memory (0.40x
//! cuBLASLt at 4096^3).
//!
//! **What it is for: matching cuBLASLt, so an exact int8 GEMM costs nothing.**
//! The mainloop is the e4m3 GEMM's (`emit_fp8_e4m3_gemm_kernel`), with its
//! helpers: K tiles copied by `cp.async` into swizzled shared stages,
//! `ldmatrix.x4` fragments. What differs:
//! - the mma accumulates in its own int32 registers (`mma d, a, b, d`): an
//!   integer sum is exact in any order, so it chains, where e4m3 adds every
//!   mma's result in float32;
//! - each substep's fragments are loaded during the previous substep's MMAs,
//!   and the barrier that publishes the next K tile comes before a tile's
//!   last substep, as CUTLASS orders its multistage mainloop: 0.97x -> 1.03x
//!   cuBLASLt at 4096^3;
//! - a launch that does not split K STORES each output; one that splits it
//!   over `%ctaid.z` adds its partials into a zeroed C with
//!   `red.global.add.s32`. Reducing every output cost a quarter of the time
//!   at 4096^3 (0.74x cuBLASLt against 0.97x storing);
//! - output tiles are grid-strided by a linear id in grouped raster order and
//!   K tiles striped over `%ctaid.z`, so any grid gives the same matrix; a
//!   block of any size but the one the header names traps.
//!
//! Against cuBLASLt (`torch._int_mm`) on an RTX 4070 Ti SUPER, two runs:
//! 1.02-1.10x at 2048^3-4096^3, 0.97-1.00x at 8192^3, 1.2-1.8x on smaller
//! and K-heavy shapes, 2.2-6.4x the kernel it replaced from 512^3 up (1.4x
//! at 256^3). On 24 LLM shapes
//! (M = 32-2048), with the operands L2-resident or read from DRAM, 83 of
//! 96 results are 0.98x or better and the lowest 0.91x: 128 rows against
//! K = 11008-14336 (the 64x64 tile gives 0.99x there, and loses 20% where
//! B fits in L2), 512 x 11008 x 4096, and 512 x 1024 x 4096 from DRAM.
//! 0.93-0.95x where K % 64 == 32 (every second row starts 32 bytes into a
//! 64-byte segment). docs/benchmarks/int8_gemm_2026-11-06/.

use super::*;

/// One CTA tile of the staged int8 GEMM: `cta_m x cta_n` outputs on
/// `warps_m x warps_n` warps, `cta_k` bytes of K per stage, `stages` shared
/// stages, the CTAs per SM its registers are capped for, and the M tiles a
/// launch group covers (grouped raster order, for L2 reuse).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Int8Tile {
    pub cta_m: u32,
    pub cta_n: u32,
    pub warps_m: u32,
    pub warps_n: u32,
    pub cta_k: u32,
    pub stages: u32,
    pub ctas_per_sm: u32,
    pub group_m: u32,
}

impl Int8Tile {
    pub(crate) fn threads(&self) -> u32 {
        self.warps_m * self.warps_n * 32
    }
    pub(crate) fn smem_bytes(&self) -> u32 {
        self.stages * (self.cta_m + self.cta_n) * self.cta_k
    }
}

/// 128x128 outputs on 2x2 warps of 64x64, 64 bytes of K a stage, 3 stages
/// (48 KB), two CTAs an SM, launch groups of 32 M tiles.
pub(crate) const INT8_TILE_LARGE: Int8Tile =
    Int8Tile { cta_m: 128, cta_n: 128, warps_m: 2, warps_n: 2, cta_k: 64, stages: 3, ctas_per_sm: 2, group_m: 32 };
/// 64x128 on 2x2 warps of 32x64, 3 stages (36 KB), registers for three CTAs.
pub(crate) const INT8_TILE_MID: Int8Tile =
    Int8Tile { cta_m: 64, cta_n: 128, warps_m: 2, warps_n: 2, cta_k: 64, stages: 3, ctas_per_sm: 3, group_m: 16 };
/// 64x64 on 2x2 warps of 32x32, 4 stages (32 KB), registers for four CTAs.
pub(crate) const INT8_TILE_SMALL: Int8Tile =
    Int8Tile { cta_m: 64, cta_n: 64, warps_m: 2, warps_n: 2, cta_k: 64, stages: 4, ctas_per_sm: 4, group_m: 16 };

/// A tile is taken when it leaves at least this many output tiles: about one
/// CTA for each SM of a mid-range card (this one has 66). A fixed number,
/// not the local SM count, so the PTX does not depend on the machine that
/// compiled it; a wrong value gives a slower kernel, never a different matrix.
pub(crate) const INT8_MIN_TILES: u32 = 64;

/// The staged kernel's tile for an `m x n x k` problem: the largest of
/// [`INT8_TILE_LARGE`], [`INT8_TILE_MID`] that is no taller than M, no wider
/// than N, and leaves [`INT8_MIN_TILES`] output tiles; else
/// [`INT8_TILE_SMALL`]. Without the first two conditions 32 x 11008 x 4096
/// took the 128x128 tile, three quarters of it past M, and ran at 0.60x
/// cuBLASLt against 1.13x on a 64-row tile. The tile is a function of the
/// shape, never of the machine; `docs/benchmarks/int8_gemm_2026-11-06/`
/// has every tile at every shape swept, and where the rule's is not the
/// fastest. `Y_INT8_TILE=cta_m,cta_n,cta_k,warps_m,warps_n,stages,ctas_per_sm,group_m`
/// names another (for sweeps and tests; a malformed value is an error).
pub(crate) fn int8_staged_tile(m: u32, n: u32, _k: u32) -> Result<Int8Tile, String> {
    if let Ok(v) = std::env::var("Y_INT8_TILE") {
        let f: Result<Vec<u32>, _> = v.split(',').map(|x| x.trim().parse::<u32>()).collect();
        return match f.as_deref() {
            Ok(&[cta_m, cta_n, cta_k, warps_m, warps_n, stages, ctas_per_sm, group_m]) => {
                Ok(Int8Tile { cta_m, cta_n, warps_m, warps_n, cta_k, stages, ctas_per_sm, group_m })
            }
            _ => Err(format!(
                "Y_INT8_TILE='{}' is not cta_m,cta_n,cta_k,warps_m,warps_n,stages,ctas_per_sm,group_m",
                v
            )),
        };
    }
    let tiles = |t: &Int8Tile| u64::from(m.div_ceil(t.cta_m)) * u64::from(n.div_ceil(t.cta_n));
    Ok([INT8_TILE_LARGE, INT8_TILE_MID]
        .into_iter()
        .find(|t| t.cta_m <= m && t.cta_n <= n && tiles(t) >= u64::from(INT8_MIN_TILES))
        .unwrap_or(INT8_TILE_SMALL))
}

impl PtxEmitter {
    /// Why a tile cannot be staged, or `None`.
    fn int8_staged_geometry_error(t: &Int8Tile) -> Option<String> {
        let Int8Tile { cta_m, cta_n, warps_m, warps_n, cta_k, stages, ctas_per_sm, group_m } = *t;
        if let Err(why) = validate_warp_tiling(cta_m, cta_n, cta_k, warps_m, warps_n, 16, 8, 32) {
            return Some(why);
        }
        let threads = t.threads();
        let cpr = cta_k / 16;
        let num_j = cta_n / warps_n / 8;
        if threads > 1024 {
            Some(format!("{} threads, past a CTA's 1024", threads))
        } else if ctas_per_sm == 0 || group_m == 0 {
            Some("CTAs per SM and the launch group's M tiles must be at least 1".to_string())
        } else if t.smem_bytes() > FP8_E4M3_MAX_SMEM {
            Some(format!("{} stages take {} bytes of shared memory, past the {} a CTA can have", stages, t.smem_bytes(), FP8_E4M3_MAX_SMEM))
        } else if !matches!(cta_k, 32 | 64 | 128) {
            Some(format!("cta_k {} is not 32, 64 or 128, the widths the chunk swizzle covers", cta_k))
        } else if stages < 2 {
            Some(format!("{} stages; the K loop needs at least 2", stages))
        } else if num_j % 2 != 0 {
            Some(format!("a warp's {} columns are not whole pairs of n8 tiles, which one `ldmatrix.x4` loads", cta_n / warps_n))
        } else if threads % cpr != 0 || (threads / cpr) % 8 != 0 {
            Some(format!("{} threads do not copy whole groups of 8 rows of {} chunks, so a thread's chunks would differ in swizzle", threads, cpr))
        } else if (cta_m * cpr) % threads != 0 || (cta_n * cpr) % threads != 0 {
            Some(format!("{} threads do not split the {}x{} and {}x{} byte tiles evenly", threads, cta_m, cta_k, cta_n, cta_k))
        } else {
            None
        }
    }

    /// The staged int8 GEMM (see the module comment). `epi` is
    /// `Some((scale_a, scale_b, bias))` for the scaled form. The shape and
    /// K refusals are `emit_int8_gemm_kernel`'s, made before this is called.
    /// Returns the threads per CTA.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit_int8_staged_gemm_kernel(
        &mut self,
        m: u32,
        n: u32,
        k: u32,
        a_ptr: &str,
        b_ptr: &str,
        c_ptr: &str,
        kernel_name: &str,
        epi: Option<(&str, &str, &str)>,
        t: Int8Tile,
    ) -> u32 {
        if let Some(why) = Self::int8_staged_geometry_error(&t) {
            self.emit_errors.push(format!(
                "[PTX] `{}`: the {}x{}x{} int8 tile on {}x{} warps cannot be staged: {}",
                kernel_name, t.cta_m, t.cta_n, t.cta_k, t.warps_m, t.warps_n, why
            ));
            return 32;
        }
        let Int8Tile { cta_m, cta_n, warps_m, warps_n, cta_k, stages, ctas_per_sm, group_m } = t;
        let threads = t.threads();
        let per_warp_m = cta_m / warps_m;
        let per_warp_n = cta_n / warps_n;
        let num_i = per_warp_m / 16;
        let num_j = per_warp_n / 8;
        let k_substeps = cta_k / 32;
        let cpr = cta_k / 16;
        let rows_per_pass = threads / cpr;
        let chunks_a = cta_m / rows_per_pass;
        let chunks_b = cta_n / rows_per_pass;
        let k_tiles = k.div_ceil(cta_k);
        let tiles_m = m.div_ceil(cta_m);
        let tiles_n = n.div_ceil(cta_n);
        // The tile ids and C's row stride in bytes are 32-bit in the kernel.
        if u64::from(tiles_m) * u64::from(tiles_n) > u64::from(u32::MAX) || u64::from(n) * 4 > u64::from(u32::MAX) {
            self.emit_errors.push(format!(
                "[PTX] `{}`: M={} N={} is past the staged int8 GEMM's 32-bit tile ids and row strides",
                kernel_name, m, n
            ));
            return 32;
        }
        let tiles = tiles_m * tiles_n;
        let stage_a_bytes = cta_m * cta_k;
        let stage_bytes = (cta_m + cta_n) * cta_k;
        let smem_bytes = t.smem_bytes();
        let stores = epi.is_some();
        // Offsets the copies and stores carry as immediates.
        let widest = [
            u64::from(chunks_a.max(chunks_b)) * u64::from(rows_per_pass) * u64::from(k),
            (u64::from(per_warp_m) * u64::from(n) + u64::from(per_warp_n)) * 4,
        ];
        if widest.iter().any(|&b| b > i32::MAX as u64) {
            self.emit_errors.push(format!(
                "[PTX] `{}`: M={} N={} K={} puts a copy or store offset of the staged int8 GEMM past a 32-bit immediate",
                kernel_name, m, n, k
            ));
            return 32;
        }

        let w = |e: &mut Self, line: String| {
            writeln!(&mut e.ptx_buffer, "    {}", line).unwrap();
        };
        w(self, "// ========================================================".to_string());
        w(self, format!(
            "// [Y INT8 TENSOR CORE GEMM] staged | M={} N={} K={} | CTA {}x{}x{} | {}x{} warps of {}x{} | {} stages | mma.sync.m16n8k32.row.col.s32.s8.s8.s32",
            m, n, k, cta_m, cta_n, cta_k, warps_m, warps_n, per_warp_m, per_warp_n, stages
        ));
        w(self, "// A [M][K] and B [N][K] int8, copied by cp.async into swizzled shared stages; every output is the exact int32 sum.".to_string());
        w(self, format!(
            "// Launch: block ({},1,1); any other block traps. Grid: any; ({}, 1, z) is one {}x{} tile per CTA.",
            threads, tiles, cta_m, cta_n
        ));
        if stores {
            w(self, "// [Y INT8 GEMM] the fused epilogue STORES, so every CTA walks the whole contraction and %ctaid.z is ignored.".to_string());
        } else {
            w(self, "// nctaid.z = 1: each output is stored. K split over z: CTA z takes K tiles z, z + nctaid.z, ... and adds its partial sums into C by red.global.add.s32, so C must then be zeroed by the caller.".to_string());
        }
        w(self, format!("// {} cp.async stages of int8 K tiles. Dynamic shared memory required: {} bytes.", stages, smem_bytes));
        w(self, "// ========================================================".to_string());
        let smem_sym = format!("smem_int8_{}", kernel_name);
        self.pending_extern_decls.push(format!(".extern .shared .align 128 .b8 {}[];", smem_sym));
        self.tile_gemm_ctas_per_sm = Some(ctas_per_sm);

        let a_g = self.alloc_reg64();
        w(self, format!("cvta.to.global.u64 {}, {};", a_g, a_ptr));
        let b_g = self.alloc_reg64();
        w(self, format!("cvta.to.global.u64 {}, {};", b_g, b_ptr));
        let c_g = self.alloc_reg64();
        w(self, format!("cvta.to.global.u64 {}, {};", c_g, c_ptr));
        let epi_g = epi.map(|(sa, sb, bi)| {
            let (gsa, gsb, gbi) = (self.alloc_reg64(), self.alloc_reg64(), self.alloc_reg64());
            w(self, format!("cvta.to.global.u64 {}, {};", gsa, sa));
            w(self, format!("cvta.to.global.u64 {}, {};", gsb, sb));
            w(self, format!("cvta.to.global.u64 {}, {};", gbi, bi));
            (gsa, gsb, gbi)
        });

        // THE LAUNCH CONTRACT. Every thread has a copy role and a warp tile:
        // a smaller block leaves rows of a stage uncopied and part of every
        // tile uncomputed, a larger one would put threads with no role across
        // `bar.sync`. Either is a wrong answer, so either traps.
        let tmp = self.alloc_reg32();
        let bad = self.alloc_pred();
        for (sreg, want) in [("%ntid.x", threads), ("%ntid.y", 1), ("%ntid.z", 1)] {
            w(self, format!("mov.u32 {}, {};", tmp, sreg));
            w(self, format!("setp.ne.u32 {}, {}, {};", bad, tmp, want));
            w(self, format!("@{} trap;", bad));
        }
        // The output tiles are grid-strided over the x*y plane by a linear
        // id; past 2^32 CTAs that product wraps and tiles would be skipped.
        let ncx = self.alloc_reg32();
        let ncy = self.alloc_reg32();
        w(self, format!("mov.u32 {}, %nctaid.x;", ncx));
        w(self, format!("mov.u32 {}, %nctaid.y;", ncy));
        let g64 = self.alloc_reg64();
        w(self, format!("mul.wide.u32 {}, {}, {};", g64, ncx, ncy));
        w(self, format!("setp.gt.u64 {}, {}, 4294967295;", bad, g64));
        w(self, format!("@{} trap;", bad));
        let n_ctas = self.alloc_reg32();
        w(self, format!("mul.lo.u32 {}, {}, {};", n_ctas, ncx, ncy));
        let pid = self.alloc_reg32();
        let cy = self.alloc_reg32();
        w(self, format!("mov.u32 {}, %ctaid.y;", cy));
        w(self, format!("mov.u32 {}, %ctaid.x;", pid));
        w(self, format!("mad.lo.u32 {}, {}, {}, {};", pid, cy, ncx, pid));

        let smem = self.alloc_reg32();
        w(self, format!("mov.u32 {}, {};", smem, smem_sym));
        let tid = self.alloc_reg32();
        w(self, format!("mov.u32 {}, %tid.x;", tid));
        let lane = self.alloc_reg32();
        w(self, format!("and.b32 {}, {}, 31;", lane, tid));
        let warp_id = self.alloc_reg32();
        w(self, format!("shr.u32 {}, {}, 5;", warp_id, tid));
        let warp_m = self.alloc_reg32();
        w(self, format!("rem.u32 {}, {}, {};", warp_m, warp_id, warps_m));
        let warp_n = self.alloc_reg32();
        w(self, format!("div.u32 {}, {}, {};", warp_n, warp_id, warps_m));
        let warp_row0 = self.alloc_reg32();
        w(self, format!("mul.lo.u32 {}, {}, {};", warp_row0, warp_m, per_warp_m));
        let warp_col0 = self.alloc_reg32();
        w(self, format!("mul.lo.u32 {}, {}, {};", warp_col0, warp_n, per_warp_n));
        let (a_lane, b_lane) = self.emit_fp8_e4m3_ldmatrix_offsets(&lane, &warp_row0, &warp_col0, cta_k, stage_a_bytes);

        // Each thread's copies: row tid / cpr, chunk tid % cpr, then every
        // rows_per_pass rows below, which have the same swizzle.
        let lr = self.alloc_reg32();
        w(self, format!("div.u32 {}, {}, {};", lr, tid, cpr));
        let lc = self.alloc_reg32();
        w(self, format!("rem.u32 {}, {}, {};", lc, tid, cpr));
        let swz = self.emit_fp8_e4m3_swizzle(&lr, cta_k);
        let sch = self.alloc_reg32();
        w(self, format!("xor.b32 {}, {}, {};", sch, lc, swz));
        let s_thread = self.alloc_reg32();
        w(self, format!("mad.lo.u32 {}, {}, {}, {};", s_thread, lr, cta_k, smem));
        w(self, format!("mad.lo.u32 {}, {}, 16, {};", s_thread, sch, s_thread));
        let k_byte = self.alloc_reg32();
        w(self, format!("shl.b32 {}, {}, 4;", k_byte, lc));
        let k_byte64 = self.alloc_reg64();
        w(self, format!("cvt.u64.u32 {}, {};", k_byte64, k_byte));
        let group_id = self.alloc_reg32();
        w(self, format!("shr.u32 {}, {}, 2;", group_id, lane));
        let tig = self.alloc_reg32();
        w(self, format!("and.b32 {}, {}, 3;", tig, lane));

        // K: this CTA's stripe is K tiles cz, cz + nz, ... below k_tiles,
        // n_my of them. The storing epilogue takes the whole contraction.
        let (cz, nz) = (self.alloc_reg32(), self.alloc_reg32());
        if stores {
            w(self, format!("mov.u32 {}, 0;", cz));
            w(self, format!("mov.u32 {}, 1;", nz));
        } else {
            w(self, format!("mov.u32 {}, %ctaid.z;", cz));
            w(self, format!("mov.u32 {}, %nctaid.z;", nz));
        }
        let n_my = self.alloc_reg32();
        let has = self.alloc_pred();
        w(self, format!("setp.lt.u32 {}, {}, {};", has, cz, k_tiles));
        w(self, format!("sub.u32 {}, {}, {};", n_my, k_tiles, cz));
        w(self, format!("add.u32 {}, {}, {};", n_my, n_my, nz));
        w(self, format!("sub.u32 {}, {}, 1;", n_my, n_my));
        w(self, format!("div.u32 {}, {}, {};", n_my, n_my, nz));
        w(self, format!("selp.u32 {}, {}, 0, {};", n_my, n_my, has));
        let k_base = self.alloc_reg32();
        w(self, format!("mul.lo.u32 {}, {}, {};", k_base, cz, cta_k));
        let k_stride = self.alloc_reg32();
        w(self, format!("mul.lo.u32 {}, {}, {};", k_stride, nz, cta_k));

        // ---- the tile loop ----
        let tile = self.alloc_reg32();
        w(self, format!("mov.u32 {}, {};", tile, pid));
        let tile_top = self.alloc_label(&format!("I8S_TILE_{}", kernel_name));
        let tile_next = self.alloc_label(&format!("I8S_TILE_NEXT_{}", kernel_name));
        let done = self.alloc_label(&format!("I8S_DONE_{}", kernel_name));
        writeln!(&mut self.ptx_buffer, "{}:", tile_top).unwrap();
        let p_end = self.alloc_pred();
        w(self, format!("setp.ge.u32 {}, {}, {};", p_end, tile, tiles));
        w(self, format!("@{} bra {};", p_end, done));
        let (tile_m, tile_n) = self.emit_int8_grouped_tile(&tile, tiles_m, tiles_n, group_m);
        let cta_row0 = self.alloc_reg32();
        w(self, format!("mul.lo.u32 {}, {}, {};", cta_row0, tile_m, cta_m));
        let cta_col0 = self.alloc_reg32();
        w(self, format!("mul.lo.u32 {}, {}, {};", cta_col0, tile_n, cta_n));
        let operand_base = |this: &mut Self, g: &str, row0: &str| -> String {
            let row = this.alloc_reg32();
            writeln!(&mut this.ptx_buffer, "    add.u32 {}, {}, {};", row, row0, lr).unwrap();
            let off = this.alloc_reg64();
            writeln!(&mut this.ptx_buffer, "    mul.wide.u32 {}, {}, {};", off, row, k).unwrap();
            let ga = this.alloc_reg64();
            writeln!(&mut this.ptx_buffer, "    add.s64 {}, {}, {};", ga, g, off).unwrap();
            writeln!(&mut this.ptx_buffer, "    add.s64 {}, {}, {};", ga, ga, k_byte64).unwrap();
            ga
        };
        let ga = operand_base(self, &a_g, &cta_row0);
        let gb = operand_base(self, &b_g, &cta_col0);
        let row_preds = |this: &mut Self, row0: &str, chunks: u32, bound: u32, ragged: bool| -> Vec<Option<String>> {
            (0..chunks)
                .map(|u| {
                    if !ragged {
                        return None;
                    }
                    let r = this.alloc_reg32();
                    writeln!(&mut this.ptx_buffer, "    add.u32 {}, {}, {};", r, row0, u * rows_per_pass).unwrap();
                    writeln!(&mut this.ptx_buffer, "    add.u32 {}, {}, {};", r, r, lr).unwrap();
                    let p = this.alloc_pred();
                    writeln!(&mut this.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p, r, bound).unwrap();
                    Some(p)
                })
                .collect()
        };
        let pa = row_preds(self, &cta_row0, chunks_a, m, m % cta_m != 0);
        let pb = row_preds(self, &cta_col0, chunks_b, n, n % cta_n != 0);
        let copy = Fp8E4m3Copies {
            ga,
            gb,
            k_byte: k_byte.clone(),
            pa,
            pb,
            k_ragged: k % cta_k != 0,
            chunks_a,
            chunks_b,
            rows_per_pass,
            cta_k,
            k,
            stage_a_bytes,
        };

        let acc: Vec<Vec<Vec<String>>> = (0..num_i)
            .map(|_| (0..num_j).map(|_| (0..4).map(|_| self.alloc_reg32()).collect()).collect())
            .collect();
        for r in acc.iter().flatten().flatten() {
            w(self, format!("mov.b32 {}, 0;", r));
        }

        // ---- prologue: this stripe's tiles 0 .. stages-2 ----
        for s in 0..stages - 1 {
            let p = self.alloc_pred();
            w(self, format!("setp.gt.u32 {}, {}, {};", p, n_my, s));
            let k0 = self.alloc_reg32();
            w(self, format!("mad.lo.u32 {}, {}, {}, {};", k0, k_stride, s, k_base));
            let slot = self.alloc_reg32();
            w(self, format!("add.u32 {}, {}, {};", slot, s_thread, s * stage_bytes));
            self.emit_fp8_e4m3_copy_tile(&copy, &k0, &slot, Some(&p));
            w(self, "cp.async.commit_group;".to_string());
        }

        // ---- the K loop, its fragment loads one substep ahead ----
        // As CUTLASS's multistage mainloop orders it: the barrier that
        // publishes tile it+1 comes after the MMAs of tile it's second-last
        // 32-wide substep are issued, and tile it+1's first fragments are
        // loaded before the MMAs of tile it's last substep, so the tensor
        // cores have queued work across the barrier and the loads' latency.
        //
        // Hazards: the copy of tile it+S-1 (iteration it, substep 0) goes
        // into the stage tile it-1 used, whose last reads were before the
        // barrier of iteration it-1. That barrier's wait_group S-2 leaves
        // tile it+1 landed: S+it groups are committed by then.
        // Fragments one substep ahead: the first substep's are carried
        // across the loop's back edge in these registers.
        let carried_a: Vec<Vec<String>> = (0..num_i).map(|_| (0..4).map(|_| self.alloc_reg32()).collect()).collect();
        let carried_b: Vec<Vec<String>> = (0..num_j / 2).map(|_| (0..4).map(|_| self.alloc_reg32()).collect()).collect();
        let load_frags = |this: &mut Self, stage: &str, ks: u32, fa: &[Vec<String>], fb: &[Vec<String>]| {
            let a_base = this.alloc_reg32();
            writeln!(&mut this.ptx_buffer, "    add.u32 {}, {}, {};", a_base, stage, a_lane[ks as usize]).unwrap();
            let b_base = this.alloc_reg32();
            writeln!(&mut this.ptx_buffer, "    add.u32 {}, {}, {};", b_base, stage, b_lane[ks as usize]).unwrap();
            for (jp, r) in fb.iter().enumerate() {
                writeln!(&mut this.ptx_buffer, "    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{}}}, [{}+{}];", r.join(","), b_base, jp as u32 * 16 * cta_k).unwrap();
            }
            for (i, r) in fa.iter().enumerate() {
                writeln!(&mut this.ptx_buffer, "    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{}}}, [{}+{}];", r.join(","), a_base, i as u32 * 16 * cta_k).unwrap();
            }
        };
        let fresh = |this: &mut Self, groups: u32| -> Vec<Vec<String>> {
            (0..groups).map(|_| (0..4).map(|_| this.alloc_reg32()).collect()).collect()
        };
        let mmas = |this: &mut Self, fa: &[Vec<String>], fb: &[Vec<String>]| {
            for i in 0..num_i as usize {
                for j in 0..num_j as usize {
                    let d = acc[i][j].join(",");
                    let b = &fb[j / 2][(j % 2) * 2..(j % 2) * 2 + 2];
                    writeln!(
                        &mut this.ptx_buffer,
                        "    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {{{d}}}, {{{}}}, {{{}}}, {{{d}}};",
                        fa[i].join(","),
                        b.join(","),
                    )
                    .unwrap();
                }
            }
        };

        w(self, format!("cp.async.wait_group {};", stages - 2));
        w(self, "bar.sync 0;".to_string());
        load_frags(self, &smem, 0, &carried_a, &carried_b);

        let it = self.alloc_reg32();
        w(self, format!("mov.u32 {}, 0;", it));
        let read_off = self.alloc_reg32();
        w(self, format!("mov.u32 {}, 0;", read_off));
        let write_off = self.alloc_reg32();
        w(self, format!("mov.u32 {}, {};", write_off, (stages - 1) * stage_bytes));
        let k_top = self.alloc_label(&format!("I8S_K_{}", kernel_name));
        let k_done = self.alloc_label(&format!("I8S_K_DONE_{}", kernel_name));
        writeln!(&mut self.ptx_buffer, "{}:", k_top).unwrap();
        let p_exit = self.alloc_pred();
        w(self, format!("setp.ge.u32 {}, {}, {};", p_exit, it, n_my));
        w(self, format!("@{} bra {};", p_exit, k_done));
        let stage = self.alloc_reg32();
        w(self, format!("add.u32 {}, {}, {};", stage, smem, read_off));
        // The stage offsets one tile on, wrapping at `stages`.
        let advance = |this: &mut Self, off: &str| -> String {
            let nx = this.alloc_reg32();
            let p_wrap = this.alloc_pred();
            writeln!(&mut this.ptx_buffer, "    add.u32 {}, {}, {};", nx, off, stage_bytes).unwrap();
            writeln!(&mut this.ptx_buffer, "    setp.eq.u32 {}, {}, {};", p_wrap, nx, smem_bytes).unwrap();
            writeln!(&mut this.ptx_buffer, "    selp.u32 {}, 0, {}, {};", nx, nx, p_wrap).unwrap();
            nx
        };
        let read_next = advance(self, &read_off);
        let write_next = advance(self, &write_off);
        let next_stage = self.alloc_reg32();
        w(self, format!("add.u32 {}, {}, {};", next_stage, smem, read_next));
        let (mut cur_a, mut cur_b) = (carried_a.clone(), carried_b.clone());
        for s in 0..k_substeps {
            let last = s + 1 == k_substeps;
            let (nxt_a, nxt_b) = if last {
                (Vec::new(), Vec::new())
            } else {
                let (fa, fb) = (fresh(self, num_i), fresh(self, num_j / 2));
                load_frags(self, &stage, s + 1, &fa, &fb);
                (fa, fb)
            };
            if s == 0 {
                let next = self.alloc_reg32();
                w(self, format!("add.u32 {}, {}, {};", next, it, stages - 1));
                let p_next = self.alloc_pred();
                w(self, format!("setp.lt.u32 {}, {}, {};", p_next, next, n_my));
                let k0 = self.alloc_reg32();
                w(self, format!("mad.lo.u32 {}, {}, {}, {};", k0, next, k_stride, k_base));
                let slot = self.alloc_reg32();
                w(self, format!("add.u32 {}, {}, {};", slot, s_thread, write_off));
                self.emit_fp8_e4m3_copy_tile(&copy, &k0, &slot, Some(&p_next));
                w(self, "cp.async.commit_group;".to_string());
            }
            if last && k_substeps > 1 {
                // Tile it+1's first fragments (published by the barrier one
                // substep back), ahead of these MMAs.
                load_frags(self, &next_stage, 0, &carried_a, &carried_b);
            }
            mmas(self, &cur_a, &cur_b);
            if s + 2 == k_substeps || k_substeps == 1 {
                w(self, format!("cp.async.wait_group {};", stages - 2));
                w(self, "bar.sync 0;".to_string());
            }
            if last && k_substeps == 1 {
                // One substep: the carried registers were its own, so they
                // are reloaded after its MMAs.
                load_frags(self, &next_stage, 0, &carried_a, &carried_b);
            }
            cur_a = nxt_a;
            cur_b = nxt_b;
        }
        w(self, format!("mov.u32 {}, {};", read_off, read_next));
        w(self, format!("mov.u32 {}, {};", write_off, write_next));
        w(self, format!("add.u32 {}, {}, 1;", it, it));
        w(self, format!("bra {};", k_top));
        writeln!(&mut self.ptx_buffer, "{}:", k_done).unwrap();
        // Every copy has landed and every warp is past its last ldmatrix
        // before the next tile's prologue writes the stages again.
        w(self, "cp.async.wait_all;".to_string());
        w(self, "bar.sync 0;".to_string());

        // ---- epilogue ----
        let row0 = self.alloc_reg32();
        w(self, format!("add.u32 {}, {}, {};", row0, cta_row0, warp_row0));
        w(self, format!("add.u32 {}, {}, {};", row0, row0, group_id));
        let col0 = self.alloc_reg32();
        w(self, format!("add.u32 {}, {}, {};", col0, cta_col0, warp_col0));
        w(self, format!("mad.lo.u32 {}, {}, 2, {};", col0, tig, col0));
        let row_bytes = self.alloc_reg64();
        w(self, format!("mul.wide.u32 {}, {}, {};", row_bytes, row0, n * 4));
        let col_bytes = self.alloc_reg64();
        w(self, format!("mul.wide.u32 {}, {}, 4;", col_bytes, col0));
        let c_thread = self.alloc_reg64();
        w(self, format!("add.s64 {}, {}, {};", c_thread, c_g, row_bytes));
        w(self, format!("add.s64 {}, {}, {};", c_thread, c_thread, col_bytes));
        // M % 16 == 0 and N % 8 == 0 (the shape refusal), so an m16 block of
        // rows and an n8 block of columns are in the matrix or out of it whole.
        let row_ok: Vec<Option<String>> = (0..num_i)
            .map(|i| {
                if m % cta_m == 0 {
                    return None;
                }
                let r = self.alloc_reg32();
                writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", r, row0, i * 16).unwrap();
                let p = self.alloc_pred();
                writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p, r, m).unwrap();
                Some(p)
            })
            .collect();
        let col_ok: Vec<Option<String>> = (0..num_j)
            .map(|j| {
                if n % cta_n == 0 {
                    return None;
                }
                let c = self.alloc_reg32();
                writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", c, col0, j * 8).unwrap();
                let p = self.alloc_pred();
                writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p, c, n).unwrap();
                Some(p)
            })
            .collect();
        let guard = |this: &mut Self, i: u32, j: u32| -> String {
            match (&row_ok[i as usize], &col_ok[j as usize]) {
                (None, None) => String::new(),
                (Some(p), None) | (None, Some(p)) => format!("@{} ", p),
                (Some(r), Some(c)) => {
                    let p = this.alloc_pred();
                    writeln!(&mut this.ptx_buffer, "    and.pred {}, {}, {};", p, r, c).unwrap();
                    format!("@{} ", p)
                }
            }
        };

        if let Some((gsa, gsb, gbi)) = &epi_g {
            // C[row][col] = f32(acc) * scale_a[row] * scale_b[col] + bias[col]:
            // `cvt.rn`, `mul`, then one `fma.rn` -- the register-tiled kernel's
            // arithmetic, so the two give the same bits.
            let mut sa = Vec::new();
            for i in 0..num_i {
                for half in [0u32, 8] {
                    let off64 = self.alloc_reg64();
                    let r = self.alloc_reg32();
                    w(self, format!("add.u32 {}, {}, {};", r, row0, i * 16 + half));
                    w(self, format!("mul.wide.u32 {}, {}, 4;", off64, r));
                    let addr = self.alloc_reg64();
                    w(self, format!("add.s64 {}, {}, {};", addr, gsa, off64));
                    let v = self.alloc_regf32();
                    let g = match &row_ok[i as usize] {
                        Some(p) => format!("@{} ", p),
                        None => String::new(),
                    };
                    w(self, format!("mov.f32 {}, 0f00000000;", v));
                    w(self, format!("{}ld.global.f32 {}, [{}];", g, v, addr));
                    sa.push(v);
                }
            }
            let mut sb = Vec::new();
            let mut bi = Vec::new();
            for j in 0..num_j {
                let off64 = self.alloc_reg64();
                let c = self.alloc_reg32();
                w(self, format!("add.u32 {}, {}, {};", c, col0, j * 8));
                w(self, format!("mul.wide.u32 {}, {}, 4;", off64, c));
                let (a1, a2) = (self.alloc_reg64(), self.alloc_reg64());
                w(self, format!("add.s64 {}, {}, {};", a1, gsb, off64));
                w(self, format!("add.s64 {}, {}, {};", a2, gbi, off64));
                let g = match &col_ok[j as usize] {
                    Some(p) => format!("@{} ", p),
                    None => String::new(),
                };
                let (s0, s1, b0, b1) = (self.alloc_regf32(), self.alloc_regf32(), self.alloc_regf32(), self.alloc_regf32());
                for r in [&s0, &s1, &b0, &b1] {
                    w(self, format!("mov.f32 {}, 0f00000000;", r));
                }
                // Scalar loads: the scales and bias need only 4-byte alignment.
                w(self, format!("{}ld.global.f32 {}, [{}];", g, s0, a1));
                w(self, format!("{}ld.global.f32 {}, [{}+4];", g, s1, a1));
                w(self, format!("{}ld.global.f32 {}, [{}];", g, b0, a2));
                w(self, format!("{}ld.global.f32 {}, [{}+4];", g, b1, a2));
                sb.push([s0, s1]);
                bi.push([b0, b1]);
            }
            for i in 0..num_i {
                for j in 0..num_j {
                    let g = guard(self, i, j);
                    for h in 0..2u32 {
                        let fr = &acc[i as usize][j as usize];
                        let mut v = Vec::new();
                        for e in 0..2usize {
                            let f = self.alloc_regf32();
                            w(self, format!("cvt.rn.f32.s32 {}, {};", f, fr[2 * h as usize + e]));
                            w(self, format!("mul.f32 {}, {}, {};", f, f, sa[(i * 2 + h) as usize]));
                            w(self, format!("fma.rn.f32 {}, {}, {}, {};", f, f, sb[j as usize][e], bi[j as usize][e]));
                            v.push(f);
                        }
                        let off = (u64::from(i * 16 + 8 * h) * u64::from(n) + u64::from(j * 8)) * 4;
                        w(self, format!("{}st.global.v2.f32 [{}+{}], {{{}, {}}};", g, c_thread, off, v[0], v[1]));
                    }
                }
            }
        } else {
            // A launch that does not split K (nctaid.z = 1) gives each output
            // to one CTA, which STORES it: the whole contraction, the value a
            // reduction into a zeroed C would leave. With K split over z
            // several CTAs own one output, so each REDUCES its partial into C
            // (zeroed by the caller); an integer add lands the same in any
            // order. A CTA whose stripe is empty adds nothing and skips them.
            // The stores are why this is a branch: 16.7M `red.global.add.s32`
            // at 4096^3 took a quarter of the kernel's time (0.74x cuBLASLt
            // against 0.97x storing).
            let p_none = self.alloc_pred();
            w(self, format!("setp.eq.u32 {}, {}, 0;", p_none, n_my));
            w(self, format!("@{} bra {};", p_none, tile_next));
            let reduce = self.alloc_label(&format!("I8S_REDUCE_{}", kernel_name));
            let p_split = self.alloc_pred();
            w(self, format!("setp.ne.u32 {}, {}, 1;", p_split, nz));
            w(self, format!("@{} bra {};", p_split, reduce));
            for reducing in [false, true] {
                if reducing {
                    writeln!(&mut self.ptx_buffer, "{}:", reduce).unwrap();
                }
                for i in 0..num_i {
                    for j in 0..num_j {
                        let g = guard(self, i, j);
                        for h in 0..2u32 {
                            let fr = &acc[i as usize][j as usize];
                            let off = (u64::from(i * 16 + 8 * h) * u64::from(n) + u64::from(j * 8)) * 4;
                            let (v0, v1) = (&fr[2 * h as usize], &fr[2 * h as usize + 1]);
                            if reducing {
                                w(self, format!("{}red.global.add.s32 [{}+{}], {};", g, c_thread, off, v0));
                                w(self, format!("{}red.global.add.s32 [{}+{}], {};", g, c_thread, off + 4, v1));
                            } else {
                                w(self, format!("{}st.global.v2.s32 [{}+{}], {{{}, {}}};", g, c_thread, off, v0, v1));
                            }
                        }
                    }
                }
                if !reducing {
                    w(self, format!("bra {};", tile_next));
                }
            }
        }
        writeln!(&mut self.ptx_buffer, "{}:", tile_next).unwrap();
        // The next tile id; a wrap past 2^32 (a grid of nearly 2^32 CTAs) ends
        // the loop rather than revisiting tile 0, which a reduction would add twice.
        let tile_nx = self.alloc_reg32();
        w(self, format!("add.u32 {}, {}, {};", tile_nx, tile, n_ctas));
        let p_wrapped = self.alloc_pred();
        w(self, format!("setp.lt.u32 {}, {}, {};", p_wrapped, tile_nx, tile));
        w(self, format!("@{} bra {};", p_wrapped, done));
        w(self, format!("mov.u32 {}, {};", tile, tile_nx));
        w(self, format!("bra {};", tile_top));
        writeln!(&mut self.ptx_buffer, "{}:", done).unwrap();
        w(self, "ret;".to_string());
        threads
    }

    /// The output tile a linear tile id names, in grouped raster order:
    /// consecutive ids cover `group_m` M tiles of one N tile, then the next
    /// N tile, so the CTAs resident together share A's and B's tiles in L2.
    /// The last group holds the M tiles left (`min(group_m, tiles_m - first)`).
    /// Returns `(tile_m, tile_n)` registers.
    fn emit_int8_grouped_tile(&mut self, t: &str, tiles_m: u32, tiles_n: u32, group_m: u32) -> (String, String) {
        let (tm, tn) = (self.alloc_reg32(), self.alloc_reg32());
        if tiles_m == 1 {
            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, 0;", tm).unwrap();
            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", tn, t).unwrap();
            return (tm, tn);
        }
        if tiles_n == 1 {
            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", tm, t).unwrap();
            writeln!(&mut self.ptx_buffer, "    mov.u32 {}, 0;", tn).unwrap();
            return (tm, tn);
        }
        let per_group = group_m * tiles_n;
        let group = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    div.u32 {}, {}, {};", group, t, per_group).unwrap();
        let first = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    mul.lo.u32 {}, {}, {};", first, group, group_m).unwrap();
        let gsize = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    sub.u32 {}, {}, {};", gsize, tiles_m, first).unwrap();
        writeln!(&mut self.ptx_buffer, "    min.u32 {}, {}, {};", gsize, gsize, group_m).unwrap();
        let r = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    rem.u32 {}, {}, {};", r, t, per_group).unwrap();
        let rm = self.alloc_reg32();
        writeln!(&mut self.ptx_buffer, "    rem.u32 {}, {}, {};", rm, r, gsize).unwrap();
        writeln!(&mut self.ptx_buffer, "    add.u32 {}, {}, {};", tm, first, rm).unwrap();
        writeln!(&mut self.ptx_buffer, "    div.u32 {}, {}, {};", tn, r, gsize).unwrap();
        (tm, tn)
    }
}
