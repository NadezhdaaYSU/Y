//! The FP32 GEMM on CUDA cores: `@tile(M, N, K) kernel k(A, B, C:
//! GlobalMemory<F32>)`, `C = A B` with A `[M][K]`, B `[K][N]`, C `[M][N]`, all
//! row-major f32.
//!
//! **What it is for: matching cuBLAS, so a Y program does not trade speed for
//! Y's checks.** The design is the one cuBLAS itself runs at these shapes
//! (CUTLASS's SIMT sgemm, `cutlass_80_simt_sgemm_256x128_8x4_nn_align1`):
//! a multistage `cp.async` pipeline, A transposed into shared memory by 4-byte
//! copies, B by 16-byte copies, each thread an outer product of float4
//! fragments. This kernel against cuBLAS (TF32 off) on an RTX 4070 Ti SUPER:
//! 0.93-1.04x from 2048 to 8192, 1.19-1.20x at 512 and 1024, 0.94-1.04x on
//! 64-1024-row and 128-column shapes, 0.81-0.85x at 768^3 where cuBLAS
//! splits K (`docs/benchmarks/f32_gemm_2026-11-05/`).
//!
//! What made it competitive, in the order it was found (each measured; the
//! last three are about how ptxas schedules this emitter's PTX, and are
//! commented where they are emitted):
//! - registers: a fragment double buffer that pushed a 128x128 kernel past 128
//!   registers halved its occupancy and lost 10-20%;
//! - **compile-time shapes**: with M, N and K constants, each copy is one
//!   `cp.async` with immediate offsets from a pointer advanced once per K tile.
//!   Computing the addresses at run time put ~150 integer instructions in a
//!   row at every tile and cost 6-12%;
//! - BK = 16 rather than 8, one barrier per 2048 FMAs;
//! - grouped raster order (8 tile rows), which keeps B's columns in L2:
//!   +3-5% at 4096-8192;
//! - a fresh fragment register per k step, all of a tile's loads before its
//!   FMAs, and branches (not predicates) around the copies and the pointer
//!   advance: 0.87x -> 1.01x at 4096, equal to NVVM's code for the same
//!   kernel.
//!
//! **Every output element is one `fma.rn.f32` chain over k = 0..K-1, in that
//! order**, whichever tile, grid or shape: the bits of `C[i][j]` depend on row
//! i of A and column j of B and nothing else - not on M, not on the tile the
//! shape selects, not on the launch. That is what `f32::mul_add` folded in
//! the same order computes, so the tests compare bit for bit. It is also why
//! there is no split-K here: splitting K changes the summation order, so the
//! answer would depend on the shape.
use super::*;

/// One tile configuration. All sizes are powers of two, so the thread
/// decomposition is shifts and masks (ptxas does not reduce a division by an
/// immediate).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct F32GemmTile {
    pub name: &'static str,
    /// CTA tile.
    pub bm: u32,
    pub bn: u32,
    pub bk: u32,
    /// Warp tile and each thread's outputs (multiples of 4: float4 fragments).
    pub wm: u32,
    pub wn: u32,
    pub tm: u32,
    pub tn: u32,
    /// `cp.async` pipeline depth.
    pub stages: u32,
    /// CTAs per SM the register budget (`.maxnreg`) is sized for. Shared
    /// memory may allow fewer; this is the register cap the configuration was
    /// measured with.
    pub reg_ctas: u32,
}

impl F32GemmTile {
    pub(crate) fn threads(&self) -> u32 {
        (self.bm / self.wm) * (self.bn / self.wn) * 32
    }
    /// Dynamic shared memory: `stages` copies of A^T (padded rows) and B.
    pub(crate) fn smem_bytes(&self) -> u32 {
        self.stages * self.bk * (self.bm + F32_GEMM_PAD_A + self.bn) * 4
    }
}

/// A's shared rows are padded by 4 floats: the 8 or 16 threads that copy one
/// row of A write 8 or 16 different rows of A^T, which without the pad all
/// fall in one bank. 4 keeps every fragment load 16-byte aligned.
const F32_GEMM_PAD_A: u32 = 4;
/// Grouped raster: this many tile rows share B's columns in L2.
const F32_GEMM_GROUP: u32 = 8;

/// Large shapes: 8 warps of 64x64, 8x16 outputs per thread, one CTA per SM.
pub(crate) const F32_TILE_LARGE: F32GemmTile =
    F32GemmTile { name: "large", bm: 128, bn: 256, bk: 16, wm: 64, wn: 64, tm: 8, tn: 16, stages: 3, reg_ctas: 1 };
/// Mid and skinny shapes: 4 warps of 32x64, 8x8 per thread.
pub(crate) const F32_TILE_MID: F32GemmTile =
    F32GemmTile { name: "mid", bm: 64, bn: 128, bk: 16, wm: 32, wn: 64, tm: 8, tn: 8, stages: 3, reg_ctas: 3 };
/// Small shapes: 4 warps of 32x32, 4x8 per thread.
pub(crate) const F32_TILE_SMALL: F32GemmTile =
    F32GemmTile { name: "small", bm: 64, bn: 64, bk: 16, wm: 32, wn: 32, tm: 4, tn: 8, stages: 3, reg_ctas: 4 };
impl PtxEmitter {
    /// The tile for an `m x n x k` FP32 GEMM on `sms` SMs, or why there is none.
    ///
    /// The rule was read off measurements (the module doc): the large tile
    /// needs about two waves of its one-CTA-per-SM tiles, the mid tile about
    /// one wave, and the small tile takes the rest. The choice changes speed
    /// only, never a bit of the answer.
    pub(crate) fn f32_gemm_tile(m: u32, n: u32, k: u32, sms: u32) -> Result<F32GemmTile, String> {
        if m == 0 || n == 0 || k == 0 || m % 64 != 0 || n % 64 != 0 || k % 16 != 0 {
            return Err(format!(
                "an FP32 GEMM needs M % 64 == 0, N % 64 == 0 and K % 16 == 0 (its smallest tile is \
                 64x64x16; a ragged edge is not handled, and padding the shape would compute a \
                 different matrix); got M={}, N={}, K={}",
                m, n, k
            ));
        }
        // The copies address A, B and C with 32-bit immediate offsets from
        // per-tile 64-bit bases; keep every offset inside i32.
        if (k as u64) * 4 * 128 >= 1 << 31 || (n as u64) * 4 * 128 >= 1 << 31 {
            return Err(format!(
                "an FP32 GEMM addresses one tile's rows with 32-bit offsets, so K and N must stay \
                 below 2^22; got N={}, K={}",
                n, k
            ));
        }
        let sms = if sms == 0 { 66 } else { sms } as u64;
        let tiles = |t: &F32GemmTile| (m / t.bm) as u64 * (n / t.bn) as u64;
        let fits = |t: &F32GemmTile| m % t.bm == 0 && n % t.bn == 0;
        if fits(&F32_TILE_LARGE) && tiles(&F32_TILE_LARGE) * 10 >= sms * 19 {
            return Ok(F32_TILE_LARGE);
        }
        if fits(&F32_TILE_MID) && tiles(&F32_TILE_MID) * 10 >= sms * 9 {
            return Ok(F32_TILE_MID);
        }
        Ok(F32_TILE_SMALL)
    }

    /// Recognise `@tile(M, N, K) kernel f(A, B, C: GlobalMemory<F32>)`.
    /// `type_checker::check_tile_gemm_signature` accepts the same shape; the
    /// two must agree (a shape accepted there and not recognised here would be
    /// lowered as an empty body).
    pub(crate) fn tile_gemm_f32_operands(&self, kernel: &KernelDecl) -> Option<(u32, u32, u32, String, String, String)> {
        let tile = kernel.tile.as_ref()?;
        fn as_positive_u32(e: &Expr) -> Option<u32> {
            match e {
                Expr::IntLit(v, _) if *v > 0 => u32::try_from(*v).ok(),
                _ => None,
            }
        }
        let m = as_positive_u32(&tile.block_m)?;
        let n = as_positive_u32(&tile.block_n)?;
        let k = as_positive_u32(tile.block_k.as_deref()?)?;
        fn is_global_f32(ty: &Type) -> bool {
            matches!(
                ty,
                Type::Generic { base, args, .. }
                    if base == "GlobalMemory"
                        && matches!(args.as_slice(), [GenericArg::Type(Type::Primitive(p, _))] if p == "F32")
            )
        }
        if kernel.params.len() != 3 || !kernel.params.iter().all(|p| is_global_f32(&p.ty)) {
            return None;
        }
        let a = self.variables.get(&kernel.params[0].name)?.clone();
        let b = self.variables.get(&kernel.params[1].name)?.clone();
        let c = self.variables.get(&kernel.params[2].name)?.clone();
        Some((m, n, k, a, b, c))
    }

    /// `log2` of a power of two, for the shift-and-mask decomposition.
    fn f32_gemm_log2(v: u32) -> u32 {
        assert!(v.is_power_of_two(), "F32 GEMM tile sizes are powers of two, got {}", v);
        v.trailing_zeros()
    }

    /// Emit the kernel body. Returns the CTA's thread count (`emit_kernel`
    /// sizes `.maxnreg` from it).
    pub(crate) fn emit_f32_simt_gemm_kernel(
        &mut self,
        m: u32,
        n: u32,
        k: u32,
        a_ptr: &str,
        b_ptr: &str,
        c_ptr: &str,
        hw_profile: &HardwareProfile,
        kernel_name: &str,
    ) -> u32 {
        let t = match Self::f32_gemm_tile(m, n, k, hw_profile.sm_count) {
            Ok(t) => t,
            Err(why) => {
                self.emit_errors.push(format!("[PTX] `{}`: {}", kernel_name, why));
                return 32;
            }
        };
        let nt = t.threads();
        let (lm, ln) = (t.wm / t.tm, t.wn / t.tn);
        let sa_stride = t.bm + F32_GEMM_PAD_A; // floats per row of A^T
        let kt_count = k / t.bk;
        let a_copies = t.bm * t.bk / nt; // 4-byte copies of A per thread per stage
        let b_copies = t.bk * t.bn / 4 / nt; // 16-byte copies of B
        let copies = a_copies + b_copies;
        // The copies are spread one per k step, all issued before the
        // commit at k = BK - 2.
        assert!(
            lm * ln == 32 && t.tm % 4 == 0 && t.tn % 4 == 0 && nt % t.bk == 0 && nt % (t.bn / 4) == 0 && copies <= t.bk - 1,
            "F32 GEMM tile {:?} is malformed",
            t
        );
        let (nbx, nby) = (n / t.bn, m / t.bm);
        let tiles = nbx * nby;
        let stage_a = t.bk * sa_stride * 4; // bytes
        let stage_b = t.bk * t.bn * 4;
        let smem = t.smem_bytes();
        let s = t.stages;

        writeln!(
            &mut self.ptx_buffer,
            "    // ========================================================\n\
             \x20   // [Y F32 SIMT GEMM] M={m} N={n} K={k} | CTA {}x{}x{} | {}x{} warps of {}x{}, {}x{} outputs per thread | tile `{}`\n\
             \x20   // {} stages of cp.async, A transposed by 4-byte copies, B by 16-byte copies, grouped raster of {} tile rows.\n\
             \x20   // Each C element is one fma.rn.f32 chain over k = 0..K-1 in order: the same bits for every shape, tile and grid.\n\
             \x20   // Launch: block ({nt},1,1), anything else traps. {tiles} output tiles, grid-strided: ANY grid works, ({nbx}, {nby}, 1) is one tile per CTA.\n\
             \x20   // A row-major [M][K], B row-major [K][N], C row-major [M][N], f32. Dynamic shared memory required: {smem} bytes.\n\
             \x20   // ========================================================",
            t.bm, t.bn, t.bk, t.bm / t.wm, t.bn / t.wn, t.wm, t.wn, t.tm, t.tn, t.name, s, F32_GEMM_GROUP
        )
        .unwrap();

        let smem_sym = format!("smem_f32_gemm_{}", kernel_name);
        self.pending_extern_decls.push(format!(".extern .shared .align 16 .b8 {}[];", smem_sym));
        self.tile_gemm_ctas_per_sm = Some(t.reg_ctas);

        let w = |e: &mut Self, line: String| {
            e.ptx_buffer.push_str("    ");
            e.ptx_buffer.push_str(&line);
            e.ptx_buffer.push('\n');
        };

        // ---- Global bases and the launch contract.
        let (ga0, gb0, gc0) = (self.alloc_reg64(), self.alloc_reg64(), self.alloc_reg64());
        w(self, format!("cvta.to.global.u64 {}, {};", ga0, a_ptr));
        w(self, format!("cvta.to.global.u64 {}, {};", gb0, b_ptr));
        w(self, format!("cvta.to.global.u64 {}, {};", gc0, c_ptr));
        let (tid, tmp) = (self.alloc_reg32(), self.alloc_reg32());
        let bad = self.alloc_pred();
        // Every thread has a copy role and an output tile: a block of any
        // other size leaves part of C unwritten or reads past a tile, and a
        // larger one would split `bar.sync` across threads with no role.
        for (sreg, want) in [("%ntid.x", nt), ("%ntid.y", 1), ("%ntid.z", 1)] {
            w(self, format!("mov.u32 {}, {};", tmp, sreg));
            w(self, format!("setp.ne.u32 {}, {}, {};", bad, tmp, want));
            w(self, format!("@{} trap;", bad));
        }
        w(self, format!("mov.u32 {}, %tid.x;", tid));

        // ---- Thread roles (all from tid; sizes are powers of two).
        let r = |e: &mut Self| e.alloc_reg32();
        let (warp, lane, wm_i, wn_i, lm_i, ln_i) = (r(self), r(self), r(self), r(self), r(self), r(self));
        w(self, format!("shr.u32 {}, {}, 5;", warp, tid));
        w(self, format!("and.b32 {}, {}, 31;", lane, tid));
        let wgm = t.bm / t.wm;
        w(self, format!("and.b32 {}, {}, {};", wm_i, warp, wgm - 1));
        w(self, format!("shr.u32 {}, {}, {};", wn_i, warp, Self::f32_gemm_log2(wgm)));
        w(self, format!("shr.u32 {}, {}, {};", lm_i, lane, Self::f32_gemm_log2(ln)));
        w(self, format!("and.b32 {}, {}, {};", ln_i, lane, ln - 1));

        let (s_a, s_b) = (r(self), r(self));
        w(self, format!("mov.u32 {}, {};", s_a, smem_sym));
        w(self, format!("add.u32 {}, {}, {};", s_b, s_a, s * stage_a));
        // Fragment bases: this thread's float4 columns of A^T and B.
        let (frag_a, frag_b) = (r(self), r(self));
        w(self, format!("mul.lo.u32 {}, {}, {};", tmp, wm_i, t.wm * 4));
        w(self, format!("mad.lo.u32 {}, {}, 16, {};", tmp, lm_i, tmp));
        w(self, format!("add.u32 {}, {}, {};", frag_a, s_a, tmp));
        w(self, format!("mul.lo.u32 {}, {}, {};", tmp, wn_i, t.wn * 4));
        w(self, format!("mad.lo.u32 {}, {}, 16, {};", tmp, ln_i, tmp));
        w(self, format!("add.u32 {}, {}, {};", frag_b, s_b, tmp));
        // Copy roles. A: element (row tid/BK + c*(NT/BK), col tid%BK) of the
        // tile, written transposed. B: 16 bytes at (row tid/(BN/4) +
        // i*(NT/(BN/4)), col (tid%(BN/4))*4).
        let (row_a, col_a, row_b, col_b, copy_a, copy_b) = (r(self), r(self), r(self), r(self), r(self), r(self));
        w(self, format!("shr.u32 {}, {}, {};", row_a, tid, Self::f32_gemm_log2(t.bk)));
        w(self, format!("and.b32 {}, {}, {};", col_a, tid, t.bk - 1));
        w(self, format!("shr.u32 {}, {}, {};", row_b, tid, Self::f32_gemm_log2(t.bn / 4)));
        w(self, format!("and.b32 {}, {}, {};", col_b, tid, t.bn / 4 - 1));
        w(self, format!("shl.b32 {}, {}, 2;", col_b, col_b));
        // Shared byte offsets of the copies within a stage.
        w(self, format!("mad.lo.u32 {}, {}, {}, {};", tmp, col_a, sa_stride, row_a));
        w(self, format!("shl.b32 {}, {}, 2;", tmp, tmp));
        w(self, format!("add.u32 {}, {}, {};", copy_a, s_a, tmp));
        w(self, format!("mad.lo.u32 {}, {}, {}, {};", tmp, row_b, t.bn, col_b));
        w(self, format!("shl.b32 {}, {}, 2;", tmp, tmp));
        w(self, format!("add.u32 {}, {}, {};", copy_b, s_b, tmp));

        // ---- Tile loop: linear tile id, grid-strided.
        let (pid, stride) = (r(self), r(self));
        let (cx, cy, cz, ncx, ncy, ncz) = (r(self), r(self), r(self), r(self), r(self), r(self));
        for (reg, sreg) in [(&cx, "%ctaid.x"), (&cy, "%ctaid.y"), (&cz, "%ctaid.z"), (&ncx, "%nctaid.x"), (&ncy, "%nctaid.y"), (&ncz, "%nctaid.z")] {
            w(self, format!("mov.u32 {}, {};", reg, sreg));
        }
        w(self, format!("mad.lo.u32 {}, {}, {}, {};", pid, cz, ncy, cy));
        w(self, format!("mad.lo.u32 {}, {}, {}, {};", pid, pid, ncx, cx));
        w(self, format!("mul.lo.u32 {}, {}, {};", stride, ncx, ncy));
        w(self, format!("mul.lo.u32 {}, {}, {};", stride, stride, ncz));

        let acc: Vec<Vec<String>> = (0..t.tm).map(|_| (0..t.tn).map(|_| self.alloc_regf32()).collect()).collect();
        // A fresh fragment register for every k step of the tile, never two
        // buffers reused: reuse ties each load to the end of the FMAs that
        // read the buffer before, and ptxas then issued the shared loads in
        // clusters - MIO throttle 0.19 against 0.03 and 0.87x cuBLAS at 4096,
        // where the same kernel with fresh registers (NVVM's SSA output) ran
        // at 1.04x. ptxas allocates the physical registers either way.
        let fa: Vec<Vec<String>> = (0..t.bk).map(|_| (0..t.tm).map(|_| self.alloc_regf32()).collect()).collect();
        let fb: Vec<Vec<String>> = (0..t.bk).map(|_| (0..t.tn).map(|_| self.alloc_regf32()).collect()).collect();

        let tile_top = self.alloc_label("f32_gemm_tile");
        let k_top = self.alloc_label("f32_gemm_k");
        let done = self.alloc_label("f32_gemm_done");
        let (p_done, p_more, p_loop, p_wrap) = (self.alloc_pred(), self.alloc_pred(), self.alloc_pred(), self.alloc_pred());
        let (by, bx, grp, first, gm, rr) = (r(self), r(self), r(self), r(self), r(self), r(self));
        let (ga, gb) = (self.alloc_reg64(), self.alloc_reg64());
        let (off64, off64b) = (self.alloc_reg64(), self.alloc_reg64());
        let (kt, rs, ws, rd_a, rd_b, wr_a, wr_b) = (r(self), r(self), r(self), r(self), r(self), r(self), r(self));

        self.ptx_buffer.push_str(&format!("{}:\n", tile_top));
        w(self, format!("setp.ge.u32 {}, {}, {};", p_done, pid, tiles));
        w(self, format!("@{} bra {};", p_done, done));
        // Grouped raster: tile rows in groups of GROUP, column-major inside a
        // group, so consecutive CTAs share B's columns.
        let gsz = F32_GEMM_GROUP * nbx;
        w(self, format!("div.u32 {}, {}, {};", grp, pid, gsz));
        w(self, format!("mul.lo.u32 {}, {}, {};", first, grp, F32_GEMM_GROUP));
        w(self, format!("sub.u32 {}, {}, {};", gm, nby, first));
        w(self, format!("min.u32 {}, {}, {};", gm, gm, F32_GEMM_GROUP));
        w(self, format!("mul.lo.u32 {}, {}, {};", tmp, grp, gsz));
        w(self, format!("sub.u32 {}, {}, {};", rr, pid, tmp));
        w(self, format!("rem.u32 {}, {}, {};", by, rr, gm));
        w(self, format!("add.u32 {}, {}, {};", by, by, first));
        w(self, format!("div.u32 {}, {}, {};", bx, rr, gm));

        // Per-tile global bases of this thread's copies.
        w(self, format!("mad.lo.u32 {}, {}, {}, {};", tmp, by, t.bm, row_a));
        w(self, format!("mul.wide.u32 {}, {}, {};", off64, tmp, k * 4));
        w(self, format!("mul.wide.u32 {}, {}, 4;", off64b, col_a));
        w(self, format!("add.u64 {}, {}, {};", off64, off64, off64b));
        w(self, format!("add.u64 {}, {}, {};", ga, ga0, off64));
        w(self, format!("mul.wide.u32 {}, {}, {};", off64, row_b, n * 4));
        w(self, format!("mad.lo.u32 {}, {}, {}, {};", tmp, bx, t.bn, col_b));
        w(self, format!("mul.wide.u32 {}, {}, 4;", off64b, tmp));
        w(self, format!("add.u64 {}, {}, {};", off64, off64, off64b));
        w(self, format!("add.u64 {}, {}, {};", gb, gb0, off64));

        for row in &acc {
            for a in row {
                w(self, format!("mov.f32 {}, 0f00000000;", a));
            }
        }

        // One copy of this thread's share of stage `stage_off` (bytes, as an
        // immediate) from the K tile `g_off` bytes along.
        let a_row_step = nt / t.bk; // rows of A between copy c and c+1
        let b_row_step = nt / (t.bn / 4);
        let copy = |e: &mut Self, c: u32, pred: Option<&str>, sa: &str, sb: &str, ga_r: &str, gb_r: &str, s_off_a: u32, s_off_b: u32, g_off_a: u64, g_off_b: u64| {
            let guard = pred.map_or(String::new(), |p| format!("@{} ", p));
            if c < a_copies {
                let so = s_off_a + c * a_row_step * 4;
                let go = g_off_a + (c * a_row_step) as u64 * k as u64 * 4;
                w(e, format!("{}cp.async.ca.shared.global [{}+{}], [{}+{}], 4;", guard, sa, so, ga_r, go));
            } else {
                let i = c - a_copies;
                let so = s_off_b + i * b_row_step * t.bn * 4;
                let go = g_off_b + (i * b_row_step) as u64 * n as u64 * 4;
                w(e, format!("{}cp.async.cg.shared.global [{}+{}], [{}+{}], 16;", guard, sb, so, gb_r, go));
            }
        };

        // ---- Prologue: stages 0..S-2.
        for st in 0..s - 1 {
            if st < kt_count {
                for c in 0..copies {
                    copy(self, c, None, &copy_a, &copy_b, &ga, &gb, st * stage_a, st * stage_b, (st * t.bk * 4) as u64, st as u64 * t.bk as u64 * n as u64 * 4);
                }
            }
            w(self, "cp.async.commit_group;".to_string());
        }
        // The next K tile to load is S-1 tiles in.
        w(self, format!("add.u64 {}, {}, {};", ga, ga, (s - 1) * t.bk * 4));
        w(self, format!("add.u64 {}, {}, {};", gb, gb, (s - 1) as u64 * t.bk as u64 * n as u64 * 4));
        w(self, format!("mov.u32 {}, 0;", kt));
        w(self, format!("mov.u32 {}, 0;", rs));
        w(self, format!("mov.u32 {}, {};", ws, s - 1));

        // ---- K loop.
        self.ptx_buffer.push_str(&format!("{}:\n", k_top));
        w(self, format!("cp.async.wait_group {};", s - 2));
        w(self, "bar.sync 0;".to_string());
        // `more`: this iteration loads K tile kt + S - 1.
        let last_load = kt_count as i64 - (s as i64 - 1);
        if last_load > 0 {
            w(self, format!("setp.lt.u32 {}, {}, {};", p_more, kt, last_load));
        } else {
            w(self, format!("setp.ne.u32 {}, {}, {};", p_more, kt, kt));
        }
        w(self, format!("mad.lo.u32 {}, {}, {}, {};", rd_a, rs, stage_a, frag_a));
        w(self, format!("mad.lo.u32 {}, {}, {}, {};", rd_b, rs, stage_b, frag_b));
        w(self, format!("mad.lo.u32 {}, {}, {}, {};", wr_a, ws, stage_a, copy_a));
        w(self, format!("mad.lo.u32 {}, {}, {}, {};", wr_b, ws, stage_b, copy_b));

        let frag = |e: &mut Self, buf: usize, kk: u32| {
            for c in 0..t.tm / 4 {
                let off = (kk * sa_stride + c * lm * 4) * 4;
                let r4 = &fa[buf][(c * 4) as usize..(c * 4 + 4) as usize];
                w(e, format!("ld.shared.v4.f32 {{{}, {}, {}, {}}}, [{}+{}];", r4[0], r4[1], r4[2], r4[3], rd_a, off));
            }
            for c in 0..t.tn / 4 {
                let off = (kk * t.bn + c * ln * 4) * 4;
                let r4 = &fb[buf][(c * 4) as usize..(c * 4 + 4) as usize];
                w(e, format!("ld.shared.v4.f32 {{{}, {}, {}, {}}}, [{}+{}];", r4[0], r4[1], r4[2], r4[3], rd_b, off));
            }
        };
        // Every shared load and copy of the tile first, then every FMA:
        // the order NVVM writes for the same kernel, which leaves ptxas free
        // to interleave them (it does; the emitted text order is not the
        // issue order).
        //
        // The copies and the pointer advance sit behind BRANCHES, not
        // predicates. Measured at 4096^3 against the same kernel from NVVM
        // (which branches): predicated, 0.92x cuBLAS; predicated copies with
        // an unconditional advance, 0.97x; both branched, 1.01-1.02x, equal to
        // NVVM's. It is the predicated 64-bit adds that ptxas schedules badly.
        for kk in 0..t.bk {
            frag(self, kk as usize, kk);
            if kk < copies {
                let skip = self.alloc_label("f32_gemm_nocopy");
                w(self, format!("@!{} bra {};", p_more, skip));
                copy(self, kk, None, &wr_a, &wr_b, &ga, &gb, 0, 0, 0, 0);
                self.ptx_buffer.push_str(&format!("{}:\n", skip));
            }
            if kk == t.bk - 2 {
                let skip = self.alloc_label("f32_gemm_noadvance");
                w(self, format!("@!{} bra {};", p_more, skip));
                w(self, format!("add.u64 {}, {}, {};", ga, ga, t.bk * 4));
                w(self, format!("add.u64 {}, {}, {};", gb, gb, t.bk as u64 * n as u64 * 4));
                self.ptx_buffer.push_str(&format!("{}:\n", skip));
                w(self, "cp.async.commit_group;".to_string());
            }
        }
        // k outer: NVVM's order (each output 4 k steps at a time) was measured
        // too, with groups of 2, 4, 8 and 16; none was faster than this.
        for kk in 0..t.bk as usize {
            for i in 0..t.tm as usize {
                for j in 0..t.tn as usize {
                    w(self, format!("fma.rn.f32 {0}, {1}, {2}, {0};", acc[i][j], fa[kk][i], fb[kk][j]));
                }
            }
        }
        for reg in [&rs, &ws] {
            w(self, format!("add.u32 {}, {}, 1;", reg, reg));
            w(self, format!("setp.eq.u32 {}, {}, {};", p_wrap, reg, s));
            w(self, format!("selp.u32 {}, 0, {}, {};", reg, reg, p_wrap));
        }
        w(self, format!("add.u32 {}, {}, 1;", kt, kt));
        w(self, format!("setp.lt.u32 {}, {}, {};", p_loop, kt, kt_count));
        w(self, format!("@{} bra {};", p_loop, k_top));

        // ---- Epilogue: float4 stores of this thread's TM x TN outputs.
        let gcb = self.alloc_reg64();
        w(self, format!("mul.lo.u32 {}, {}, {};", tmp, wm_i, t.wm));
        w(self, format!("mad.lo.u32 {}, {}, 4, {};", tmp, lm_i, tmp));
        w(self, format!("mad.lo.u32 {}, {}, {}, {};", tmp, by, t.bm, tmp));
        w(self, format!("mul.wide.u32 {}, {}, {};", off64, tmp, n * 4));
        w(self, format!("mul.lo.u32 {}, {}, {};", tmp, wn_i, t.wn));
        w(self, format!("mad.lo.u32 {}, {}, 4, {};", tmp, ln_i, tmp));
        w(self, format!("mad.lo.u32 {}, {}, {}, {};", tmp, bx, t.bn, tmp));
        w(self, format!("mul.wide.u32 {}, {}, 4;", off64b, tmp));
        w(self, format!("add.u64 {}, {}, {};", off64, off64, off64b));
        w(self, format!("add.u64 {}, {}, {};", gcb, gc0, off64));
        for ci in 0..t.tm / 4 {
            for rw in 0..4 {
                let i = (ci * 4 + rw) as usize;
                for cj in 0..t.tn / 4 {
                    let off = ((ci * lm * 4 + rw) as u64 * n as u64 + (cj * ln * 4) as u64) * 4;
                    let j = (cj * 4) as usize;
                    w(self, format!(
                        "st.global.v4.f32 [{}+{}], {{{}, {}, {}, {}}};",
                        gcb, off, acc[i][j], acc[i][j + 1], acc[i][j + 2], acc[i][j + 3]
                    ));
                }
            }
        }
        // Every warp is done with the stages before the next tile's prologue
        // writes them.
        w(self, "bar.sync 0;".to_string());
        w(self, format!("add.u32 {}, {}, {};", pid, pid, stride));
        w(self, format!("bra.uni {};", tile_top));
        self.ptx_buffer.push_str(&format!("{}:\n", done));
        nt
    }
}
