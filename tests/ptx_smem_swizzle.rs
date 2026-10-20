//! The F16 GEMM's swizzled shared-memory tiles.
//!
//! The tensor-core GEMM padded each shared-memory row by 8 elements, which
//! keeps `ldmatrix` free of bank conflicts but made every warp's `cp.async`
//! copies read 1.5 times the L2 sectors they need (24 for each 16-sector copy
//! instruction, where cuBLAS's read 16). `SmemTile::swizzled` packs the rows
//! and XORs a row's bits into each chunk's index instead. This file checks the
//! layout's own properties, and that a GEMM emitted with it computes, on the
//! device and bit for bit, what the same GEMM emitted with the padded layout
//! computes, on every path that writes or reads the tiles: the synchronous
//! single-stage copy, the hoisted `cp.async` copy (one and several chunks a
//! thread), the per-chunk copy a tail or the swizzle period forces, two and
//! more stages, and the bias + ReLU epilogue.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

use y::ptx_emitter::SmemTile;

#[path = "common/pinned.rs"]
mod pinned;

/// The checks a layout has to pass, as a list of what fails: every chunk of
/// a row stored inside that row and at a different place; 8 consecutive rows
/// from a multiple of 8 putting each chunk in 8 different 16-byte slots of a
/// 128-byte line (an `ldmatrix` matrix, conflict-free); and each 32
/// consecutive chunks in row-major order, a warp's copies, filling whole
/// 128-byte lines.
fn layout_faults(t: &SmemTile, cols: u32) -> Vec<String> {
    let chunks = cols / 8;
    let row_bytes = t.stride * 2;
    let mut faults = Vec::new();
    for row in 0..64 {
        let mut offs: Vec<u32> = (0..chunks).map(|c| t.chunk_offset(row, c)).collect();
        if offs.iter().any(|&o| o < row * row_bytes || o + 16 > row * row_bytes + cols * 2) {
            faults.push(format!("row {row} stores a chunk outside its {cols} columns"));
        }
        offs.sort_unstable();
        offs.dedup();
        if offs.len() != chunks as usize {
            faults.push(format!("row {row} stores two chunks in one place"));
        }
    }
    for r0 in (0..64).step_by(8) {
        for c in 0..chunks {
            let mut slots: Vec<u32> = (r0..r0 + 8).map(|r| t.chunk_offset(r, c) / 16 % 8).collect();
            slots.sort_unstable();
            slots.dedup();
            if slots.len() != 8 {
                faults.push(format!("rows {r0}..{} chunk {c}: {} slots, a bank conflict", r0 + 8, slots.len()));
            }
        }
    }
    for first in (0..64 * chunks).step_by(32) {
        let mut lines: BTreeMap<u32, u32> = BTreeMap::new();
        for i in first..first + 32 {
            *lines.entry(t.chunk_offset(i / chunks, i % chunks) / 128).or_insert(0) += 1;
        }
        if lines.values().any(|&n| n != 8) {
            faults.push(format!("the warp copying chunks {first}.. fills lines partly: {lines:?}"));
        }
    }
    faults
}

#[test]
fn the_swizzle_permutes_each_row_without_conflicts_on_either_side() {
    for cols in [16u32, 32, 64, 128, 256] {
        let t = SmemTile::swizzled(cols).expect("a power of two of at least 2 chunks");
        assert_eq!((t.stride, t.period()), (cols, 8), "cols {cols}");
        let faults = layout_faults(&t, cols);
        assert!(faults.is_empty(), "cols {cols}: {faults:#?}");
        for r in 0..56 {
            for c in 0..cols / 8 {
                assert_eq!(t.chunk_offset(r + 8, c), t.chunk_offset(r, c) + 8 * cols * 2, "cols {cols}: no period 8");
            }
        }
    }
    // The control: the padded layout is conflict-free for ldmatrix, and fails
    // only the copy check - the 1.5x sectors.
    for cols in [32u32, 64, 128] {
        let faults = layout_faults(&SmemTile::padded(cols, 8), cols);
        assert!(!faults.is_empty() && faults.iter().all(|f| f.contains("fills lines partly")), "cols {cols}: {faults:#?}");
    }
    // An unpacked identity layout passes the copies and fails ldmatrix.
    let flat = SmemTile { stride: 64, swizzle: None };
    assert!(layout_faults(&flat, 64).iter().any(|f| f.contains("bank conflict")));
    for cols in [8u32, 20, 24, 48, 96] {
        assert_eq!(SmemTile::swizzled(cols), None, "cols {cols}");
    }
}

/// With no knob set a GEMM gets the swizzled tiles; `Y_SMEM_SWIZZLE=0` and
/// `Y_SMEM_PAD`, the knobs that A/B the layouts, give the padded ones.
#[test]
fn the_default_layout_is_the_swizzled_one() {
    let dir = pinned::pinned_scratch("smem_swizzle_default", pinned::SM_PINNED);
    let src = pinned::copy_fixture(&dir, "tests/gemm_f16_512.ysu");
    let emit = |knobs: &[(&str, &str)]| {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_Y"));
        cmd.arg(&src).arg("--emit-ptx").current_dir(&dir);
        for knob in ["Y_SMEM_SWIZZLE", "Y_SMEM_PAD", "Y_CTA_OVERRIDE"] {
            cmd.env_remove(knob);
        }
        for (k, v) in knobs {
            cmd.env(k, v);
        }
        let out = cmd.output().expect("run Y");
        assert!(out.status.success(), "{knobs:?}: {}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        std::fs::read_to_string(src.with_extension("ptx")).expect("no .ptx")
    };
    let default = emit(&[]);
    let off = emit(&[("Y_SMEM_SWIZZLE", "0")]);
    let padded = emit(&[("Y_SMEM_PAD", "8")]);
    assert!(default.contains("mma.sync") && default.contains("xor.b32"), "the default GEMM is not swizzled");
    assert!(!off.contains("xor.b32") && off == padded, "Y_SMEM_SWIZZLE=0 and Y_SMEM_PAD=8 differ");
    let smem = |p: &str| number_after(p, "Dynamic shared memory required: ");
    assert!(smem(&default) < smem(&off), "{} vs {} bytes", smem(&default), smem(&off));
}

/// `Autotuner::score_candidate` ranks tiles by the stages they will get, so
/// its count must be the emitter's. Under the pinned profile's 48 KB a
/// 64x64x32 tile asking for 6 stages gets 5 swizzled and 4 padded, and a K
/// that is not a multiple of the tile's depth rounds its K-tiles up. The
/// compiler inherits this process's environment, so both see the same layout.
#[test]
fn the_scorer_counts_the_stages_the_emitter_uses() {
    use y::autotuner::{AutotuneCandidate, Autotuner};
    let dir = pinned::pinned_scratch("smem_swizzle_stages", pinned::SM_PINNED);
    let profile = y::sentinel::HardwareProfile { sm_count: 66, max_smem_per_sm_bytes: 49152, ..Default::default() };
    // (m, n, k, cta_m, cta_n, cta_k, warps_m, warps_n, stages)
    let cases: [(u32, u32, u32, u32, u32, u32, u32, u32, u32); 5] = [
        (256, 256, 512, 64, 64, 32, 4, 2, 6),
        (256, 256, 1024, 128, 128, 32, 2, 2, 4),
        (256, 512, 2048, 64, 128, 32, 2, 2, 4),
        (32, 4096, 4096, 32, 64, 64, 2, 2, 4),
        (64, 64, 80, 64, 64, 32, 4, 2, 6),
    ];
    let mut differ_from_padded = false;
    for (i, &(m, n, k, cm, cn, ck, wm, wn, st)) in cases.iter().enumerate() {
        let src = dir.join(format!("g{i}.ysu"));
        std::fs::write(&src, format!("@tile({m}, {n}, {k})\nkernel gemm_f16_stages_{i}(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>) {{\n}}\n\nfn main() {{\n}}\n")).unwrap();
        let out = Command::new(env!("CARGO_BIN_EXE_Y"))
            .arg(&src)
            .arg("--emit-ptx")
            .env("Y_CTA_OVERRIDE", format!("{cm},{cn},{ck},{wm},{wn},{st}"))
            .current_dir(&dir)
            .output()
            .expect("run Y");
        assert!(out.status.success(), "case {i}: {}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        let ptx = std::fs::read_to_string(src.with_extension("ptx")).expect("no .ptx");
        let used = number_after(&ptx, &format!("Autotuner selected {st} pipeline stages ("));
        let smem = number_after(&ptx, "Dynamic shared memory required: ");
        let cand = AutotuneCandidate { cta_m: cm, cta_n: cn, cta_k: ck, warps_m: wm, warps_n: wn, num_stages: st, num_warps: wm * wn };
        let (per_stage, stages) = Autotuner::pipeline_footprint(&cand, k, &profile);
        assert_eq!(stages, used, "case {i}: the scorer counts {stages} stages, the emitter used {used}");
        assert!(smem >= stages * per_stage, "case {i}: {stages} stages of {per_stage} bytes do not fit the {smem} the kernel asks for");
        // The padded layout's count, as the scorer used to take it.
        let padded_stage = cm * (ck + 8) * 2 + ck * (cn + 8) * 2;
        differ_from_padded |= (45056 / padded_stage).max(1).min(st).min(k.div_ceil(ck)) != stages;
    }
    let swizzled = std::env::var("Y_SMEM_SWIZZLE").map_or(true, |v| v != "0") && std::env::var("Y_SMEM_PAD").is_err();
    assert_eq!(differ_from_padded, swizzled, "the cases do not separate the two layouts' stage counts");
}

/// A GEMM compiled at a pinned tile, and the path through the emitter that
/// tile takes (a line of the emitted module that only that path writes, and
/// one it must not).
struct Case {
    tag: &'static str,
    m: u32,
    n: u32,
    k: u32,
    tile: &'static str,
    bias: bool,
    path_has: &'static str,
    path_lacks: &'static str,
}

const HOISTED: &str = "addresses hoisted";
const SYNC: &str = "stages synchronously";

const CASES: &[Case] = &[
    // Four stages, the barrier between a tile's last two K steps; one chunk
    // of A (4 to a row) and of B (8 to a row) a thread.
    Case { tag: "xbar", m: 256, n: 256, k: 512, tile: "64,64,32,4,2,4", bias: false, path_has: HOISTED, path_lacks: SYNC },
    // Rows of 8 chunks on both sides, four chunks a thread 16 rows apart.
    Case { tag: "k64", m: 256, n: 256, k: 512, tile: "64,64,64,2,2,3", bias: false, path_has: HOISTED, path_lacks: SYNC },
    // Two stages; B rows of 16 chunks, four chunks a thread 8 rows apart.
    Case { tag: "two_stage", m: 256, n: 512, k: 256, tile: "128,128,32,2,2,2", bias: false, path_has: HOISTED, path_lacks: SYNC },
    // One K-tile: the synchronous ld.global / st.shared copy.
    Case { tag: "sync", m: 128, n: 128, k: 32, tile: "64,64,32,2,2,2", bias: false, path_has: SYNC, path_lacks: HOISTED },
    // M not a multiple of the tile: the per-chunk copy with bound checks.
    Case { tag: "m_tail", m: 200, n: 256, k: 256, tile: "64,64,32,4,2,3", bias: false, path_has: "[Y ASYNC TILE LOAD] A |", path_lacks: HOISTED },
    // 64 threads and B rows of 16 chunks: a thread's chunks would be 4 rows
    // apart, half the swizzle period, so the copy may not be hoisted.
    Case { tag: "period", m: 128, n: 256, k: 256, tile: "64,128,32,1,2,3", bias: false, path_has: "[Y ASYNC TILE LOAD] B |", path_lacks: HOISTED },
    // B rows of 4 chunks, and the bias + ReLU epilogue.
    Case { tag: "bias", m: 128, n: 128, k: 256, tile: "64,32,32,2,2,3", bias: true, path_has: HOISTED, path_lacks: SYNC },
];

/// `name`'s source and the module the compiler emits for it with the layout
/// chosen by `swizzle`.
fn compile(dir: &Path, case: &Case, swizzle: bool) -> String {
    let name = format!("gemm_f16_swz_{}", case.tag);
    let params = if case.bias {
        "A: GlobalMemory<F16>, B: GlobalMemory<F16>, Bias: GlobalMemory<F32>, C: GlobalMemory<F32>"
    } else {
        "A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>"
    };
    let stem = format!("{}_{}", name, if swizzle { "on" } else { "off" });
    let src = dir.join(format!("{stem}.ysu"));
    std::fs::write(&src, format!("@tile({}, {}, {})\nkernel {name}({params}) {{\n}}\n\nfn main() {{\n}}\n", case.m, case.n, case.k)).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&src)
        .arg("--emit-ptx")
        .env("Y_CTA_OVERRIDE", case.tile)
        .env("Y_SMEM_SWIZZLE", if swizzle { "1" } else { "0" })
        .env_remove("Y_SMEM_PAD")
        .current_dir(dir)
        .output()
        .expect("run Y");
    assert!(out.status.success(), "{}: {}{}", case.tag, String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    std::fs::read_to_string(src.with_extension("ptx")).expect("no .ptx")
}

/// The number after `key` in `text`.
fn number_after(text: &str, key: &str) -> u32 {
    let at = text.find(key).unwrap_or_else(|| panic!("no {key:?}")) + key.len();
    text[at..].chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().expect("a number")
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The f16 bit pattern of a small integer.
fn f16_of_int(v: i32) -> u16 {
    if v == 0 {
        return 0;
    }
    let (sign, a) = (if v < 0 { 0x8000u16 } else { 0 }, v.unsigned_abs());
    let e = 31 - a.leading_zeros(); // a = 1.f * 2^e, a < 2^11
    let mantissa = ((a << (10 - e)) & 0x3FF) as u16;
    sign | (((e + 15) as u16) << 10) | mantissa
}

#[test]
fn a_swizzled_gemm_computes_what_the_padded_one_does_on_the_gpu() {
    use y::cuda_runtime::CudaContext;
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver - no swizzled GEMM was executed.");
        return;
    };
    let dir = pinned::pinned_scratch("smem_swizzle", pinned::SM_PINNED);
    let mut state = 0x5157_1A2B_3C4D_5E6Fu64;
    for case in CASES {
        let (m, n, k) = (case.m as usize, case.n as usize, case.k as usize);
        let off = compile(&dir, case, false);
        let on = compile(&dir, case, true);
        let tile: Vec<u32> = case.tile.split(',').map(|x| x.parse().unwrap()).collect();
        let (cta_m, cta_n, cta_k) = (tile[0], tile[1], tile[2]);
        // Both modules take the tile, the swizzled one the path the case is
        // for; the padded one hoists where only the swizzle period refuses.
        for (mode, ptx) in [("padded", &off), ("swizzled", &on)] {
            assert!(ptx.contains(&format!("CTA {cta_m}x{cta_n}x{cta_k} | {}x{} warps", tile[3], tile[4])), "{}: {mode}: tile not taken", case.tag);
        }
        assert!(on.contains(case.path_has) && !on.contains(case.path_lacks), "{}: not the path the case is for", case.tag);
        if case.tag == "period" {
            assert!(off.contains(HOISTED), "period: the padded tile is not hoisted either, so the period is not what refuses it");
        }
        assert!(!off.contains("xor.b32") && on.contains("xor.b32"), "{}: the layouts did not switch", case.tag);
        let (smem_off, smem_on) = (number_after(&off, "Dynamic shared memory required: "), number_after(&on, "Dynamic shared memory required: "));
        assert!(smem_on < smem_off, "{}: the swizzled tiles are not packed ({smem_on} vs {smem_off} bytes)", case.tag);

        let threads = tile[3] * tile[4] * 32;
        let grid = ((case.n + cta_n - 1) / cta_n, (case.m + cta_m - 1) / cta_m, 1);
        let modules: Vec<_> = [(&off, smem_off), (&on, smem_on)]
            .into_iter()
            .map(|(ptx, smem)| {
                let module = ctx.load_ptx(ptx, &format!("gemm_f16_swz_{}", case.tag)).unwrap_or_else(|e| panic!("{}: {e}", case.tag));
                if smem > 48 * 1024 {
                    module.set_max_dynamic_smem(smem).unwrap();
                }
                (module, smem)
            })
            .collect();
        let d_a = ctx.alloc(m * k * 2).unwrap();
        let d_b = ctx.alloc(k * n * 2).unwrap();
        let d_bias = ctx.alloc(n * 4).unwrap();
        let d_c = ctx.alloc(m * n * 4).unwrap();
        let run = |a: &[u16], b: &[u16], bias: &[f32]| -> Vec<Vec<u32>> {
            ctx.memcpy_htod_at(&d_a, 0, &a.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
            ctx.memcpy_htod_at(&d_b, 0, &b.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
            ctx.memcpy_htod_at(&d_bias, 0, &bias.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
            modules
                .iter()
                .map(|(module, smem)| {
                    ctx.memset_u8(&d_c, 0xFF).unwrap(); // NaN
                    let args: Vec<u64> = if case.bias {
                        vec![d_a.device_ptr(), d_b.device_ptr(), d_bias.device_ptr(), d_c.device_ptr()]
                    } else {
                        vec![d_a.device_ptr(), d_b.device_ptr(), d_c.device_ptr()]
                    };
                    ctx.launch(module, grid, (threads, 1, 1), *smem, &args).expect("launch");
                    ctx.synchronize().expect("kernel did not complete");
                    let mut raw = vec![0u8; m * n * 4];
                    ctx.memcpy_dtoh_at(&mut raw, &d_c, 0).unwrap();
                    raw.chunks_exact(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect()
                })
                .collect()
        };

        // Random f16 values of either sign in [0.25, 4): the two layouts
        // must give the same bits, as they move no arithmetic.
        let rand_f16 = |s: &mut u64| {
            let r = splitmix(s);
            ((r & 1) << 15) as u16 | ((13 + (r >> 1) % 4) << 10) as u16 | ((r >> 8) & 0x3FF) as u16
        };
        let a: Vec<u16> = (0..m * k).map(|_| rand_f16(&mut state)).collect();
        let b: Vec<u16> = (0..k * n).map(|_| rand_f16(&mut state)).collect();
        let bias: Vec<f32> = (0..n).map(|_| (splitmix(&mut state) % 64) as f32 - 32.0).collect();
        let got = run(&a, &b, &bias);
        assert!(got[0].iter().all(|&w| f32::from_bits(w).is_finite()), "{}: the padded GEMM left NaNs", case.tag);
        let differ = got[0].iter().zip(&got[1]).filter(|(x, y)| x != y).count();
        assert_eq!(differ, 0, "{}: {differ} of {} outputs differ from the padded layout's", case.tag, m * n);

        // Small integers: every partial sum exact, so both equal the CPU.
        let ia: Vec<i32> = (0..m * k).map(|_| (splitmix(&mut state) % 9) as i32 - 4).collect();
        let ib: Vec<i32> = (0..k * n).map(|_| (splitmix(&mut state) % 9) as i32 - 4).collect();
        let got = run(&ia.iter().map(|&v| f16_of_int(v)).collect::<Vec<_>>(), &ib.iter().map(|&v| f16_of_int(v)).collect::<Vec<_>>(), &bias);
        let mut want = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0i64;
                for kk in 0..k {
                    acc += (ia[i * k + kk] * ib[kk * n + j]) as i64;
                }
                let v = acc as f32;
                want[i * n + j] = if case.bias { (v + bias[j]).max(0.0) } else { v };
            }
        }
        for (mode, out) in ["padded", "swizzled"].iter().zip(&got) {
            let bad: Vec<usize> = (0..m * n).filter(|&i| out[i] != want[i].to_bits()).collect();
            assert!(bad.is_empty(), "{}: {mode}: {} of {} wrong, first at {} (GPU {}, CPU {})", case.tag, bad.len(), m * n,
                bad[0], f32::from_bits(out[bad[0]]), want[bad[0]]);
        }
    }
}
