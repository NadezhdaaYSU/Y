//! A tile measured at a nearby shape is borrowed (`load_persisted_nearest`),
//! but never across M = 32. Up to 32 rows the F16 GEMM adds each mma's result
//! in float32 and its tiles were measured for decode; the 2x bound alone let
//! 64 x 4096 x 4096 borrow the M = 32 tile, which took 1.21-1.22x cuBLAS's
//! time against the analytic model's 1.02-1.03x
//! (`docs/benchmarks/f16_llm_shapes_2026-10-24/m64/`). Each refusal here is
//! paired with a borrow from the same profile that is still taken.

use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

/// A pinned scratch directory whose profile holds `entry`, a measured tile
/// for `M x 4096 x 4096`, if any.
fn profile_with(tag: &str, entry: Option<(u32, &str)>) -> PathBuf {
    let dir = pinned::pinned_scratch(tag, pinned::SM_PINNED);
    if let Some((m, tile)) = entry {
        let profile = dir.join(".ysu_hw_profile");
        let mut text = std::fs::read_to_string(&profile).unwrap();
        text.push_str(&format!("AUTOTUNE_F16_{m}x4096x4096_{}={tile}\n", pinned::GPU_NAME));
        std::fs::write(&profile, text).unwrap();
    }
    dir
}

/// The compiler's output and the tile the module's header states
/// (`CTA 32x64x64 | 2x2 warps`) for an `m x 4096 x 4096` F16 GEMM compiled
/// in `dir`.
fn compile(dir: &Path, m: u32) -> (String, String) {
    let src = dir.join(format!("g{m}.ysu"));
    std::fs::write(&src, format!("@tile({m}, 4096, 4096)\nkernel g{m}(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>) {{\n}}\n\nfn main() {{\n}}\n")).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&src)
        .arg("--emit-ptx")
        .current_dir(dir)
        .env_remove("Y_CTA_OVERRIDE")
        .env_remove("Y_F16_ACCUMULATE")
        .output()
        .expect("run Y");
    let log = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success(), "M = {m}: {log}");
    let ptx = std::fs::read_to_string(src.with_extension("ptx")).expect("no .ptx");
    let header = ptx.lines().find(|l| l.contains("[Y TENSOR CORE GEMM]")).unwrap_or_else(|| panic!("M = {m}: no GEMM header"));
    (log, header.split(" | ").skip(1).take(2).collect::<Vec<_>>().join(" | "))
}

#[test]
fn a_measured_tile_is_not_lent_across_m_32() {
    const DECODE: &str = "CTA 32x64x64 | 1x2 warps";
    const PREFILL: &str = "CTA 64x128x32 | 2x4 warps";
    let lent = |log: &str, from: u32| log.contains(&format!("borrowing the measured tile from the nearest shape M={from} N=4096 K=4096"));

    // What the analytic model picks with nothing to borrow.
    let bare = profile_with("borrow_none", None);
    let [model_32, model_33, model_64] = [32, 33, 64].map(|m| compile(&bare, m).1);
    for (m, model) in [(33, &model_33), (64, &model_64)] {
        assert_ne!(model, DECODE, "the M = {m} refusal below would look like a borrow");
    }
    assert_ne!(model_32, PREFILL, "the M = 32 refusal below would look like a borrow");

    // A decode entry, M = 32: lent to M = 16, not to M = 33 or 64.
    let dir = profile_with("borrow_decode", Some((32, "32,64,64,1,2,3")));
    let (log, tile) = compile(&dir, 16);
    assert!(lent(&log, 32) && tile == DECODE, "M = 16 did not borrow the M = 32 tile: {tile}\n{log}");
    for (m, model) in [(33, &model_33), (64, &model_64)] {
        let (log, tile) = compile(&dir, m);
        assert!(!lent(&log, 32) && &tile == model, "M = {m} borrowed the M = 32 tile: {tile}\n{log}");
    }

    // And the reverse: an M = 64 entry is lent to M = 128, not to M = 32.
    let dir2 = profile_with("borrow_prefill", Some((64, "64,128,32,2,4,3")));
    let (log, tile) = compile(&dir2, 128);
    assert!(lent(&log, 64) && tile == PREFILL, "M = 128 did not borrow the M = 64 tile: {tile}\n{log}");
    let (log, tile) = compile(&dir2, 32);
    assert!(!lent(&log, 64) && tile == model_32, "M = 32 borrowed the M = 64 tile: {tile}\n{log}");

    for d in [bare, dir, dir2] {
        let _ = std::fs::remove_dir_all(d);
    }
}
