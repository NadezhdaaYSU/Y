//! Does the PTX backend refuse a target that lacks an instruction it emits?
//!
//! Every target below sm_80 used to be accepted and given sm_80 code: with a
//! profile naming sm_61 (GTX 10xx), sm_70 (V100) or sm_75 (RTX 20xx), all 25
//! f16 and int8 GEMMs in `tests/` compiled to `cp.async`, `ldmatrix` and
//! m16n8k16 `mma.sync` under "Compilation Successful!", and `ptxas` rejected
//! every one at every one of the three. `check_target_features` now refuses
//! them by name, and keeps the 36 programs those cards can run.
//!
//! Its table is only as good as its numbers, so each row is held to `ptxas`
//! here, both ways. That needs a `ptxas` that still knows sm_61 and sm_70,
//! which CUDA 13 does not: Triton bundles a CUDA 12 one (`venv/.../triton`),
//! or point `Y_PTXAS_OLD_ARCH` at one. Without it those checks SKIP.
//!
//! Run with:  cargo test --test ptx_target_features

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use y::ptx_emitter::{arch_features, check_target_features};

#[path = "common/pinned.rs"]
mod pinned;
#[path = "common/verification.rs"]
mod verification;

/// One instruction per row of `ARCH_FEATURES`: `(row, instruction, exact)`.
/// `exact` is false where the row is deliberately wider than the instruction
/// its floor was measured on, so one arch below may still assemble.
const PROBES: &[(&str, &str, bool)] = &[
    ("`cp.async.bulk` (TMA)", "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [sm], [%rd1], 16, [sm+1024];", true),
    ("`cp.async`", "cp.async.ca.shared.global [sm], [%rd1], 16;\n cp.async.commit_group;\n cp.async.wait_group 0;", true),
    ("`ldmatrix`", "mov.u32 %r1, sm;\n ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r2,%r3,%r4,%r5}, [%r1];", true),
    ("FP8 `mma.sync`", "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%f0,%f1,%f2,%f3}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%f0,%f1,%f2,%f3};", true),
    ("`mma.sync` m16n8k8 on bf16 or tf32", "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%f0,%f1,%f2,%f3}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%f0,%f1,%f2,%f3};", true),
    ("`mma.sync` m16n8k8", "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%f0,%f1,%f2,%f3}, {%r1,%r2}, {%r3}, {%f0,%f1,%f2,%f3};", true),
    ("`mma.sync` m8n8k4", "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%f0,%f1,%f2,%f3,%f4,%f5,%f6,%f7}, {%r1,%r2}, {%r3,%r4}, {%f0,%f1,%f2,%f3,%f4,%f5,%f6,%f7};", true),
    ("`mma.sync` m16n8k16 / m16n8k32", "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f0,%f1,%f2,%f3}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%f0,%f1,%f2,%f3};", true),
    ("`wmma` on bf16 or tf32", "wmma.load.a.sync.aligned.row.m16n16k16.global.bf16 {%r1,%r2,%r3,%r4}, [%rd1];", true),
    ("integer `wmma`", "wmma.load.a.sync.aligned.row.m16n16k16.global.s8 {%r1,%r2}, [%rd1];", false),
    ("`wmma`", "mov.u32 %r1, sm;\n wmma.store.d.sync.aligned.row.m16n16k16.shared.f32 [%r1], {%f0,%f1,%f2,%f3,%f4,%f5,%f6,%f7}, 16;", true),
    ("packed `cvt` to f16x2/bf16x2", "cvt.rn.f16x2.f32 %r1, %f1, %f2;", true),
    ("FP8 `cvt`", "cvt.rn.satfinite.e4m3x2.f32 %h1, %f1, %f2;", true),
    ("`createpolicy`", "createpolicy.fractional.L2::evict_last.b64 %rd2, 1.0;", true),
    ("an `.L2::cache_hint` access", "ld.global.L2::cache_hint.f32 %f1, [%rd1], %rd2;", true),
    ("`redux.sync`", "redux.sync.add.u32 %r1, %r2, 0xffffffff;", true),
    ("`mbarrier`", "mbarrier.init.shared.b64 [sm], 1;", true),
    ("`tanh.approx`", "tanh.approx.f32 %f1, %f2;", true),
    ("`nanosleep`", "nanosleep.u32 100;", true),
    ("a scoped memory-model access (.acquire/.release/.acq_rel)", "ld.acquire.gpu.global.u32 %r1, [%rd1];", true),
    ("`shfl.sync`", "shfl.sync.bfly.b32 %r1, %r2, 1, 31, 0xffffffff;", true),
    ("`vote.sync`", "setp.ne.u32 %p1, %r2, 0;\n vote.sync.ballot.b32 %r1, %p1, 0xffffffff;", true),
    ("`dp4a` / `dp2a`", "dp4a.s32.s32 %r1, %r2, %r3, %r4;\n dp2a.lo.s32.s32 %r1, %r2, %r3, %r1;", true),
];

/// Architectures CUDA 12's `ptxas` targets, and the first PTX ISA for each.
const ARCHS: &[(u32, &str)] = &[
    (60, "5.0"), (61, "5.0"), (62, "5.0"), (70, "6.0"), (72, "6.1"), (75, "6.3"),
    (80, "7.0"), (86, "7.1"), (87, "7.4"), (89, "7.8"), (90, "7.8"),
];
const VERSIONS: &[&str] = &[
    "5.0", "6.0", "6.1", "6.2", "6.3", "6.4", "6.5", "7.0", "7.1", "7.2", "7.3", "7.4", "7.5",
    "7.6", "7.7", "7.8", "8.0", "8.1", "8.2", "8.3", "8.4", "8.5", "8.6", "8.7",
];

fn ge(a: &str, b: &str) -> bool {
    VERSIONS.iter().position(|v| *v == a) >= VERSIONS.iter().position(|v| *v == b)
}

fn intro(sm: u32) -> &'static str {
    ARCHS.iter().find(|(a, _)| *a == sm).map(|(_, v)| *v).unwrap()
}

fn module(sm: u32, version: &str, body: &str) -> String {
    format!(
        ".version {version}\n.target sm_{sm}\n.address_size 64\n\
         .visible .entry probe(.param .u64 p)\n{{\n\
         \x20.reg .b32 %r<16>; .reg .b64 %rd<8>; .reg .f32 %f<16>; .reg .b16 %h<8>; .reg .pred %p<4>;\n\
         \x20.shared .align 16 .b8 sm[4096];\n\
         \x20ld.param.u64 %rd1, [p];\n {body}\n ret;\n}}\n"
    )
}

static SALT: AtomicUsize = AtomicUsize::new(0);

fn assembles(tool: &Path, sm: u32, version: &str, body: &str) -> Result<(), String> {
    let p = std::env::temp_dir().join(format!(
        "y_target_features_{}_{}.ptx",
        std::process::id(),
        SALT.fetch_add(1, Ordering::SeqCst)
    ));
    std::fs::write(&p, module(sm, version, body)).unwrap();
    let out = Command::new(tool)
        .arg(format!("-arch=sm_{sm}"))
        .arg(&p)
        .args(["-o", "/dev/null"])
        .output();
    let _ = std::fs::remove_file(&p);
    match out {
        Ok(o) if o.status.success() => Ok(()),
        Ok(o) => Err(String::from_utf8_lossy(&o.stderr).into_owned()),
        Err(e) => Err(e.to_string()),
    }
}

/// A `ptxas` that still assembles sm_61: CUDA 13's does not.
fn old_arch_ptxas() -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(p) = std::env::var_os("Y_PTXAS_OLD_ARCH") {
        candidates.push(p.into());
    }
    for venv in ["venv", ".venv"] {
        if let Ok(dir) = std::fs::read_dir(pinned::repo().join(venv).join("lib")) {
            for py in dir.flatten() {
                candidates.push(py.path().join("site-packages/triton/backends/nvidia/bin/ptxas"));
            }
        }
    }
    candidates.extend(["ptxas".into(), "/opt/cuda/bin/ptxas".into(), "/usr/local/cuda/bin/ptxas".into()]);
    let found = candidates
        .into_iter()
        .find(|t| assembles(t, 61, "6.0", "mov.u32 %r1, 1;").is_ok());
    verification::prerequisite_available(
        found.is_some(),
        "a ptxas that targets sm_61 (CUDA 12 or older; Triton bundles one)",
    );
    found
}

fn row(what: &str) -> (u32, &'static str) {
    arch_features()
        .find(|(w, _, _)| *w == what)
        .map(|(_, sm, v)| (sm, v))
        .unwrap_or_else(|| panic!("probe names `{what}`, which is not a row of ARCH_FEATURES"))
}

#[test]
fn every_row_has_a_probe() {
    for (what, _, _) in arch_features() {
        assert!(
            PROBES.iter().any(|(w, _, _)| *w == what),
            "row `{what}` of ARCH_FEATURES has no probe here, so nothing holds its numbers to ptxas"
        );
    }
}

/// The scan, with no assembler: below a row's arch its probe is refused by
/// name, at it the probe passes and raises `.version` to the row's.
#[test]
fn the_check_finds_every_probe() {
    for (what, body, _) in PROBES {
        let (min_sm, min_ptx) = row(what);
        if let Some(&(below, _)) = ARCHS.iter().rev().find(|(a, _)| *a < min_sm) {
            let c = check_target_features(&module(below, "8.7", body));
            assert!(
                c.errors.iter().any(|e| e.contains(what) && e.contains("`probe`")),
                "{what} at sm_{below}: not refused by name: {:?}",
                c.errors
            );
        }
        let at = min_sm.max(60);
        let c = check_target_features(&module(at, "5.0", body));
        assert!(c.errors.is_empty(), "{what} at sm_{at}: refused: {:?}", c.errors);
        assert!(
            c.required_ptx.is_some_and(|v| ge(v, min_ptx)),
            "{what} at sm_{at}: .version not raised to {min_ptx}: {:?}",
            c.required_ptx
        );
    }
}

/// A comment or a label naming an instruction is not the instruction: the
/// FP8 GEMM's header comment spells its `mma.sync`.
#[test]
fn comments_and_labels_are_not_instructions() {
    let body = "// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 in a comment\n \
                cp_async_done:\n mov.u32 %r1, 1; // cp.async.ca.shared.global";
    let c = check_target_features(&module(61, "6.0", body));
    assert!(c.errors.is_empty() && c.required_ptx.is_none(), "{c:?}");
    // and a predicated instruction is still one, named after the kernel it is
    // in, not one a comment mentions
    let body = "// split out of .entry old_kernel\n @%p1 cp.async.ca.shared.global [sm], [%rd1], 16;";
    let c = check_target_features(&module(61, "6.0", body));
    assert!(c.errors.len() == 1 && c.errors[0].contains("`probe`"), "{c:?}");
}

/// Each row's numbers against `ptxas`: the probe assembles at the row's arch
/// and version, and is rejected one arch below (unless the row is marked
/// wider than its probe) and one ISA version below.
#[test]
fn every_row_is_what_ptxas_enforces() {
    let Some(tool) = old_arch_ptxas() else { return };
    let mut wrong = Vec::new();
    for (what, body, exact) in PROBES {
        let (min_sm, min_ptx) = row(what);
        let at = min_sm.max(60);
        let v = if ge(min_ptx, intro(at)) { min_ptx } else { intro(at) };
        if let Err(e) = assembles(&tool, at, v, body) {
            wrong.push(format!("{what}: rejected at its own floor sm_{at} / {v}: {e}"));
        }
        if let Some(&(below, _)) = ARCHS.iter().rev().find(|(a, _)| *a < min_sm) {
            if *exact && assembles(&tool, below, "8.7", body).is_ok() {
                wrong.push(format!("{what}: assembles at sm_{below}, below the row's sm_{min_sm}"));
            }
        }
        let i = VERSIONS.iter().position(|x| *x == min_ptx).unwrap();
        if i > 0 && ge(VERSIONS[i - 1], intro(at)) && assembles(&tool, at, VERSIONS[i - 1], body).is_ok() {
            wrong.push(format!("{what}: assembles at .version {}, below the row's {min_ptx}", VERSIONS[i - 1]));
        }
    }
    assert!(wrong.is_empty(), "ARCH_FEATURES disagrees with ptxas:\n  {}", wrong.join("\n  "));
}

fn compile(sm: &str, fixture: &str) -> (std::process::Output, PathBuf) {
    let dir = pinned::pinned_scratch("target_features", sm);
    let src = pinned::copy_fixture(&dir, fixture);
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&src)
        .args(["--emit-ptx", "--no-autotune"])
        .current_dir(&dir)
        .output()
        .expect("run Y");
    (out, src.with_extension("ptx"))
}

/// The compiler end to end, on the cards the check is for: a tensor-core GEMM
/// is refused below sm_80 and writes nothing; a kernel the card can run
/// still compiles, declares the measured `.version`, and assembles there.
#[test]
fn old_targets_refuse_tensor_core_gemms_and_keep_what_they_can_run() {
    let tool = old_arch_ptxas();
    // The control: the same GEMM compiles at the floor.
    let (out, ptx) = compile(pinned::SM_PINNED, "tests/gemm_f16_256.ysu");
    assert!(out.status.success() && ptx.exists(), "{}", String::from_utf8_lossy(&out.stderr));

    for (sm, version) in [("6.1", ".version 6.0"), ("7.0", ".version 6.3"), ("7.5", ".version 6.3")] {
        let (out, ptx) = compile(sm, "tests/gemm_f16_256.ysu");
        let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        assert!(!out.status.success(), "sm {sm}: the f16 GEMM compiled:\n{text}");
        assert!(
            text.contains("needs sm_80 or later") && text.contains("`gemm_f16_256`"),
            "sm {sm}: refused without naming the kernel and the arch it needs:\n{text}"
        );
        assert!(!ptx.exists(), "sm {sm}: a refused module was written to {}", ptx.display());

        // `shfl.sync`, which every one of these cards has.
        let (out, ptx) = compile(sm, "tests/paged_decode_attention_128_32_8_16.ysu");
        assert!(out.status.success(), "sm {sm}: {}", String::from_utf8_lossy(&out.stderr));
        let module = std::fs::read_to_string(&ptx).expect("the module");
        let arch: u32 = sm.replace('.', "").parse().unwrap();
        assert!(module.contains(&format!(".target sm_{arch}")) && module.contains(version), "sm {sm}: header {:?}",
            module.lines().take(3).collect::<Vec<_>>());
        if let Some(tool) = &tool {
            let p = std::env::temp_dir().join(format!("y_target_features_attn_{}_{arch}.ptx", std::process::id()));
            std::fs::write(&p, &module).unwrap();
            let a = Command::new(tool).arg(format!("-arch=sm_{arch}")).arg(&p).args(["-o", "/dev/null"]).output().unwrap();
            let _ = std::fs::remove_file(&p);
            assert!(a.status.success(), "sm {sm}: ptxas rejects it: {}", String::from_utf8_lossy(&a.stderr));
        }
    }
}

/// Pascal's `.version 6.0` is `shfl.sync`'s: a kernel using it fails at 5.0.
#[test]
fn the_pascal_floor_is_shfl_sync() {
    let Some(tool) = old_arch_ptxas() else { return };
    let shfl = "shfl.sync.bfly.b32 %r1, %r2, 1, 31, 0xffffffff;";
    for sm in [60, 61, 62] {
        assert!(assembles(&tool, sm, "6.0", shfl).is_ok(), "sm_{sm}: shfl.sync fails at 6.0");
        assert!(assembles(&tool, sm, "5.0", shfl).is_err(), "sm_{sm}: shfl.sync assembles at 5.0");
    }
}
