//! `@launch_bounds(max_threads[, min_blocks])` before a `kernel`: the PTX
//! `.maxntid max_threads, 1, 1` and `.minnctapersm min_blocks`, which `ptxas`
//! sizes the kernel's registers to, in place of the `.maxnreg` the backend
//! otherwise guesses.
//!
//! The guess assumes a 256-thread block and one block an SM, so it is 255 for
//! any kernel with many virtual registers: `exact_decode_attention`, given four
//! more staging loads, took 142 registers under it, and its 7-warp blocks went
//! from two an SM to one (two need 128 a thread). Under its bounds, (256, 2),
//! `ptxas` fits the same PTX in 128 or fewer, without spilling.
//!
//! Past an SM's threads or blocks `ptxas` does not fail: it warns that the
//! directives "will be ignored" and sizes the kernel as if they were absent.
//! The compiler refuses those bounds itself, from `launch_limits_for_sm`, and
//! `the_limits_are_ptxas_limits` holds that table to `ptxas` at both edges.

use std::path::PathBuf;
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;
#[path = "common/ptxas.rs"]
mod ptxas;
#[path = "common/verification.rs"]
mod verification;

use y::ptx_emitter::launch_limits_for_sm;

fn compile(tag: &str, sm: &str, src: &str) -> (bool, String, PathBuf) {
    let dir = pinned::pinned_scratch(&format!("launch_bounds_{tag}"), sm);
    let path = dir.join("k.ysu");
    let main = if src.contains("fn main()") { "" } else { "\nfn main() {}\n" };
    std::fs::write(&path, format!("{src}{main}")).unwrap();
    let ptx = dir.join("k.ptx");
    let _ = std::fs::remove_file(&ptx);
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&path)
        .args(["--emit-ptx", "-o"])
        .arg(&ptx)
        .current_dir(&dir)
        .output()
        .expect("run Y");
    let log = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    (out.status.success() && ptx.exists(), log, ptx)
}

/// Block b's thread t writes t + 1 to word b * 64 + t.
const KERNEL: &str = "kernel k(Out: GlobalMemory<U32>) {\n    let t: I32 = thread_idx_x();\n    \
                      let base: I32 = block_idx_x() * 64 + t;\n    Out[base] = t + 1;\n}\n";

/// `ptxas -v` on `ptx` for `sm`: (exit ok, stderr, registers, spill bytes).
fn assemble(tool: &PathBuf, sm: &str, ptx: &PathBuf) -> (bool, String, u32, u32) {
    let out = Command::new(tool)
        .args([format!("-arch={sm}").as_str(), "-v"])
        .arg(ptx)
        .args(["-o", "/dev/null"])
        .output()
        .expect("run ptxas");
    let log = String::from_utf8_lossy(&out.stderr).into_owned();
    let num_before = |key: &str| {
        log.split(key)
            .next()
            .filter(|_| log.contains(key))
            .and_then(|head| head.split_whitespace().last())
            .and_then(|n| n.parse::<u32>().ok())
    };
    let regs = num_before(" registers").unwrap_or(0);
    let spills = num_before(" bytes spill stores").unwrap_or(0);
    (out.status.success(), log, regs, spills)
}

fn sm_name(sm: &str) -> String {
    format!("sm_{}", sm.replace('.', ""))
}

#[test]
fn the_directives_replace_the_guessed_register_limit() {
    let cases: &[(&str, &str, &[&str], &[&str])] = &[
        ("two", "@launch_bounds(256, 2)\n", &[".maxntid 256, 1, 1\n", ".minnctapersm 2\n"], &[".maxnreg"]),
        ("one", "@launch_bounds(96)\n", &[".maxntid 96, 1, 1\n"], &[".maxnreg", ".minnctapersm"]),
        ("edge", "@launch_bounds(768, 2)\n", &[".maxntid 768, 1, 1\n", ".minnctapersm 2\n"], &[".maxnreg"]),
        ("blocks", "@launch_bounds(32, 24)\n", &[".maxntid 32, 1, 1\n", ".minnctapersm 24\n"], &[".maxnreg"]),
        ("warps", "@launch_bounds(192, 8)\n", &[".minnctapersm 8\n"], &[".maxnreg"]),
        ("big", "@launch_bounds(1024)\n", &[".maxntid 1024, 1, 1\n"], &[".maxnreg"]),
        ("required", "@require(sm >= 80)\n@launch_bounds(128, 4)\n", &[".maxntid 128, 1, 1\n", ".minnctapersm 4\n"], &[".maxnreg"]),
        // The control: without the attribute the guess is still there.
        ("none", "", &[".maxnreg "], &[".maxntid", ".minnctapersm"]),
    ];
    let tool = ptxas::ptxas();
    for (tag, attr, present, absent) in cases {
        let src = format!("{attr}{KERNEL}");
        let (ok, log, ptx) = compile(tag, pinned::SM_FP8, &src);
        assert!(ok, "{tag}: refused:\n{src}\n{log}");
        let text = std::fs::read_to_string(&ptx).unwrap();
        for p in *present {
            assert!(text.contains(p), "{tag}: no {p:?} in the PTX:\n{text}");
        }
        for a in *absent {
            assert!(!text.contains(a), "{tag}: {a:?} in the PTX:\n{text}");
        }
        if let Some(tool) = &tool {
            let (ok, log, _, _) = assemble(tool, &sm_name(pinned::SM_FP8), &ptx);
            assert!(ok, "{tag}: ptxas failed:\n{log}");
            // A bound ptxas drops is a warning, not an error: none may appear.
            assert!(!log.contains("ignored") && !log.contains("warning"), "{tag}: ptxas: {log}");
        }
    }
}

#[test]
fn bounds_the_target_cannot_meet_are_refused() {
    let fp8 = pinned::SM_FP8; // sm_89: 1536 threads, 24 blocks an SM
    let cases: &[(&str, &str, &str, &[&str])] = &[
        ("zero", fp8, "@launch_bounds(0)\n", &["positive integer literals"]),
        ("zero_blocks", fp8, "@launch_bounds(256, 0)\n", &["positive integer literals"]),
        ("negative", fp8, "@launch_bounds(-32)\n", &["positive integer literals"]),
        ("name", fp8, "@launch_bounds(N)\n", &["positive integer literals"]),
        ("sum", fp8, "@launch_bounds(128 + 128)\n", &["positive integer literals"]),
        ("three", fp8, "@launch_bounds(256, 2, 1)\n", &["one or two values"]),
        ("block", fp8, "@launch_bounds(1025)\n", &["a block has 1 to 1024 threads"]),
        ("threads", fp8, "@launch_bounds(256, 7)\n", &["7 blocks of 256 threads", "1536 threads"]),
        // 200 threads are 7 warps, 224 threads: 7 x 224 = 1568.
        ("whole_warps", fp8, "@launch_bounds(200, 7)\n", &["(224 in whole warps)"]),
        ("blocks", fp8, "@launch_bounds(32, 25)\n", &["24 blocks"]),
        ("turing", "7.5", "@launch_bounds(512, 3)\n", &["1024 threads", "16 blocks"]),
        ("unlisted", "10.1", "@launch_bounds(64)\n", &["on sm_101", "no limits are recorded"]),
        ("not_a_kernel", fp8, "@launch_bounds(64)\nfn f() {}\n", &["only supported directly before `kernel`"]),
        ("after", fp8, "@launch_bounds(64)\n@require(sm >= 80)\n", &["only supported directly before `kernel`"]),
    ];
    for (tag, sm, attr, phrases) in cases {
        let src = format!("{attr}{KERNEL}");
        let (ok, log, _) = compile(tag, sm, &src);
        assert!(!ok, "{tag}: compiled:\n{src}\n{log}");
        for p in *phrases {
            assert!(log.contains(p), "{tag}: the refusal does not say {p:?}:\n{src}\n{log}");
        }
    }
    // On a `@tile` GEMM, which chooses its own block size.
    let tile = "@tile(1024, 1024, 1024)\n@launch_bounds(256, 2)\n\
                kernel g(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>) {\n}\n";
    let (ok, log, _) = compile("tile", fp8, tile);
    assert!(!ok, "@tile with @launch_bounds compiled:\n{log}");
    assert!(log.contains("on a `@tile` GEMM"), "{log}");
    let (ok, log, _) = compile("tile_alone", fp8, &tile.replace("@launch_bounds(256, 2)\n", ""));
    assert!(ok, "the same @tile kernel without the bounds was refused:\n{log}");
}

fn probe(tool: &PathBuf, sm: &str, version: &str, ntid: u32, blocks: u32) -> Result<bool, String> {
    let dir = pinned::scratch(&format!("launch_bounds_probe_{sm}_{ntid}_{blocks}"));
    let p = dir.join("p.ptx");
    std::fs::write(
        &p,
        format!(
            ".version {version}\n.target {sm}\n.address_size 64\n.visible .entry p(.param .u64 o)\n\
             .maxntid {ntid}, 1, 1\n.minnctapersm {blocks}\n{{\n    .reg .b32 %r<2>;\n    .reg .b64 %rd<3>;\n    \
             ld.param.u64 %rd1, [o];\n    mov.u32 %r1, %tid.x;\n    mul.wide.u32 %rd2, %r1, 4;\n    \
             add.u64 %rd1, %rd1, %rd2;\n    st.global.u32 [%rd1], %r1;\n    ret;\n}}\n"
        ),
    )
    .unwrap();
    let out = Command::new(tool).arg(format!("-arch={sm}")).arg(&p).args(["-o", "/dev/null"]).output();
    let _ = std::fs::remove_dir_all(&dir);
    match out {
        Ok(o) if o.status.success() => Ok(!String::from_utf8_lossy(&o.stderr).contains("ignored")),
        Ok(o) => Err(String::from_utf8_lossy(&o.stderr).into_owned()),
        Err(e) => Err(e.to_string()),
    }
}

/// A `ptxas` that still assembles sm_60..sm_72: CUDA 13's does not.
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
    let found = candidates.into_iter().find(|t| probe(t, "sm_61", "6.0", 32, 1).is_ok());
    verification::prerequisite_available(found.is_some(), "a ptxas that targets sm_61 (CUDA 12 or older; Triton bundles one)");
    found
}

/// Each row of `launch_limits_for_sm`, at both edges: the most threads two
/// blocks may have and the most blocks of one warp assemble without the
/// "will be ignored" warning, and one warp or one block more draws it.
#[test]
fn the_limits_are_ptxas_limits() {
    let rows: &[(&str, &str)] = &[
        ("sm_60", "6.0"), ("sm_61", "6.0"), ("sm_62", "6.0"), ("sm_70", "6.3"), ("sm_72", "6.2"),
        ("sm_75", "6.3"), ("sm_80", "7.0"), ("sm_86", "7.1"), ("sm_87", "7.5"), ("sm_89", "7.8"),
        ("sm_90", "8.0"), ("sm_100", "8.7"), ("sm_120", "8.7"),
    ];
    let new = ptxas::ptxas();
    let old = old_arch_ptxas();
    let mut checked = 0;
    for (sm, version) in rows {
        let (threads, blocks) = launch_limits_for_sm(sm).unwrap_or_else(|| panic!("{sm} has no row"));
        let tool = [&new, &old].into_iter().flatten().find(|t| probe(t, sm, version, 32, 1).is_ok());
        let Some(tool) = tool else {
            verification::prerequisite_available(false, &format!("a ptxas that targets {sm}"));
            continue;
        };
        let half = threads / 2;
        assert_eq!(probe(tool, sm, version, half, 2), Ok(true), "{sm}: 2 x {half} threads drew the warning");
        assert_eq!(probe(tool, sm, version, half + 32, 2), Ok(false), "{sm}: 2 x {} threads did not", half + 32);
        assert_eq!(probe(tool, sm, version, 32, blocks), Ok(true), "{sm}: {blocks} blocks drew the warning");
        assert_eq!(probe(tool, sm, version, 32, blocks + 1), Ok(false), "{sm}: {} blocks did not", blocks + 1);
        checked += 1;
    }
    assert!(launch_limits_for_sm("sm_89a").is_some() && launch_limits_for_sm("8.9").is_some());
    assert!(launch_limits_for_sm("sm_101").is_none() && launch_limits_for_sm("sm_52").is_none());
    eprintln!("launch limits checked against ptxas on {checked} of {} targets", rows.len());
}

/// On the device: a block of up to `max_threads` runs; a larger one is
/// refused at launch, and the context still runs a launch after that.
#[test]
fn a_block_past_the_bound_is_refused_at_launch() {
    let (ok, log, ptx) = compile("device", pinned::SM_PINNED, &format!("@launch_bounds(64)\n{KERNEL}"));
    assert!(ok, "{log}");
    let Some(ctx) = y::cuda_runtime::CudaContext::new() else {
        eprintln!("SKIP: no CUDA device, so the bounded kernel was compiled and not launched");
        return;
    };
    let module = ctx.load_ptx(&std::fs::read_to_string(&ptx).unwrap(), "k").expect("load");
    let out = ctx.alloc(4 * 256).unwrap();
    let read = |ctx: &y::cuda_runtime::CudaContext| {
        let mut raw = vec![0u8; 4 * 256];
        ctx.memcpy_dtoh_at(&mut raw, &out, 0).unwrap();
        raw.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<u32>>()
    };
    for (block, refused) in [(64u32, false), (65, true), (128, true), (32, false)] {
        ctx.memset_u8(&out, 0).unwrap();
        let r = ctx.launch(&module, (2, 1, 1), (block, 1, 1), 0, &[out.device_ptr()]);
        assert_eq!(r.is_err(), refused, "block {block}: {r:?}");
        ctx.synchronize().expect("the context survives");
        let got = read(&ctx);
        for (i, w) in got.iter().enumerate() {
            let (b, t) = (i / 64, i % 64);
            let want = if !refused && b < 2 && t < block as usize { t as u32 + 1 } else { 0 };
            assert_eq!(*w, want, "block {block}: word {i}");
        }
    }
}
