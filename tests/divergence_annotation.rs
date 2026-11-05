//! `@divergence(uniform)` claims every lane of a warp takes a branch the same
//! way. The claim is checked by `crate::uniformity`, the analysis the PTX
//! backend refuses divergent collectives with.
//!
//! It was checked by looking for identifiers containing `threadIdx` or
//! `laneId` - which no Y program spells, since the thread index is the call
//! `thread_idx_x()` - so `@divergence(uniform) if thread_idx_x() < 20` and
//! every condition on a value computed from it were accepted.
//!
//! Each case below is refused naming what makes the lanes differ, and the
//! controls - block-uniform, warp-uniform (a warp reduction's result), a
//! constant `for`, host code, and the two corpus programs that carry the
//! annotation - compile.
use std::fs;
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

fn compile(dir: &std::path::Path, name: &str, src: &str, args: &[&str]) -> (bool, String) {
    let path = dir.join(format!("{}.ysu", name));
    fs::write(&path, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .current_dir(dir)
        .arg(&path)
        .args(args)
        .output()
        .expect("run Y");
    (out.status.success(), format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)))
}

fn kernel(body: &str) -> String {
    format!("kernel k(X: GlobalMemory<F32>, Out: GlobalMemory<F32>, N: I32) {{\n{}\n}}\n\nfn main() {{}}\n", body)
}

const REFUSED: &[(&str, &str, &str)] = &[
    (
        "thread_index",
        "    @divergence(uniform)\n    if thread_idx_x() < 20 {\n        Out[0] = 1.0;\n    }",
        "it reads `thread_idx_x()`",
    ),
    (
        "derived_from_it",
        "    let t: I32 = thread_idx_x();\n    let u: I32 = t * 2;\n    @divergence(uniform)\n    if u < 4 {\n        Out[0] = 1.0;\n    }",
        "it reads `u` (from `t` on line 3)",
    ),
    (
        "assigned_under_a_divergent_branch",
        "    let f: I32 = 0;\n    if thread_idx_x() < 3 {\n        f = 1;\n    }\n    @divergence(uniform)\n    if f == 1 {\n        Out[0] = 1.0;\n    }",
        "it reads `f` (assigned on line 4 inside the `if` on line 3)",
    ),
    (
        "a_memory_read",
        "    @divergence(uniform)\n    if X[0] > 0.0 {\n        Out[0] = 1.0;\n    }",
        "it reads a value loaded from memory",
    ),
    (
        "while_on_the_thread_index",
        "    let t: I32 = thread_idx_x();\n    let i: I32 = 0;\n    @invariant(i >= 0)\n    @divergence(uniform)\n    while i < t {\n        i = i + 1;\n    }\n    Out[0] = 1.0;",
        "claims every lane of a warp takes this while the same way, but its condition",
    ),
    (
        "for_bounded_by_the_thread_index",
        "    let t: I32 = thread_idx_x();\n    let s: I32 = 0;\n    @invariant(i >= 0)\n    @divergence(uniform)\n    for i in 0..t {\n        s = s + i;\n    }\n    Out[0] = 1.0;",
        "claims every lane of a warp takes this for the same way, but its bounds",
    ),
];

#[test]
fn a_branch_the_lanes_can_disagree_on_is_refused() {
    let dir = pinned::pinned_scratch("divergence_refused", pinned::SM_PINNED);
    for (name, body, why) in REFUSED {
        // The type checker refuses it, so every backend does: the PTX one and
        // the default (LLVM) one.
        for args in [&["--emit-ptx", "-o", "k.ptx"][..], &["--emit-llvm", "-o", "k.ll"][..]] {
            let (ok, text) = compile(&dir, name, &kernel(body), args);
            assert!(!ok, "{} {:?} was accepted:\n{}", name, args, text);
            assert!(text.contains("`@divergence(uniform)` claims every lane"), "{}: not refused by name:\n{}", name, text);
            assert!(text.contains(why), "{}: does not say `{}`:\n{}", name, why, text);
        }
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_branch_every_lane_takes_alike_is_accepted() {
    let dir = pinned::pinned_scratch("divergence_accepted", pinned::SM_PINNED);
    let cases = [
        (
            "block_uniform",
            kernel("    let b: I32 = block_idx_x() * 2;\n    @divergence(uniform)\n    if b < N {\n        Out[0] = 1.0;\n    }"),
        ),
        (
            "warp_uniform",
            kernel("    let t: I32 = thread_idx_x();\n    let s: F32 = warp_reduce_sum(X[t]);\n    @divergence(uniform)\n    if s > 0.0 {\n        Out[t] = s;\n    }"),
        ),
        (
            "constant_for",
            kernel("    let s: F32 = 0.0;\n    @invariant(i >= 0)\n    @divergence(uniform)\n    for i in 0..8 {\n        s = s + 1.0;\n    }\n    Out[0] = s;"),
        ),
    ];
    for (name, src) in &cases {
        let (ok, text) = compile(&dir, name, src, &["--emit-ptx", "-o", "k.ptx"]);
        assert!(ok, "{} was refused:\n{}", name, text);
    }
    // Host code: one thread runs it, so any condition is uniform.
    let host = "fn pick(a: I32) -> I32 {\n    let r: I32 = 0;\n    @divergence(uniform)\n    if a > 3 {\n        r = a;\n    }\n    return r;\n}\n\nfn main() {\n    let x: I32 = pick(5);\n    print_int(x);\n}\n";
    let (ok, text) = compile(&dir, "host", host, &["--emit-llvm", "-o", "h.ll"]);
    assert!(ok, "host code was refused:\n{}", text);
    // The corpus programs that carry the annotation.
    for (fixture, args) in [("tests/train_spec.ysu", &["--emit-ptx", "-o", "t.ptx"][..]), ("tests/math.ysu", &["--emit-llvm", "-o", "m.ll"][..])] {
        let path = pinned::copy_fixture(&dir, fixture);
        let out = Command::new(env!("CARGO_BIN_EXE_Y")).current_dir(&dir).arg(&path).args(args).output().unwrap();
        assert!(out.status.success(), "{} was refused:\n{}", fixture, String::from_utf8_lossy(&out.stdout));
    }
    let _ = fs::remove_dir_all(&dir);
}
