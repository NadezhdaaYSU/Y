//! `--emit-cpu`'s Rust computes in the types the program declares.
//!
//! Its `let`s were untyped, so Rust picked each local's type from how it was
//! used, not from its declaration: `let x: U8 = 200; x = x + 100; return x;`
//! returned 300 where the LLVM backend's `u8` wraps to 44. Now a numeric local
//! carries its declared Rust type, and a value crossing into one - a `let`, an
//! assignment, `op=`, a `return`, an argument - converts the way the LLVM
//! backend converts it; a mixed operation computes in Y's common type (the
//! wider, unsigned at equal width, a float over an integer).
//!
//! Each program runs on the LLVM backend and through rustc, against a constant.
use std::path::Path;
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

const CASES: &[(&str, &str, i32)] = &[
    // Halved: an exit code is taken mod 256, so 300 and a wrapped 44 exit alike.
    ("u8_wraps", "fn main() -> I32 {\n    let x: U8 = 200;\n    x = x + 100;\n    return x / 2;\n}\n", 22),
    ("u8_op_assign", "fn main() -> I32 {\n    let w: U8 = 250;\n    w += 10;\n    return w / 2;\n}\n", 2),
    ("narrowing_let", "fn main() -> I32 {\n    let big: I64 = 300;\n    let small: U8 = big;\n    return small / 2;\n}\n", 22),
    // Only the declared type makes this an i64: untyped, Rust gives the
    // literal its default, i32, which cannot hold it.
    ("wide_literal", "fn main() -> I32 {\n    let big: I64 = 3000000000;\n    return big / 1000000000;\n}\n", 3),
    (
        "mixed_widths",
        "fn main() -> I32 {\n    let a: I64 = 5000000000;\n    let b: I32 = 7;\n    let c: I64 = a + b;\n    return c / 1000000000;\n}\n",
        5,
    ),
    (
        "unsigned_compare",
        "fn main() -> I32 {\n    let u: U32 = 3000000000;\n    let s: I32 = 5;\n    if u > s {\n        return 1;\n    }\n    return 0;\n}\n",
        1,
    ),
    (
        "argument",
        "fn twice(n: I64) -> I64 {\n    return n * 2;\n}\n\nfn main() -> I32 {\n    let k: I32 = 21;\n    return twice(k);\n}\n",
        42,
    ),
];

fn compile(dir: &Path, tag: &str, src: &str, args: &[&str]) -> String {
    let path = dir.join(format!("{tag}.ysu"));
    std::fs::write(&path, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_Y")).arg(&path).args(args).current_dir(dir).output().expect("run Y");
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success(), "{tag} {args:?}: {text}");
    text
}

#[test]
fn every_case_runs_on_the_llvm_backend_as_stated() {
    for (tag, src, want) in CASES {
        let dir = pinned::pinned_scratch(&format!("cputy_llvm_{tag}"), pinned::SM_PINNED);
        let bin = dir.join(format!("{tag}.bin"));
        compile(&dir, tag, src, &["-o", bin.to_str().unwrap()]);
        let code = Command::new(&bin).status().unwrap().code();
        assert_eq!(code, Some(*want), "{tag} on the LLVM backend");
    }
}

#[test]
fn emit_cpu_computes_what_the_llvm_backend_computes() {
    for (tag, src, want) in CASES {
        let dir = pinned::pinned_scratch(&format!("cputy_cpu_{tag}"), pinned::SM_PINNED);
        let out = compile(&dir, tag, src, &["--emit-cpu"]);
        let mut lines = out.lines().skip_while(|l| !l.contains("GENERATED RUST BLOB")).skip(1);
        let blob: Vec<&str> = lines
            .by_ref()
            .take_while(|l| !(l.trim_start_matches('=').is_empty() && l.len() > 20 && !l.starts_with("//")))
            .collect();
        let blob = blob.join("\n").replace("pub fn main() -> i32", "pub fn y_main() -> i32");
        let rs = dir.join("blob.rs");
        std::fs::write(&rs, format!("#![allow(unused, non_snake_case)]\n{blob}\nfn main() {{ std::process::exit(y_main()); }}\n"))
            .unwrap();
        let bin = dir.join("blob");
        let Ok(built) = Command::new("rustc").args(["--edition", "2021", "-O", "-o"]).arg(&bin).arg(&rs).output() else {
            eprintln!("SKIP: no rustc, so the emitted Rust was not run");
            return;
        };
        assert!(built.status.success(), "{tag}: the emitted Rust does not compile:\n{}\n{blob}", String::from_utf8_lossy(&built.stderr));
        let code = Command::new(&bin).status().unwrap().code();
        assert_eq!(code, Some(*want), "{tag} through --emit-cpu:\n{blob}");
    }
}

/// A struct field is typed too. `p.a + 100` on a `U8` field was written with
/// Rust's `+` because the emitter knew no field's type, so a debug build of
/// the blob panicked where Y wraps (`200 + 100` is 44 in a `U8`). The CPU
/// backend cannot build a struct literal, so the blob is called from a Rust
/// `main` that does, and built WITHOUT `-O`: overflow checks on.
#[test]
fn field_arithmetic_wraps_in_a_debug_build_as_the_llvm_backend_does() {
    let fns = "struct Pair {\n    a: U8,\n    b: I32,\n}\n\nfn sum(p: &Pair) -> I32 {\n    return p.a + p.b;\n}\n\n\
               fn bump(p: &mut Pair) {\n    p.a = p.a + 100;\n}\n";
    let llvm = format!("{fns}\nfn main() -> I32 {{\n    let p = Pair {{ a: 200, b: 5 }};\n    bump(&mut p);\n    return sum(&p);\n}}\n");
    let dir = pinned::pinned_scratch("cputy_fields_llvm", pinned::SM_PINNED);
    let bin = dir.join("fields.bin");
    compile(&dir, "fields", &llvm, &["-o", bin.to_str().unwrap()]);
    assert_eq!(Command::new(&bin).status().unwrap().code(), Some(49), "the LLVM backend's answer moved");

    let dir = pinned::pinned_scratch("cputy_fields_cpu", pinned::SM_PINNED);
    let out = compile(&dir, "fields", fns, &["--emit-cpu"]);
    let blob: Vec<&str> = out
        .lines()
        .skip_while(|l| !l.contains("GENERATED RUST BLOB"))
        .skip(1)
        .take_while(|l| !(l.trim_start_matches('=').is_empty() && l.len() > 20 && !l.starts_with("//")))
        .collect();
    let blob = blob.join("\n");
    let rs = dir.join("fields.rs");
    std::fs::write(
        &rs,
        format!("#![allow(unused, non_snake_case)]\n{blob}\nfn main() {{\n    let mut p = Pair {{ a: 200, b: 5 }};\n    \
                 bump(&mut p);\n    std::process::exit(sum(&p));\n}}\n"),
    )
    .unwrap();
    let rbin = dir.join("fields");
    let Ok(built) = Command::new("rustc").args(["--edition", "2021", "-o"]).arg(&rbin).arg(&rs).output() else {
        eprintln!("SKIP: no rustc, so the emitted Rust was not run");
        return;
    };
    assert!(built.status.success(), "the emitted Rust does not compile:\n{}\n{blob}", String::from_utf8_lossy(&built.stderr));
    let run = Command::new(&rbin).output().unwrap();
    assert_eq!(run.status.code(), Some(49), "--emit-cpu, debug build:\n{}\n{blob}", String::from_utf8_lossy(&run.stderr));
}
