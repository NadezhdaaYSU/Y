//! `dp4a`, `dp2a_lo`, `dp2a_hi`: PTX's packed integer dot products
//! (`dp4a.s32.s32`, `dp2a.{lo,hi}.s32.s32`), through every backend that lowers
//! them, against a Rust model of the PTX ISA's definition:
//!
//!   dp4a(a, b, c)    = c + sum_i  sbyte(a, i) * sbyte(b, i),          i = 0..3
//!   dp2a_lo(a, b, c) = c + shalf(a, 0) * sbyte(b, 0) + shalf(a, 1) * sbyte(b, 1)
//!   dp2a_hi(a, b, c) = c + shalf(a, 0) * sbyte(b, 2) + shalf(a, 1) * sbyte(b, 3)
//!
//! with signed bytes / halves and a sum that wraps at 32 bits. The GPU run is
//! the arbiter of the model (PTX is what the builtins are named after); the
//! LLVM and emitted-Rust lowerings must then agree with both. Operands are
//! every combination of words whose bytes and halves sit at their limits
//! (0x7F, 0x80, 0xFF, 0x7FFF, 0x8000), mixed signs, and accumulators that make
//! the sum wrap.
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

const OPS: [&str; 3] = ["dp4a", "dp2a_lo", "dp2a_hi"];

fn words() -> Vec<i32> {
    [0u32, 1, 0xFFFF_FFFF, 0x8000_0000, 0x7FFF_FFFF, 0x7F7F_7F7F, 0x8080_8080, 0x7FFF_8000,
     0x8000_7FFF, 0x0102_0304, 0xFF00_FF00, 0x00FF_00FF]
        .iter().map(|w| *w as i32).collect()
}

fn accumulators() -> Vec<i32> {
    vec![0, 1, -1, i32::MAX, i32::MIN]
}

fn triples() -> Vec<(i32, i32, i32)> {
    let mut v = Vec::new();
    for a in words() {
        for b in words() {
            for c in accumulators() {
                v.push((a, b, c));
            }
        }
    }
    v
}

fn sbyte(x: i32, i: u32) -> i32 {
    ((x as u32 >> (8 * i)) as u8) as i8 as i32
}

fn shalf(x: i32, i: u32) -> i32 {
    ((x as u32 >> (16 * i)) as u16) as i16 as i32
}

fn model(op: &str, a: i32, b: i32, c: i32) -> i32 {
    match op {
        "dp4a" => (0..4).fold(c, |s, i| s.wrapping_add(sbyte(a, i).wrapping_mul(sbyte(b, i)))),
        "dp2a_lo" => c.wrapping_add(shalf(a, 0) * sbyte(b, 0)).wrapping_add(shalf(a, 1) * sbyte(b, 1)),
        "dp2a_hi" => c.wrapping_add(shalf(a, 0) * sbyte(b, 2)).wrapping_add(shalf(a, 1) * sbyte(b, 3)),
        _ => unreachable!("{op}"),
    }
}

fn y_source() -> String {
    let mut s = String::new();
    for op in OPS {
        writeln!(s, "fn r_{op}(a: I32, b: I32, c: I32) -> I32 {{\n    return {op}(a, b, c);\n}}").unwrap();
    }
    s.push_str("fn main() -> I32 {\n    return 0;\n}\n");
    s
}

fn scratch(tag: &str) -> PathBuf {
    pinned::pinned_scratch(&format!("dot_{tag}"), pinned::SM_PINNED)
}

fn y(dir: &Path, file: &str, args: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_Y")).arg(dir.join(file)).args(args)
        .current_dir(dir).output().expect("run Y");
    (out.status.success(),
     format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)))
}

/// Lines `r_<op> a b c result` (hex words), checked against `model`.
fn check_results(what: &str, printed: &str) -> usize {
    let mut seen = 0;
    let mut wrong = Vec::new();
    for line in printed.lines() {
        let p: Vec<&str> = line.split_whitespace().collect();
        if p.len() != 5 || !p[0].starts_with("r_") {
            continue;
        }
        let w = |s: &str| u32::from_str_radix(s, 16).unwrap() as i32;
        let (a, b, c, got) = (w(p[1]), w(p[2]), w(p[3]), w(p[4]));
        let want = model(&p[0][2..], a, b, c);
        if got != want {
            wrong.push(format!("{}({a:#x}, {b:#x}, {c:#x}) = {got:#x}, the model gives {want:#x}", &p[0][2..]));
        }
        seen += 1;
    }
    let total = OPS.len() * triples().len();
    assert_eq!(seen, total, "{what}: the harness printed {seen} of {total} results");
    assert!(wrong.is_empty(), "{what}: {} of {seen} differ:\n{}", wrong.len(),
            wrong.iter().take(20).cloned().collect::<Vec<_>>().join("\n"));
    seen
}

#[test]
fn every_dot_product_on_the_gpu_matches_the_ptx_definition() {
    let Some(ctx) = y::cuda_runtime::CudaContext::new() else {
        eprintln!("SKIP: no CUDA device, so the PTX dot products were not run");
        return;
    };
    let dir = scratch("ptx");
    let mut src = String::new();
    for op in OPS {
        writeln!(src, "kernel r_{op}(A: GlobalMemory<I32>, B: GlobalMemory<I32>, C: GlobalMemory<I32>, \
                       Out: GlobalMemory<I32>) {{\n    let i: I32 = thread_idx_x();\n    \
                       let a: I32 = A[i];\n    let b: I32 = B[i];\n    let c: I32 = C[i];\n    \
                       let r: I32 = {op}(a, b, c);\n    Out[i] = r;\n}}").unwrap();
    }
    src.push_str("fn main() {}\n");
    std::fs::write(dir.join("dot.ysu"), &src).unwrap();
    let (ok, log) = y(&dir, "dot.ysu", &["--emit-ptx", "-o", "dot.ptx"]);
    assert!(ok, "--emit-ptx failed:\n{log}");
    let ptx = std::fs::read_to_string(dir.join("dot.ptx")).unwrap();
    for want in ["dp4a.s32.s32", "dp2a.lo.s32.s32", "dp2a.hi.s32.s32"] {
        assert!(ptx.contains(want), "no {want} in:\n{ptx}");
    }
    let t = triples();
    let bytes = |f: &dyn Fn(&(i32, i32, i32)) -> i32| -> Vec<u8> {
        t.iter().flat_map(|x| f(x).to_le_bytes()).collect()
    };
    let (ab, bb, cb) = (bytes(&|x| x.0), bytes(&|x| x.1), bytes(&|x| x.2));
    let mut printed = String::new();
    for op in OPS {
        let module = ctx.load_ptx(&ptx, &format!("r_{op}")).expect("load");
        let bufs: Vec<_> = [&ab, &bb, &cb].iter().map(|b| {
            let d = ctx.alloc(b.len()).unwrap();
            ctx.memcpy_htod_at(&d, 0, b).unwrap();
            d
        }).collect();
        let out = ctx.alloc(t.len() * 4).unwrap();
        ctx.memset_u8(&out, 0xA5).unwrap();
        ctx.launch(&module, (1, 1, 1), (t.len() as u32, 1, 1), 0,
                   &[bufs[0].device_ptr(), bufs[1].device_ptr(), bufs[2].device_ptr(), out.device_ptr()])
            .expect("launch");
        ctx.synchronize().unwrap();
        let mut got = vec![0u8; t.len() * 4];
        ctx.memcpy_dtoh_at(&mut got, &out, 0).unwrap();
        for (k, (a, b, c)) in t.iter().enumerate() {
            let r = i32::from_le_bytes(got[k * 4..k * 4 + 4].try_into().unwrap());
            writeln!(printed, "r_{op} {:x} {:x} {:x} {:x}", *a as u32, *b as u32, *c as u32, r as u32).unwrap();
        }
    }
    let n = check_results("PTX on the GPU", &printed);
    eprintln!("PTX on {}: {n} dot products equal the model", ctx.device_name());
}

#[test]
fn every_dot_product_through_llvm_matches_the_model() {
    let have_clang = Command::new("clang").arg("--version").output().is_ok_and(|o| o.status.success());
    if !have_clang {
        eprintln!("SKIP: no clang, so the emitted LLVM was not linked and run");
        return;
    }
    let dir = scratch("llvm");
    std::fs::write(dir.join("dot.ysu"), y_source()).unwrap();
    let (ok, log) = y(&dir, "dot.ysu", &["--emit-llvm", "-o", "dot.ll"]);
    assert!(ok, "--emit-llvm failed:\n{log}");
    let mut c = String::from("#include <stdint.h>\n#include <stdio.h>\n");
    for op in OPS {
        writeln!(c, "int32_t r_{op}(int32_t, int32_t, int32_t);").unwrap();
    }
    c.push_str("int main(void) {\n");
    for op in OPS {
        for (a, b, cc) in triples() {
            writeln!(c, "  printf(\"r_{op} %x %x %x %x\\n\", 0x{:x}u, 0x{:x}u, 0x{:x}u, \
                         (uint32_t)r_{op}((int32_t)0x{:x}u, (int32_t)0x{:x}u, (int32_t)0x{:x}u));",
                     a as u32, b as u32, cc as u32, a as u32, b as u32, cc as u32).unwrap();
        }
    }
    c.push_str("  return 0;\n}\n");
    std::fs::write(dir.join("harness.c"), c).unwrap();
    for opt in ["-O0", "-O2"] {
        let bin = dir.join(format!("dot{opt}"));
        let built = Command::new("clang").args([opt, "-Wno-override-module", "-o"]).arg(&bin)
            .arg(dir.join("harness.c")).arg(dir.join("dot.ll")).output().unwrap();
        assert!(built.status.success(), "clang {opt}: {}", String::from_utf8_lossy(&built.stderr));
        let run = Command::new(&bin).output().unwrap();
        assert!(run.status.success());
        let n = check_results(&format!("LLVM {opt}"), &String::from_utf8_lossy(&run.stdout));
        eprintln!("LLVM {opt}: {n} dot products equal the model");
    }
}

#[test]
fn every_dot_product_through_emitted_rust_matches_the_model() {
    let dir = scratch("rust");
    std::fs::write(dir.join("dot.ysu"), y_source()).unwrap();
    let (ok, log) = y(&dir, "dot.ysu", &["--emit-cpu", "-o", "dot_emitted.rs"]);
    assert!(ok, "--emit-cpu failed:\n{log}");
    let emitted = std::fs::read_to_string(dir.join("dot_emitted.rs")).unwrap();
    let mut rs = emitted.replace("pub fn main() -> i32", "pub fn y_main() -> i32");
    rs.push_str("\nfn main() {\n    let _ = y_main();\n");
    for op in OPS {
        for (a, b, c) in triples() {
            writeln!(rs, "    println!(\"r_{op} {:x} {:x} {:x} {{:x}}\", r_{op}({a}, {b}, {c}) as u32);",
                     a as u32, b as u32, c as u32).unwrap();
        }
    }
    rs.push_str("}\n");
    std::fs::write(dir.join("dot_harness.rs"), rs).unwrap();
    let bin = dir.join("dot_rs");
    let Ok(built) = Command::new("rustc").args(["--edition", "2021", "-O", "-A", "warnings", "-o"])
        .arg(&bin).arg(dir.join("dot_harness.rs")).output() else {
        eprintln!("SKIP: no rustc, so the emitted Rust was not run");
        return;
    };
    assert!(built.status.success(), "rustc: {}", String::from_utf8_lossy(&built.stderr));
    let run = Command::new(&bin).output().unwrap();
    assert!(run.status.success());
    let n = check_results("emitted Rust", &String::from_utf8_lossy(&run.stdout));
    eprintln!("emitted Rust: {n} dot products equal the model");
}

#[test]
fn wrong_operands_are_refused_by_name() {
    let dir = scratch("refuse");
    let cases: &[(&str, &str, &str)] = &[
        ("unsigned", "let a: U32 = 1;\n    let r: I32 = dp4a(a, 1, 0);", "`dp4a` takes a I32, got U32"),
        ("wide", "let a: I64 = 1;\n    let r: I32 = dp2a_lo(a, 1, 0);", "`dp2a_lo` takes a I32, got I64"),
        ("float", "let a: F32 = 1.0;\n    let r: I32 = dp2a_hi(1, a, 0);", "`dp2a_hi` takes a I32, got F32"),
        ("arity", "let a: I32 = 1;\n    let r: I32 = dp4a(a, a);", "`dp4a` takes 3 I32 arguments, got 2"),
    ];
    for (tag, body, needle) in cases {
        let src = format!("fn main() -> I32 {{\n    {}\n    return 0;\n}}\n", body);
        std::fs::write(dir.join(format!("{tag}.ysu")), &src).unwrap();
        let (ok, log) = y(&dir, &format!("{tag}.ysu"), &["--emit-llvm", "-o", "x.ll"]);
        assert!(!ok, "{tag} compiled:\n{src}\n{log}");
        assert!(log.contains(needle), "{tag}: expected {needle:?} in:\n{log}");
    }
    // Control: I32 operands, literals included, compile.
    let src = "fn main() -> I32 {\n    let a: I32 = 258;\n    let r: I32 = dp4a(a, a, 0);\n    \
               let s: I32 = dp2a_lo(a, r, 1);\n    return 0;\n}\n";
    std::fs::write(dir.join("ok.ysu"), src).unwrap();
    let (ok, log) = y(&dir, "ok.ysu", &["--emit-llvm", "-o", "ok.ll"]);
    assert!(ok, "{log}");
}
