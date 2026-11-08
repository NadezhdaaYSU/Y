//! `f64_to_bits`, `f64_from_bits`, `f32_to_bits`, `f32_from_bits`: Rust's bit
//! reinterpretations, through every backend that lowers them, bit for bit
//! against Rust's own.
//!
//! Unlike `as`, a reinterpretation has no hardware latitude: NaN payloads, a
//! signalling NaN, negative zero and subnormals come back exactly, so they are
//! compared as bits, NaNs included. The results are unsigned (U64, U32) on
//! every backend: `f64_to_bits(x) >> 63` is a logical shift and a compare of
//! two results is unsigned, which a lowering that typed them as signed
//! integers would get wrong for every negative float.
//!
//! As in `numeric_casts.rs`, each operation is its own Y function called on
//! chosen bit patterns: `--emit-llvm` linked with a C harness at -O0 and -O2,
//! `--emit-cpu`'s Rust compiled with rustc, and one PTX kernel per operation
//! on the GPU.
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

/// (Y function, argument type, result type, Y body expression of `x`).
const OPS: [(&str, &str, &str, &str); 8] = [
    ("r_f64_to_bits", "F64", "U64", "f64_to_bits(x)"),
    ("r_f64_from_bits", "U64", "F64", "f64_from_bits(x)"),
    ("r_f32_to_bits", "F32", "U32", "f32_to_bits(x)"),
    ("r_f32_from_bits", "U32", "F32", "f32_from_bits(x)"),
    ("r_sign64", "F64", "U64", "f64_to_bits(x) >> 63"),
    ("r_sign32", "F32", "U32", "f32_to_bits(x) >> 31"),
    // Unsigned: every negative float's bits are above 1.0's.
    ("r_above_one64", "F64", "U64", "f64_to_bits(x) > f64_to_bits(one)"),
    ("r_above_one32", "F32", "U32", "f32_to_bits(x) > f32_to_bits(one)"),
];

fn f64_inputs() -> Vec<u64> {
    let mut v: Vec<u64> = [0.0f64, -0.0, 1.0, -1.0, 1.5, -2.5, 0.1, f64::MAX, f64::MIN,
                           f64::MIN_POSITIVE, f64::INFINITY, f64::NEG_INFINITY, 1e-310, -1e-310]
        .iter().map(|x| x.to_bits()).collect();
    // Smallest subnormal, a quiet NaN, a negative quiet NaN with a payload, a
    // signalling NaN (quiet bit clear), and the all-ones NaN.
    v.extend([1, 0x7FF8_0000_0000_0000, 0xFFF8_DEAD_BEEF_0001, 0x7FF0_0000_0000_0001,
              0x7FF4_0000_0000_0000, u64::MAX]);
    v
}

fn f32_inputs() -> Vec<u64> {
    let mut v: Vec<u64> = [0.0f32, -0.0, 1.0, -1.0, 1.5, -2.5, 0.1, f32::MAX, f32::MIN,
                           f32::MIN_POSITIVE, f32::INFINITY, f32::NEG_INFINITY, 1e-40, -1e-40]
        .iter().map(|x| x.to_bits() as u64).collect();
    v.extend([1, 0x7FC0_0000, 0xFFC0_BEEF, 0x7F80_0001, 0x7FA0_0000, 0xFFFF_FFFF]);
    v
}

fn inputs(t: &str) -> Vec<u64> {
    match t {
        "F64" => f64_inputs(),
        "F32" => f32_inputs(),
        // Every float pattern is also an integer to reinterpret, plus a few
        // that are not round as floats.
        "U64" => { let mut v = f64_inputs(); v.extend([0x0123_4567_89AB_CDEF, 1 << 63, 2]); v }
        _ => { let mut v = f32_inputs(); v.extend([0x0123_4567, 1 << 31, 2]); v }
    }
}

/// Rust's answer, as the bit pattern of the result.
fn expected(op: &str, bits: u64) -> u64 {
    match op {
        "r_f64_to_bits" | "r_f64_from_bits" => bits,
        "r_f32_to_bits" | "r_f32_from_bits" => bits & 0xFFFF_FFFF,
        "r_sign64" => f64::from_bits(bits).to_bits() >> 63,
        "r_sign32" => (f32::from_bits(bits as u32).to_bits() >> 31) as u64,
        "r_above_one64" => (f64::from_bits(bits).to_bits() > 1.0f64.to_bits()) as u64,
        "r_above_one32" => (f32::from_bits(bits as u32).to_bits() > 1.0f32.to_bits()) as u64,
        _ => unreachable!("{op}"),
    }
}

fn y_source() -> String {
    let mut s = String::new();
    for (name, from, to, body) in OPS {
        if name.starts_with("r_above_one") {
            writeln!(s, "fn {name}(x: {from}) -> {to} {{\n    let one: {from} = 1.0;\n    \
                         if {body} {{\n        return 1;\n    }}\n    return 0;\n}}").unwrap();
        } else {
            writeln!(s, "fn {name}(x: {from}) -> {to} {{\n    return {body};\n}}").unwrap();
        }
    }
    s.push_str("fn main() -> I32 {\n    return 0;\n}\n");
    s
}

fn c_type(t: &str) -> &'static str {
    match t { "F64" => "double", "F32" => "float", "U64" => "uint64_t", _ => "uint32_t" }
}

fn width(t: &str) -> usize {
    if t.ends_with("64") { 8 } else { 4 }
}

/// The text of the function or kernel whose header contains `header`, up to
/// its closing brace.
fn body_of<'a>(text: &'a str, header: &str) -> &'a str {
    let start = text.find(header).unwrap_or_else(|| panic!("no {header} in:\n{text}"));
    let len = text[start..].find("\n}").unwrap_or(text.len() - start);
    &text[start..start + len]
}

fn scratch(tag: &str) -> PathBuf {
    pinned::pinned_scratch(&format!("bits_{tag}"), pinned::SM_PINNED)
}

fn y(dir: &Path, file: &str, args: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_Y")).arg(dir.join(file)).args(args)
        .current_dir(dir).output().expect("run Y");
    (out.status.success(),
     format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)))
}

/// Lines `<function> <input hex> <output hex>`, checked against `expected`.
fn check_results(what: &str, printed: &str) -> usize {
    let mut seen = 0;
    let mut wrong = Vec::new();
    for line in printed.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 3 || !parts[0].starts_with("r_") {
            continue;
        }
        let input = u64::from_str_radix(parts[1], 16).unwrap();
        let got = u64::from_str_radix(parts[2], 16).unwrap();
        let want = expected(parts[0], input);
        if got != want {
            wrong.push(format!("{} {:#x}: got {:#x}, Rust gives {:#x}", parts[0], input, got, want));
        }
        seen += 1;
    }
    let total: usize = OPS.iter().map(|(_, from, _, _)| inputs(from).len()).sum();
    assert_eq!(seen, total, "{what}: the harness printed {seen} of {total} results:\n{printed}");
    assert!(wrong.is_empty(), "{what}: {} of {seen} results differ:\n{}", wrong.len(), wrong.join("\n"));
    seen
}

#[test]
fn every_reinterpretation_through_llvm_matches_rust() {
    let have_clang = Command::new("clang").arg("--version").output().is_ok_and(|o| o.status.success());
    if !have_clang {
        eprintln!("SKIP: no clang, so the emitted LLVM was not linked and run");
        return;
    }
    let dir = scratch("llvm");
    std::fs::write(dir.join("bits.ysu"), y_source()).unwrap();
    let (ok, log) = y(&dir, "bits.ysu", &["--emit-llvm", "-o", "bits.ll"]);
    assert!(ok, "--emit-llvm failed:\n{log}");
    let ll = std::fs::read_to_string(dir.join("bits.ll")).unwrap();
    for (name, ..) in &OPS[..4] {
        let body = body_of(&ll, &format!("@{name}("));
        assert!(body.contains("bitcast"), "{name} has no bitcast:\n{body}");
        for conv in ["fptoui", "fptosi", "uitofp", "sitofp", "fpext", "fptrunc"] {
            assert!(!body.contains(conv), "{name} lowered through `{conv}`:\n{body}");
        }
    }
    let mut c = String::from("#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n");
    for (name, from, to, _) in OPS {
        writeln!(c, "{} {}({});", c_type(to), name, c_type(from)).unwrap();
    }
    c.push_str("int main(void) {\n");
    for (name, from, to, _) in OPS {
        for bits in inputs(from) {
            writeln!(c, "  {{ uint64_t b = 0x{bits:x}ull; {ft} x; memcpy(&x, &b, sizeof x); \
                         {tt} r = {name}(x); uint64_t o = 0; memcpy(&o, &r, sizeof r); \
                         printf(\"{name} %llx %llx\\n\", (unsigned long long)b, (unsigned long long)o); }}",
                     ft = c_type(from), tt = c_type(to)).unwrap();
        }
    }
    c.push_str("  return 0;\n}\n");
    std::fs::write(dir.join("harness.c"), c).unwrap();
    for opt in ["-O0", "-O2"] {
        let bin = dir.join(format!("bits{opt}"));
        let built = Command::new("clang").args([opt, "-Wno-override-module", "-o"]).arg(&bin)
            .arg(dir.join("harness.c")).arg(dir.join("bits.ll")).output().unwrap();
        assert!(built.status.success(), "clang {opt}: {}", String::from_utf8_lossy(&built.stderr));
        let run = Command::new(&bin).output().unwrap();
        assert!(run.status.success());
        let n = check_results(&format!("LLVM {opt}"), &String::from_utf8_lossy(&run.stdout));
        eprintln!("LLVM {opt}: {n} results equal Rust's");
    }
}

#[test]
fn every_reinterpretation_through_emitted_rust_matches_rust() {
    let dir = scratch("rust");
    std::fs::write(dir.join("bits.ysu"), y_source()).unwrap();
    let (ok, log) = y(&dir, "bits.ysu", &["--emit-cpu", "-o", "bits_emitted.rs"]);
    assert!(ok, "--emit-cpu failed:\n{log}");
    let emitted = std::fs::read_to_string(dir.join("bits_emitted.rs")).unwrap();
    assert!(emitted.contains("pub fn main() -> i32"), "{emitted}");
    let mut rs = emitted.replace("pub fn main() -> i32", "pub fn y_main() -> i32");
    rs.push_str("\nfn main() {\n    let _ = y_main();\n");
    for (name, from, to, _) in OPS {
        for bits in inputs(from) {
            let decode = match from {
                "F64" => format!("f64::from_bits(0x{bits:x}u64)"),
                "F32" => format!("f32::from_bits(0x{bits:x}u64 as u32)"),
                "U64" => format!("0x{bits:x}u64"),
                _ => format!("0x{bits:x}u64 as u32"),
            };
            let encode = match to {
                "F64" => "r.to_bits()",
                "F32" => "r.to_bits() as u64",
                _ => "r as u64",
            };
            writeln!(rs, "    {{ let r = {name}({decode}); println!(\"{name} {bits:x} {{:x}}\", {encode}); }}").unwrap();
        }
    }
    rs.push_str("}\n");
    std::fs::write(dir.join("bits_harness.rs"), rs).unwrap();
    let bin = dir.join("bits_rs");
    let Ok(built) = Command::new("rustc").args(["--edition", "2021", "-O", "-A", "warnings", "-o"])
        .arg(&bin).arg(dir.join("bits_harness.rs")).output() else {
        eprintln!("SKIP: no rustc, so the emitted Rust was not run");
        return;
    };
    assert!(built.status.success(), "rustc: {}", String::from_utf8_lossy(&built.stderr));
    let run = Command::new(&bin).output().unwrap();
    assert!(run.status.success());
    let n = check_results("emitted Rust", &String::from_utf8_lossy(&run.stdout));
    eprintln!("emitted Rust: {n} results equal Rust's");
}

/// One kernel per operation, `Out[i] = <op>(X[i])`. Skips without a GPU.
#[test]
fn every_reinterpretation_on_the_gpu_matches_rust() {
    let Some(ctx) = y::cuda_runtime::CudaContext::new() else {
        eprintln!("SKIP: no CUDA device, so the PTX reinterpretations were not run");
        return;
    };
    let dir = scratch("ptx");
    let mut src = String::new();
    for (name, from, to, body) in OPS {
        if name.starts_with("r_above_one") {
            writeln!(src, "kernel {name}(X: GlobalMemory<{from}>, Out: GlobalMemory<{to}>) {{\n    \
                           let i: I32 = thread_idx_x();\n    let x: {from} = X[i];\n    \
                           let one: {from} = 1.0;\n    let r: {to} = 0;\n    if {body} {{\n        \
                           r = 1;\n    }}\n    Out[i] = r;\n}}").unwrap();
        } else {
            writeln!(src, "kernel {name}(X: GlobalMemory<{from}>, Out: GlobalMemory<{to}>) {{\n    \
                           let i: I32 = thread_idx_x();\n    let x: {from} = X[i];\n    \
                           let r: {to} = {body};\n    Out[i] = r;\n}}").unwrap();
        }
    }
    src.push_str("fn main() {}\n");
    std::fs::write(dir.join("bits.ysu"), &src).unwrap();
    let (ok, log) = y(&dir, "bits.ysu", &["--emit-ptx", "-o", "bits.ptx"]);
    assert!(ok, "--emit-ptx failed:\n{log}");
    let ptx = std::fs::read_to_string(dir.join("bits.ptx")).unwrap();
    for (name, ..) in &OPS[..4] {
        let body = body_of(&ptx, &format!(".entry {name}("));
        assert!(body.contains("mov.b"), "{name} has no mov.b:\n{body}");
        assert!(!body.contains(".f32;") && !body.contains("cvt.rn") && !body.contains("cvt.rzi")
                && !body.contains("cvt.f"), "{name} lowered through a conversion:\n{body}");
    }

    let mut printed = String::new();
    for (name, from, to, _) in OPS {
        let xs = inputs(from);
        let mut bytes = Vec::new();
        for b in &xs {
            bytes.extend_from_slice(&b.to_le_bytes()[..width(from)]);
        }
        let module = ctx.load_ptx(&ptx, name).expect("load");
        let x = ctx.alloc(bytes.len()).unwrap();
        let out = ctx.alloc(xs.len() * width(to)).unwrap();
        ctx.memcpy_htod_at(&x, 0, &bytes).unwrap();
        ctx.memset_u8(&out, 0xA5).unwrap();
        ctx.launch(&module, (1, 1, 1), (xs.len() as u32, 1, 1), 0,
                   &[x.device_ptr(), out.device_ptr()]).expect("launch");
        ctx.synchronize().unwrap();
        let mut got = vec![0u8; xs.len() * width(to)];
        ctx.memcpy_dtoh_at(&mut got, &out, 0).unwrap();
        for (k, b) in xs.iter().enumerate() {
            let w = width(to);
            let mut v = [0u8; 8];
            v[..w].copy_from_slice(&got[k * w..(k + 1) * w]);
            writeln!(printed, "{name} {b:x} {:x}", u64::from_le_bytes(v)).unwrap();
        }
    }
    let n = check_results("PTX on the GPU", &printed);
    eprintln!("PTX on {}: {n} results equal Rust's", ctx.device_name());
}

#[test]
fn wrong_operands_are_refused_by_name() {
    let dir = scratch("refuse");
    let cases: &[(&str, &str, &str)] = &[
        ("literal", "let u: U64 = f64_to_bits(1.5);", "`f64_to_bits` of a literal"),
        ("narrow_float", "let x: F32 = 1.5;\n    let u: U64 = f64_to_bits(x);", "`f64_to_bits` takes a F64, got F32"),
        ("wide_float", "let x: F64 = 1.5;\n    let u: U32 = f32_to_bits(x);", "`f32_to_bits` takes a F32, got F64"),
        ("signed", "let i: I64 = 5;\n    let f: F64 = f64_from_bits(i);", "`f64_from_bits` takes a U64, got I64"),
        ("narrow_int", "let i: U32 = 5;\n    let f: F64 = f64_from_bits(i);", "`f64_from_bits` takes a U64, got U32"),
        ("arity", "let x: F64 = 1.5;\n    let u: U64 = f64_to_bits(x, x);", "`f64_to_bits` takes one F64 argument, got 2"),
    ];
    for (tag, body, needle) in cases {
        let src = format!("fn main() -> I32 {{\n    {}\n    return 0;\n}}\n", body);
        std::fs::write(dir.join(format!("{tag}.ysu")), &src).unwrap();
        for backend in [["--emit-llvm", "-o", "x.ll"], ["--emit-ptx", "-o", "x.ptx"]] {
            let (ok, log) = y(&dir, &format!("{tag}.ysu"), &backend);
            assert!(!ok, "{tag} compiled with {}:\n{src}\n{log}", backend[0]);
            assert!(log.contains(needle), "{tag} ({}): expected {needle:?} in:\n{log}", backend[0]);
        }
    }
    // Control: the same shapes with operands of the right type compile.
    let src = "fn main() -> I32 {\n    let x: F64 = 1.5;\n    let u: U64 = f64_to_bits(x);\n    \
               let y: F64 = f64_from_bits(u);\n    let s: F32 = 2.5;\n    let v: U32 = f32_to_bits(s);\n    \
               let t: F32 = f32_from_bits(v);\n    return 0;\n}\n";
    std::fs::write(dir.join("ok.ysu"), src).unwrap();
    let (ok, log) = y(&dir, "ok.ysu", &["--emit-llvm", "-o", "ok.ll"]);
    assert!(ok, "{log}");
}
