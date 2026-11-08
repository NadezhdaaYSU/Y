//! `as`: every conversion between I32, I64, U32, U64, F32 and F64, through
//! the CPU backends, bit for bit against Rust's own `as` - which is the
//! semantics Y adopts (`ast::Expr::Cast`): integer to float rounds to nearest,
//! ties to even; float to integer truncates toward zero, saturates at the
//! target's range and takes NaN to 0; F64 to F32 rounds to nearest even;
//! integer to integer truncates or extends by the source's signedness.
//!
//! Each conversion is its own Y function, so a backend's output is called
//! directly on chosen bit patterns rather than printed by the program:
//! `--emit-llvm` is linked with a C harness at -O0 and at -O2 (the optimiser
//! is what would exploit poison, if the float-to-integer lowering produced
//! any), and `--emit-cpu`'s Rust is compiled with rustc. The inputs are the
//! values where a conversion can go wrong: ties, the first integer a float
//! cannot hold (2^24 + 1, 2^53 + 1), each type's limits and the values just
//! past them, NaN, infinities, signed zeros and subnormals.
//!
//! A NaN result is compared as "is a NaN": what payload a float-to-float
//! conversion gives a NaN is the hardware's (x86 keeps it, PTX's `cvt`
//! canonicalises it), and Y makes no claim about it.
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

const TYPES: [&str; 6] = ["I32", "I64", "U32", "U64", "F32", "F64"];

fn c_type(t: &str) -> &'static str {
    match t {
        "I32" => "int32_t",
        "I64" => "int64_t",
        "U32" => "uint32_t",
        "U64" => "uint64_t",
        "F32" => "float",
        _ => "double",
    }
}

fn rust_type(t: &str) -> &'static str {
    match t {
        "I32" => "i32",
        "I64" => "i64",
        "U32" => "u32",
        "U64" => "u64",
        "F32" => "f32",
        _ => "f64",
    }
}

fn fname(from: &str, to: &str) -> String {
    format!("c_{}_{}", from.to_lowercase(), to.to_lowercase())
}

/// Inputs for a source type, as the bit pattern of a value of that type.
fn inputs(t: &str) -> Vec<u64> {
    match t {
        "I32" => [0i32, 1, -1, i32::MIN, i32::MAX, i32::MIN + 1, 16_777_217, -16_777_217,
                  16_777_219, 123_456_789, -2_147_483_520, 2_147_483_520]
            .iter().map(|v| *v as u32 as u64).collect(),
        "I64" => [0i64, 1, -1, i64::MIN, i64::MAX, i64::MIN + 1, (1 << 53) + 1, -((1 << 53) + 1),
                  (1 << 24) + 1, 1 << 31, -(1 << 31) - 1, 1 << 32, (1 << 32) + 5,
                  i32::MAX as i64 + 1, i32::MIN as i64 - 1, 9_007_199_254_740_993,
                  -9_223_372_036_854_775_807]
            .iter().map(|v| *v as u64).collect(),
        "U32" => [0u32, 1, u32::MAX, 1 << 31, (1 << 31) + 1, 16_777_217, 16_777_219,
                  u32::MAX - 128, u32::MAX - 127, i32::MAX as u32]
            .iter().map(|v| *v as u64).collect(),
        "U64" => vec![0, 1, u64::MAX, 1 << 63, (1 << 63) + 1, (1 << 53) + 1, u64::MAX - 1024,
                      u64::MAX - 1023, u64::MAX - 2047, 1 << 32, (1 << 32) - 1, u32::MAX as u64 + 1,
                      i64::MAX as u64, 18_446_744_073_709_550_592],
        "F32" => [0.0f32, -0.0, 0.5, -0.5, 1.5, 2.5, -2.5, 0.499_999_97, 0.999_999_94, -1.0,
                  -0.999_999_94, f32::from_bits(1), -f32::from_bits(1), 16_777_216.0,
                  2_147_483_520.0, 2_147_483_648.0, -2_147_483_648.0, -2_147_483_904.0,
                  4_294_967_040.0, 4_294_967_296.0, 9.223_371_5e18, 9.223_372e18, 1.844_674_3e19,
                  1.844_674_4e19, f32::MAX, f32::MIN, f32::INFINITY, f32::NEG_INFINITY, f32::NAN,
                  -f32::NAN, f32::MIN_POSITIVE]
            .iter().map(|v| v.to_bits() as u64).collect(),
        _ => [0.0f64, -0.0, 0.5, -0.5, 1.5, 2.5, -2.5, -1.0, -0.999_999_999_9, 0.1,
              2_147_483_647.0, 2_147_483_647.5, 2_147_483_648.0, -2_147_483_648.0,
              -2_147_483_648.9, -2_147_483_649.0, 4_294_967_295.0, 4_294_967_295.9,
              4_294_967_296.0, 9_223_372_036_854_774_784.0, 9_223_372_036_854_775_808.0,
              -9_223_372_036_854_775_808.0, -9_223_372_036_854_777_856.0,
              18_446_744_073_709_549_568.0, 18_446_744_073_709_551_616.0, 9_007_199_254_740_993.0,
              f64::INFINITY, f64::NEG_INFINITY, f64::NAN, -f64::NAN, 1e300, -1e300,
              f64::from_bits(1), 1e-40, -1e-45, 3.402_823_466_385_288_6e38,
              3.402_823_567_797_336_6e38, 3.402_823_567_797_336_7e38, 1.000_000_059_604_644_8,
              1.000_000_178_813_934_3, 1.000_000_059_604_645, f64::MIN_POSITIVE]
            .iter().map(|v| v.to_bits()).collect(),
    }
}

/// Rust's `as`, from the bit pattern of a `from` to the bit pattern of a `to`.
fn expected(from: &str, to: &str, bits: u64) -> u64 {
    macro_rules! out {
        ($v:expr) => {
            match to {
                "I32" => ($v as i32) as u32 as u64,
                "I64" => ($v as i64) as u64,
                "U32" => ($v as u32) as u64,
                "U64" => $v as u64,
                "F32" => ($v as f32).to_bits() as u64,
                _ => ($v as f64).to_bits(),
            }
        };
    }
    match from {
        "I32" => out!(bits as u32 as i32),
        "I64" => out!(bits as i64),
        "U32" => out!(bits as u32),
        "U64" => out!(bits),
        "F32" => out!(f32::from_bits(bits as u32)),
        _ => out!(f64::from_bits(bits)),
    }
}

fn is_nan_bits(t: &str, bits: u64) -> bool {
    match t {
        "F32" => f32::from_bits(bits as u32).is_nan(),
        "F64" => f64::from_bits(bits).is_nan(),
        _ => false,
    }
}

fn y_source() -> String {
    let mut s = String::new();
    for from in TYPES {
        for to in TYPES {
            writeln!(s, "fn {}(x: {}) -> {} {{\n    return x as {};\n}}", fname(from, to), from, to, to).unwrap();
        }
    }
    s.push_str("fn main() -> I32 {\n    return 0;\n}\n");
    s
}

fn scratch(tag: &str) -> PathBuf {
    pinned::pinned_scratch(&format!("casts_{tag}"), pinned::SM_PINNED)
}

fn y(dir: &Path, file: &str, args: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(dir.join(file))
        .args(args)
        .current_dir(dir)
        .output()
        .expect("run Y");
    (out.status.success(),
     format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)))
}

/// Lines `<function> <input hex> <output hex>` printed by a harness, checked
/// against `expected`. Returns how many results were compared.
fn check_results(what: &str, printed: &str) -> usize {
    let mut seen = 0;
    let mut wrong = Vec::new();
    for line in printed.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 3 || !parts[0].starts_with("c_") {
            continue;
        }
        let (from, to) = parts[0][2..].split_once('_').unwrap();
        let (from, to) = (from.to_uppercase(), to.to_uppercase());
        let input = u64::from_str_radix(parts[1], 16).unwrap();
        let got = u64::from_str_radix(parts[2], 16).unwrap();
        let want = expected(&from, &to, input);
        let ok = got == want || (is_nan_bits(&to, want) && is_nan_bits(&to, got));
        if !ok {
            wrong.push(format!("{} {:#x}: got {:#x}, Rust's `as` gives {:#x}", parts[0], input, got, want));
        }
        seen += 1;
    }
    let total: usize = TYPES.iter().map(|f| inputs(f).len() * TYPES.len()).sum();
    assert_eq!(seen, total, "{what}: the harness printed {seen} of {total} results:\n{printed}");
    assert!(wrong.is_empty(), "{what}: {} of {seen} conversions differ:\n{}", wrong.len(), wrong.join("\n"));
    seen
}

#[test]
fn every_conversion_through_llvm_matches_rust_as() {
    let have_clang = Command::new("clang").arg("--version").output().is_ok_and(|o| o.status.success());
    if !have_clang {
        eprintln!("SKIP: no clang, so the emitted LLVM was not linked and run");
        return;
    }
    let dir = scratch("llvm");
    std::fs::write(dir.join("casts.ysu"), y_source()).unwrap();
    let (ok, log) = y(&dir, "casts.ysu", &["--emit-llvm", "-o", "casts.ll"]);
    assert!(ok, "--emit-llvm failed:\n{log}");

    let mut c = String::from("#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n");
    for from in TYPES {
        for to in TYPES {
            writeln!(c, "{} {}({});", c_type(to), fname(from, to), c_type(from)).unwrap();
        }
    }
    c.push_str("int main(void) {\n");
    for from in TYPES {
        for to in TYPES {
            for bits in inputs(from) {
                let call = format!("{}(x)", fname(from, to));
                writeln!(c, "  {{ uint64_t b = 0x{bits:x}ull; {ft} x; memcpy(&x, &b, sizeof x); \
                             {tt} r = {call}; uint64_t o = 0; memcpy(&o, &r, sizeof r); \
                             printf(\"{n} %llx %llx\\n\", (unsigned long long)b, (unsigned long long)o); }}",
                         ft = c_type(from), tt = c_type(to), n = fname(from, to)).unwrap();
            }
        }
    }
    c.push_str("  return 0;\n}\n");
    std::fs::write(dir.join("harness.c"), c).unwrap();
    for opt in ["-O0", "-O2"] {
        let bin = dir.join(format!("casts{opt}"));
        let built = Command::new("clang")
            .args([opt, "-Wno-override-module", "-o"]).arg(&bin)
            .arg(dir.join("harness.c")).arg(dir.join("casts.ll"))
            .output().unwrap();
        assert!(built.status.success(), "clang {opt}: {}", String::from_utf8_lossy(&built.stderr));
        let run = Command::new(&bin).output().unwrap();
        assert!(run.status.success());
        let n = check_results(&format!("LLVM {opt}"), &String::from_utf8_lossy(&run.stdout));
        eprintln!("LLVM {opt}: {n} conversions equal Rust's `as`");
    }
}

#[test]
fn every_conversion_through_emitted_rust_matches_rust_as() {
    let dir = scratch("rust");
    std::fs::write(dir.join("casts.ysu"), y_source()).unwrap();
    let (ok, log) = y(&dir, "casts.ysu", &["--emit-cpu", "-o", "casts_emitted.rs"]);
    assert!(ok, "--emit-cpu failed:\n{log}");
    let emitted = std::fs::read_to_string(dir.join("casts_emitted.rs")).unwrap();
    // The emitted program's own `main` returns i32; the harness needs its own.
    assert!(emitted.contains("pub fn main() -> i32"), "{emitted}");
    let mut rs = emitted.replace("pub fn main() -> i32", "pub fn y_main() -> i32");
    rs.push_str("\nfn main() {\n    let _ = y_main();\n");
    for from in TYPES {
        for to in TYPES {
            for bits in inputs(from) {
                let decode = match from {
                    "F32" => format!("f32::from_bits(0x{bits:x}u64 as u32)"),
                    "F64" => format!("f64::from_bits(0x{bits:x}u64)"),
                    t => format!("0x{bits:x}u64 as {}", rust_type(t)),
                };
                let encode = match to {
                    "F32" => "r.to_bits() as u64",
                    "F64" => "r.to_bits()",
                    "I32" => "r as u32 as u64",
                    _ => "r as u64",
                };
                writeln!(rs, "    {{ let r = {}({}); println!(\"{} {:x} {{:x}}\", {}); }}",
                         fname(from, to), decode, fname(from, to), bits, encode).unwrap();
            }
        }
    }
    rs.push_str("}\n");
    std::fs::write(dir.join("casts_harness.rs"), rs).unwrap();
    let bin = dir.join("casts_rs");
    let Ok(built) = Command::new("rustc")
        .args(["--edition", "2021", "-O", "-A", "warnings", "-o"]).arg(&bin)
        .arg(dir.join("casts_harness.rs")).output() else {
        eprintln!("SKIP: no rustc, so the emitted Rust was not run");
        return;
    };
    assert!(built.status.success(), "rustc: {}", String::from_utf8_lossy(&built.stderr));
    let run = Command::new(&bin).output().unwrap();
    assert!(run.status.success());
    let n = check_results("emitted Rust", &String::from_utf8_lossy(&run.stdout));
    eprintln!("emitted Rust: {n} conversions equal Rust's `as`");
}

/// Programs whose answer depends on how `as` parses. Each is a function the
/// C harness calls; the wrong grouping gives a different number.
#[test]
fn as_binds_like_rust() {
    let have_clang = Command::new("clang").arg("--version").output().is_ok_and(|o| o.status.success());
    if !have_clang {
        eprintln!("SKIP: no clang, so the precedence programs were not run");
        return;
    }
    let dir = scratch("prec");
    let src = "\
// (-a) as I64: negating I32::MIN wraps to itself first.
fn neg_first(a: I32) -> I64 {
    return -a as I64;
}
// (x as F32) as F64: 2^24 + 1 rounds to 2^24 on the way through F32.
fn chain(x: I32) -> F64 {
    return x as F32 as F64;
}
// a + (b as I64), not (a + b) as I64: the I32 sum would wrap.
fn add(a: I64, b: I32) -> I64 {
    return a + b as I64;
}
fn main() -> I32 {
    return 0;
}
";
    std::fs::write(dir.join("prec.ysu"), src).unwrap();
    let (ok, log) = y(&dir, "prec.ysu", &["--emit-llvm", "-o", "prec.ll"]);
    assert!(ok, "{log}");
    let c = "#include <stdint.h>\n#include <stdio.h>\n\
             int64_t neg_first(int32_t); double chain(int32_t); int64_t add(int64_t, int32_t);\n\
             int main(void) { printf(\"%lld %.1f %lld\\n\", (long long)neg_first(INT32_MIN), \
             chain(16777217), (long long)add(4294967296LL, 2147483647)); return 0; }\n";
    std::fs::write(dir.join("prec.c"), c).unwrap();
    let bin = dir.join("prec");
    let built = Command::new("clang").args(["-O1", "-Wno-override-module", "-o"]).arg(&bin)
        .arg(dir.join("prec.c")).arg(dir.join("prec.ll")).output().unwrap();
    assert!(built.status.success(), "{}", String::from_utf8_lossy(&built.stderr));
    let out = Command::new(&bin).output().unwrap();
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(),
               "-2147483648 16777216.0 6442450943");
}

#[test]
fn as_binds_tighter_than_a_binary_operator() {
    // a * (b as F64) is an I32 times an F64: a type error. Were it parsed as
    // (a * b) as F64 it would compile.
    let dir = scratch("tight");
    std::fs::write(dir.join("t.ysu"),
        "fn f(a: I32, b: I32) -> F64 {\n    return a * b as F64;\n}\nfn main() -> I32 {\n    return 0;\n}\n").unwrap();
    let (ok, log) = y(&dir, "t.ysu", &["--emit-llvm", "-o", "t.ll"]);
    assert!(!ok, "a * b as F64 compiled:\n{log}");
    assert!(log.contains("mismatch"), "{log}");
}

#[test]
fn refused_casts_are_refused_by_name() {
    let dir = scratch("refuse");
    let cases: &[(&str, &str, &str)] = &[
        ("literal", "let y: F64 = 0.1 as F64;", "Cast of a literal"),
        ("literal_expr", "let y: F64 = (1 + 2) as F64;", "Cast of a literal"),
        ("to_u8", "let x: I32 = 300;\n    let y: U8 = x as U8;", "Cannot cast to `U8`"),
        ("to_f16", "let x: F32 = 1.5;\n    let y: F16 = x as F16;", "Cannot cast to `F16`"),
        ("to_bool", "let x: I32 = 1;\n    let y: bool = x as bool;", "Cannot cast to `bool`"),
        ("from_bool", "let x: bool = true;\n    let y: I32 = x as I32;", "Cannot cast a `bool`"),
    ];
    for (tag, body, needle) in cases {
        let src = format!("fn main() -> I32 {{\n    {}\n    return 0;\n}}\n", body);
        std::fs::write(dir.join(format!("{tag}.ysu")), &src).unwrap();
        let (ok, log) = y(&dir, &format!("{tag}.ysu"), &["--emit-llvm", "-o", "x.ll"]);
        assert!(!ok, "{tag} compiled:\n{src}\n{log}");
        assert!(log.contains(needle), "{tag}: expected {needle:?} in:\n{log}");
    }
    // Control: the same shapes with a typed operand and a supported target compile.
    let src = "fn main() -> I32 {\n    let x: I32 = 300;\n    let y: F64 = x as F64;\n    let z: U32 = x as U32;\n    return 0;\n}\n";
    std::fs::write(dir.join("ok.ysu"), src).unwrap();
    let (ok, log) = y(&dir, "ok.ysu", &["--emit-llvm", "-o", "ok.ll"]);
    assert!(ok, "{log}");
}

fn elem_bytes(t: &str) -> usize {
    if matches!(t, "I32" | "U32" | "F32") { 4 } else { 8 }
}

/// The PTX backend, on the GPU: one kernel per conversion, `Out[i] = X[i] as
/// To` over the same inputs, against the same oracle. Skips without a GPU.
#[test]
fn every_conversion_on_the_gpu_matches_rust_as() {
    let Some(ctx) = y::cuda_runtime::CudaContext::new() else {
        eprintln!("SKIP: no CUDA device, so the PTX conversions were not run");
        return;
    };
    let dir = scratch("ptx");
    let mut src = String::new();
    for from in TYPES {
        for to in TYPES {
            writeln!(src, "kernel {}(X: GlobalMemory<{from}>, Out: GlobalMemory<{to}>) {{\n    \
                           let i: I32 = thread_idx_x();\n    let x: {from} = X[i];\n    \
                           Out[i] = x as {to};\n}}", fname(from, to)).unwrap();
        }
    }
    src.push_str("fn main() {}\n");
    std::fs::write(dir.join("casts.ysu"), &src).unwrap();
    let (ok, log) = y(&dir, "casts.ysu", &["--emit-ptx", "-o", "casts.ptx"]);
    assert!(ok, "--emit-ptx failed:\n{log}");
    let ptx = std::fs::read_to_string(dir.join("casts.ptx")).unwrap();

    let mut printed = String::new();
    for from in TYPES {
        let xs = inputs(from);
        let mut bytes = Vec::new();
        for b in &xs {
            bytes.extend_from_slice(&b.to_le_bytes()[..elem_bytes(from)]);
        }
        for to in TYPES {
            let module = ctx.load_ptx(&ptx, &fname(from, to)).expect("load");
            let x = ctx.alloc(bytes.len()).unwrap();
            let out = ctx.alloc(xs.len() * elem_bytes(to)).unwrap();
            ctx.memcpy_htod_at(&x, 0, &bytes).unwrap();
            ctx.memset_u8(&out, 0xA5).unwrap();
            ctx.launch(&module, (1, 1, 1), (xs.len() as u32, 1, 1), 0,
                       &[x.device_ptr(), out.device_ptr()]).expect("launch");
            ctx.synchronize().unwrap();
            let mut got = vec![0u8; xs.len() * elem_bytes(to)];
            ctx.memcpy_dtoh_at(&mut got, &out, 0).unwrap();
            for (k, b) in xs.iter().enumerate() {
                let w = elem_bytes(to);
                let mut v = [0u8; 8];
                v[..w].copy_from_slice(&got[k * w..(k + 1) * w]);
                writeln!(printed, "{} {:x} {:x}", fname(from, to), b, u64::from_le_bytes(v)).unwrap();
            }
        }
    }
    let n = check_results("PTX on the GPU", &printed);
    eprintln!("PTX on {}: {n} conversions equal Rust's `as`", ctx.device_name());
}
