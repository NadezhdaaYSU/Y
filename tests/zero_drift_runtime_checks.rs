//! A `@ZeroDrift` accumulator stops the program instead of computing a
//! different number.
//!
//! The accumulator is an exact integer or fixed-point value whose format the
//! backend picks to hold the declared `@bounds`. Three things went wrong
//! silently, on the LLVM backend and the GPU alike:
//!
//! * **The declared range was trusted.** `@bounds(-1024.0, 1024.0)` sized the
//!   accumulator and nothing tested it: 20 terms of 100 reached 2000 and the
//!   program carried on.
//! * **A term was converted unchecked.** LLVM's `fptosi` of a value the format
//!   cannot hold is poison: one term of 1e12 made the total 0, exit 0. PTX's
//!   `cvt` saturates, and a NaN converts to 0.
//! * **The accumulate wrapped.** A plain `add` (`add.s64` on the GPU): two
//!   I64 terms of 5e18 overflowed into a negative total.
//!
//! Now a term is tested before it is converted, the accumulate is checked for
//! overflow, and the declared range after every store; any of them stops the
//! program (a message and exit 1 on the host, a trap on the GPU). Each case
//! has an in-range control.
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

fn source(tag: &str, src: &str) -> PathBuf {
    let dir = pinned::pinned_scratch(&format!("zdrt_{tag}"), pinned::SM_PINNED);
    let path = dir.join(format!("{tag}.ysu"));
    std::fs::write(&path, src).unwrap();
    path
}

fn y(path: &Path, args: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(path)
        .args(args)
        .current_dir(path.parent().unwrap())
        .output()
        .expect("run Y");
    (out.status.success(), format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)))
}

/// Build with the default backend and run: (exit code, output).
fn run(tag: &str, src: &str) -> (Option<i32>, String) {
    let path = source(tag, src);
    let bin = path.with_extension("bin");
    let (ok, out) = y(&path, &["-o", bin.to_str().unwrap()]);
    assert!(ok, "{tag}: {out}");
    let r = Command::new(&bin).output().expect("run the program");
    (r.status.code(), format!("{}{}", String::from_utf8_lossy(&r.stdout), String::from_utf8_lossy(&r.stderr)))
}

fn bounded(count: u32, term: &str) -> String {
    format!(
        "fn main() -> I32 {{\n    @ZeroDrift\n    @bounds(-1024.0, 1024.0)\n    let acc: F32 = 0.0;\n    \
         @invariant(i >= 0)\n    for i in 0..{count} {{\n        acc += {term};\n    }}\n    print_int(acc);\n    \
         println(\"\");\n    return 0;\n}}\n"
    )
}

const WIDE: &str = "@unsafe
fn big(n: I64) -> I64 {
    return n;
}

fn main() -> I32 {
    @ZeroDrift
    let acc: I64 = 0;
    acc OP big(FIRST);
    acc OP big(SECOND);
    print_int(acc);
    println(\"\");
    return 0;
}
";

fn wide(op: &str, first: &str, second: &str) -> String {
    WIDE.replace("OP", op).replace("FIRST", first).replace("SECOND", second)
}

#[test]
fn the_declared_range_is_tested_after_every_store() {
    let (code, out) = run("in_range", &bounded(5, "100.0"));
    assert_eq!((code, out.trim()), (Some(0), "500"), "the control");
    let (code, out) = run("past_range", &bounded(20, "100.0"));
    assert_eq!(code, Some(1), "{out}");
    assert!(out.contains("a @ZeroDrift accumulator reached 1100, outside its @bounds(-1024, 1024); stopping"), "{out}");
}

#[test]
fn a_term_the_format_cannot_hold_stops_the_program() {
    // 1e12 in the accumulator's units is beyond any i64; the conversion was
    // poison and the program printed 0.
    let (code, out) = run("huge_term", &bounded(1, "1000000000000.0"));
    assert_eq!(code, Some(1), "{out}");
    assert!(out.contains("does not fit its exact representation; stopping"), "{out}");
    // A U64 term at 2^63 does not fit the I64 an integer accumulator is.
    let u64_term = "@unsafe\nfn big(n: U64) -> U64 {\n    return n + n;\n}\n\nfn main() -> I32 {\n    @ZeroDrift\n    \
                    let acc: I64 = 0;\n    acc += big(4611686018427387904);\n    print_int(acc);\n    println(\"\");\n    return 0;\n}\n";
    let (code, out) = run("u64_term", u64_term);
    assert_eq!(code, Some(1), "{out}");
    assert!(out.contains("does not fit its exact representation"), "{out}");
}

#[test]
fn an_accumulator_that_overflows_stops_the_program() {
    let (code, out) = run("wide_in", &wide("+=", "4000000000000000000", "4000000000000000000"));
    assert_eq!((code, out.trim()), (Some(0), "8000000000000000000"), "the control");
    for (tag, op, a, b) in [
        ("add_over", "+=", "5000000000000000000", "5000000000000000000"),
        ("sub_over", "-=", "-5000000000000000000", "-5000000000000000000"),
    ] {
        let (code, out) = run(tag, &wide(op, a, b));
        assert_eq!(code, Some(1), "{tag}: {out}");
        assert!(out.contains("a @ZeroDrift accumulator overflowed its exact representation; stopping"), "{tag}: {out}");
    }
}

#[test]
fn the_jit_stops_too() {
    let path = source("jit", &bounded(20, "100.0"));
    let out = Command::new(env!("CARGO_BIN_EXE_Y")).arg(&path).arg("--jit").current_dir(path.parent().unwrap()).output().unwrap();
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert_eq!(out.status.code(), Some(1), "{text}");
    assert!(text.contains("outside its @bounds(-1024, 1024)"), "{text}");
}

/// The kernels sum `COUNT` loaded values. `ELEM` is the buffer type.
fn kernel(elem: &str, op: &str, count: u32, bounds: &str) -> String {
    format!(
        "kernel k(A: GlobalMemory<{elem}>, Out: GlobalMemory<{elem}>) {{\n    @ZeroDrift\n    {bounds}\n    \
         let mut sum: {elem} = {zero};\n    @invariant(i >= 0)\n    for i in 0..{count} {{\n        sum {op} A[i];\n    }}\n    \
         Out[0] = sum;\n}}\n\nfn main() {{}}\n",
        zero = if elem == "F32" { "0.0" } else { "0" }
    )
}

/// On the device, each launch in a child process (this test re-run with
/// `Y_DRIFT_GPU_CASE`): a trap leaves the device unusable for its process.
#[test]
fn the_gpu_traps_where_the_host_stops() {
    if let Ok(case) = std::env::var("Y_DRIFT_GPU_CASE") {
        let mut parts = case.split('|');
        let (ptx_path, elem) = (parts.next().unwrap(), parts.next().unwrap());
        let values: Vec<f64> = parts.map(|v| v.parse().unwrap()).collect();
        let ptx = std::fs::read_to_string(ptx_path).unwrap();
        let ctx = y::cuda_runtime::CudaContext::new().expect("a context");
        let module = ctx.load_ptx(&ptx, "k").expect("load");
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| if elem == "F32" { (*v as f32).to_le_bytes().to_vec() } else { (*v as i64).to_le_bytes().to_vec() })
            .collect();
        let (a, o) = (ctx.alloc(bytes.len()).unwrap(), ctx.alloc(8).unwrap());
        ctx.memcpy_htod_at(&a, 0, &bytes).unwrap();
        ctx.memcpy_htod_at(&o, 0, &[0u8; 8]).unwrap();
        match ctx.launch(&module, (1, 1, 1), (1, 1, 1), 0, &[a.device_ptr(), o.device_ptr()]).and_then(|_| ctx.synchronize()) {
            Ok(()) => {
                let mut got = [0u8; 8];
                ctx.memcpy_dtoh_at(&mut got, &o, 0).unwrap();
                if elem == "F32" {
                    println!("RESULT {}", f32::from_le_bytes(got[..4].try_into().unwrap()));
                } else {
                    println!("RESULT {}", i64::from_le_bytes(got));
                }
            }
            Err(e) => println!("TRAPPED {e}"),
        }
        return;
    }
    if y::cuda_runtime::CudaContext::new().is_none() {
        eprintln!("SKIP: no CUDA device, so no kernel was run");
        return;
    }
    let range = "@bounds(-1024.0, 1024.0)";
    // An F16 accumulator needs no `@bounds` (Q32.32 holds F16's range), so
    // nothing but the term test stands between a NaN term and the total:
    // the device converts NaN to i64::MIN (measured), which a declared range
    // would also have caught.
    let unbounded_f16 = "kernel k(A: GlobalMemory<F32>, Out: GlobalMemory<F32>) {\n    @ZeroDrift\n    \
                         let mut sum: F16 = 0.0;\n    @invariant(i >= 0)\n    for i in 0..1 {\n        sum += A[i];\n    \
                         }\n    Out[0] = sum;\n}\n\nfn main() {}\n";
    let cases: [(&str, String, &str, Vec<f64>, Option<&str>); 8] = [
        ("f_in", kernel("F32", "+=", 5, range), "F32", vec![100.0; 5], Some("RESULT 500")),
        ("f_past", kernel("F32", "+=", 20, range), "F32", vec![100.0; 20], None),
        ("f_nan", kernel("F32", "+=", 1, range), "F32", vec![f64::NAN], None),
        ("h_in", unbounded_f16.to_string(), "F32", vec![2.5], Some("RESULT 2.5")),
        ("h_nan", unbounded_f16.to_string(), "F32", vec![f64::NAN], None),
        ("i_in", kernel("I64", "+=", 3, ""), "I64", vec![1000.0; 3], Some("RESULT 3000")),
        ("i_add_over", kernel("I64", "+=", 2, ""), "I64", vec![5e18; 2], None),
        ("i_sub_over", kernel("I64", "-=", 2, ""), "I64", vec![-5e18; 2], None),
    ];
    for (tag, src, elem, values, want) in cases {
        let path = source(&format!("gpu_{tag}"), &src);
        let ptx_path = path.with_extension("ptx");
        let (ok, out) = y(&path, &["--emit-ptx", "-o", ptx_path.to_str().unwrap()]);
        assert!(ok, "{tag}: {out}");
        let case: Vec<String> = [ptx_path.display().to_string(), elem.to_string()]
            .into_iter()
            .chain(values.iter().map(|v| v.to_string()))
            .collect();
        let child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "the_gpu_traps_where_the_host_stops", "--nocapture"])
            .env("Y_DRIFT_GPU_CASE", case.join("|"))
            .output()
            .expect("run the launch in a child");
        let text = String::from_utf8_lossy(&child.stdout).into_owned();
        match want {
            Some(line) => assert!(text.contains(&format!("{line}\n")), "{tag}: {text}"),
            None => assert!(text.contains("TRAPPED"), "{tag}: the kernel ran to completion: {text}"),
        }
    }
}
