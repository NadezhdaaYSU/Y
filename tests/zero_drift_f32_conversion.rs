//! A `@ZeroDrift` term in `f32` is converted with integer arithmetic on the
//! GPU, and must still mean exactly what it meant through `f64`.
//!
//! The conversion used to widen every term to f64, scale, add ±0.5 and
//! truncate: five f64 instructions per term, on cards that run f64 at 1/64 of
//! the f32 rate. It now takes the float's bits apart (`emit_drift_f32_to_fixed`).
//! Three things must hold, and each is checked here by running the kernel:
//!
//! * **The value.** Round to nearest, ties away from zero, exactly as the
//!   documented semantics and the LLVM backend: exact ties `(k + 0.5) * 2^-32`,
//!   their neighbours one ulp either side, subnormals, both zeros, and values
//!   across the declared range, each compared bit for bit with a reference
//!   computed here from the definition.
//! * **The order.** Two threads sum the same 4001 terms in opposite orders and
//!   get the same bits, where an f32 control kernel on the same input must not.
//! * **The stops.** With narrow bounds the accumulate's overflow test is
//!   elided and the term traps at 2^62 instead of 2^63; a program must stop in
//!   exactly the cases it stopped before. Each case runs in a child process,
//!   because a trap leaves the device unusable for its process.
//!
//! Hermetic PTX checks pin the shape: no f64 in the conversion, the overflow
//! test absent only under narrow bounds, one compare for the range.
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

const NARROW: &str = "@bounds(min=-10000.0, max=10000.0)";
// 1e9 in Q32.32 is about 4.3e18, above 2^61: the overflow test stays.
const WIDE: &str = "@bounds(min=-1000000000.0, max=1000000000.0)";

fn compile(tag: &str, src: &str) -> String {
    let dir = pinned::pinned_scratch(&format!("zdf32_{tag}"), pinned::SM_PINNED);
    let path = dir.join(format!("{tag}.ysu"));
    std::fs::write(&path, src).unwrap();
    let ptx = path.with_extension("ptx");
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&path)
        .args(["--emit-ptx", "-o", ptx.to_str().unwrap()])
        .current_dir(&dir)
        .output()
        .expect("run Y");
    assert!(
        out.status.success(),
        "{tag}: {}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    ptx.display().to_string()
}

/// Thread i adds the single term `A[i]` and stores the accumulator.
fn single_terms(n: usize, bounds: &str) -> String {
    format!(
        "kernel k(A: GlobalMemory<F32>, Out: GlobalMemory<F32>) {{\n    let i: I32 = global_thread_id();\n    \
         if i < {n} {{\n        @ZeroDrift\n        {bounds}\n        let mut acc: F32 = 0.0;\n        acc += A[i];\n        \
         Out[i] = acc;\n    }}\n}}\n\nfn main() {{}}\n"
    )
}

/// Thread 0 adds `A[0..n]` in order; the stops cases.
fn sequence(n: usize, bounds: &str) -> String {
    sequence_op(n, bounds, "+=")
}

/// The same with `op`: `-=` takes the subtracting accumulate.
fn sequence_op(n: usize, bounds: &str, op: &str) -> String {
    format!(
        "kernel k(A: GlobalMemory<F32>, Out: GlobalMemory<F32>) {{\n    @ZeroDrift\n    {bounds}\n    \
         let mut acc: F32 = 0.0;\n    @invariant(j >= 0)\n    for j in 0..{n} step 1 {{\n        acc {op} A[j];\n    }}\n    \
         Out[0] = acc;\n}}\n\nfn main() {{}}\n"
    )
}

/// Thread 0 adds 4001 terms forwards, thread 1 backwards.
fn two_orders(drift: bool) -> String {
    let decl = if drift {
        format!("@ZeroDrift\n        {NARROW}\n        let mut acc: F32 = 0.0;")
    } else {
        "let mut acc: F32 = 0.0;".to_string()
    };
    format!(
        "kernel k(A: GlobalMemory<F32>, Out: GlobalMemory<F32>) {{\n    let t: I32 = global_thread_id();\n    \
         if t < 2 {{\n        {decl}\n        @invariant(j >= 0)\n        for j in 0..4001 step 1 {{\n            \
         let idx: I32 = j + t * (4000 - 2 * j);\n            acc += A[idx];\n        }}\n        Out[t] = acc;\n    }}\n}}\n\n\
         fn main() {{}}\n"
    )
}

/// The documented semantics: round(x * 2^32), ties away from zero, summed
/// exactly, read back as the f32 nearest to S * 2^-32.
fn reference(terms: &[f32]) -> f32 {
    let s: i64 = terms
        .iter()
        .map(|&x| {
            let y = x as f64 * 4294967296.0;
            (y.abs() + 0.5).floor().copysign(y) as i64
        })
        .sum();
    (s as f64 / 4294967296.0) as f32
}

/// The child: launch `ptx` once over `values` with `threads` threads and print
/// the outputs, or `TRAPPED`.
fn child_run() -> Option<()> {
    let case = std::env::var("Y_ZD_F32_CASE").ok()?;
    let mut parts = case.split('|');
    let ptx_path = parts.next().unwrap();
    let threads: u32 = parts.next().unwrap().parse().unwrap();
    let outs: usize = parts.next().unwrap().parse().unwrap();
    let values: Vec<f32> = parts.map(|v| f32::from_bits(v.parse::<u32>().unwrap())).collect();
    let ctx = y::cuda_runtime::CudaContext::new().expect("a context");
    let module = ctx.load_ptx(&std::fs::read_to_string(ptx_path).unwrap(), "k").expect("load");
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let (a, o) = (ctx.alloc(bytes.len().max(4)).unwrap(), ctx.alloc(outs * 4).unwrap());
    ctx.memcpy_htod_at(&a, 0, &bytes).unwrap();
    ctx.memcpy_htod_at(&o, 0, &vec![0u8; outs * 4]).unwrap();
    let grid = (threads + 255) / 256;
    match ctx
        .launch(&module, (grid, 1, 1), (threads.min(256), 1, 1), 0, &[a.device_ptr(), o.device_ptr()])
        .and_then(|_| ctx.synchronize())
    {
        Ok(()) => {
            let mut got = vec![0u8; outs * 4];
            ctx.memcpy_dtoh_at(&mut got, &o, 0).unwrap();
            let words: Vec<String> =
                got.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap()).to_string()).collect();
            println!("RESULT {}", words.join(" "));
        }
        Err(e) => println!("TRAPPED {e}"),
    }
    Some(())
}

/// Runs a case in a child process: `Ok(bits of the outputs)` or `Err(text)`
/// when it trapped. `None` without a GPU (a reported skip).
fn launch(test: &str, ptx: &str, threads: u32, outs: usize, values: &[f32]) -> Option<Result<Vec<u32>, String>> {
    if y::cuda_runtime::CudaContext::new().is_none() {
        eprintln!("SKIP: no CUDA device, so no kernel was run");
        return None;
    }
    let case: Vec<String> = [ptx.to_string(), threads.to_string(), outs.to_string()]
        .into_iter()
        .chain(values.iter().map(|v| v.to_bits().to_string()))
        .collect();
    let child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--nocapture"])
        .env("Y_ZD_F32_CASE", case.join("|"))
        .output()
        .expect("run the launch in a child");
    let text = String::from_utf8_lossy(&child.stdout).into_owned();
    if let Some(line) = text.lines().find(|l| l.starts_with("RESULT ")) {
        return Some(Ok(line[7..].split(' ').map(|w| w.parse().unwrap()).collect()));
    }
    assert!(text.contains("TRAPPED"), "the child neither finished nor trapped: {text}");
    Some(Err(text))
}

fn adversarial() -> Vec<f32> {
    let mut v = Vec::new();
    let step = 2f64.powi(-32);
    for k in -40i32..=40 {
        let t = ((k as f64) + 0.5) * step;
        let f = t as f32; // exact: a small odd multiple of 2^-33
        assert_eq!(f as f64, t);
        v.extend([f, f32::from_bits(f.to_bits() + 1), f32::from_bits(f.to_bits().saturating_sub(1))]);
    }
    v.extend([0.0, -0.0, f32::from_bits(1), -f32::from_bits(1), f32::MIN_POSITIVE, -f32::MIN_POSITIVE]);
    v.extend([1.0, -1.0, 0.1, -0.1, 9999.5, -9999.5, 1234.5678, 2f32.powi(-33), -(2f32.powi(-33))]);
    v.extend([0.499_999_97 * step as f32, 3.0 * 2f32.powi(-34), 5.0 * 2f32.powi(-34), -3.0 * 2f32.powi(-34)]);
    // Values of every magnitude inside the bounds, deterministic.
    let mut x: u32 = 0x9E37_79B9;
    for _ in 0..600 {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        let mag = 2f32.powi((x % 48) as i32 - 34) * (1.0 + (x >> 9) as f32 / 8_388_608.0);
        let s = if x & 1 == 0 { mag } else { -mag };
        if s.abs() < 9999.0 {
            v.push(s);
        }
    }
    v
}

#[test]
fn every_term_rounds_to_the_documented_value() {
    if child_run().is_some() {
        return;
    }
    let vals = adversarial();
    let ptx = compile("single", &single_terms(vals.len(), NARROW));
    let Some(r) = launch("every_term_rounds_to_the_documented_value", &ptx, vals.len() as u32, vals.len(), &vals) else {
        return;
    };
    let got = r.expect("no trap: every term is in range");
    let mut bad = Vec::new();
    for (i, &x) in vals.iter().enumerate() {
        let want = reference(&[x]).to_bits();
        if got[i] != want {
            bad.push(format!("  {x:e} ({:#010x}): got {:e}, want {:e}", x.to_bits(), f32::from_bits(got[i]), f32::from_bits(want)));
        }
    }
    // The ties are what distinguishes ties-away from ties-to-even: make sure
    // the set has them, so a passing run is not a vacuous one.
    let ties = vals.iter().filter(|&&x| {
        let y = x as f64 * 4294967296.0;
        y.fract().abs() == 0.5
    });
    assert!(ties.count() >= 80, "the input lost its exact ties");
    assert!(bad.is_empty(), "{} of {} terms converted wrongly:\n{}", bad.len(), vals.len(), bad.join("\n"));
}

#[test]
fn opposite_orders_agree_where_f32_does_not() {
    if child_run().is_some() {
        return;
    }
    let mut vals = vec![1.0f32];
    vals.extend(std::iter::repeat(1.0e-7f32).take(4000));
    let zd = compile("orders_zd", &two_orders(true));
    let fl = compile("orders_f32", &two_orders(false));
    let test = "opposite_orders_agree_where_f32_does_not";
    let Some(z) = launch(test, &zd, 2, 2, &vals) else { return };
    let z = z.expect("in range");
    assert_eq!(z[0], z[1], "@ZeroDrift: the two orders disagree ({} vs {})", f32::from_bits(z[0]), f32::from_bits(z[1]));
    assert_eq!(z[0], reference(&vals).to_bits(), "and the total is the documented one");
    let f = launch(test, &fl, 2, 2, &vals).unwrap().expect("in range");
    assert_ne!(f[0], f[1], "the f32 control must drift on this input, or the test proves nothing");
}

#[test]
fn programs_stop_in_the_same_cases() {
    if child_run().is_some() {
        return;
    }
    let test = "programs_stop_in_the_same_cases";
    let two62 = 2f32.powi(30); // 2^30 * 2^32 = 2^62: the narrow-bounds term limit
    // (bounds, op, terms, Some(result) or None for a stop)
    let cases: Vec<(&str, &str, Vec<f32>, Option<f32>)> = vec![
        (NARROW, "+=", vec![9999.0], Some(9999.0)),
        // Exactly on each bound is inside it.
        (NARROW, "+=", vec![10000.0], Some(10000.0)),
        (NARROW, "+=", vec![-10000.0], Some(-10000.0)),
        // `-=` takes the subtracting accumulate, overflow test elided too.
        (NARROW, "-=", vec![3.5, -1.25], Some(-2.25)),
        (NARROW, "-=", vec![-6000.0, -6000.0], None),
        (NARROW, "+=", vec![f32::NAN], None),
        (NARROW, "+=", vec![f32::INFINITY], None),
        (NARROW, "+=", vec![-f32::INFINITY], None),
        (NARROW, "+=", vec![two62], None),                        // the term test, at 2^62
        (NARROW, "+=", vec![f32::from_bits(two62.to_bits() - 1)], None), // just below it: the range test
        (NARROW, "+=", vec![10001.0], None),
        (NARROW, "+=", vec![6000.0, 6000.0], None),
        (NARROW, "+=", vec![6000.0, -6000.0, 9000.0], Some(9000.0)),
        (WIDE, "+=", vec![1.5e9], None),                          // range
        (WIDE, "+=", vec![3.1e9], None),                          // the term test, at 2^63
        // 3.9e9 * 2^32 = 1.8 * 2^63: past 2^63 it would wrap to -0.2 * 2^63,
        // INSIDE the wide bounds, so only the term test can stop it.
        (WIDE, "+=", vec![3.9e9], None),
        (WIDE, "+=", vec![f32::NAN], None),
        (WIDE, "+=", vec![9e8, 9e8], None),
        (WIDE, "+=", vec![9e8, -9e8, 5e8], Some(5e8)),
        // A term of 1.4e9 * 2^32 = 6.0e18 >= 2^62 that brings the total back
        // inside: it ran before, so it must run now. The wide bounds are why
        // the overflow test is kept and the term limit stays at 2^63.
        (WIDE, "+=", vec![-9e8, 1.4e9], Some(5e8)),
    ];
    let mut bad = Vec::new();
    for (i, (bounds, op, terms, want)) in cases.iter().enumerate() {
        let ptx = compile(&format!("stop{i}"), &sequence_op(terms.len(), bounds, op));
        let Some(r) = launch(test, &ptx, 1, 1, terms) else { return };
        match (r, want) {
            (Ok(got), Some(w)) if got[0] == w.to_bits() => {}
            (Err(_), None) => {}
            (r, w) => bad.push(format!("  {bounds} {op} {terms:?}: got {:?}, want {w:?}", r.map(|g| f32::from_bits(g[0])))),
        }
    }
    assert!(bad.is_empty(), "a program stopped (or ran) where it did not before:\n{}", bad.join("\n"));
}

/// The emitted shape, hermetic: what a reviewer would grep for.
#[test]
fn the_conversion_uses_no_double_and_the_overflow_test_only_where_needed() {
    let body = |p: &str| std::fs::read_to_string(p).unwrap();
    let narrow = body(&compile("shape_narrow", &sequence(4, NARROW)));
    let wide = body(&compile("shape_wide", &sequence(4, WIDE)));
    for (tag, ptx) in [("narrow", &narrow), ("wide", &wide)] {
        for op in ["cvt.f64.f32", "add.f64", "selp.f64", "cvt.rzi.s64.f64", "div.rn.f64"] {
            assert!(!ptx.contains(op), "{tag}: the f32 term path emitted `{op}`:\n{ptx}");
        }
        assert!(ptx.contains("mov.b32"), "{tag}: the term's bits are not taken apart:\n{ptx}");
        assert!(ptx.contains("mul.f64"), "{tag}: the read-back multiplies by 2^-32:\n{ptx}");
    }
    assert!(!narrow.contains("xor.b64"), "narrow bounds: the overflow test is provably dead:\n{narrow}");
    assert!(wide.contains("xor.b64"), "wide bounds: the overflow test must stay:\n{wide}");
    assert!(narrow.contains("setp.gt.u64"), "the range is one unsigned compare:\n{narrow}");
    // 2^62 for the narrow term test (s > 38), 2^63 for the wide (s > 39).
    assert!(narrow.contains(", 38;"), "narrow: the term traps at 2^62:\n{narrow}");
    assert!(wide.contains(", 39;"), "wide: the term traps at 2^63:\n{wide}");
    let Some(tool) = which("ptxas") else {
        eprintln!("SKIP: no ptxas, so the shape was not assembled");
        return;
    };
    for (tag, src) in [("narrow", sequence(4, NARROW)), ("wide", sequence(4, WIDE))] {
        let p = compile(&format!("asm_{tag}"), &src);
        let out = Command::new(&tool).args(["-arch=sm_80", &p, "-o", "/dev/null"]).output().unwrap();
        assert!(out.status.success(), "{tag}: ptxas: {}", String::from_utf8_lossy(&out.stderr));
    }
}

fn which(tool: &str) -> Option<PathBuf> {
    std::env::var_os("PATH")?
        .to_str()?
        .split(':')
        .map(|d| Path::new(d).join(tool))
        .chain([PathBuf::from("/opt/cuda/bin").join(tool), PathBuf::from("/usr/local/cuda/bin").join(tool)])
        .find(|p| p.is_file())
}
