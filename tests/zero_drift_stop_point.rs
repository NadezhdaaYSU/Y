//! Where a `@ZeroDrift` loop stops on the GPU (decided 2026-10-28).
//!
//! Inside a `for` that nothing outside its thread can watch, a failed check
//! (a term that does not fit, an accumulate that overflows, a value outside
//! its `@bounds`) only sets a flag, and the program stops after the loop. The
//! loop then has no branch out, and ptxas unrolls it. Any other loop still
//! stops at the term. Checked here:
//! * the emitted shape, hermetic: which loops move their stop to the end,
//!   which keep it at the term, and that a nested loop shares the flag;
//! * on the GPU: the same inputs stop, wherever the bad term is, and the
//!   others give the exact value. Each launch runs in a child process,
//!   because a trap leaves the device unusable for its process.
//!
//! What a GPU cannot show, the stores a kernel made before it stopped, is
//! checked under the CPU interpreter in
//! `python/y_lang/tests/test_drift_stop_point.py`.
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

const NARROW: &str = "@bounds(min=-10000.0, max=10000.0)";
// 1e9 in Q32.32 is above 2^61: the overflow test stays.
const WIDE: &str = "@bounds(min=-1000000000.0, max=1000000000.0)";

fn compile(tag: &str, src: &str) -> String {
    let dir = pinned::pinned_scratch(&format!("zdstop_{tag}"), pinned::SM_PINNED);
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

/// `acc` declared with `bounds`, a `for j in 0..N` (with `step`) around
/// `body`, and `acc` stored after it.
fn with_loop(bounds: &str, step: &str, body: &str) -> String {
    format!(
        "kernel k(A: GlobalMemory<F32>, Out: GlobalMemory<F32>, C: GlobalMemory<I32>, N: I32, S: I32) {{\n    \
         @ZeroDrift\n    {bounds}\n    let mut acc: F32 = 0.0;\n    @invariant(j >= 0)\n    for j in 0..N{step} {{\n\
         {body}\n    }}\n    Out[0] = acc;\n}}\n\nfn main() {{}}\n"
    )
}

/// `M` rows of `N` terms, an inner loop in an outer one; `after_inner` follows the inner loop.
fn nested(after_inner: &str) -> String {
    format!(
        "kernel k(A: GlobalMemory<F32>, Out: GlobalMemory<F32>, M: I32, N: I32) {{\n    @ZeroDrift\n    {NARROW}\n    \
         let mut acc: F32 = 0.0;\n    @invariant(i >= 0)\n    for i in 0..M step 1 {{\n        @invariant(j >= 0)\n        \
         for j in 0..N step 1 {{\n            acc += A[i * N + j];\n        }}\n{after_inner}\n    }}\n    Out[0] = acc;\n}}\n\n\
         fn main() {{}}\n"
    )
}

const I64_LOOP: &str = "kernel k(A: GlobalMemory<I64>, Out: GlobalMemory<I64>, N: I32) {\n    @ZeroDrift\n    \
    let mut acc: I64 = 0;\n    @invariant(j >= 0)\n    for j in 0..N step 1 {\n        acc += A[j];\n    }\n    \
    Out[0] = acc;\n}\n\nfn main() {}\n";

/// One block's threads sum `X` exactly with `drift_add_to`, the full rounds
/// apart from the guarded last one.
const GRID: &str = "kernel k(X: GlobalMemory<F32>, ACC: GlobalMemory<I64>, OUT: GlobalMemory<F32>, N: I32, R: I32) {\n    \
    @ZeroDrift\n    @bounds(min=-1000000.0, max=1000000.0)\n    let mut acc: F32 = 0.0;\n    let t: I32 = thread_idx_x();\n    \
    @invariant(j >= 0)\n    for j in 0..R step 1 {\n        let i: I32 = t + j * block_dim_x();\n        if i < N {\n            \
    acc += X[i];\n        }\n    }\n    drift_add_to(ACC, OUT, acc);\n}\n\nfn main() {}\n";

/// Each `for` in `ptx`: its body (from `$LOOP_START_n` to `$LOOP_END_n+1`)
/// and the line after its end.
fn loops(ptx: &str) -> Vec<(String, String)> {
    let lines: Vec<&str> = ptx.lines().map(str::trim).collect();
    let label = |l: &str, kind: &str| l.strip_prefix(kind)?.strip_suffix(':')?.parse::<usize>().ok();
    let mut found = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        let Some(n) = label(l, "$LOOP_START_") else { continue };
        let e = (i..lines.len()).find(|&k| label(lines[k], "$LOOP_END_") == Some(n + 1)).unwrap_or_else(|| panic!("no end for loop {n}:\n{ptx}"));
        let after = lines[e + 1..].iter().find(|l| !l.is_empty()).unwrap_or(&"").to_string();
        found.push((lines[i + 1..e].join("\n"), after));
    }
    found
}

/// The flag a deferring loop sets: the predicate of its `or.pred`s, which must all agree.
fn flag(body: &str) -> Option<String> {
    let flags: Vec<&str> = body.lines().filter_map(|l| l.trim().strip_prefix("or.pred ")?.split(',').next()).collect();
    let first = *flags.first()?;
    assert!(flags.iter().all(|f| *f == first), "one loop sets two flags: {flags:?}\n{body}");
    Some(first.to_string())
}

/// `src` (whose kernel is `k`) defers: no trap inside any loop, a flag set
/// inside, cleared before the outermost loop and trapped on right after it.
fn assert_defers(tag: &str, src: &str) {
    let ptx = std::fs::read_to_string(compile(tag, src)).unwrap();
    let all = loops(&ptx);
    let (outer, after) = &all[0];
    assert!(!outer.contains("trap;"), "{tag}: the loop still traps at the term:\n{ptx}");
    let f = flag(outer).unwrap_or_else(|| panic!("{tag}: no check sets a flag in the loop:\n{ptx}"));
    assert_eq!(after, &format!("@{f} trap;"), "{tag}: no trap on the flag right after the loop:\n{ptx}");
    let start = ptx.find("$LOOP_START_").unwrap();
    assert!(ptx[..start].contains(&format!("setp.ne.u32 {f}, 0, 0;")), "{tag}: the flag is not cleared first:\n{ptx}");
    assert_eq!(ptx.matches(&format!("setp.ne.u32 {f}, 0, 0;")).count(), 1, "{tag}: the flag is cleared twice:\n{ptx}");
}

/// `src` keeps every stop at the term: no flag, and the drift checks trap inside the loop.
fn assert_keeps(tag: &str, src: &str) {
    let ptx = std::fs::read_to_string(compile(tag, src)).unwrap();
    assert!(!ptx.contains("or.pred"), "{tag}: a loop that can be seen moved its stop:\n{ptx}");
    assert!(loops(&ptx).iter().any(|(body, _)| body.contains("trap;")), "{tag}: no trap at the term:\n{ptx}");
}

#[test]
fn a_loop_that_stays_in_its_thread_stops_at_its_end() {
    let cases = [
        ("plain", with_loop(NARROW, " step 1", "        acc += A[j];")),
        ("no_step", with_loop(NARROW, "", "        acc += A[j];")),
        ("minus", with_loop(NARROW, " step 1", "        acc -= A[j];")),
        // The overflow test and both range compares move too.
        ("wide", with_loop(WIDE, " step 1", "        acc += A[j];")),
        ("branch", with_loop(NARROW, " step 1", "        let i: I32 = j * 2 + thread_idx_x();\n        if i < N {\n            acc += A[i];\n        }")),
        // Its only accumulator is declared inside it.
        ("let_inside", with_loop(NARROW, " step 1", &format!("        @ZeroDrift\n        {NARROW}\n        let mut part: F32 = 0.0;\n        part += A[j];"))),
        // Bounds past 2^63 raw units: the range takes two compares.
        ("very_wide", with_loop("@bounds(min=-2000000000.0, max=2000000000.0)", " step 1", "        acc += A[j];")),
        // An integer term goes through f64, with its own fit tests.
        ("int_term", with_loop(NARROW, " step 1", "        acc += C[j];")),
        ("i64", I64_LOOP.to_string()),
        ("u64_term", I64_LOOP.replace("A: GlobalMemory<I64>", "A: GlobalMemory<U64>")),
        ("grid_partial", GRID.to_string()),
    ];
    for (tag, src) in &cases {
        assert_defers(tag, src);
    }
    // The checks of each kind are all there, now setting the flag.
    for (tag, want, what) in [
        ("wide", 3, "the term, the overflow and the range"),
        ("very_wide", 4, "the term, the overflow and both bounds"),
        ("int_term", 4, "the two f64 fit tests, the overflow and the range"),
        ("u64_term", 2, "the U64 term and the overflow"),
    ] {
        let src = &cases.iter().find(|(t, _)| *t == tag).unwrap().1;
        let ptx = std::fs::read_to_string(compile(&format!("{tag}_count"), src)).unwrap();
        let (body, _) = &loops(&ptx)[0];
        assert_eq!(body.matches("or.pred").count(), want, "{tag}: {what}:\n{body}");
    }
    // The flag ends with its loop: a term after it stops at the term.
    let after = with_loop(NARROW, " step 1", "        acc += A[j];").replace("    Out[0] = acc;", "    acc += A[N];\n    Out[0] = acc;");
    let ptx = std::fs::read_to_string(compile("after", &after)).unwrap();
    let end = ptx.lines().position(|l| l.trim().starts_with("$LOOP_END_") && l.trim().ends_with(':')).unwrap();
    let tail = ptx.lines().skip(end).collect::<Vec<_>>().join("\n");
    assert_eq!(tail.matches("or.pred").count(), 0, "a term after the loop sets a flag nobody reads:\n{ptx}");
    assert!(tail.matches("trap;").count() >= 2, "the loop's stop and the later term's:\n{ptx}");
    // The hand-over comes after the stop, so a partial with a bad term is never added.
    let grid = std::fs::read_to_string(compile("grid_order", GRID)).unwrap();
    let f = flag(&loops(&grid)[0].0).unwrap();
    let (stop, atom) = (grid.find(&format!("@{f} trap;")).unwrap(), grid.find("atom.").expect("the hand-over's atomic"));
    assert!(stop < atom, "the partial is handed over before the stop:\n{grid}");
    if let Some(tool) = which("ptxas") {
        for (tag, src) in &cases {
            let p = compile(&format!("asm_{tag}"), src);
            let out = Command::new(&tool).args(["-arch=sm_80", &p, "-o", "/dev/null"]).output().unwrap();
            assert!(out.status.success(), "{tag}: ptxas: {}", String::from_utf8_lossy(&out.stderr));
        }
    } else {
        eprintln!("SKIP: no ptxas, so the deferring loops were not assembled");
    }
}

#[test]
fn loops_that_can_be_seen_or_may_not_end_stop_at_the_term() {
    let keep = [
        ("store", with_loop(NARROW, " step 1", "        acc += A[j];\n        Out[j] = acc;")),
        ("atomic", with_loop(NARROW, " step 1", "        acc += A[j];\n        atomic_add(C[1], 0, 1);")),
        ("return", with_loop(NARROW, " step 1", "        acc += A[j];\n        if j > 5 {\n            return;\n        }")),
        // A `while` may not end; the term beside it would stop no one.
        ("while", with_loop(NARROW, " step 1", "        acc += A[j];\n        let mut m: I32 = 0;\n        @invariant(m >= 0)\n        while m < 2 {\n            m = m + 1;\n        }")),
        ("while_term_inside", with_loop(NARROW, " step 1", "        let mut m: I32 = 0;\n        @invariant(m >= 0)\n        while m < 2 {\n            acc += A[j];\n            m = m + 1;\n        }")),
        ("store_in_else", with_loop(NARROW, " step 1", "        if A[j] > 0.0 {\n            acc += A[j];\n        } else {\n            Out[j] = acc;\n        }")),
        ("safe_store", with_loop(NARROW, " step 1", "        acc += A[j];\n        @safe {\n            Out[j] = acc;\n        }")),
        // A shuffle reads other threads' registers.
        ("shuffle", with_loop(NARROW, " step 1", "        let v: F32 = shfl_sync_bfly(A[j], 1);\n        acc += v;")),
        ("shuffle_in_if", with_loop(NARROW, " step 1", "        if shfl_sync_bfly(A[j], 1) > 0.0 {\n            acc += A[j];\n        }")),
        ("inner_step_2", with_loop(NARROW, " step 1", "        @invariant(m >= 0)\n        for m in 0..8 step 2 {\n            acc += A[j + m];\n        }")),
        // A counter stepping by more than 1 could wrap past `i32::MAX` and
        // never end. (The front end refuses that unless the end is bounded,
        // and a step it cannot bound at all.)
        ("step_2", with_loop(NARROW, " step 2", "        acc += A[j];").replace("0..N step 2", "0..64 step 2")),
        ("counter_assigned", with_loop(NARROW, " step 1", "        acc += A[j];\n        j = 3;")),
        // The inner loop stores, so neither loop moves its stop.
        ("nested_store", nested("        Out[i] = acc;").replace("acc += A[i * N + j];", "acc += A[i * N + j];\n            Out[j] = acc;")),
    ];
    for (tag, src) in &keep {
        assert_keeps(tag, src);
    }
    // A loop with no `@ZeroDrift` accumulation is untouched.
    let plain = "kernel k(A: GlobalMemory<F32>, Out: GlobalMemory<F32>, N: I32) {\n    let mut s: F32 = 0.0;\n    \
                 @invariant(j >= 0)\n    for j in 0..N step 1 {\n        s += A[j];\n    }\n    Out[0] = s;\n}\n\nfn main() {}\n";
    let ptx = std::fs::read_to_string(compile("no_drift", plain)).unwrap();
    assert!(!ptx.contains("or.pred") && !ptx.contains("trap;"), "a loop with no accumulator changed:\n{ptx}");
}

#[test]
fn a_nested_loop_shares_the_flag_and_the_outer_end_stops() {
    let ptx = std::fs::read_to_string(compile("nested", &nested(""))).unwrap();
    let all = loops(&ptx);
    assert_eq!(all.len(), 2, "{ptx}");
    let ((outer, outer_after), (inner, inner_after)) = (&all[0], &all[1]);
    let f = flag(outer).expect("the outer loop sets a flag");
    assert_eq!(flag(inner).as_deref(), Some(f.as_str()), "the inner loop sets another flag:\n{ptx}");
    assert!(!outer.contains("trap;"), "a trap inside the outer loop:\n{ptx}");
    assert!(!inner_after.contains("trap"), "the inner loop's end stops, inside the outer one:\n{ptx}");
    assert_eq!(outer_after, &format!("@{f} trap;"), "{ptx}");
    assert_defers("nested_whole", &nested(""));
    // An outer loop that stores keeps its stops; the pure inner loop still
    // defers, and stops at its own end, before the store.
    let ptx = std::fs::read_to_string(compile("inner_only", &nested("        Out[i] = acc;"))).unwrap();
    let all = loops(&ptx);
    let ((_, outer_after), (inner, inner_after)) = (&all[0], &all[1]);
    let f = flag(inner).expect("the inner loop defers");
    assert_eq!(inner_after, &format!("@{f} trap;"), "{ptx}");
    assert!(!outer_after.contains("trap"), "{ptx}");
}

/// One kernel argument: a buffer of 32-bit words, or a scalar.
enum Arg {
    Words(Vec<u32>),
    Scalar(u64),
}

fn f32s(v: &[f32]) -> Arg {
    Arg::Words(v.iter().map(|x| x.to_bits()).collect())
}

fn i64s(v: &[i64]) -> Arg {
    Arg::Words(v.iter().flat_map(|x| [*x as u64 as u32, (*x as u64 >> 32) as u32]).collect())
}

/// The child: launch `k` once and print every buffer afterwards, or `TRAPPED`.
fn child_run() -> Option<()> {
    let case = std::env::var("Y_ZD_STOP_CASE").ok()?;
    let mut parts = case.split('|');
    let (ptx_path, block) = (parts.next().unwrap(), parts.next().unwrap().parse::<u32>().unwrap());
    let ctx = y::cuda_runtime::CudaContext::new().expect("a context");
    let module = ctx.load_ptx(&std::fs::read_to_string(ptx_path).unwrap(), "k").expect("load");
    let mut bufs = Vec::new();
    let mut args = Vec::new();
    for a in parts {
        if let Some(s) = a.strip_prefix("s:") {
            args.push(s.parse::<u64>().unwrap());
            continue;
        }
        let words: Vec<u32> = a[2..].split(',').filter(|w| !w.is_empty()).map(|w| w.parse().unwrap()).collect();
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let b = ctx.alloc(bytes.len().max(8)).unwrap();
        ctx.memcpy_htod_at(&b, 0, &bytes).unwrap();
        args.push(b.device_ptr());
        bufs.push((b, bytes.len()));
    }
    match ctx.launch(&module, (1, 1, 1), (block, 1, 1), 0, &args).and_then(|_| ctx.synchronize()) {
        Ok(()) => {
            let text: Vec<String> = bufs
                .iter()
                .map(|(b, len)| {
                    let mut got = vec![0u8; *len];
                    ctx.memcpy_dtoh_at(&mut got, b, 0).unwrap();
                    got.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap()).to_string()).collect::<Vec<_>>().join(",")
                })
                .collect();
            println!("RESULT {}", text.join(";"));
        }
        Err(e) => println!("TRAPPED {e}"),
    }
    Some(())
}

/// Runs `k` in a child: `Ok(each buffer's words after the launch)` or `Err`
/// when it trapped. `None` without a GPU (a reported skip).
fn launch(test: &str, ptx: &str, block: u32, args: &[Arg]) -> Option<Result<Vec<Vec<u32>>, String>> {
    if y::cuda_runtime::CudaContext::new().is_none() {
        eprintln!("SKIP: no CUDA device, so no kernel was run");
        return None;
    }
    let mut case = vec![ptx.to_string(), block.to_string()];
    for a in args {
        case.push(match a {
            Arg::Words(w) => format!("w:{}", w.iter().map(u32::to_string).collect::<Vec<_>>().join(",")),
            Arg::Scalar(s) => format!("s:{s}"),
        });
    }
    let child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--nocapture"])
        .env("Y_ZD_STOP_CASE", case.join("|"))
        .output()
        .expect("run the launch in a child");
    let text = String::from_utf8_lossy(&child.stdout).into_owned();
    if let Some(line) = text.lines().find(|l| l.starts_with("RESULT ")) {
        let words = |b: &str| b.split(',').filter(|w| !w.is_empty()).map(|w| w.parse().unwrap()).collect();
        return Some(Ok(line[7..].split(';').map(words).collect()));
    }
    assert!(text.contains("TRAPPED"), "the child neither finished nor trapped: {text}");
    Some(Err(text))
}

/// Round to Q32.32, ties away from zero, sum exactly, read back through f64.
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

#[test]
fn the_same_inputs_stop_wherever_the_bad_term_is() {
    if child_run().is_some() {
        return;
    }
    let test = "the_same_inputs_stop_wherever_the_bad_term_is";
    let mut bad = Vec::new();
    // One thread, one deferring loop: `None` is a stop.
    let seq = compile("gpu_seq", &with_loop(NARROW, " step 1", "        acc += A[j];"));
    let cases: Vec<(Vec<f32>, Option<f32>)> = vec![
        (vec![1.0, 2.0, f32::NAN, 4.0], None),
        (vec![1e30, 1.0, 1.0], None),                    // the first term
        (vec![1.0, 2.0, 3.0, f32::INFINITY], None),      // the last term
        (vec![6000.0, 6000.0, -6000.0, -6000.0], None),  // only a partial sum leaves the bounds
        (vec![-6000.0, -6000.0, 6000.0, 6000.0], None),
        (vec![9999.0, -9999.0, 5000.5], Some(5000.5)),
        (vec![10000.0, -20000.0 / 2.0, 0.25], Some(0.25)),
        (vec![], Some(0.0)),
    ];
    for (terms, want) in &cases {
        let args = [f32s(terms), f32s(&[7.0]), Arg::Words(vec![0; 2]), Arg::Scalar(terms.len() as u64), Arg::Scalar(1)];
        let Some(r) = launch(test, &seq, 1, &args) else { return };
        match (r, want) {
            (Ok(got), Some(w)) if got[1][0] == w.to_bits() => {}
            (Err(_), None) => {}
            (r, w) => bad.push(format!("  loop {terms:?}: got {:?}, want {w:?}", r.map(|g| f32::from_bits(g[1][0])))),
        }
    }
    // Nested: 3 rows of 4, the bad term in the last row.
    let nest = compile("gpu_nested", &nested(""));
    let good: Vec<f32> = (0..12).map(|i| i as f32 * 0.37 - 2.0).collect();
    let mut late = good.clone();
    late[9] = f32::NAN;
    let mut over = good.clone();
    over[5] = 9000.0;
    over[6] = 9000.0;
    over[7] = -18000.0;
    for (terms, want) in [(&good, Some(reference(&good))), (&late, None), (&over, None)] {
        let Some(r) = launch(test, &nest, 1, &[f32s(terms), f32s(&[7.0]), Arg::Scalar(3), Arg::Scalar(4)]) else { return };
        match (r, want) {
            (Ok(got), Some(w)) if got[1][0] == w.to_bits() => {}
            (Err(_), None) => {}
            (r, w) => bad.push(format!("  nested {terms:?}: got {:?}, want {w:?}", r.map(|g| f32::from_bits(g[1][0])))),
        }
    }
    // An exact I64: the overflow test, deferred.
    let int = compile("gpu_i64", I64_LOOP);
    for (terms, want) in [
        (vec![i64::MAX, -5, 1], Some(i64::MAX - 4)),
        (vec![i64::MAX - 1, 1, 1], None),
        (vec![i64::MIN, -1, 5], None),
        (vec![i64::MIN + 1, -1], Some(i64::MIN)),
    ] {
        let Some(r) = launch(test, &int, 1, &[i64s(&terms), i64s(&[7]), Arg::Scalar(terms.len() as u64)]) else { return };
        let got = r.map(|g| (g[1][0] as u64 | (g[1][1] as u64) << 32) as i64);
        match (got, want) {
            (Ok(g), Some(w)) if g == w => {}
            (Err(_), None) => {}
            (g, w) => bad.push(format!("  i64 {terms:?}: got {g:?}, want {w:?}")),
        }
    }
    // A block's partials, deferred, then handed over: 200 terms over 64 threads.
    let grid = compile("gpu_grid", GRID);
    let xs: Vec<f32> = (0..200).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013).collect();
    let mut nan = xs.clone();
    nan[150] = f32::NAN;
    for (terms, want) in [(&xs, Some(reference(&xs))), (&nan, None)] {
        let args = [f32s(terms), i64s(&[0, 0, 0]), f32s(&[7.0]), Arg::Scalar(200), Arg::Scalar(4)];
        let Some(r) = launch(test, &grid, 64, &args) else { return };
        match (r, want) {
            (Ok(got), Some(w)) if got[2][0] == w.to_bits() && got[1].iter().all(|&x| x == 0) => {}
            (Err(_), None) => {}
            (r, w) => bad.push(format!("  grid partial: got {:?}, want {w:?}", r.map(|g| (f32::from_bits(g[2][0]), g[1].clone())))),
        }
    }
    assert!(bad.is_empty(), "a program stopped (or ran) where it should not:\n{}", bad.join("\n"));
}

fn which(tool: &str) -> Option<PathBuf> {
    std::env::var_os("PATH")?
        .to_str()?
        .split(':')
        .map(|d| Path::new(d).join(tool))
        .chain([PathBuf::from("/opt/cuda/bin").join(tool), PathBuf::from("/usr/local/cuda/bin").join(tool)])
        .find(|p| p.is_file())
}
