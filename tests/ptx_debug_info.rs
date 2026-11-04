//! `--emit-ptx -g`: PTX that cuda-gdb debugs as Y (`src/ptx_debug_info.rs`),
//! and `ydb --gpu`, which runs one launch of it under cuda-gdb.
//!
//! What these tests hold it to:
//!
//! * **`-g` adds a description and nothing else.** Over the whole corpus, the
//!   `-g` PTX without its labels, its `debug` target and its DWARF sections is
//!   byte-for-byte the `--lineinfo` PTX, and so its instructions are the ones
//!   the kernel runs without -g.
//! * **The description is the code's.** Every `-g` module assembles, and
//!   `ydwarf.py --check` holds each binding to the instructions: its registers
//!   are declared and written before its scope begins, two names share a
//!   register only while nothing writes it, and the scopes nest.
//! * **cuda-gdb shows each thread's values, as Y, and never a wrong one.** A
//!   kernel is stopped on the GPU at five lines and every binding read in a
//!   chosen thread is compared with what the source computes there: signed,
//!   unsigned, 64-bit and float values, a parameter the kernel assigns, a loop
//!   variable, a shadowing `let` and the binding it hides, a `@ZeroDrift`
//!   accumulator in its exact representation - and a binding whose `let` has
//!   not run yet is not shown at all.
//! * **`lanes` shows one value per thread**, and `verify` says the code being
//!   debugged is not the code the evidence is about.
//!
//! The GPU tests need cuda-gdb and a CUDA device and say SKIP without them.
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

#[path = "common/pinned.rs"]
mod pinned;

fn text(out: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
}

fn skip(test: &str, why: &str) {
    eprintln!("SKIP {}: {}, so this test checked NOTHING", test, why);
}

fn y(dir: &Path) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_Y"));
    c.current_dir(dir);
    c
}

fn tool(rel: &str) -> PathBuf {
    pinned::repo().join(rel)
}

fn have_python() -> bool {
    Command::new("python3").arg("--version").output().is_ok()
}

fn corpus() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = fs::read_dir(pinned::repo().join("tests"))
        .expect("tests/")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "ysu").unwrap_or(false))
        .collect();
    v.sort();
    v
}

/// The `-g` module as it is without -g: written here, not shared with
/// `yverify.strip_debug`, so the two can disagree.
fn strip_debug(ptx: &str) -> String {
    let mut out = String::new();
    let mut in_section = false;
    for line in ptx.lines() {
        let t = line.trim();
        if in_section {
            in_section = t != "}";
            continue;
        }
        if t.starts_with(".section") {
            in_section = true;
            continue;
        }
        if t.starts_with("$L__ydbg") && t.ends_with(':') && t[8..t.len() - 1].chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        if t.starts_with(".target ") {
            out.push_str(&line.replacen(", debug", "", 1));
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    out
}

/// Each corpus program compiled with `--lineinfo` and with `-g`, for those
/// the PTX backend compiles: (name, line-table PTX, -g PTX, -g path).
fn corpus_pairs(dir: &Path) -> (Vec<(String, String, String, PathBuf)>, Vec<String>) {
    let (mut pairs, mut failures) = (Vec::new(), Vec::new());
    for src in corpus() {
        let name = src.file_stem().unwrap().to_string_lossy().into_owned();
        let copy = dir.join(format!("{}.ysu", name));
        fs::copy(&src, &copy).expect("copy fixture");
        let li_path = dir.join(format!("{}_li.ptx", name));
        let li = y(dir).arg(&copy).arg("--emit-ptx").arg("--lineinfo").arg("-o").arg(&li_path).output().expect("run Y");
        if !li.status.success() {
            continue;
        }
        let g_path = dir.join(format!("{}_g.ptx", name));
        let g = y(dir).arg(&copy).arg("--emit-ptx").arg("-g").arg("-o").arg(&g_path).output().expect("run Y");
        if !g.status.success() {
            failures.push(format!("{}: compiles with --lineinfo and not with -g:\n{}", name, text(&g)));
            continue;
        }
        let a = fs::read_to_string(&li_path).expect("line-table PTX");
        let b = fs::read_to_string(&g_path).expect("-g PTX");
        pairs.push((name, a, b, g_path));
    }
    (pairs, failures)
}

/// THE property: `-g` is `--lineinfo` plus the description, over every corpus
/// program the PTX backend compiles.
#[test]
fn the_debug_build_is_the_line_table_build_plus_its_description() {
    let dir = pinned::pinned_scratch("ptxdebug_corpus", pinned::SM_PINNED);
    let (pairs, mut failures) = corpus_pairs(&dir);
    for (name, li, g, _) in &pairs {
        assert!(!li.contains(".debug_info") && !li.contains(", debug"), "{}: debug information without -g", name);
        // The positive control: a strip that removed everything would pass.
        assert!(g.lines().any(|l| l.starts_with(".target ") && l.ends_with(", debug")), "{}: no debug target", name);
        assert!(g.contains(".section\t.debug_info"), "{}: no .debug_info", name);
        if strip_debug(g) != *li {
            failures.push(format!("{}: -g changed more than the debug information", name));
        }
    }
    assert!(pairs.len() >= 50, "only {} corpus programs compiled; the sweep is not testing anything", pairs.len());
    assert!(failures.is_empty(), "{} of {}:\n{}", failures.len(), pairs.len(), failures.join("\n"));
    let _ = fs::remove_dir_all(&dir);
}

/// Every `-g` module assembles, and its description holds of its code.
#[test]
fn every_debug_module_assembles_and_describes_its_code() {
    let test = "every_debug_module_assembles_and_describes_its_code";
    if !have_python() {
        return skip(test, "no python3 (tools/ydb/ydwarf.py)");
    }
    let ptxas = Command::new("ptxas").arg("--version").output().is_ok();
    let dir = pinned::pinned_scratch("ptxdebug_check", pinned::SM_PINNED);
    let (pairs, mut failures) = corpus_pairs(&dir);
    let (mut described, mut vectors) = (0, 0);
    for (name, _, g, path) in &pairs {
        if ptxas {
            let arch = g.lines().find_map(|l| l.strip_prefix(".target ")).unwrap().split(',').next().unwrap().trim();
            let out = Command::new("ptxas")
                .arg(format!("-arch={}", arch))
                .arg(path)
                .arg("-o")
                .arg(dir.join(format!("{}.cubin", name)))
                .output()
                .expect("run ptxas");
            if !out.status.success() {
                failures.push(format!("{}: ptxas refused the -g module:\n{}", name, text(&out)));
            }
        }
        let out = Command::new("python3").arg(tool("tools/ydb/ydwarf.py")).arg("--check").arg(path).output().unwrap();
        if !out.status.success() {
            failures.push(format!("{}: the description does not hold:\n{}", name, text(&out)));
        }
        let json = Command::new("python3").arg(tool("tools/ydb/ydwarf.py")).arg(path).output().unwrap();
        let s = String::from_utf8_lossy(&json.stdout);
        described += s.matches("\"scope\"").count();
        vectors += s.matches("\"kind\": \"struct\"").count();
    }
    if !ptxas {
        skip(&format!("{} (the assembly half)", test), "no ptxas");
    }
    // The sweep sees bindings, U32x4s among them, or it checks nothing.
    assert!(described > 2000, "only {} bindings described over the corpus", described);
    assert!(vectors > 100, "only {} U32x4 bindings described over the corpus", vectors);
    assert!(failures.is_empty(), "{} of {}:\n{}", failures.len(), pairs.len(), failures.join("\n"));
    let _ = fs::remove_dir_all(&dir);
}

const PROBE: &str = "\
kernel probe(Out: GlobalMemory<F32>, Ints: GlobalMemory<I32>, N: I32) {
    let i: I32 = thread_idx_x();          // L:first
    let u: U32 = 4000000000;
    let big: I64 = 5000000000;
    let neg: I32 = 0 - i;
    let d: F64 = 0.1;
    let a: I32 = i * 3;
    N = N + 1;
    let s: I32 = 0;
    @invariant(j >= 0)
    for j in 0..4 {
        s = s + j;                        // L:loop
    }
    if i < 8 {
        let a: I32 = i + 100;             // L:shadow_let
        Ints[i] = a;                      // L:shadow_use
    }
    Ints[i + 32] = a + s + N;             // L:after
    @ZeroDrift
    @bounds(-1024.0, 1024.0)
    let mut acc: F32 = 0.0;
    acc += 0.25;
    acc += 0.5;
    Out[i] = acc;                         // L:acc
}

fn main() {}
";

fn line_of(src: &str, tag: &str) -> usize {
    let mark = format!("// L:{}", tag);
    let hits: Vec<usize> = src.lines().enumerate().filter(|(_, l)| l.contains(&mark)).map(|(i, _)| i + 1).collect();
    assert_eq!(hits.len(), 1, "marker {} must occur once", mark);
    hits[0]
}

/// `ydb --gpu` on `src` in `dir`, with these commands: its output split at
/// each `@@NAME` echo the commands contain, or None (and a SKIP) when there
/// is no cuda-gdb or no GPU to run on.
fn ydb_gpu(test: &str, dir: &Path, file: &str, launch: &str, cmds: &[&str]) -> Option<(String, Vec<(String, String)>)> {
    if !have_python() {
        skip(test, "no python3");
        return None;
    }
    let mut c = Command::new("python3");
    c.current_dir(dir)
        .arg(tool("tools/ydb/ydb"))
        .arg(file)
        .arg("--y")
        .arg(env!("CARGO_BIN_EXE_Y"))
        .arg("--gpu")
        .arg(launch)
        .arg("--batch")
        .arg("--nx");
    for e in cmds {
        c.arg("-ex").arg(e);
    }
    // stdout and stderr into ONE file, so gdb's errors (stderr) stay in order
    // with the echoes and the values (stdout).
    let log = dir.join("ydb.log");
    let f = fs::File::create(&log).expect("create log");
    c.stdout(f.try_clone().expect("clone log")).stderr(f);
    c.status().expect("run ydb");
    let all = fs::read_to_string(&log).expect("read log");
    if all.contains("needs cuda-gdb") {
        skip(test, "no cuda-gdb");
        return None;
    }
    if all.contains("no CUDA driver") || all.contains("cuInit failed") || all.contains("no CUDA-capable device") {
        skip(test, "no CUDA device");
        return None;
    }
    let mut parts = Vec::new();
    let mut name = String::from("(start)");
    let mut body = String::new();
    for l in all.lines() {
        if let Some(n) = l.trim().strip_prefix("@@") {
            parts.push((std::mem::replace(&mut name, n.to_string()), std::mem::take(&mut body)));
            continue;
        }
        body.push_str(l);
        body.push('\n');
    }
    parts.push((name, body));
    Some((all, parts))
}

fn part<'a>(parts: &'a [(String, String)], name: &str) -> &'a str {
    &parts.iter().find(|(n, _)| n == name).unwrap_or_else(|| panic!("no section {}", name)).1
}

fn has_line(s: &str, want: &str) -> bool {
    s.lines().any(|l| l.trim() == want)
}

#[test]
fn cuda_gdb_shows_each_threads_bindings_as_y() {
    let test = "cuda_gdb_shows_each_threads_bindings_as_y";
    let dir = pinned::pinned_scratch("ptxdebug_gdb", pinned::SM_PINNED);
    fs::write(dir.join("probe.ysu"), PROBE).unwrap();
    let at = |t: &str| format!("break probe:{}", line_of(PROBE, t));
    let (b_first, b_loop, b_inner_let, b_inner, b_after, b_acc) =
        (at("first"), at("loop"), at("shadow_let"), at("shadow_use"), at("after"), at("acc"));
    let cmds = [
        b_first.as_str(), b_loop.as_str(), b_inner_let.as_str(), b_inner.as_str(), b_after.as_str(), b_acc.as_str(),
        "run",
        "echo @@first\\n", "print i", "continue",
        "echo @@loop\\n", "cuda thread (3,0,0)", "locals", "verify", "delete 2", "continue",
        "echo @@inner_let\\n", "cuda thread (3,0,0)", "print a", "continue",
        "echo @@inner\\n", "cuda thread (3,0,0)", "locals", "continue",
        "echo @@after\\n", "cuda thread (12,0,0)", "locals", "print j", "continue",
        "echo @@acc\\n", "cuda thread (3,0,0)", "print acc", "print/r acc", "lanes i", "lanes neg", "continue",
        "echo @@end\\n",
    ];
    let Some((all, parts)) = ydb_gpu(test, &dir, "probe.ysu", "probe<<<1, 16>>>(zeros(16), zeros(48), 5)", &cmds) else {
        return;
    };
    assert!(!all.contains("Python Exception"), "a ydb command failed:\n{}", all);

    // A binding whose `let` has not run is not in scope: not a stale value.
    let first = part(&parts, "first");
    assert!(first.contains("No symbol \"i\" in current context."), "i before its let:\n{}", first);

    // Thread 3, first iteration of the loop: every binding, its Y type, and
    // N after `N = N + 1`.
    let lp = part(&parts, "loop");
    for want in [
        "(arg) N: I32 = 6",
        "j: I32 = 0",
        "s: I32 = 0",
        "a: I32 = 9",
        "d: F64 = 0.10000000000000001",
        "neg: I32 = -3",
        "big: I64 = 5000000000",
        "u: U32 = 4000000000",
        "i: I32 = 3",
    ] {
        assert!(has_line(lp, want), "missing `{}` at the loop, thread 3:\n{}", want, lp);
    }
    assert!(lp.lines().any(|l| l.trim().starts_with("(arg) Out: *F32 = 0x")), "Out:\n{}", lp);
    // `verify` on a kernel line, in a GPU session: the code running is the
    // debug build, whose instructions are compared with the plain build's.
    assert!(lp.contains("The code this process runs on the GPU (Y --emit-ptx -g for sm_80: debug code):"), "{}", lp);
    assert!(lp.contains("SAME PTX     its instructions are the ones Y --emit-ptx writes without -g"), "{}", lp);
    assert!(lp.contains("Without -g (Y --emit-ptx for sm_80, then ptxas) - what the evidence is about:"), "{}", lp);

    // On the inner `let`'s line the inner `a` does not exist yet: `a` is the
    // outer one. Then the inner one hides it, and after its block the outer
    // one is back.
    let il = part(&parts, "inner_let");
    assert!(il.lines().any(|l| l.trim().ends_with("= 9") && l.trim().starts_with('$')), "a on the inner let:\n{}", il);
    let inner = part(&parts, "inner");
    assert!(has_line(inner, "a: I32 = 103"), "the inner a:\n{}", inner);
    assert_eq!(inner.lines().filter(|l| l.trim().starts_with("a: ")).count(), 1, "a shadowed binding listed:\n{}", inner);
    let after = part(&parts, "after");
    assert!(has_line(after, "a: I32 = 36"), "the outer a, thread 12:\n{}", after);
    assert!(!after.lines().any(|l| l.trim().starts_with("j: ")), "the loop variable after its loop:\n{}", after);
    assert!(after.contains("No symbol \"j\" in current context."), "{}", after);

    // The accumulator: exact, and its representation's raw value.
    let acc = part(&parts, "acc");
    assert!(acc.lines().any(|l| l.trim().ends_with("= 0.75 (Q32.32)")), "acc:\n{}", acc);
    assert!(acc.lines().any(|l| l.trim().ends_with("= 3221225472")), "acc raw (0.75 * 2^32):\n{}", acc);
    // `lanes`: each thread's own value; the lanes the block does not have.
    for want in ["[0] 0", "[3] 3", "[15] 15", "[5] -5", "lanes 16-31: not here"] {
        assert!(acc.contains(want), "lanes: no `{}`:\n{}", want, acc);
    }
    // The launch ran to the end, and the kernel computed what it says.
    assert!(all.contains("Out: 16 x F32 = [0.75, 0.75,"), "{}", all);
    assert!(all.contains("Ints: 48 x I32 = [100, 101, 102, 103, 104, 105, 106, 107, 0, 0, 0, 0, ..., 48, 51, 54, 57]"), "{}", all);
    assert!(all.contains("exited normally"), "{}", all);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_lane_copy_that_is_assigned_leaves_its_vector_alone() {
    let test = "a_lane_copy_that_is_assigned_leaves_its_vector_alone";
    if !have_python() {
        return skip(test, "no python3");
    }
    let dir = pinned::pinned_scratch("ptxdebug_lane", pinned::SM_PINNED);
    let src = pinned::copy_fixture(&dir, "tests/v4_lane_copy_assigned.ysu");
    let ptx = dir.join("lane.ptx");
    let out = y(&dir).arg(&src).arg("--emit-ptx").arg("-g").arg("-o").arg(&ptx).output().unwrap();
    assert!(out.status.success(), "{}", text(&out));
    let run = Command::new("python3")
        .arg(tool("tools/ydb/ylaunch.py"))
        .arg(&ptx)
        .arg("v4_lane_copy_assigned<<<1, 32>>>(zeros(128), 32)")
        .output()
        .unwrap();
    let t = text(&run);
    if t.contains("no CUDA driver") || t.contains("cuInit failed") {
        return skip(test, "no CUDA device");
    }
    assert!(run.status.success(), "{}", t);
    // (v0, w, v.y, v.z) per thread: `w` is the vector's x, untouched by v0.
    assert!(
        t.contains("Out: 128 x U32 = [11, 10, 20, 30, 11, 10, 20, 30, 11, 10, 20, 30, ..., 11, 10, 20, 30]"),
        "{}",
        t
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_launcher_refuses_a_launch_that_does_not_fit() {
    let test = "the_launcher_refuses_a_launch_that_does_not_fit";
    if !have_python() {
        return skip(test, "no python3");
    }
    let dir = pinned::pinned_scratch("ptxdebug_launch", pinned::SM_PINNED);
    fs::write(dir.join("probe.ysu"), PROBE).unwrap();
    let out = y(&dir).arg("probe.ysu").arg("--emit-ptx").arg("-g").arg("-o").arg("probe.ptx").output().unwrap();
    assert!(out.status.success(), "{}", text(&out));
    let plain = y(&dir).arg("probe.ysu").arg("--emit-ptx").arg("-o").arg("plain.ptx").output().unwrap();
    assert!(plain.status.success(), "{}", text(&plain));
    // Each is refused before anything touches a GPU, by name.
    for (ptx, launch, why) in [
        ("probe.ptx", "probe<<<1, 16>>>(zeros(16), zeros(48), 2.5)", "`N`: 2.5 is not an integer, and I32 holds integers"),
        ("probe.ptx", "probe<<<1, 16>>>(zeros(16), 3, 5)", "`Ints` is a buffer of I32"),
        ("probe.ptx", "probe<<<1, 16>>>([16777217], zeros(48), 5)", "16777217.0 is not exactly representable in F32"),
        ("probe.ptx", "probe<<<1, 16>>>(zeros(16), [3000000000], 5)", "3000000000 does not fit in I32"),
        ("probe.ptx", "probe<<<1, 16>>>(zeros(16), zeros(48))", "no argument for N"),
        ("probe.ptx", "probe<<<1, 16>>>(zeros(16), zeros(48), 5, Q=1)", "probe has no parameter `Q`"),
        ("probe.ptx", "nope<<<1, 16>>>()", "the module has no kernel `nope`; it has probe"),
        ("probe.ptx", "probe<<<0, 16>>>(zeros(16), zeros(48), 5)", "`0` is not a dimension"),
        ("plain.ptx", "probe<<<1, 16>>>(zeros(16), zeros(48), 5)", "it was not compiled with -g"),
    ] {
        let r = Command::new("python3").current_dir(&dir).arg(tool("tools/ydb/ylaunch.py")).arg(ptx).arg(launch).output().unwrap();
        let t = text(&r);
        assert!(!r.status.success() && t.contains(why), "{} should be refused with `{}`:\n{}", launch, why, t);
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn debug_flags_are_refused_where_they_do_nothing() {
    let dir = pinned::pinned_scratch("ptxdebug_flags", pinned::SM_PINNED);
    fs::write(dir.join("probe.ysu"), PROBE).unwrap();
    for (args, why) in [
        (&["--emit-ptx", "--debug"][..], "--debug cannot be combined with --emit-ptx"),
        (&["--emit-cpu", "-g"][..], "-g cannot be combined with --emit-cpu"),
        (&["--emit-ptx", "-g", "-O2"][..], "-O2 cannot be combined with --emit-ptx"),
    ] {
        let out = y(&dir).arg("probe.ysu").args(args).output().unwrap();
        let t = text(&out);
        assert!(!out.status.success() && t.contains(why), "{:?} should be refused with `{}`:\n{}", args, why, t);
    }
    let _ = fs::remove_dir_all(&dir);
}
