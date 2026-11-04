//! `ydb --gpu LAUNCH --check TOOL`: compute-sanitizer on one kernel launch,
//! reported once per Y line (`tools/ydb/ycheck.py`).
//!
//! * **memcheck** names the line, the threads, and the buffer element a bad
//!   access touched - `Out[32] .. Out[35]: past the end of Out` - and the same
//!   launch in bounds reports nothing (an index into a `GlobalMemory` is not
//!   checked by the compiler or at run time, so this is where it is caught).
//! * **racecheck** names both lines of a shared-memory race, and the same
//!   kernel with a barrier between them reports nothing.
//! * **A report it cannot read is shown whole.** The sanitizer's own error
//!   count is held against what was read; a record it does not know is printed
//!   as written. The first version of the reader dropped the first report of
//!   every run (it shares a block with the sanitizer's banner) and said
//!   "nothing found" for a racecheck run whose summary counted an error.
//!
//! The GPU tests need compute-sanitizer and a CUDA device, and say SKIP
//! without them.
use std::fs;
use std::path::Path;
use std::process::{Command, Output};

#[path = "common/pinned.rs"]
mod pinned;

fn text(out: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
}

fn skip(test: &str, why: &str) {
    eprintln!("SKIP {}: {}, so this test checked NOTHING", test, why);
}

fn have_python() -> bool {
    Command::new("python3").arg("--version").output().is_ok()
}

const OOB: &str = "\
kernel oob(X: GlobalMemory<F32>, Out: GlobalMemory<F32>, N: I32) {
    let i: I32 = thread_idx_x();
    let v: F32 = X[i];
    Out[i + 4] = v * 2.0;
}

fn main() {}
";

const RACE: &str = "\
kernel race(Out: GlobalMemory<U32>, N: I32) {
    let smem: U64 = shared_alloc_u32(1024);
    let t: I32 = thread_idx_x();
    shared_store_v4(smem, t, 1, 2, 3, 4);
    //BARRIER
    let peer: I32 = 63 - t;
    let v: U32x4 = shared_load_v4(smem, peer);
    let x: U32 = v.x;
    block_ptr2d_store_v4(Out, 0, t * 4, 4 * N, 1, 4 * N, x, x, x, x);
}

fn main() {}
";

/// `ydb --gpu LAUNCH --check TOOL` on `src`: (exit success, output), or None
/// with a SKIP when there is no sanitizer or no GPU.
fn check(test: &str, dir: &Path, name: &str, src: &str, launch: &str, tool: &str) -> Option<(bool, String)> {
    if !have_python() {
        skip(test, "no python3");
        return None;
    }
    fs::write(dir.join(format!("{}.ysu", name)), src).unwrap();
    let out = Command::new("python3")
        .current_dir(dir)
        .arg(pinned::repo().join("tools/ydb/ydb"))
        .arg(format!("{}.ysu", name))
        .arg("--y")
        .arg(env!("CARGO_BIN_EXE_Y"))
        .arg("--gpu")
        .arg(launch)
        .arg("--check")
        .arg(tool)
        .output()
        .expect("run ydb");
    let t = text(&out);
    if t.contains("needs compute-sanitizer") {
        skip(test, "no compute-sanitizer");
        return None;
    }
    if t.contains("no CUDA driver") || t.contains("cuInit failed") || t.contains("no CUDA-capable device") {
        skip(test, "no CUDA device");
        return None;
    }
    assert!(!t.contains("Traceback"), "ycheck failed:\n{}", t);
    assert!(!t.contains(", and this report read"), "the report disagrees with the sanitizer's count:\n{}", t);
    Some((out.status.success(), t))
}

fn has_line(s: &str, want: &str) -> bool {
    s.lines().any(|l| l.trim() == want)
}

#[test]
fn memcheck_names_the_line_the_threads_and_the_element() {
    let test = "memcheck_names_the_line_the_threads_and_the_element";
    let dir = pinned::pinned_scratch("ycheck_mem", pinned::SM_PINNED);
    // 32 threads write Out[i + 4]: threads 28-31 write Out[32..35], past the
    // end of a 32-element buffer.
    let Some((ok, t)) = check(test, &dir, "oob", OOB, "oob<<<1, 32>>>(arange(32), zeros(32), 32)", "memcheck") else {
        return;
    };
    assert!(!ok, "an out-of-bounds write passed:\n{}", t);
    for want in [
        "oob.ysu:4  Out[i + 4] = v * 2.0;",
        "out-of-bounds global write, 4 bytes, 4 times: threads (28,0,0)-(31,0,0) of block (0,0,0)",
        "at Out[32] .. Out[35]: past the end of Out (32 x F32)",
        "ERROR SUMMARY: 5 errors",
    ] {
        assert!(has_line(&t, want), "missing `{}`:\n{}", want, t);
    }
    // The control: 28 threads stay inside.
    let (ok, t) = check(test, &dir, "oob", OOB, "oob<<<1, 28>>>(arange(32), zeros(32), 32)", "memcheck").unwrap();
    assert!(ok && has_line(&t, "nothing found"), "the in-bounds launch:\n{}", t);
    assert!(t.contains("Out: 32 x F32 = [0, 0, 0, 0, 0, 2, 4, 6,"), "{}", t);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn racecheck_names_both_lines_of_a_race() {
    let test = "racecheck_names_both_lines_of_a_race";
    let dir = pinned::pinned_scratch("ycheck_race", pinned::SM_PINNED);
    // Thread t writes slot t and reads slot 63 - t with no barrier between.
    let Some((ok, t)) = check(test, &dir, "race", RACE, "race<<<1, 64>>>(zeros(256), 64)", "racecheck") else {
        return;
    };
    assert!(!ok, "a race passed:\n{}", t);
    for want in [
        "race.ysu:4  shared_store_v4(smem, t, 1, 2, 3, 4);",
        "race.ysu:7  let v: U32x4 = shared_load_v4(smem, peer);",
        "error: a race on shared memory between the write on line 4 and the read on line 7, 256 hazards: no barrier orders them",
    ] {
        assert!(has_line(&t, want), "missing `{}`:\n{}", want, t);
    }
    // The control: the same kernel with the barrier.
    let fenced = RACE.replace("    //BARRIER", "    barrier_sync();");
    let (ok, t) = check(test, &dir, "race", &fenced, "race<<<1, 64>>>(zeros(256), 64)", "racecheck").unwrap();
    assert!(ok && has_line(&t, "nothing found"), "the fenced kernel:\n{}", t);
    let _ = fs::remove_dir_all(&dir);
}

/// No GPU: the reader on sanitizer output written here.
#[test]
fn a_report_it_cannot_read_is_shown_whole() {
    let test = "a_report_it_cannot_read_is_shown_whole";
    if !have_python() {
        return skip(test, "no python3");
    }
    let script = r#"
import sys
sys.path.insert(0, sys.argv[1])
import ycheck
P = "========= "
known = [P + "COMPUTE-SANITIZER",
         P + "Invalid __global__ write of size 4 bytes",
         P + "    at k+0x90 in k.ysu:3",
         P + "    by thread (5,0,0) in block (0,0,0)",
         P + "    Access to 0x1000 is out of bounds",
         P + "    and is 0 bytes after the nearest allocation at 0xf00 of size 256 bytes",
         P]
inside = [P + "Invalid __global__ read of size 4 bytes",
          P + "    at k+0xb0 in k.ysu:5",
          P + "    by thread (2,0,0) in block (0,0,0)",
          P + "    Address 0xf0a is misaligned",
          P]
odd = [P + "Something new the sanitizer reports",
       P + "    at k+0xa0 in k.ysu:4",
       P]
def run(lines):
    text, errors = ycheck.report("\n".join(lines), [{"name": "B", "address": 0xf00, "count": 64, "type": "F32", "size": 4}],
                                 {}, "memcheck", "k<<<1, 8>>>(..)")
    print(text)
    print("@@errors", errors)
    print("@@@@")
run(known + inside + odd + [P + "ERROR SUMMARY: 3 errors"])
run(known + [P + "ERROR SUMMARY: 3 errors"])
run(known)
"#;
    let out = Command::new("python3").arg("-c").arg(script).arg(pinned::repo().join("tools/ydb")).output().unwrap();
    let t = text(&out);
    assert!(out.status.success(), "{}", t);
    let runs: Vec<&str> = t.split("@@@@").collect();
    // The first report shares a block with the banner, and is read; the
    // unknown one is printed as written; the counts agree, so nothing more.
    let (a, b, c) = (runs[0], runs[1], runs[2]);
    assert!(has_line(a, "k.ysu:3"), "{}", a);
    assert!(has_line(a, "at B[64]: past the end of B (64 x F32)"), "{}", a);
    // Inside the buffer: 0xf0a is byte 2 of B[2].
    assert!(has_line(a, "misaligned global read, 4 bytes, 1 time: thread (2,0,0) of block (0,0,0)"), "{}", a);
    assert!(has_line(a, "at B[2] (not on an element boundary), inside B (64 x F32)"), "{}", a);
    assert!(a.contains("Reported as the sanitizer wrote it") && a.contains("Something new the sanitizer reports"), "{}", a);
    assert!(!a.contains(", and this report read"), "{}", a);
    // A count the report does not reach: everything, as written.
    assert!(b.contains("The sanitizer counted 3, and this report read 1: what it wrote, in full:"), "{}", b);
    assert!(b.contains("  ========= Invalid __global__ write of size 4 bytes"), "{}", b);
    // No summary at all is no evidence of a clean run.
    assert!(c.contains("The sanitizer printed no summary, and this report read 1"), "{}", c);
}

#[test]
fn check_needs_a_launch() {
    if !have_python() {
        return skip("check_needs_a_launch", "no python3");
    }
    let dir = pinned::pinned_scratch("ycheck_flags", pinned::SM_PINNED);
    fs::write(dir.join("oob.ysu"), OOB).unwrap();
    let out = Command::new("python3")
        .current_dir(&dir)
        .arg(pinned::repo().join("tools/ydb/ydb"))
        .arg("oob.ysu")
        .arg("--check")
        .arg("memcheck")
        .output()
        .unwrap();
    let t = text(&out);
    assert!(!out.status.success() && t.contains("--check runs a kernel launch: give it with --gpu LAUNCH"), "{}", t);
    let _ = fs::remove_dir_all(&dir);
}
