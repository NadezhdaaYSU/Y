//! An index computed with `%`, `&`, `|`, `^`, `<<` or `>>` is proved in bounds
//! where every backend computes the same non-negative value, and refused where
//! it is not proved - `eval_interval` in `src/type_checker.rs`.
//!
//! Until 2026-11-05 only `x & c` with a constant mask had an interval, so
//! `t[(x & 1023) % 8]`, `t[(x & 255) >> 5]` - a lane, a swizzle, a bucket -
//! were refused as unprovable in strict mode. The rules' soundness is
//! checked value by value in `type_checker::tests::
//! bitwise_and_remainder_intervals_hold_every_value_they_cover`; this holds the
//! compiler to them from the command line and runs what it builds.
use std::fs;
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version").output().is_ok()
}

fn compile(dir: &std::path::Path, name: &str, src: &str, args: &[&str]) -> (bool, String) {
    fs::write(dir.join(format!("{}.ysu", name)), src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .current_dir(dir)
        .arg(format!("{}.ysu", name))
        .args(args)
        .output()
        .expect("run Y");
    (out.status.success(), format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)))
}

const TABLE: &str = "\
fn main() {
    let t: [I32; 8] = {};
    t[0] = 0;
    t[1] = 10;
    t[2] = 20;
    t[3] = 30;
    t[4] = 40;
    t[5] = 50;
    t[6] = 60;
    t[7] = 70;
    let x: I32 = 1234;
    print_int(t[(x & 1023) % 8]);
    print_int(t[(x & 255) >> 5]);
    print_int(t[(x & 3) << 1]);
    print_int(t[(x & 3) | 4]);
    print_int(t[(x & 7) ^ 5]);
    print_int(t[(x & 15) & (x & 7)]);
}
";

/// 1234 = 0b100_1101_0010: & 1023 = 210, % 8 = 2; & 255 = 210, >> 5 = 6;
/// & 3 = 2, << 1 = 4; | 4 = 6; & 7 = 2, ^ 5 = 7; & 15 = 2, & 2 = 2.
const PICKED: [&str; 6] = ["20", "60", "40", "60", "70", "20"];

#[test]
fn each_form_is_proved_and_picks_the_right_element() {
    let test = "each_form_is_proved_and_picks_the_right_element";
    let dir = pinned::pinned_scratch("bitwise_index", pinned::SM_PINNED);
    let (ok, text) = compile(&dir, "table", TABLE, &["--emit-llvm", "-o", "table.ll"]);
    assert!(ok, "the proved forms were refused:\n{}", text);
    if !have("clang") {
        eprintln!("SKIP {} (the run): no clang, so the built program was not run", test);
        return;
    }
    let bin = dir.join("table");
    let (ok, text) = compile(&dir, "table", TABLE, &["-o", bin.to_str().unwrap()]);
    assert!(ok, "{}", text);
    let run = Command::new(&bin).output().expect("run the program");
    assert!(run.status.success(), "{:?}", run);
    // `print_int` writes the digits and nothing else.
    let printed = String::from_utf8_lossy(&run.stdout).trim().to_string();
    assert_eq!(printed, PICKED.concat(), "the elements the program picked");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn what_is_not_proved_is_refused() {
    let dir = pinned::pinned_scratch("bitwise_index_refused", pinned::SM_PINNED);
    let cases = [
        // No interval: an unbounded or negative dividend, a divisor that can
        // be 0, a shift amount that can reach the width, a negative shiftee.
        ("x % 8", "index has no statically provable bounds"),
        ("(0 - (x & 7)) % 8", "index has no statically provable bounds"),
        ("(x & 7) % (y & 3)", "index has no statically provable bounds"),
        ("(x & 255) >> (y & 63)", "index has no statically provable bounds"),
        ("(0 - (x & 3)) << 1", "index has no statically provable bounds"),
        // Proved, and too large.
        ("(x & 3) << 2", "inferred max: 12 >= array size 8"),
        ("(x & 7) | 8", "inferred max: 15 >= array size 8"),
        ("(x & 7) ^ 8", "inferred max: 15 >= array size 8"),
    ];
    for (i, (index, why)) in cases.iter().enumerate() {
        let src = format!(
            "fn f(x: I32, y: I32) -> I32 {{\n    let t: [I32; 8] = {{}};\n    return t[{}];\n}}\n\nfn main() {{\n    print_int(f(5, 2));\n}}\n",
            index
        );
        let (ok, text) = compile(&dir, &format!("r{}", i), &src, &["--emit-llvm", "-o", "r.ll"]);
        assert!(!ok, "t[{}] was accepted", index);
        assert!(text.contains(why), "t[{}]: does not say `{}`:\n{}", index, why, text);
    }
    let _ = fs::remove_dir_all(&dir);
}
