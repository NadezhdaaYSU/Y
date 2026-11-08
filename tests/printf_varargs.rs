//! `printf` from Y passes its arguments by C's variadic rules.
//!
//! Until 2026-11-08 it went through the generic call path, which typed every
//! argument past the format `i32` - an F64 reached printf as `fptosi` of
//! itself - and turned each string literal into a runtime string handle, so
//! the format was a pointer to the handle, not to text: every Y `printf`
//! printed garbage. The program below prints each argument kind with an
//! exact format (`%a` for floats, so a value converted to an integer cannot
//! pass for itself) through the JIT and through `--emit-llvm` linked by clang.
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

const PROGRAM: &str = "fn main() -> I32 {
    let n: I32 = -42;
    let u: U32 = 4000000000;
    let big: I64 = 9000000000;
    let x: F64 = 0.1;
    let y: F32 = 1.25;
    let flag: bool = true;
    printf(\"int %d uint %u i64 %lld bool %d str %s\\n\", n, u, big, flag, \"ok\");
    printf(\"dbl %a flt %a\\n\", x, y);
    return 0;
}
";
const EXPECTED: &str = "int -42 uint 4000000000 i64 9000000000 bool 1 str ok\ndbl 0x1.999999999999ap-4 flt 0x1.4p+0\n";

fn y(dir: &std::path::Path, file: &str, args: &[&str]) -> (bool, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_Y")).arg(dir.join(file)).args(args)
        .current_dir(dir).output().expect("run Y");
    (out.status.success(), String::from_utf8_lossy(&out.stdout).into_owned(),
     String::from_utf8_lossy(&out.stderr).into_owned())
}

/// The program's own output: what follows the JIT's "Running main..." line.
fn program_output(stdout: &str) -> String {
    let start = stdout.find("Running main...").map(|i| i + stdout[i..].find('\n').unwrap() + 1).unwrap_or(0);
    stdout[start..].to_string()
}

#[test]
fn printf_prints_every_argument_kind_through_the_jit() {
    let dir = pinned::pinned_scratch("printf_jit", pinned::SM_PINNED);
    std::fs::write(dir.join("p.ysu"), PROGRAM).unwrap();
    let (ok, out, err) = y(&dir, "p.ysu", &["--jit"]);
    assert!(ok, "{out}{err}");
    assert_eq!(program_output(&out), EXPECTED, "{out}");
}

#[test]
fn printf_prints_every_argument_kind_through_emit_llvm() {
    let have_clang = Command::new("clang").arg("--version").output().is_ok_and(|o| o.status.success());
    if !have_clang {
        eprintln!("SKIP: no clang, so the emitted LLVM was not linked and run");
        return;
    }
    let dir = pinned::pinned_scratch("printf_llvm", pinned::SM_PINNED);
    std::fs::write(dir.join("p.ysu"), PROGRAM).unwrap();
    let (ok, out, err) = y(&dir, "p.ysu", &["--emit-llvm", "-o", "p.ll"]);
    assert!(ok, "{out}{err}");
    let ll = std::fs::read_to_string(dir.join("p.ll")).unwrap();
    assert!(ll.contains("call i32 (ptr, ...) @printf(ptr @.str"), "{ll}");
    assert!(!ll.contains("fptosi"), "a float argument was converted to an integer:\n{ll}");
    // The runtime's string functions are not linked here; nothing may need them.
    std::fs::write(dir.join("main.c"), "int ysu_main(void);\nint main(void) { return ysu_main(); }\n").unwrap();
    let bin = dir.join("p");
    let built = Command::new("clang").args(["-O1", "-Wno-override-module", "-o"]).arg(&bin)
        .arg(dir.join("p.ll")).arg(dir.join("main.c")).output().unwrap();
    assert!(built.status.success(), "{}", String::from_utf8_lossy(&built.stderr));
    let run = Command::new(&bin).output().unwrap();
    assert!(run.status.success());
    assert_eq!(String::from_utf8_lossy(&run.stdout), EXPECTED);
}

#[test]
fn a_format_that_is_not_a_literal_is_refused() {
    let dir = pinned::pinned_scratch("printf_fmt", pinned::SM_PINNED);
    std::fs::write(dir.join("p.ysu"),
        "fn main() -> I32 {\n    let n: I32 = 1;\n    printf(n, n);\n    return 0;\n}\n").unwrap();
    let (ok, out, err) = y(&dir, "p.ysu", &["--emit-llvm", "-o", "p.ll"]);
    assert!(!ok, "{out}{err}");
    assert!(format!("{out}{err}").contains("format must be a string literal"), "{out}{err}");
}
