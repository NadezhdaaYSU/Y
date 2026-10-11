//! A host program's `fn main` is called the way its entry calls it.
//!
//! The LLVM backend's C runtime calls `int32_t ysu_main(void)` and passes the
//! result to `exit`; the native backend's `_start` calls `main` with no
//! arguments and exits with `eax`. Measured before this file existed:
//!
//! * `fn main(x: I32) -> I32 { return x + 5; }` compiled on both and exited 5:
//!   `x` was whatever its argument register held, which happened to be 0.
//! * `fn main() -> F64 { return 7.0; }` compiled on the LLVM backend and exited
//!   0: the result sat in `xmm0` and the runtime read `eax`.
//!
//! Both are refused now. A circuit's `main` does take parameters - they are its
//! inputs - but that is `--target=r1cs`, and the JIT, which calls `main` with
//! explicit arguments, keeps its own entry check (`tests/cpu_jit_cli.rs`).
use std::path::Path;
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

/// Compiles `src` with `args` in a pinned scratch directory.
fn compile(dir: &Path, tag: &str, src: &str, args: &[&str]) -> (bool, String) {
    let path = dir.join(format!("{tag}.ysu"));
    std::fs::write(&path, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_Y")).arg(&path).args(args).current_dir(dir).output().expect("run Y");
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    (out.status.success(), text)
}

#[test]
fn a_main_with_parameters_is_refused_by_every_host_entry() {
    let src = "fn main(x: I32) -> I32 {\n    return x + 5;\n}\n";
    for (tag, args, phrase) in [
        ("default", vec!["-o", "main_arg.bin"], "`fn main` takes 1 parameter(s)"),
        ("emit_llvm", vec!["--emit-llvm"], "`fn main` takes 1 parameter(s)"),
        ("native", vec!["--emit-native", "-o", "main_arg.elf"], "`fn main` with parameters"),
    ] {
        let dir = pinned::pinned_scratch(&format!("main_contract_args_{tag}"), pinned::SM_PINNED);
        let (ok, out) = compile(&dir, "main_arg", src, &args);
        assert!(!ok, "{tag}: a `main` the entry calls with no arguments compiled:\n{out}");
        assert!(out.contains(phrase), "{tag}: refused without naming `fn main`'s parameters:\n{out}");
        for artifact in ["main_arg.bin", "main_arg.ll", "main_arg.elf"] {
            assert!(!dir.join(artifact).exists(), "{tag}: refused, but wrote {artifact}");
        }
    }
}

#[test]
fn a_main_returning_a_float_is_refused() {
    for (ty, value) in [("F64", "7.0"), ("F32", "7.0")] {
        let src = format!("fn main() -> {ty} {{\n    return {value};\n}}\n");
        for (tag, args) in [("default", vec!["-o", "main_float.bin"]), ("emit_llvm", vec!["--emit-llvm"])] {
            let dir = pinned::pinned_scratch(&format!("main_contract_{ty}_{tag}"), pinned::SM_PINNED);
            let (ok, out) = compile(&dir, "main_float", &src, &args);
            assert!(!ok, "{ty} {tag}: a `main` whose result the runtime cannot read compiled:\n{out}");
            assert!(out.contains(&format!("`fn main` returns `{ty}`")), "{ty} {tag}: refused without naming the return type:\n{out}");
        }
    }
}

/// The positive half: every result the runtime CAN read still reaches the exit
/// status, so the refusal above is not a refusal of `main` in general. Each
/// expected status differs from the others and from 0 mod 256 where it can.
#[test]
fn an_integer_or_absent_result_is_the_exit_status() {
    let cases: &[(&str, &str, i32)] = &[
        ("i32", "fn main() -> I32 {\n    return 7;\n}\n", 7),
        ("void", "fn main() {\n    let x: I32 = 32;\n    x = x + 1;\n}\n", 0),
        // The runtime reads 32 bits and `exit` keeps 8: 2^32 + 9 exits 9.
        ("i64", "fn main() -> I64 {\n    return 4294967305;\n}\n", 9),
        ("u8", "fn main() -> U8 {\n    let x: U8 = 200;\n    return x;\n}\n", 200),
        ("bool", "fn main() -> bool {\n    return 3 > 2;\n}\n", 1),
    ];
    for (tag, src, want) in cases {
        let dir = pinned::pinned_scratch(&format!("main_contract_ok_{tag}"), pinned::SM_PINNED);
        let bin = dir.join(format!("{tag}.bin"));
        let (ok, out) = compile(&dir, tag, src, &["-o", bin.to_str().unwrap()]);
        assert!(ok, "{tag}: a `main` the runtime can call was refused:\n{out}");
        let code = Command::new(&bin).status().unwrap().code();
        assert_eq!(code, Some(*want), "{tag}: wrong exit status\n{src}");
    }
}
