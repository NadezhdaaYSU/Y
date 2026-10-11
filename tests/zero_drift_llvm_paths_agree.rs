//! `--emit-llvm` shows the `@ZeroDrift` representation the default build runs.
//!
//! The LLVM backend has two paths: `--emit-llvm` writes the module, and the
//! default build writes the same module to a temp file and links it. Only the
//! first loaded the drift costs `.ysu_hw_profile` records - GPU accumulate
//! timings, keyed by the GPU's name - so the two chose a representation from
//! different inputs. Measured with a profile recording `I64` as cheapest: for
//! one `@ZeroDrift @bounds(-1024, 1024) let acc: I32` program, `--emit-llvm`
//! reported `acc: I32 -> I64 (measured 1 ps/acc)` while the binary was built
//! with `Q16.16`. Neither was wrong - both are exact - but the module a user
//! inspected was not the one that ran. Both take the narrowest sufficient
//! representation now: the costs are GPU timings and this backend emits CPU
//! code.
use std::path::Path;
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

const PROGRAM: &str = "fn main() -> I32 {\n    @ZeroDrift\n    @bounds(-1024, 1024)\n    let acc: I32 = 0;\n    \
                       @invariant(i >= 0)\n    for i in 0..10 {\n        acc += 3;\n    }\n    print_int(acc);\n    \
                       println(\"\");\n    return 0;\n}\n";

fn reports(dir: &Path, args: &[&str]) -> Vec<String> {
    let out = Command::new(env!("CARGO_BIN_EXE_Y")).arg(dir.join("d.ysu")).args(args).current_dir(dir).output().unwrap();
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success(), "{args:?}:\n{text}");
    text.lines().filter(|l| l.contains("@ZeroDrift acc:")).map(|l| l.trim().to_string()).collect()
}

#[test]
fn both_llvm_paths_choose_the_same_representation() {
    let dir = pinned::pinned_scratch("drift_paths", pinned::SM_PINNED);
    // A profile whose costs, if a path read them, would pick `I64` over the
    // narrower `Q16.16` that also holds this accumulator.
    let profile = std::fs::read_to_string(dir.join(".ysu_hw_profile")).unwrap();
    let skewed = profile.replace(&format!("DRIFT_ACC_I64_{}=2106.000", pinned::GPU_NAME), &format!("DRIFT_ACC_I64_{}=1.000", pinned::GPU_NAME));
    assert_ne!(skewed, profile, "the pinned profile no longer records an I64 cost to skew");
    std::fs::write(dir.join(".ysu_hw_profile"), skewed).unwrap();
    std::fs::write(dir.join("d.ysu"), PROGRAM).unwrap();

    let emitted = reports(&dir, &["--emit-llvm"]);
    let bin = dir.join("d.bin");
    let built = reports(&dir, &["-o", bin.to_str().unwrap()]);
    assert_eq!(emitted.len(), 1, "--emit-llvm reported no single choice: {emitted:?}");
    assert_eq!(emitted, built, "--emit-llvm and the default build chose differently");
    assert!(emitted[0].contains("acc: I32 -> Q16.16") && emitted[0].contains("narrowest sufficient"), "{emitted:?}");

    // And the binary that representation was built into computes the sum.
    let run = Command::new(&bin).output().unwrap();
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "30");
}
