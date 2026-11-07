//! The `Y` executable still prints the hardware profile it loads.
//!
//! A library call (the C API, so the Python package) no longer does: it sets
//! `sentinel::set_quiet`, and `python/tests/test_gpu_probe_lookup.py` checks
//! that side. Nothing checked this one, so a switch that silenced the command
//! line too would have passed.

use std::fs;
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

#[test]
fn the_command_line_names_the_profile_it_loaded() {
    let dir = pinned::pinned_scratch("cli_prints_its_profile", pinned::SM_PINNED);
    fs::write(dir.join("p.ysu"), "fn main() {\n    print_int(1);\n}\n").unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .current_dir(&dir)
        .args(["p.ysu", "--emit-llvm", "-o", "p.ll"])
        .env_remove("Y_VERBOSE")
        .env_remove("YSU_HW_PROFILE")
        .output()
        .expect("run Y");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{}{}", stdout, String::from_utf8_lossy(&out.stderr));
    assert!(
        stdout.contains("Found existing .ysu_hw_profile, skipping Sentinel Probe"),
        "the command line no longer names the profile it loaded:\n{}",
        stdout
    );
    assert!(stdout.contains("-> Loaded SM (Multiprocessor) Count"), "{}", stdout);
    let _ = fs::remove_dir_all(&dir);
}
