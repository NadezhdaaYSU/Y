//! Verify skip policy in isolated child processes, never changing the parent
//! environment shared by Cargo's parallel tests.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

#[allow(dead_code)]
#[path = "common/ptxas.rs"]
mod ptxas;
#[path = "common/validator.rs"]
mod validator;
#[path = "common/verification.rs"]
mod verification;

fn child_probe(mode: &str, scenario: &str, python: Option<&Path>) -> Output {
    let mut child = Command::new(std::env::current_exe().expect("test executable"));
    child
        .args(["--exact", "verification_policy_child", "--nocapture"])
        .env("Y_VERIFICATION_STRICT", mode)
        .env("Y_VERIFICATION_POLICY_PROBE", scenario)
        .env_remove("Y_TVAL_PYTHON");
    if let Some(python) = python {
        child.env("Y_TVAL_PYTHON", python);
    }
    child
        .output()
        .expect("run isolated verification policy probe")
}

fn transcript(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

#[test]
fn verification_policy_child() {
    match std::env::var("Y_VERIFICATION_POLICY_PROBE").as_deref() {
        Ok("missing") => {
            assert!(!verification::prerequisite_available(
                false,
                "fixture proof solver"
            ));
        }
        Ok("present") => {
            assert!(verification::prerequisite_available(
                true,
                "fixture proof solver"
            ));
        }
        Ok("python") => {
            let python = validator::validator_python(Path::new(env!("CARGO_MANIFEST_DIR")));
            assert_eq!(Some(python), std::env::var_os("Y_TVAL_PYTHON"));
        }
        Ok("ptxas") => {
            let candidate =
                PathBuf::from(std::env::var_os("Y_VERIFICATION_PTXAS_FIXTURE").unwrap());
            let tool = ptxas::find_ptxas(&[candidate]);
            assert_eq!(
                tool.is_some(),
                std::env::var("Y_VERIFICATION_PTXAS_EXPECTED").as_deref() == Ok("present")
            );
        }
        _ => {}
    }
}

#[test]
fn local_missing_dependency_reports_a_skip() {
    let output = child_probe("0", "missing", None);
    assert!(output.status.success(), "{}", transcript(&output));
    assert!(transcript(&output).contains("SKIP: fixture proof solver is unavailable"));
}

#[test]
fn strict_missing_dependency_fails_the_test_process() {
    let output = child_probe("1", "missing", None);
    assert!(
        !output.status.success(),
        "a strict gate accepted absent proof evidence"
    );
    assert!(transcript(&output).contains("Y_VERIFICATION_STRICT=1 requires fixture proof solver"));
}

#[test]
fn strict_available_dependency_still_runs() {
    let output = child_probe("1", "present", None);
    assert!(output.status.success(), "{}", transcript(&output));
    assert!(!transcript(&output).contains("SKIP:"));
}

#[test]
fn strict_mode_requires_the_explicit_one_value() {
    for mode in ["", "true", "yes"] {
        let output = child_probe(mode, "missing", None);
        assert!(output.status.success(), "{mode:?}: {}", transcript(&output));
    }
}

struct TemporaryCommand(PathBuf);

impl Drop for TemporaryCommand {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[cfg(unix)]
fn python_without_z3() -> TemporaryCommand {
    command_fixture("python", 3)
}

#[cfg(unix)]
fn command_fixture(name: &str, exit_code: u8) -> TemporaryCommand {
    use std::os::unix::fs::PermissionsExt;
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    // Tests on parallel threads can read the clock in the same nanosecond.
    let sequence = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let directory = std::env::temp_dir().join(format!(
        "y-verification-command-fixture-{}-{nonce}-{sequence}",
        std::process::id()
    ));
    std::fs::create_dir(&directory).unwrap();
    let command = directory.join(name);
    // Reproduce a dependency probe's exit status independently of installed tools.
    std::fs::write(&command, format!("#!/bin/sh\nexit {exit_code}\n")).unwrap();
    std::fs::set_permissions(&command, std::fs::Permissions::from_mode(0o755)).unwrap();
    TemporaryCommand(directory)
}

#[cfg(unix)]
#[test]
fn fixtures_made_at_the_same_moment_get_their_own_directories() {
    // Cargo runs this file's tests on parallel threads, and two of them can
    // read the clock in the same nanosecond.
    let start = std::sync::Arc::new(std::sync::Barrier::new(8));
    for _ in 0..200 {
        let threads: Vec<_> = (0..8u8)
            .map(|i| {
                let start = start.clone();
                std::thread::spawn(move || {
                    start.wait();
                    command_fixture("ptxas", i)
                })
            })
            .collect();
        let fixtures: Vec<TemporaryCommand> = threads
            .into_iter()
            .map(|t| t.join().expect("creating a fixture failed"))
            .collect();
        let distinct: std::collections::HashSet<_> = fixtures.iter().map(|f| &f.0).collect();
        assert_eq!(distinct.len(), fixtures.len(), "two fixtures share a directory");
        for (i, fixture) in fixtures.iter().enumerate() {
            let script = std::fs::read_to_string(fixture.0.join("ptxas")).unwrap();
            assert_eq!(script, format!("#!/bin/sh\nexit {i}\n"));
        }
    }
}

#[cfg(unix)]
fn ptxas_probe(mode: &str, candidate: &Path, expected: &str) -> Output {
    Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "verification_policy_child", "--nocapture"])
        .env("Y_VERIFICATION_STRICT", mode)
        .env("Y_VERIFICATION_POLICY_PROBE", "ptxas")
        .env("Y_VERIFICATION_PTXAS_FIXTURE", candidate)
        .env("Y_VERIFICATION_PTXAS_EXPECTED", expected)
        .output()
        .expect("run isolated PTXAS dependency probe")
}

#[cfg(unix)]
#[test]
fn a_spawnable_but_failing_ptxas_does_not_satisfy_the_strict_gate() {
    let fixture = command_fixture("ptxas", 7);
    let candidate = fixture.0.join("ptxas");
    let optional = ptxas_probe("0", &candidate, "absent");
    assert!(optional.status.success(), "{}", transcript(&optional));
    assert!(transcript(&optional).contains("SKIP: working ptxas"));
    let strict = ptxas_probe("1", &candidate, "absent");
    assert!(
        !strict.status.success(),
        "failed version probe was accepted as a working assembler"
    );
    assert!(transcript(&strict).contains("Y_VERIFICATION_STRICT=1 requires working ptxas"));
}

#[cfg(unix)]
#[test]
fn strict_ptxas_discovery_tries_a_working_explicit_candidate() {
    let fixture = command_fixture("ptxas", 0);
    let output = ptxas_probe("1", &fixture.0.join("ptxas"), "present");
    assert!(output.status.success(), "{}", transcript(&output));
}

#[cfg(unix)]
#[test]
fn explicit_python_without_z3_is_rejected_in_strict_mode() {
    let interpreter = python_without_z3();
    let output = child_probe("1", "python", Some(&interpreter.0.join("python")));
    assert!(
        !output.status.success(),
        "the explicit interpreter lacked the required solver"
    );
    assert!(transcript(&output).contains("with z3-solver for translation validation"));
}

#[cfg(unix)]
#[test]
fn local_python_override_preserves_dependency_free_regressions() {
    let interpreter = python_without_z3();
    let output = child_probe("0", "python", Some(&interpreter.0.join("python")));
    assert!(output.status.success(), "{}", transcript(&output));
}

#[test]
fn strict_nonexistent_python_override_fails() {
    let missing = std::env::temp_dir().join(format!("y-missing-python-{}", std::process::id()));
    assert!(
        !missing.exists(),
        "fixture unexpectedly exists: {missing:?}"
    );
    let output = child_probe("1", "python", Some(&missing));
    assert!(
        !output.status.success(),
        "a nonexistent interpreter passed the strict gate"
    );
    assert!(transcript(&output).contains("with z3-solver for translation validation"));
}
