//! Shared policy for optional local checks and required verification runs.

pub fn strict() -> bool {
    std::env::var_os("Y_VERIFICATION_STRICT").is_some_and(|value| value == "1")
}

/// Missing dependencies may skip in ordinary local tests. The verification
/// runner sets `Y_VERIFICATION_STRICT=1` so a skip cannot certify an unchecked
/// proof, solver obligation, or executable artifact.
pub fn prerequisite_available(available: bool, requirement: &str) -> bool {
    if available {
        return true;
    }
    assert!(
        !strict(),
        "Y_VERIFICATION_STRICT=1 requires {requirement}; this verification check cannot be skipped"
    );
    eprintln!("SKIP: {requirement} is unavailable");
    false
}

/// Y's exact CPU GEMM (`vpdpwssd`) needs AVX-512 VNNI. On a host without it
/// the compiler keeps the scalar exact lowering and says why, as it must, so a
/// test of the fast path has nothing to test there: on an AVX2-only Xeon
/// (Kaggle, 2026-11-12) 28 such tests failed rather than reporting a skip.
/// `Y_NO_AVX512_VNNI=1` makes any host such a one.
#[allow(dead_code)]
pub fn exact_cpu_gemm_host() -> bool {
    prerequisite_available(
        y::sentinel::host_has_avx512_vnni(),
        "a host with AVX-512 VNNI, which Y's exact CPU GEMM (`vpdpwssd`) needs",
    )
}

/// The same for code the compiler emits only for an AVX-512 host.
#[allow(dead_code)]
pub fn avx512_host() -> bool {
    prerequisite_available(y::sentinel::host_has_avx512(), "a host with AVX-512")
}

/// Y's LLVM backend links a program with `clang` from `PATH`; without one it
/// refuses ("clang not found or failed to execute"), and a test that runs the
/// program has nothing to run (16 failed so on a Kaggle machine, 2026-11-12).
#[allow(dead_code)]
pub fn clang_host() -> bool {
    let found = std::process::Command::new("clang")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success());
    prerequisite_available(found, "clang on PATH, which Y's LLVM backend links with")
}
