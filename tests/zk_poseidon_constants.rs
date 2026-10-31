//! `src/zk_poseidon_constants.rs` is generated, and this gate regenerates it.
//!
//! `tools/gen_poseidon_constants.py --check` derives the table from the
//! Poseidon reference procedure (Grain-LFSR round constants, Cauchy MDS), then
//! refuses unless the reference permutation reproduces circomlib's published
//! digests, the optimized tables agree with the reference permutation, and the
//! committed file is byte-identical to what it would write. A hand edit to one
//! constant fails here before it fails an interop test.

use std::process::Command;

#[test]
fn poseidon_constants_are_the_reference_procedure() {
    let out = Command::new("python3")
        .arg("-I")
        .arg("tools/gen_poseidon_constants.py")
        .arg("--check")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("run python3");
    assert!(
        out.status.success(),
        "the committed Poseidon table is not the reference procedure's:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}
