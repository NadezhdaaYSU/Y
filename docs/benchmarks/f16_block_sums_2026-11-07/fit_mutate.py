"""Mutants of the block-sum tile fitting; each must fail ptx_f16_accumulate.

Restores each file by rewriting its original text (a fresh mtime, so cargo
rebuilds), and checks the tree is back to the original bytes at the end.
"""
import pathlib
import subprocess
import sys

REPO = pathlib.Path("/home/yumin/NVME files/YSU-engine-main/YSU-engine-main/src/Y_lang/Y")
EMIT = REPO / "src/ptx_emitter.rs"
TUNE = REPO / "src/autotuner.rs"

MUTANTS = {
    "never_required": (EMIT,
        "std::env::var_os(\"Y_F16_ACCUMULATE\").is_none() && m > 32 && k >= 8192 && k > Self::f16_block_k()",
        "false && std::env::var_os(\"Y_F16_ACCUMULATE\").is_none() && m > 32 && k >= 8192 && k > Self::f16_block_k()"),
    "from_k_4096": (EMIT,
        "&& m > 32 && k >= 8192 && k > Self::f16_block_k()",
        "&& m > 32 && k >= 4096 && k > Self::f16_block_k()"),
    "ignores_the_knob": (EMIT,
        "std::env::var_os(\"Y_F16_ACCUMULATE\").is_none() && m > 32 && k >= 8192",
        "m > 32 && k >= 8192"),
    "widens_n_first": (TUNE,
        "[(tile.warps_m * 2, tile.warps_n), (tile.warps_m, tile.warps_n * 2)]",
        "[(tile.warps_m, tile.warps_n * 2), (tile.warps_m * 2, tile.warps_n)]"),
    "always_widens": (TUNE,
        "if tile.holds_f16_block_sums() || !crate::ptx_emitter::PtxEmitter::f16_block_sums_required(m, k) {",
        "if !crate::ptx_emitter::PtxEmitter::f16_block_sums_required(m, k) {"),
    "measured_not_fitted": (TUNE,
        "let cached = fit(cached, \"measured\");",
        "let cached = cached;"),
    "analytic_not_fitted": (TUNE,
        "let best_candidate = fit(best_candidate, \"analytic\");",
        "let best_candidate = best_candidate;"),
    "keeps_2_stages": (TUNE,
        "num_stages: tile.num_stages.max(3),",
        "num_stages: tile.num_stages,"),
    "override_fitted": (TUNE,
        "                return cand.to_cta_tile_config();\n            }\n            eprintln!(\n                \"[Y autotuner] ignoring malformed Y_CTA_OVERRIDE",
        "                return Self::fit_f16_block_sums(m, k, cand).to_cta_tile_config();\n            }\n            eprintln!(\n                \"[Y autotuner] ignoring malformed Y_CTA_OVERRIDE"),
}


def main():
    originals = {p: p.read_text() for p in (EMIT, TUNE)}
    results = {}
    try:
        only = sys.argv[1:]
        for name, (path, old, new) in MUTANTS.items():
            if only and name not in only:
                continue
            text = originals[path]
            assert text.count(old) == 1, f"{name}: the pattern occurs {text.count(old)} times"
            path.write_text(text.replace(old, new))
            run = subprocess.run(["cargo", "test", "--release", "--test", "ptx_f16_accumulate"],
                                 cwd=REPO, capture_output=True, text=True)
            path.write_text(originals[path])
            out = run.stdout + run.stderr
            failed = [l.split()[1] for l in out.splitlines() if l.startswith("test ") and l.endswith("FAILED")]
            built = "error[" not in out
            results[name] = (run.returncode != 0, built, failed)
            print(f"{name}: {'KILLED' if run.returncode else 'SURVIVED'}"
                  f"{'' if built else ' (did not compile)'} {failed}", flush=True)
    finally:
        for p, t in originals.items():
            if p.read_text() != t:
                p.write_text(t)
    assert all(p.read_text() == t for p, t in originals.items()), "a file was not restored"
    survived = [n for n, (killed, _, _) in results.items() if not killed]
    print(f"{len(results) - len(survived)} of {len(results)} killed; survived: {survived}")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
