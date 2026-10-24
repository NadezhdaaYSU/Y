"""Mutants of the F16 accumulation default: each must fail the selection tests in
tests/ptx_f16_accumulate.rs or tests/ptx_f16_split_k.rs.

Run under the GPU lock. The original ptx_emitter.rs is restored by writing its text
back, then touched and rebuilt, whatever happens.
"""
import json, os, pathlib, subprocess, sys

X = pathlib.Path("/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/xbar")
OUT = pathlib.Path(__file__).parent
SRC = X / "src/ptx_emitter.rs"
ARMS = ("            Err(_) if split_k => F16Accumulation::F32,\n"
        "            Err(_) if m <= 32 => F16Accumulation::Kahan,\n"
        "            Err(_) => F16Accumulation::Chain,\n")
MUTANTS = {
    "old_default_f32": ARMS.replace("m <= 32 => F16Accumulation::Kahan", "m <= 32 => F16Accumulation::F32"),
    "no_split_arm": ARMS.replace("            Err(_) if split_k => F16Accumulation::F32,\n", ""),
    "decode_arm_first": ("            Err(_) if m <= 32 => F16Accumulation::Kahan,\n"
                         "            Err(_) if split_k => F16Accumulation::F32,\n"
                         "            Err(_) => F16Accumulation::Chain,\n"),
    "strict_lt_32": ARMS.replace("m <= 32", "m < 32"),
    "le_33": ARMS.replace("m <= 32", "m <= 33"),
    "kahan_everywhere": ARMS.replace("Err(_) => F16Accumulation::Chain", "Err(_) => F16Accumulation::Kahan"),
}
TEST = ["cargo", "test", "--release", "--test", "ptx_f16_accumulate", "--test", "ptx_f16_split_k", "--",
        "the_default_follows_the_row_count_and_the_knob_overrides_it",
        "split_k_names_select_the_shape_and_everything_else_is_refused"]

def run(cmd, log):
    with open(log, "w") as f:
        return subprocess.run(cmd, cwd=X, stdout=f, stderr=subprocess.STDOUT).returncode

original = SRC.read_text()
assert original.count(ARMS) == 1, "default arms not found exactly once"
results = {}
try:
    for name, text in MUTANTS.items():
        SRC.write_text(original.replace(ARMS, text))
        os.utime(SRC)
        rc = run(TEST, OUT / f"mut_{name}.log")
        results[name] = "killed" if rc != 0 else "SURVIVED"
        print(name, results[name], flush=True)
finally:
    SRC.write_text(original)
    os.utime(SRC)
    rc = run(TEST, OUT / "restored.log")
    results["restored_passes"] = rc == 0
    print("restored passes", rc == 0, flush=True)
(OUT / "mutants.json").write_text(json.dumps(results, indent=1) + "\n")
sys.exit(0 if all(v == "killed" for k, v in results.items() if k != "restored_passes") and results["restored_passes"] else 1)
