"""Mutants of the odd-N C stores; each must fail tests/f32_simt_gemm.rs."""
import pathlib, subprocess, sys
REPO = pathlib.Path("/home/yumin/NVME files/YSU-engine-main/YSU-engine-main/src/Y_lang/Y")
F = REPO / "src/ptx_emitter/f32_simt_gemm.rs"
MUTANTS = {
    "v4_at_place_2": ("2 => &[(0, 2), (2, 2)],", "2 => &[(0, 4)],"),
    "v2_halves_swapped": ('2 => w(self, format!("{}st.global.v2.f32 [{}+{}], {{{}, {}}};", guard, gcb, at, v(0), v(1))),',
                          '2 => w(self, format!("{}st.global.v2.f32 [{}+{}], {{{}, {}}};", guard, gcb, at, v(1), v(0))),'),
    "no_row_guard": ('let guard = if rag_m { format!("@{} ", p_row) } else { String::new() };\n                        for &(e, len) in pieces {',
                     'let guard = String::new();\n                        for &(e, len) in pieces {'),
}
orig = F.read_text()
survived = []
try:
    for name, (old, new) in MUTANTS.items():
        assert orig.count(old) == 1, f"{name}: pattern occurs {orig.count(old)} times"
        F.write_text(orig.replace(old, new))
        r = subprocess.run(["cargo", "test", "--release", "--test", "f32_simt_gemm"], cwd=REPO, capture_output=True, text=True)
        F.write_text(orig)
        failed = [l.split()[1] for l in (r.stdout + r.stderr).splitlines() if l.startswith("test ") and l.endswith("FAILED")]
        print(f"{name}: {'KILLED' if r.returncode else 'SURVIVED'} {failed}", flush=True)
        if not r.returncode:
            survived.append(name)
finally:
    if F.read_text() != orig:
        F.write_text(orig)
assert F.read_text() == orig
print(f"{len(MUTANTS) - len(survived)} of {len(MUTANTS)} killed; survived: {survived}")
sys.exit(1 if survived else 0)
