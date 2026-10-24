"""mutants.py OUT.json: each mutant of the Kahan change, applied to the worktree in turn,
rebuilt, and run against tests/ptx_f16_accumulate.rs; the file is restored by writing its
original text back, touching it and rebuilding."""
import json, os, subprocess, sys
X = "/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/xbar"
E = "src/ptx_emitter.rs"
MUTANTS = [
 ("kahan selects plain f32", E, 'Ok("kahan") => F16Accumulation::Kahan,', 'Ok("kahan") => F16Accumulation::F32,'),
 ("the default is kahan", E, "Err(_) if m <= 32 || split_k => F16Accumulation::F32,", "Err(_) if m <= 32 || split_k => F16Accumulation::Kahan,"),
 ("no compensation registers", E, "        if Self::f16_accumulation(m, split_k) != F16Accumulation::Kahan {\n            return;\n        }\n        self.f16_kahan_tile_mmas",
  "        if true {\n            return;\n        }\n        self.f16_kahan_tile_mmas"),
 ("GEMM accumulators uncompensated", E, "        self.emit_f16_kahan_compensations(m, split.is_some(), &acc, k_substeps);\n", ""),
 ("SwiGLU gate uncompensated", E, "        self.emit_f16_kahan_compensations(m, false, &acc_gate, k_substeps);\n", ""),
 ("SwiGLU up uncompensated", E, "        self.emit_f16_kahan_compensations(m, false, &acc_up, k_substeps);\n", ""),
 ("the error drops x", E, '"    add.rn.f32 {}, {}, {};", e, back, x)', '"    mov.f32 {}, {};", e, back)'),
 ("the error's sign flipped", E, '"    sub.rn.f32 {}, {}, {};", back, s, sum)', '"    sub.rn.f32 {}, {}, {};", back, sum, s)'),
 ("the errors never summed", E, '        writeln!(&mut self.ptx_buffer, "    add.rn.f32 {}, {}, {};", c, c, e).unwrap();\n', ""),
 ("the compensation never added in", E, '            writeln!(&mut self.ptx_buffer, "    @{} add.rn.f32 {}, {}, {};", finite, s, s, c).unwrap();\n', ""),
 ("the compensation added to an infinite sum", E, '"    @{} add.rn.f32 {}, {}, {};", finite, s, s, c)', '"    add.rn.f32 {}, {}, {};", s, s, c)'),
 ("the finite check inverted", E, '"    setp.eq.b32 {}, {}, 0;", finite, bits)', '"    setp.ne.b32 {}, {}, 0;", finite, bits)'),
 ("the finish keeps the compensations", E, "        self.f16_kahan.clear();\n", ""),
 ("tile sums folded only after the loop", E, "            if self.f16_kahan[i].pending == self.f16_kahan_tile_mmas {", "            if false {"),
 ("a fold after every mma", E, "            if self.f16_kahan[i].pending == self.f16_kahan_tile_mmas {", "            if self.f16_kahan[i].pending >= 1 {"),
 ("the tile sum kept after its fold", E, '        writeln!(&mut self.ptx_buffer, "    mov.f32 {}, 0f00000000;", x).unwrap();\n', ""),
 ("no fold after the loop", E, "            self.emit_f16_kahan_fold(i);\n            let (s, c)", "            let (s, c)"),
]
TESTS = [["cargo", "test", "--release", "--test", "ptx_f16_accumulate", "--", "--test-threads=1"]]
def run(cmd):
    p = subprocess.run(cmd, cwd=X, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr
results = []
for name, f, old, new in MUTANTS:
    path = os.path.join(X, f)
    orig = open(path).read()
    if orig.count(old) != 1:
        results.append({"mutant": name, "status": f"NOT APPLIED ({orig.count(old)} matches)"})
        print(results[-1], flush=True); continue
    open(path, "w").write(orig.replace(old, new))
    try:
        killed_by = []
        for t in TESTS:
            rc, out = run(t)
            if rc != 0:
                fails = [l for l in out.splitlines() if l.startswith("test ") and "FAILED" in l] or [l for l in out.splitlines() if "error" in l][:2]
                msgs = [l.strip() for l in out.splitlines() if "panicked" in l or l.strip().startswith(("assertion", "left", "right")) or "M = " in l or "x4096" in l][:4]
                killed_by.append("; ".join(fails + msgs)[:500])
        results.append({"mutant": name, "status": "KILLED" if killed_by else "SURVIVED", "by": killed_by})
    finally:
        open(path, "w").write(orig)
        os.utime(path, None)
    print(results[-1], flush=True)
rc, out = run(["cargo", "build", "--release"])
results.append({"restore_build": rc})
json.dump(results, open(sys.argv[1], "w"), indent=1)
print("restore build", rc)
