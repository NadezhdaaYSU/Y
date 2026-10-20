"""mutants.py: each mutant of the split-K change, applied to the worktree in
turn, rebuilt, and run against the split-K tests; the file is restored by
writing its original text back, touching it and rebuilding."""
import json, os, subprocess, sys, time
X = "/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/xbar"
E, T, A, U, R, M = "src/ptx_emitter.rs", "src/type_checker.rs", "src/autotuner.rs", "src/empirical_autotune.rs", "src/cuda_runtime.rs", "src/main.rs"
MUTANTS = [
 ("reduce reads a misplaced slot", E, "writeln!(&mut self.ptx_buffer, \"    add.s64 {}, {}, {};\", next, addr, slot_bytes).unwrap();", "writeln!(&mut self.ptx_buffer, \"    add.s64 {}, {}, {};\", next, addr, slot_bytes - 16).unwrap();"),
 ("reduce drops the last slot", E, "            for z in z0..splits.min(z0 + 8) {", "            for z in z0..splits.min(z0 + 8).min(splits.max(2) - 1) {"),
 ("A offset uses the row length", E, "[(&a_g, 2 * k as u64), (&b_g, 2 * k as u64 * n as u64), (&c_g, 4 * m as u64 * n as u64)]", "[(&a_g, 2 * lda as u64), (&b_g, 2 * k as u64 * n as u64), (&c_g, 4 * m as u64 * n as u64)]"),
 ("B offset forgets N", E, "[(&a_g, 2 * k as u64), (&b_g, 2 * k as u64 * n as u64), (&c_g, 4 * m as u64 * n as u64)]", "[(&a_g, 2 * k as u64), (&b_g, 2 * k as u64), (&c_g, 4 * m as u64 * n as u64)]"),
 ("every split writes slot 0", E, "[(&a_g, 2 * k as u64), (&b_g, 2 * k as u64 * n as u64), (&c_g, 4 * m as u64 * n as u64)]", "[(&a_g, 2 * k as u64), (&b_g, 2 * k as u64 * n as u64), (&c_g, 0)]"),
 ("hoisted A stride is the split depth", E, "&a_g, Some(&cta_m_start), None, lda, cta_m, cta_k, smem_a, threads_per_cta, 2,", "&a_g, Some(&cta_m_start), None, k, cta_m, cta_k, smem_a, threads_per_cta, 2,"),
 ("per-chunk A stride is the split depth", E, "\"A\", &a_g, &cta_m_start, &k0_next, lda, m, k, cta_m, cta_k,", "\"A\", &a_g, &cta_m_start, &k0_next, k, m, k, cta_m, cta_k,"),
 ("splits chain by default", E, "Err(_) => m <= 32 || split_k,", "Err(_) => m <= 32,"),
 ("leading zeros accepted", E, "&& !digits.starts_with('0');", ";"),
 ("parts need not be multiples of 8", E, "if k % s != 0 || (k / s) % 8 != 0 {", "if k % s != 0 {"),
 ("type checker skips the parameter check", T, "                Ok(_) if !shape_ok => Some(format!(", "                Ok(_) if false => Some(format!("),
 ("split tag ignores the split", A, "    if splits > 1 {\n        format!(\"{}_SPLITK{}\", precision.tag(), splits)", "    if false {\n        format!(\"{}_SPLITK{}\", precision.tag(), splits)"),
 ("split candidates are the unsplit ones", A, "            if c.cta_m <= m && c.cta_n <= n && !candidates.contains(&c) {", "            if false && c.cta_m <= m && c.cta_n <= n && !candidates.contains(&c) {"),
 ("tuner never launches the reduction", U, "        if let Some(r) = &c.reduce {\n            steps.push(", "        if let Some(r) = c.reduce.as_ref().filter(|_| false) {\n            steps.push("),
 ("emitter tunes the unsplit shape", E, "            Some(s) => Autotuner::autotune_split(m, n, lda, s, hw_profile, Precision::F16),", "            Some(_) => Autotuner::autotune(m, n, k, hw_profile, Precision::F16),"),
 ("sequence timer launches only the first step", R, "                    for &(kernel, grid, block, shared_bytes) in steps {", "                    for &(kernel, grid, block, shared_bytes) in &steps[..1] {"),
]
TESTS = [["cargo", "test", "--release", "--test", "ptx_f16_split_k"],
         ["cargo", "test", "--release", "--lib", "--", "split", "persist_key", "probe_source"]]
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
                killed_by.append(" ".join(t[3:]) + ": " + "; ".join(fails)[:300])
        results.append({"mutant": name, "status": "KILLED" if killed_by else "SURVIVED", "by": killed_by})
    finally:
        open(path, "w").write(orig)
        os.utime(path, None)
    print(results[-1], flush=True)
rc, out = run(["cargo", "build", "--release"])
results.append({"restore_build": rc})
json.dump(results, open(sys.argv[1], "w"), indent=1)
print("restore build", rc)
