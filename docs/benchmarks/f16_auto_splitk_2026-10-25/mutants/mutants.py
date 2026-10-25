"""Mutants of the split-K rule, the `_splitk` form and compensated block folds: each must fail
tests/f16_split_k_rule.rs, tests/ptx_f16_split_k.rs or tests/ptx_f16_accumulate.rs (GPU tests included
unless SKIP_GPU is set).

Run under the GPU lock. Every mutated file is restored by writing its text back, then touched and the
suites rerun, whatever happens. `mutants.py NAME...` reruns those alone (into mutants_rerun.json).
"""
import json, os, pathlib, subprocess, sys

X = pathlib.Path("/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/xbar")
OUT = pathlib.Path(__file__).parent
AT, PE, TC, MAIN = (X / p for p in ("src/autotuner.rs", "src/ptx_emitter.rs", "src/type_checker.rs", "src/main.rs"))
PLANNED = """        if let Some(planned) = planned {
            return planned.to_cta_tile_config();
        }
"""
STEP4 = "        if mode.may_use_cache() {\n            if let Some(cached) = load_persisted("
STEP7 = "        // (7) Analytic fallback.\n"


def rep(old, new):
    def f(t):
        assert t.count(old) == 1, (old, t.count(old))
        return t.replace(old, new)
    return f


def chain(*fs):
    def f(t):
        for g in fs:
            t = g(t)
        return t
    return f


MUTANTS = {
    # the rule (autotuner.rs)
    "trigger_gt": (AT, rep("if unsplit_ctas >= sm_count as u64 {", "if unsplit_ctas > sm_count as u64 {")),
    "depth_le": (AT, rep("k / s < SPLIT_K_MIN_DEPTH", "k / s <= SPLIT_K_MIN_DEPTH")),
    "depth_2048": (AT, rep("pub const SPLIT_K_MIN_DEPTH: u32 = 4096;", "pub const SPLIT_K_MIN_DEPTH: u32 = 2048;")),
    "no_lower_bound": (AT, rep("if c < unsplit_ctas || c > sm_count as u64 {", "if c > sm_count as u64 {")),
    "upper_ge": (AT, rep("if c < unsplit_ctas || c > sm_count as u64 {", "if c < unsplit_ctas || c >= sm_count as u64 {")),
    "ties_keep_the_last": (AT, rep("if best.as_ref().is_none_or(|(b, ..)| c > *b) {", "if best.as_ref().is_none_or(|(b, ..)| c >= *b) {")),
    "no_intensity_check": (AT, rep("if flops < SPLIT_K_MIN_INTENSITY as u128 * bytes {", "if false && flops < SPLIT_K_MIN_INTENSITY as u128 * bytes {")),
    "intensity_48": (AT, rep("pub const SPLIT_K_MIN_INTENSITY: u64 = 96;", "pub const SPLIT_K_MIN_INTENSITY: u64 = 48;")),
    "intensity_128": (AT, rep("pub const SPLIT_K_MIN_INTENSITY: u64 = 96;", "pub const SPLIT_K_MIN_INTENSITY: u64 = 128;")),
    "intensity_without_c": (AT, rep("2 * m64 * k64 + 2 * k64 * n64 + 4 * m64 * n64", "2 * m64 * k64 + 2 * k64 * n64")),
    # Expected to survive: with the intensity at least 96 neither M nor N is 64 or less, and only then
    # could a tile larger than M x N have the most CTAs (split_k_choice's doc comment).
    "no_row_fit_equivalent": (AT, rep(".filter(|t| t.cta_m <= m && t.cta_n <= n)", ".filter(|t| t.cta_n <= n)")),
    "no_column_fit_equivalent": (AT, rep(".filter(|t| t.cta_m <= m && t.cta_n <= n)", ".filter(|t| t.cta_m <= m)")),
    "no_shape_check": (AT, rep("if crate::ptx_emitter::PtxEmitter::gemm_split_k_shape(m, n, k, s).is_err() || k / s < SPLIT_K_MIN_DEPTH {", "if k / s < SPLIT_K_MIN_DEPTH {")),
    "counts_2_4_16": (AT, rep("pub const SPLIT_K_COUNTS: [u32; 3] = [2, 4, 8];", "pub const SPLIT_K_COUNTS: [u32; 3] = [2, 4, 16];")),
    "tiles_misordered": (AT, rep("    AutotuneCandidate { cta_m: 128, cta_n: 128, cta_k: 32, warps_m: 4, warps_n: 2, num_stages: 3, num_warps: 8 },\n    AutotuneCandidate { cta_m: 64, cta_n: 128, cta_k: 32, warps_m: 4, warps_n: 2, num_stages: 4, num_warps: 8 },\n",
                                 "    AutotuneCandidate { cta_m: 64, cta_n: 128, cta_k: 32, warps_m: 4, warps_n: 2, num_stages: 4, num_warps: 8 },\n    AutotuneCandidate { cta_m: 128, cta_n: 128, cta_k: 32, warps_m: 4, warps_n: 2, num_stages: 3, num_warps: 8 },\n")),
    "mid_tile_2x4_warps": (AT, rep("AutotuneCandidate { cta_m: 64, cta_n: 128, cta_k: 32, warps_m: 4, warps_n: 2, num_stages: 4, num_warps: 8 }",
                                   "AutotuneCandidate { cta_m: 64, cta_n: 128, cta_k: 32, warps_m: 2, warps_n: 4, num_stages: 4, num_warps: 8 }")),
    "big_tile_two_warp_columns": (AT, rep("AutotuneCandidate { cta_m: 128, cta_n: 128, cta_k: 32, warps_m: 4, warps_n: 2, num_stages: 3, num_warps: 8 }",
                                          "AutotuneCandidate { cta_m: 128, cta_n: 128, cta_k: 32, warps_m: 2, warps_n: 2, num_stages: 3, num_warps: 4 }")),
    # the rule's tile against the cache, measurements and borrowing (autotuner.rs)
    "planned_takes_any_cached": (AT, rep("if *measured || planned.is_none() {", "if true {")),
    "planned_after_borrow": (AT, chain(rep(PLANNED, ""), rep(STEP7, PLANNED + STEP7))),
    "planned_before_persisted": (AT, chain(rep(PLANNED, ""), rep(STEP4, PLANNED + STEP4))),
    "planned_cached": (AT, rep(PLANNED, "        if let Some(planned) = planned {\n            if let Ok(mut cache) = get_cache().lock() {\n                cache.insert(key, (planned.clone(), false));\n            }\n            return planned.to_cta_tile_config();\n        }\n")),
    # the `_splitk` form (ptx_emitter.rs)
    "bare_suffix_refused": (PE, rep("        if digits.is_empty() {\n            return Some(Ok(SplitK::Auto));\n        }\n", "")),
    "unsplit_writes_partial": (PE, rep("self.variables.get(&kernel.params[2].name).expect(\"a split-K GEMM's C parameter\")", "self.variables.get(&kernel.params[3].name).expect(\"a split-K GEMM's C parameter\")")),
    "unsplit_without_reduce": (PE, rep("                        self.emit_gemm_noop_reduce_entry(kernel, &kernel.name);\n", "")),
    "unsplit_as_one_split": (PE, rep("GemmSplit::Auto(plan) => (plan.splits > 1).then_some(plan.splits),", "GemmSplit::Auto(plan) => Some(plan.splits),")),
    "split_on_autotuner_tile": (PE, rep("(GemmSplit::Auto(SplitKPlan { tile: Some(tile), .. }), Some(s)) => Autotuner::autotune_split_planned(m, n, lda, s, hw_profile, Precision::F16, tile),\n", "")),
    "noop_reduce_stores": (PE, rep('        writeln!(&mut entry, "    ret;").unwrap();\n', '        writeln!(&mut entry, "    st.global.u32 [0], 0;").unwrap();\n        writeln!(&mut entry, "    ret;").unwrap();\n')),
    "header_no_other_tile": (PE, rep("Some(t) if (t.cta_m, t.cta_n, t.cta_k, t.warps_m, t.warps_n, t.num_stages) != (cta_m, cta_n, cta_k, warps_m, warps_n, config.num_stages) => {", "Some(t) if false && t.cta_m == 0 => {")),
    # compensated block folds (ptx_emitter.rs, Y_F16_ACCUMULATE=blocked_kahan)
    "bk_fold_drops_error": (PE, rep('                    writeln!(&mut self.ptx_buffer, "    add.rn.f32 {}, {}, {};", comp, comp, e).unwrap();\n', "")),
    "bk_never_compensates": (PE, rep("        for (i, comp) in self.f16_block_comps.iter().enumerate() {\n            compensated.push((self.f16_blocks[i].0.clone(), comp.clone()));\n        }\n", "")),
    "bk_finish_unguarded": (PE, rep('writeln!(&mut self.ptx_buffer, "    @{} add.rn.f32 {}, {}, {};", finite, s, s, c).unwrap();', 'writeln!(&mut self.ptx_buffer, "    add.rn.f32 {}, {}, {};", s, s, c).unwrap(); let _ = finite;')),
    "bk_knob_is_blocked": (PE, rep('Ok("blocked_kahan") => F16Accumulation::BlockedKahan,', 'Ok("blocked_kahan") => F16Accumulation::Blocked,')),
    "bk_comp_not_zeroed": (PE, rep('                let comp = self.alloc_regf32();\n                writeln!(&mut self.ptx_buffer, "    mov.f32 {}, 0f00000000;", comp).unwrap();\n', '                let comp = self.alloc_regf32();\n')),
    "bk_sum_not_kept": (PE, rep('                    writeln!(&mut self.ptx_buffer, "    mov.f32 {}, {};", acc, sum).unwrap();\n', "")),
    "bk_error_reversed": (PE, rep('"    sub.rn.f32 {}, {}, {};", back, acc, sum', '"    sub.rn.f32 {}, {}, {};", back, sum, acc')),
    # the type checker and the CLI
    "checker_refuses_bare": (TC, rep("((Some(_), Some(_), Some(_)), crate::ptx_emitter::SplitK::Auto) => None,", "((Some(_), Some(_), Some(_)), crate::ptx_emitter::SplitK::Auto) => Some(\"no\".into()),")),
    "checker_bare_takes_shape_check": (TC, rep("((Some(_), Some(_), Some(_)), crate::ptx_emitter::SplitK::Auto) => None,", "((Some(m), Some(n), Some(k)), crate::ptx_emitter::SplitK::Auto) => crate::ptx_emitter::PtxEmitter::gemm_split_k_shape(m, n, k, 2).err(),")),
    "note_for_any_gemm": (MAIN, rep("None if plan.splits > 1 && k.params.len() == 3 => {", "None if plan.splits > 1 => {")),
    "note_never": (MAIN, rep("None if plan.splits > 1 && k.params.len() == 3 => {", "None if plan.splits > 99 && k.params.len() == 3 => {")),
}
TEST = ["cargo", "test", "--release", "--no-fail-fast", "--test", "f16_split_k_rule", "--test", "ptx_f16_split_k", "--test", "ptx_f16_accumulate"]
if os.environ.get("SKIP_GPU"):
    TEST += ["--", "--skip", "on_the_gpu"]


def run(cmd, log):
    with open(log, "w") as f:
        return subprocess.run(cmd, cwd=X, stdout=f, stderr=subprocess.STDOUT).returncode


originals = {p: p.read_text() for p in (AT, PE, TC, MAIN)}
ONLY = sys.argv[1:]
if ONLY:
    MUTANTS = {k: MUTANTS[k] for k in ONLY}
for name, (p, f) in MUTANTS.items():
    assert f(originals[p]) != originals[p], name
results = {"skip_gpu": bool(os.environ.get("SKIP_GPU"))}
try:
    for name, (p, f) in MUTANTS.items():
        p.write_text(f(originals[p]))
        os.utime(p)
        rc = run(TEST, OUT / f"mut_{name}.log")
        results[name] = "killed" if rc != 0 else "SURVIVED"
        print(name, results[name], flush=True)
        p.write_text(originals[p])
        os.utime(p)
finally:
    for p, t in originals.items():
        p.write_text(t)
        os.utime(p)
    rc = run(TEST, OUT / "restored.log")
    results["restored_passes"] = rc == 0
    print("restored passes", rc == 0, flush=True)
(OUT / ("mutants_rerun.json" if ONLY else "mutants.json")).write_text(json.dumps(results, indent=1) + "\n")
