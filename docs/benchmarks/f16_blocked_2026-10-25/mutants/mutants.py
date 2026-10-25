"""Mutants of blocked summation (F16Accumulation::Blocked) and its default: each must fail
tests/ptx_f16_accumulate.rs or tests/ptx_f16_split_k.rs (GPU tests included unless SKIP_GPU is set).

Run under the GPU lock. The original ptx_emitter.rs is restored by writing its text back,
then touched and rebuilt, whatever happens. `mutants.py NAME...` reruns those alone
(into mutants_rerun.json).
"""
import json, os, pathlib, subprocess, sys

X = pathlib.Path("/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/xbar")
OUT = pathlib.Path(__file__).parent
SRC = X / "src/ptx_emitter.rs"
FOLD = "            self.emit_f16_block_fold(&k_iter);\n"
TOP = "            self.emit_f16_block_loop_top(&k_iter, k_tiles);\n"
BLK_ARM = "            Err(_) if k >= 4096 && k > Self::f16_block_k() && Self::f16_block_sums_fit(accumulators, threads) => F16Accumulation::Blocked,\n"
SPLIT_ARM = "            Err(_) if split_k => F16Accumulation::F32,\n"
SPLIT_BLK = "            Err(_) if split_k && m > 32 && k > Self::f16_block_k() && Self::f16_block_sums_fit(accumulators, threads) => F16Accumulation::Blocked,\n"
CAP = ("        if f16_mode == F16Accumulation::Blocked {\n"
       "            let ctas = (hw_profile.max_smem_per_sm_bytes / (total_dyn_smem_bytes + 1024)).max(1);\n"
       "            if ctas > 1 && 2 * f16_accumulators + 64 <= 65536 / (threads_per_cta * ctas) {\n"
       "                self.tile_gemm_ctas_per_sm = Some(ctas);\n"
       "            }\n"
       "        }\n")
DEC_ARM = "            Err(_) if m <= 32 => F16Accumulation::Kahan,\n"
MIN = '        writeln!(&mut self.ptx_buffer, "    min.u32 {}, {}, {};", end, end, k_tiles).unwrap();\n'
ADD = '            writeln!(&mut self.ptx_buffer, "    add.rn.f32 {}, {}, {};", acc, acc, block).unwrap();\n'
ZERO = '            writeln!(&mut self.ptx_buffer, "    mov.f32 {}, 0f00000000;", block).unwrap();\n'
INNER = "        if let Some((inner, end)) = self.f16_block_inner.take() {\n"
NOWHERE = ('        if let Some((_inner, end)) = self.f16_block_inner.take() {\n'
           '            let inner = self.alloc_label("F16_BLOCK_NOWHERE");\n'
           '            writeln!(&mut self.ptx_buffer, "    {}:", inner).unwrap();\n')


def drop_nth(n, line):
    """Drop the n-th (0-based) of the four K loops' `line`."""
    def f(t):
        parts = t.split(line)
        assert len(parts) == 5, len(parts)
        out = parts[0]
        for i, p in enumerate(parts[1:]):
            out += ("" if i == n else line) + p
        return out
    return f


def rep(old, new):
    def f(t):
        assert t.count(old) == 1, (old, t.count(old))
        return t.replace(old, new)
    return f


MUTANTS = {
    "k_ge_2048": rep("Err(_) if k >= 4096 && k > Self::f16_block_k()", "Err(_) if k >= 2048 && k > Self::f16_block_k()"),
    "k_gt_4096": rep("Err(_) if k >= 4096 && k > Self::f16_block_k()", "Err(_) if k > 4096 && k > Self::f16_block_k()"),
    "no_fit_check": rep("k >= 4096 && k > Self::f16_block_k() && Self::f16_block_sums_fit(accumulators, threads)", "k >= 4096 && k > Self::f16_block_k()"),
    "fit_counts_once": rep("2 * accumulators + 100 <= ", "accumulators + 100 <= "),
    "blocked_before_split": rep(SPLIT_BLK + SPLIT_ARM + DEC_ARM + BLK_ARM, BLK_ARM + SPLIT_BLK + SPLIT_ARM + DEC_ARM),
    "split_never_blocked": rep(SPLIT_BLK, ""),
    "split_blocked_at_any_m": rep("Err(_) if split_k && m > 32 && k > Self::f16_block_k()", "Err(_) if split_k && k > Self::f16_block_k()"),
    "split_blocked_at_any_k": rep("Err(_) if split_k && m > 32 && k > Self::f16_block_k() &&", "Err(_) if split_k && m > 32 &&"),
    "split_no_fit_check": rep("k > Self::f16_block_k() && Self::f16_block_sums_fit(accumulators, threads) => F16Accumulation::Blocked,\n            Err(_) if split_k => F16Accumulation::F32",
                              "k > Self::f16_block_k() => F16Accumulation::Blocked,\n            Err(_) if split_k => F16Accumulation::F32"),
    "swiglu_no_register_cap": rep(CAP, ""),
    "swiglu_cap_one_cta_more": rep("let ctas = (hw_profile.max_smem_per_sm_bytes / (total_dyn_smem_bytes + 1024)).max(1);", "let ctas = (hw_profile.max_smem_per_sm_bytes / (total_dyn_smem_bytes + 1024)).max(1) + 1;"),
    "blocked_before_decode": rep(DEC_ARM + BLK_ARM, BLK_ARM + DEC_ARM),
    "block_end_off_by_one": rep('"    add.u32 {}, {}, {};", end, k_iter, self.f16_block_tiles)', '"    add.u32 {}, {}, {};", end, k_iter, self.f16_block_tiles - 1)'),
    "inner_test_le": rep('"    setp.lt.u32 {}, {}, {};", p, k_iter, end', '"    setp.le.u32 {}, {}, {};", p, k_iter, end'),
    "block_end_not_capped_at_k": rep(MIN, ""),
    "fold_no_zeroing": rep(ADD + ZERO, ADD),
    "inner_back_edge_to_nowhere_new": rep(INNER, NOWHERE),
    "no_inner_loop": rep("if self.f16_blocks.is_empty() || self.f16_block_tiles == 1 {\n            return;\n        }\n        let end = self.alloc_reg32();",
                         "if !self.f16_blocks.is_empty() || self.f16_block_tiles >= 1 {\n            return;\n        }\n        let end = self.alloc_reg32();"),
    "mma_into_accumulator": rep("Some((_, block)) => block,\n", "Some((acc, _)) => acc,\n"),
    "tiles_per_16": rep("self.f16_block_tiles = (Self::f16_block_k() / cta_k).max(1);", "self.f16_block_tiles = (Self::f16_block_k() / 16).max(1);"),
    "block_k_default_128": rep("            Err(_) => 256,\n        }", "            Err(_) => 128,\n        }"),
    "block_k_not_power_of_two": rep("b.is_power_of_two() && *b >= 16", "*b >= 16"),
    "zero_register_for_blocked": rep("if !matches!(mode, F16Accumulation::F32 | F16Accumulation::Kahan) {", "if !matches!(mode, F16Accumulation::F32 | F16Accumulation::Kahan | F16Accumulation::Blocked) {"),
    "drop_fold_gemm_single_stage": drop_nth(0, FOLD),
    "drop_fold_gemm_pipelined": drop_nth(1, FOLD),
    "drop_fold_swiglu_single_stage": drop_nth(2, FOLD),
    "drop_fold_swiglu_pipelined": drop_nth(3, FOLD),
    "drop_top_gemm_pipelined": drop_nth(1, TOP),
    "drop_top_swiglu_single_stage": drop_nth(2, TOP),
    "swiglu_counts_one_array": rep("let f16_accumulators = 2 * num_i * num_j * 8;", "let f16_accumulators = num_i * num_j * 8;"),
}
TEST = ["cargo", "test", "--release", "--test", "ptx_f16_accumulate", "--test", "ptx_f16_split_k"]
if os.environ.get("SKIP_GPU"):
    TEST += ["--", "--skip", "on_the_gpu"]


def run(cmd, log):
    with open(log, "w") as f:
        return subprocess.run(cmd, cwd=X, stdout=f, stderr=subprocess.STDOUT).returncode


original = SRC.read_text()
ONLY = sys.argv[1:]
if ONLY:
    MUTANTS = {k: MUTANTS[k] for k in ONLY}
for name, f in MUTANTS.items():
    assert f(original) != original, name
results = {"skip_gpu": bool(os.environ.get("SKIP_GPU"))}
try:
    for name, f in MUTANTS.items():
        SRC.write_text(f(original))
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
(OUT / ("mutants_rerun.json" if ONLY else "mutants.json")).write_text(json.dumps(results, indent=1) + "\n")
