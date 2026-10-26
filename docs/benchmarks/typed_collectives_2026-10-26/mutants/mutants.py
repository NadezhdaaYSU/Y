"""Mutants of the typed warp shuffles and reductions (src/ptx_emitter.rs: emit_bfly_typed,
emit_warp_reduce, the shfl_sync_bfly dispatch): each must fail tests/ptx_typed_collectives.rs or one
of the other collective tests (GPU tests included).

Run under the GPU lock. The file is restored by writing its text back, then touched and the tests
rerun, whatever happens.
"""
import json, os, pathlib, subprocess, sys

X = pathlib.Path("/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/xbar")
OUT = pathlib.Path(__file__).parent
PE = X / "src/ptx_emitter.rs"


def rep(old, new):
    def f(t):
        assert t.count(old) == 1, (old, t.count(old))
        return t.replace(old, new)
    return f


BFLY_DST = "        let dst = self.alloc_ty(ty);\n        if ty.is_64() {\n            let (lo, hi)"
MUTANTS = {
    # the shuffle
    "bfly_dst_f32": rep(BFLY_DST, BFLY_DST.replace("self.alloc_ty(ty)", "self.alloc_regf32()")),
    "wide_as_32_bits": rep(BFLY_DST, BFLY_DST.replace("if ty.is_64() {", "if false {")),
    "wide_halves_swapped": rep('"    mov.b64 {}, {{{}, {}}};", dst, lo2, hi2', '"    mov.b64 {}, {{{}, {}}};", dst, hi2, lo2'),
    "wide_high_half_kept": rep('"    shfl.sync.bfly.b32 {}, {}, {}, 0x1f, 0xffffffff;", hi2, hi, offset', '"    mov.b32 {}, {};", hi2, hi'),
    "wide_low_half_kept": rep('"    shfl.sync.bfly.b32 {}, {}, {}, 0x1f, 0xffffffff;", lo2, lo, offset', '"    mov.b32 {}, {};", lo2, lo'),
    # the reductions
    "reduce_op_f32": rep('"    {}.{} {}, {}, {};", op, ty.arith(), next, curr, tmp', '"    {}.{} {}, {}, {};", op, "f32", next, curr, tmp'),
    "reduce_result_f32": rep("let next = self.alloc_ty(ty);", "let next = self.alloc_regf32();"),
    "sum_is_max": rep('self.emit_warp_reduce(val_reg, "add", "SUM")', 'self.emit_warp_reduce(val_reg, "max", "SUM")'),
    "max_is_add": rep('self.emit_warp_reduce(val_reg, "max", "MAX")', 'self.emit_warp_reduce(val_reg, "add", "MAX")'),
    "reduce_skips_offset_1": rep("for offset in [16u32, 8, 4, 2, 1] {", "for offset in [16u32, 8, 4, 2] {"),
    # the dispatch
    "b32_takes_64_bits": rep('if fname == "shfl_sync_bfly_b32" && ty.is_64() {', "if false {"),
    "float_offset_taken": rep("if oty.is_float() {", "if false {"),
    "offset_unconverted": rep("self.emit_convert(&o, ScalarTy::U32)\n", "o\n"),
}
TESTS = [["cargo", "test", "--release", "--no-fail-fast", "--test", "ptx_typed_collectives", "--test", "ptx_intrinsics_assemble",
          "--test", "ptx_divergent_collectives", "--test", "ptx_rmsnorm_v4"]]


def run(cmds, log):
    rc = 0
    with open(log, "w") as f:
        for cmd in cmds:
            rc = rc or subprocess.run(cmd, cwd=X, stdout=f, stderr=subprocess.STDOUT).returncode
    return rc


original = PE.read_text()
ONLY = sys.argv[1:]
if ONLY:
    MUTANTS = {k: MUTANTS[k] for k in ONLY}
for name, f in MUTANTS.items():
    assert f(original) != original, name
results = {}
try:
    for name, f in MUTANTS.items():
        PE.write_text(f(original))
        os.utime(PE)
        rc = run(TESTS, OUT / f"mut_{name}.log")
        results[name] = "killed" if rc != 0 else "SURVIVED"
        print(name, results[name], flush=True)
        PE.write_text(original)
        os.utime(PE)
finally:
    PE.write_text(original)
    os.utime(PE)
    rc = run(TESTS, OUT / "restored.log")
    results["restored_passes"] = rc == 0
    print("restored passes", rc == 0, flush=True)
(OUT / ("mutants_rerun.json" if ONLY else "mutants.json")).write_text(json.dumps(results, indent=1) + "\n")
