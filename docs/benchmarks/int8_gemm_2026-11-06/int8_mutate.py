# Mutants of the staged int8 GEMM (src/ptx_emitter/int8_staged_gemm.rs), each
# run against the int8 test files. Restores the source (fresh mtime) at the end.
# Usage, from a scratch directory (the backup int8_staged_gemm.rs.orig is
# written to the current one): python3 int8_mutate.py [M1 M5 ...]
import subprocess, shutil, os, time, sys
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
SRC = REPO + "/src/ptx_emitter/int8_staged_gemm.rs"
BAK = os.path.abspath("int8_staged_gemm.rs.orig")
shutil.copy2(SRC, BAK)
B = 'w(self, "bar.sync 0;".to_string());'
M = [
 ("M1 the copies ignore the swizzle",
  'w(self, format!("xor.b32 {}, {}, {};", sch, lc, swz));', 'w(self, format!("mov.b32 {}, {};", sch, lc));'),
 ("M2 every chunk copied from K byte 0",
  'w(self, format!("shl.b32 {}, {}, 4;", k_byte, lc));', 'w(self, format!("mov.u32 {}, 0;", k_byte));'),
 ("M3 the K loop waits for one group too few",
  'if s + 2 == k_substeps || k_substeps == 1 {\n                w(self, format!("cp.async.wait_group {};", stages - 2));',
  'if s + 2 == k_substeps || k_substeps == 1 {\n                w(self, format!("cp.async.wait_group {};", stages - 1));'),
 ("M4 no barrier in the K loop",
  'w(self, format!("cp.async.wait_group {};", stages - 2));\n                ' + B,
  'w(self, format!("cp.async.wait_group {};", stages - 2));'),
 ("M5 no barrier between output tiles",
  'w(self, "cp.async.wait_all;".to_string());\n        ' + B, 'w(self, "cp.async.wait_all;".to_string());'),
 ("M6 the barrier one substep late",
  'if s + 2 == k_substeps || k_substeps == 1 {', 'if s + 1 == k_substeps {'),
 ("M7 a split launch stores",
  'w(self, format!("setp.ne.u32 {}, {}, 1;", p_split, nz));', 'w(self, format!("setp.ne.u32 {}, {}, {};", p_split, nz, nz));'),
 ("M8 an unsplit launch reduces",
  'w(self, format!("setp.ne.u32 {}, {}, 1;", p_split, nz));', 'w(self, format!("setp.ne.u32 {}, {}, 0;", p_split, nz));'),
 ("M9 tile ids stride by 1",
  'w(self, format!("add.u32 {}, {}, {};", tile_nx, tile, n_ctas));', 'w(self, format!("add.u32 {}, {}, 1;", tile_nx, tile));'),
 ("M10 no trap on a wrong block",
  '[("%ntid.x", threads), ("%ntid.y", 1), ("%ntid.z", 1)]', '[("%ntid.y", 1), ("%ntid.z", 1)]'),
 ("M11 no trap on a 2^32-CTA grid",
  'w(self, format!("setp.gt.u64 {}, {}, 4294967295;", bad, g64));', 'w(self, format!("setp.gt.u64 {}, {}, 18446744073709551615;", bad, g64));'),
 ("M12 the stripe counts one K tile too many",
  'w(self, format!("sub.u32 {}, {}, 1;", n_my, n_my));', ''),
 ("M13 the last launch group is full size",
  'writeln!(&mut self.ptx_buffer, "    min.u32 {}, {}, {};", gsize, gsize, group_m).unwrap();',
  'writeln!(&mut self.ptx_buffer, "    mov.u32 {}, {};", gsize, group_m).unwrap();'),
 ("M14 B's fragment halves swapped",
  'let b = &fb[j / 2][(j % 2) * 2..(j % 2) * 2 + 2];', 'let b = &fb[j / 2][(1 - j % 2) * 2..(1 - j % 2) * 2 + 2];'),
 ("M15 rows past M and N are copied",
  'writeln!(&mut this.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p, r, bound).unwrap();',
  'writeln!(&mut this.ptx_buffer, "    setp.lt.u32 {}, {}, 4294967295;", p, r).unwrap();'),
 ("M16 rows past M are stored",
  'writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p, r, m).unwrap();',
  'writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, 4294967295;", p, r).unwrap();'),
 ("M17 columns past N are stored",
  'writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, {};", p, c, n).unwrap();',
  'writeln!(&mut self.ptx_buffer, "    setp.lt.u32 {}, {}, 4294967295;", p, c).unwrap();'),
]
TESTS = ["int8_gemm_launch_contract", "gpu_batch_invariance", "int8_gemm_exactness", "ptx_int8_gemm_scaled", "ptx_int8_mma_layout"]
only = sys.argv[1:]


def run(cmd, t):
    try:
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=t)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return "timeout", ""


try:
    for name, old, new in M:
        if only and name.split()[0] not in only:
            continue
        s = open(BAK).read()
        assert s.count(old) == 1, (name, s.count(old))
        open(SRC, "w").write(s.replace(old, new))
        os.utime(SRC, None)
        rc, out = run(["cargo", "build", "--release", "--bin", "Y"], 900)
        if rc != 0:
            print(f"{name}: BUILD FAILED", flush=True)
            continue
        t0 = time.time()
        args = ["cargo", "test", "--release"]
        for t in TESTS:
            args += ["--test", t]
        rc, out = run(args + ["--", "--test-threads=1"], 1800)
        failed = sorted({l.split()[1] for l in out.splitlines() if l.startswith("test ") and l.endswith("FAILED")})
        verdict = "KILLED" if rc != 0 else "survived"
        print(f"{name}: {verdict} ({time.time() - t0:.0f}s) by {failed if failed else rc}", flush=True)
finally:
    shutil.copy2(BAK, SRC)
    os.utime(SRC, None)
    print("restored", flush=True)
