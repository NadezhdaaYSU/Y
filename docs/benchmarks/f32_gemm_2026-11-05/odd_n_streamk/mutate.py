"""Mutants of the odd-N stream-K and edge-copy changes, each run against
tests/f32_simt_gemm.rs (bits on every grid, the plan, compute-sanitizer).

usage: python3 mutate.py   (from anywhere; restores the source after each)
"""
import os, shutil, subprocess, sys, tempfile, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
SRC = os.path.join(ROOT, "src/ptx_emitter/f32_simt_gemm.rs")
MUTANTS = [
    ("odd reload reads the wrong element",
     'w(self, format!("{}ld.global.cg.f32 {}, [{}+{}];", guard, acc[i][j + e as usize], gcl, off + e as u64 * 4));',
     'w(self, format!("{}ld.global.cg.f32 {}, [{}+{}];", guard, acc[i][j + e as usize], gcl, off + (3 - e) as u64 * 4));'),
    ("odd reload without its column guard",
     '''                                let guard = if rag_n {
                                    w(self, format!("setp.lt.s32 {}, {}, {};", lp_col, lcol0, n as i64 - (dcol + e) as i64));''',
     '''                                let guard = if false {
                                    w(self, format!("setp.lt.s32 {}, {}, {};", lp_col, lcol0, n as i64 - (dcol + e) as i64));'''),
    ("B past N: column N itself not redirected",
     'w(self, format!("setp.ge.u32 {}, {}, {};", p_past, tmp, n));',
     'w(self, format!("setp.gt.u32 {}, {}, {};", p_past, tmp, n));'),
    ("B past N: col - N not capped at the last chunk",
     'w(self, format!("min.u32 {}, {}, {};", past, past, n - e));\n',
     ''),
    ("bar at N odd back to 1.04",
     'let bar = if n % 4 == 0 { 1.04 } else { 1.25 };',
     'let bar = if n % 4 == 0 { 1.04 } else { 1.04 };'),
    ("stream-K at N odd on the 256-column tiles too",
     'let odd_narrow = n % 2 == 1 && t.bn < 256;',
     'let odd_narrow = n % 2 == 1;'),
]

def main():
    backup = tempfile.mktemp(suffix=".rs")
    shutil.copy2(SRC, backup)
    original = open(SRC).read()
    try:
        for name, old, new in MUTANTS:
            assert original.count(old) == 1, (name, original.count(old))
            open(SRC, "w").write(original.replace(old, new))
            t = time.time()
            r = subprocess.run(["cargo", "test", "--test", "f32_simt_gemm"], cwd=ROOT, capture_output=True, text=True)
            out = r.stdout + r.stderr
            failed = [l.split()[1] for l in out.splitlines() if l.startswith("test ") and l.rstrip().endswith("FAILED")]
            verdict = "KILLED" if r.returncode != 0 else "SURVIVED"
            print(f"{verdict:8s} {name}: {', '.join(failed) or ('build failed' if 'error[' in out else '-')} ({time.time() - t:.0f} s)", flush=True)
    finally:
        open(SRC, "w").write(original)   # a fresh mtime, so cargo rebuilds
    print("restored:", open(SRC).read() == original)

main()
