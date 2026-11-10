"""One mutant at a time: replace a unique string, run the test that should
catch it, restore the file (rewritten, so its mtime is fresh), record whether
the test failed (caught) or passed (survived)."""
import os
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
PY = sys.executable
RUST = ["cargo", "test", "--test", "ptx_launch_bounds"]
INTERP = [PY, "-m", "unittest", "python/y_lang/tests/test_interpreter.py", "-k", "launch_bounds"]
ATTN = [PY, "-m", "unittest", "python/y_lang/tests/test_exact_decode_attention.py"]
ATTN_RUST = ["cargo", "test", "--test", "ptx_launch_bounds", "ptxas_sizes"]

E = "src/ptx_emitter.rs"
P = "src/parser.rs"
I = "python/y_lang/interpreter.py"
K = "tests/exact_decode_attention.ysu"

MUTANTS = [
    ("R1 whole warps not counted", E, "let warp_threads = n.div_ceil(32) * 32;", "let warp_threads = n;", RUST),
    ("R2 one block too many", E, "if m == 0 || m > sm_blocks ||", "if m == 0 || m > sm_blocks + 1 ||", RUST),
    ("R3 block up to 2048", E, "if n == 0 || n > 1024 {", "if n == 0 || n > 2048 {", RUST),
    ("R4 sm_89 32 blocks", E, '"89" | "120" => Some((1536, 24)),', '"89" | "120" => Some((1536, 32)),', RUST),
    ("R5 no .minnctapersm", E, 'writeln!(&mut self.ptx_buffer, ".minnctapersm {}", m).unwrap();', "let _ = m;", RUST),
    ("R6 @tile allowed", E, "if tile_gemm || kernel.tile.is_some() {", "if false && tile_gemm {", RUST),
    ("R7 .maxnreg kept too", E, 'self.emit_launch_bounds(kernel, lb, tile_gemm_threads_per_cta.is_some());\n',
     'self.emit_launch_bounds(kernel, lb, tile_gemm_threads_per_cta.is_some());\n            writeln!(&mut self.ptx_buffer, ".maxnreg 255").unwrap();\n', RUST),
    ("R8 bounds dropped before non-kernel", P, "            if !self.check(TokenKind::Kernel) {\n                return Err(format!(\n                    \"Line {}: `@launch_bounds(...)` is only",
     "            if false {\n                return Err(format!(\n                    \"Line {}: `@launch_bounds(...)` is only", RUST),
    ("R9 three values taken", P, "if values.len() > 2 {", "if values.len() > 3 {", RUST),
    ("R10 zero taken", P, "Expr::IntLit(v, _) if v > 0 && v <= u32::MAX as i64 => values.push(v as u32),",
     "Expr::IntLit(v, _) if v >= 0 && v <= u32::MAX as i64 => values.push(v as u32),", RUST),
    ("R11 maxntid doubled", E, 'writeln!(&mut self.ptx_buffer, ".maxntid {}, 1, 1", n).unwrap();',
     'writeln!(&mut self.ptx_buffer, ".maxntid {}, 1, 1", 2 * n).unwrap();', RUST),
    ("R12 sm_75 2048 threads", E, '"75" => Some((1024, 16)),', '"75" => Some((2048, 16)),', RUST),
    ("R13 attribute ignored", E, "if let Some(lb) = &kernel.launch_bounds {", "if let Some(lb) = None::<&LaunchBoundsAttr> {", RUST),
    ("R14 unlisted arch guessed", E, '        _ => None,\n    }\n}\n\n/// An instruction family', '        _ => Some((2048, 32)),\n    }\n}\n\n/// An instruction family', RUST),
    ("P1 one thread over taken", I, "if threads > ceiling:", "if threads > ceiling + 1:", INTERP),
    ("P2 header not read", I, r'head[m.end():]):', r'""):', INTERP),
    ("P3 largest dimension only", I, "threads = int(np.prod(block))", "threads = int(max(block))", INTERP),
    ("K1 scale from the unit before", K, "shared_load_v4(sk_s, lane * kstride + dq8);", "shared_load_v4(sk_s, lane * kstride + dq8 - 1);", ATTN),
    ("K2 third unit not stored near the end", K, "if i2 < ku {", "if i2 < ku - 32 {", ATTN),
    ("K3 V group 7 not stored", K, "if ja < 8 {", "if ja < 7 {", ATTN),
    ("K4 third unit misplaced", K, "let i2: I32 = i1 + nt;", "let i2: I32 = i1 + nt - 1;", ATTN),
    ("K5 K units from one unit on", K, "block_ptr2d_load_v4(KP, 0, kbase + i0 * 4, 0, 1, kend);", "block_ptr2d_load_v4(KP, 0, kbase + i0 * 4 + 4, 0, 1, kend);", ATTN),
    ("K6 second V group from the first", K, "let vrowa: I32 = (krow0 + c * 32 + ja * 4) * vwords;", "let vrowa: I32 = (krow0 + c * 32 + j4 * 4) * vwords;", ATTN),
    ("K7 bounds removed", K, "@launch_bounds(256, 2)\n", "", ATTN_RUST),
    ("K8 two units an iteration", K, "let nt3: I32 = 3 * nt;", "let nt3: I32 = 2 * nt;", ATTN),
]


def run(cmd):
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": "python"}, timeout=3000)
    return r.returncode, (r.stdout + r.stderr)[-3000:]


def main():
    only = set(sys.argv[1:])
    out = open(os.path.join(os.path.dirname(__file__), "results.txt"), "a")
    for name, path, old, new, cmd in MUTANTS:
        if only and name.split()[0] not in only:
            continue
        full = os.path.join(REPO, path)
        orig = open(full).read()
        assert orig.count(old) == 1, (name, orig.count(old))
        try:
            open(full, "w").write(orig.replace(old, new))
            t0 = time.time()
            rc, log = run(cmd)
        finally:
            open(full, "w").write(orig)
        verdict = "CAUGHT" if rc != 0 else "SURVIVED"
        line = f"{verdict:8s} {name} ({time.time() - t0:.0f}s)"
        print(line, flush=True)
        out.write(line + "\n")
        if rc != 0:
            tail = [l for l in log.splitlines() if "panicked" in l or "Error" in l or "FAIL" in l][:3]
            out.write("    " + " | ".join(tail)[:400] + "\n")
        out.flush()


if __name__ == "__main__":
    main()
