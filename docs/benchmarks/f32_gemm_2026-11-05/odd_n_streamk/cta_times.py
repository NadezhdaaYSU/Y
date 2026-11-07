"""Per-CTA start/end/SM of a W-form FP32 GEMM, by patching its PTX.

usage: cta_times.py YBIN MxNxK [GRID]

The kernel is compiled as usual, then given a fifth parameter: thread 0 of
each CTA stores %globaltimer at entry and exit, and %smid, after a final
bar.sync. 400 launches warm the GPU; the last one is reported. The patch
adds a barrier and a few stores, so read the spread, not the absolute time.
"""
import os, re, subprocess, sys, tempfile, shutil
import cupy as cp, numpy as np, torch
ybin, spec = sys.argv[1], sys.argv[2]
m, n, k = map(int, spec.split("x"))
d = tempfile.mkdtemp(); shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../.ysu_hw_profile"), d)
open(f"{d}/g.ysu", "w").write(f"@tile({m}, {n}, {k})\nkernel g(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, W: GlobalMemory<U64>) {{\n}}\n\nfn main() {{\n}}\n")
subprocess.run([ybin, "g.ysu", "--emit-ptx", "-o", "g.ptx"], cwd=d, check=True, capture_output=True)
ptx = open(f"{d}/g.ptx").read()
blk = int(re.search(r"Launch: block \((\d+),1,1\)", ptx).group(1))
g = re.search(r"Grid \((\d+), 1, 1\)", ptx) or re.search(r"\((\d+), 1, 1\) is one tile per CTA", ptx)
grid = int(sys.argv[3]) if len(sys.argv) > 3 else int(g.group(1))
smem = int(re.search(r"Dynamic shared memory required: (\d+) bytes", ptx).group(1))
ptx = ptx.replace(".param .u64 W_3\n", ".param .u64 W_3,\n    .param .u64 T_dbg\n", 1)
head = ("\n    .reg .b64 %dbgt<4>;\n    .reg .b32 %dbgr<4>;\n    .reg .pred %dbgp;\n"
        "    mov.u64 %dbgt0, %globaltimer;\n")
ptx = ptx.replace("{\n", "{" + head, 1)
tail = ("    bar.sync 0;\n    mov.u32 %dbgr0, %tid.x;\n    setp.ne.u32 %dbgp, %dbgr0, 0;\n    @%dbgp bra $dbg_out;\n"
        "    mov.u64 %dbgt1, %globaltimer;\n    ld.param.u64 %dbgt2, [T_dbg];\n    cvta.to.global.u64 %dbgt2, %dbgt2;\n"
        "    mov.u32 %dbgr1, %ctaid.x;\n    mul.wide.u32 %dbgt3, %dbgr1, 24;\n    add.u64 %dbgt2, %dbgt2, %dbgt3;\n"
        "    st.global.u64 [%dbgt2], %dbgt0;\n    st.global.u64 [%dbgt2+8], %dbgt1;\n    mov.u32 %dbgr2, %smid;\n"
        "    cvt.u64.u32 %dbgt3, %dbgr2;\n    st.global.u64 [%dbgt2+16], %dbgt3;\n$dbg_out:\n    ret;\n}")
i = ptx.rindex("}")
ptx = ptx[:i] + tail + ptx[i+1:]
open(f"{d}/gd.ptx", "w").write(ptx)
f = cp.RawModule(path=f"{d}/gd.ptx").get_function("g"); f.max_dynamic_shared_size_bytes = smem
a = torch.randn(m, k, device="cuda"); b = torch.randn(k, n, device="cuda"); c = torch.empty(m, n, device="cuda")
w = torch.zeros(4096, dtype=torch.int64, device="cuda"); t = torch.zeros(grid * 3, dtype=torch.int64, device="cuda")
for _ in range(400):
    f((grid, 1, 1), (blk, 1, 1), (cp.asarray(a), cp.asarray(b), cp.asarray(c), cp.asarray(w), cp.asarray(t)), shared_mem=smem)
torch.cuda.synchronize()
r = t.view(grid, 3).cpu().numpy()
t0 = r[:, 0].min()
s, e, sm = (r[:, 0] - t0) / 1e3, (r[:, 1] - t0) / 1e3, r[:, 2]
print(f"{spec} grid {grid}: kernel span {e.max():.1f} us; start spread {s.max():.1f} us")
dur = e - s
order = np.argsort(-e)
print("latest-ending CTAs (cta: start..end us, sm):", ", ".join(f"{i}: {s[i]:.0f}..{e[i]:.0f} sm{sm[i]}" for i in order[:12]))
print("duration percentiles us: min %.0f p50 %.0f p90 %.0f max %.0f" % (dur.min(), np.median(dur), np.percentile(dur, 90), dur.max()))
print("end time by CTA index (every 8th):", " ".join(f"{e[i]:.0f}" for i in range(0, grid, 8)))
