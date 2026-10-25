"""occupancy.py PTX...: ptxas -v registers and spills of each kernel (sm_89), its CTA's threads and dynamic
shared memory (from the PTX header), and the CTAs an SM can hold by each limit: registers (65,536 a SM,
allocated per warp in units of 256), shared memory (100 KB a SM, 1 KB reserved a CTA), threads (1,536)."""
import re, subprocess, sys
for p in sys.argv[1:]:
    ptx = open(p).read()
    m = re.search(r"CTA (\d+)x(\d+)x(\d+) (?:\(fixed\) )?\| (\d+)x(\d+) warps", ptx)
    wm, wn = int(m.group(4)), int(m.group(5))
    threads = wm * wn * 32
    smem = int(re.search(r"Dynamic shared memory required: (\d+)", ptx).group(1))
    out = subprocess.run(["ptxas", "-arch=sm_89", "-v", p, "-o", "/dev/null"], capture_output=True, text=True).stderr
    regs = int(re.search(r"Used (\d+) registers", out).group(1))
    spill = int(re.search(r"(\d+) bytes spill stores", out).group(1))
    per_warp = -(-regs * 32 // 256) * 256
    by_regs = 65536 // (per_warp * (threads // 32))
    by_smem = 102400 // (smem + 1024)
    by_thr = 1536 // threads
    print(f"{p.split('/blk/')[-1]:60s} {m.group(1)}x{m.group(2)}x{m.group(3)} {wm}x{wn} regs {regs:3d} spill {spill:4d} "
          f"smem {smem:6d} | CTAs/SM regs {by_regs} smem {by_smem} threads {by_thr} -> {min(by_regs, by_smem, by_thr)}")
