"""table.py OUT: per shape and split, the tile, CTAs, registers, CTAs per SM, and the tile's source."""
import glob, os, re, subprocess, sys
d = sys.argv[1]
rows = []
for p in sorted(glob.glob(os.path.join(d, "*_s*.ptx"))):
    sh, s = re.match(r".*/(\d+x\d+x\d+)_s(\d+)\.ptx", p).groups()
    m, n, k = map(int, sh.split("x")); s = int(s)
    ptx = open(p).read(); log = open(p[:-4] + ".log").read()
    cm, cn, ck, wm, wn = map(int, re.search(r"CTA (\d+)x(\d+)x(\d+) \| (\d+)x(\d+) warps", ptx).groups())
    smem = int(re.search(r"Dynamic shared memory required: (\d+)", ptx).group(1))
    v = subprocess.run(["ptxas", "-arch=sm_89", "-v", p, "-o", "/dev/null"], capture_output=True, text=True).stderr
    name = [x for x in re.findall(r"\.visible \.entry (\w+)\(", ptx) if not x.endswith("_reduce")][0]
    blk = v.split(f"Compiling entry function '{name}'")[1]
    regs = int(re.search(r"Used (\d+) registers", blk).group(1))
    spill = int(re.search(r"(\d+) bytes spill stores", blk).group(1))
    thr = wm * wn * 32
    regs_r = -(-regs // 8) * 8
    per_sm = min(65536 // (regs_r * thr) if regs_r * thr else 99, 102400 // (smem + 1024), 1536 // thr, 24)
    src = "measured" if "measured tile from" in log else ("borrowed " + re.search(r"nearest shape (M=\d+ N=\d+ K=\d+)", log).group(1) if "borrowing" in log else "analytic")
    tiles = -(-m // cm) * -(-n // cn)
    rows.append((sh, s, f"{cm}x{cn}x{ck} {wm}x{wn}", tiles, tiles * s, regs, spill, per_sm, src))
print(f"{'shape':18s} S  {'tile':16s} tiles CTAs regs spill CTA/SM slots(66*CTA/SM) fill src")
for r in sorted(rows, key=lambda r: (r[0], r[1])):
    sh, s, t, tiles, ctas, regs, spill, per_sm, src = r
    slots = 66 * per_sm
    waves = -(-ctas // slots)
    print(f"{sh:18s} {s}  {t:16s} {tiles:5d} {ctas:4d} {regs:4d} {spill:5d} {per_sm:6d} {slots:5d} {ctas / (waves * slots):5.2f} {src}")
