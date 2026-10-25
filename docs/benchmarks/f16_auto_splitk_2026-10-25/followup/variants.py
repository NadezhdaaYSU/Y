"""variants.py: candidate split-K rules scored against every measured (shape, regime): for each, the time of the
column the rule would pick over the best measured split or unsplit column (1 = the best; per run, then the
worst and mean over runs). Only shapes with fewer than 66 unsplit CTAs above 32 rows, where splits exist."""
import glob, json, os, re, sys
from statistics import mean
SMS = 66
SRC = {  # regime: directories
    "L2": [sys.argv[2] + "/l2b", sys.argv[1] + "/speed/ad", sys.argv[2] + "/l2"],
    "DRAM": [sys.argv[2] + "/dramb", sys.argv[1] + "/speed/ad_dram", sys.argv[2] + "/dram"],
}
SHAPES = ["512x512x4096", "512x512x16384", "512x512x65536", "256x256x16384", "256x256x32768", "256x256x65536",
          "256x512x16384", "256x384x32768"]
UNSPLIT_TILE = {"512x512": (64, 64), "256x256": (32, 32), "256x512": (32, 64), "256x384": (32, 64)}
OCC = {(128, 128): 1, (64, 128): 1, (64, 64): 2}  # CTAs an SM by registers (blocked splits: 134-168, 90-106 regs x 256 threads)
def col(s, cm, cn, names):
    for n in names:
        m = re.fullmatch(rf"s{s}_{cm}x{cn}(w42s\d)?", n)
        if m: return n
    return None
def runs(shape, regime):
    for d in SRC[regime]:
        fs = sorted(glob.glob(f"{d}/{shape}_run*_*.json"))
        if fs: return [json.load(open(f)) for f in fs]
    return []
def cands(m, n, k, s_depth):
    cm1, cn1 = UNSPLIT_TILE[f"{m}x{n}"]
    t1 = -(-m // cm1) * -(-n // cn1)
    out = []
    for s in (2, 4, 8):
        if k % s or (k // s) % 8 or k // s < s_depth: continue
        for (cm, cn) in [(128, 128), (64, 128), (64, 64)]:
            if cm > m or cn > n: continue
            out.append((s, cm, cn, -(-m // cm) * -(-n // cn) * s, t1))
    return out
def pick(variant, m, n, k):
    if variant == "unsplit": return None
    depth = 2048 if variant.endswith("_d2048") else 4096
    v = variant.replace("_d2048", "")
    cs = cands(m, n, k, depth)
    ok = []
    for s, cm, cn, c, t1 in cs:
        cap = SMS * (OCC[(cm, cn)] if "occ" in v else 1)
        if t1 <= c <= cap: ok.append((s, cm, cn, c))
    if not ok: return None
    if v.startswith("now"):          # most CTAs, then smaller S
        return max(ok, key=lambda x: (x[3], -x[0]))
    if v.startswith("bigtile"):      # most CTAs, then the larger tile
        return max(ok, key=lambda x: (x[3], x[1] * x[2]))
    if v.startswith("fill"):         # most of the capacity filled (CTAs / cap), then the larger tile
        return max(ok, key=lambda x: (x[3] / (SMS * (OCC[(x[1], x[2])] if "occ" in v else 1)), x[1] * x[2]))
    raise ValueError(v)
VARIANTS = ["unsplit", "now", "now_d2048", "bigtile", "bigtile_d2048", "now_occ", "now_occ_d2048", "bigtile_occ", "bigtile_occ_d2048", "fill_occ", "fill_occ_d2048"]
rows = {v: [] for v in VARIANTS}
missing = set()
for shape in SHAPES:
    m, n, k = map(int, shape.split("x"))
    for regime in SRC:
        rs = runs(shape, regime)
        if not rs: continue
        names = list(rs[0]["samples"])
        for v in VARIANTS:
            p = pick(v, m, n, k)
            c = "unsplit" if p is None else col(p[0], p[1], p[2], names)
            if c is None:
                missing.add((v, shape, regime, p)); continue
            t = lambda r, x: min(r["samples"][x])
            vals = []
            for r in rs:
                best = min(t(r, x) for x in names if x != "cuBLAS")
                vals.append(best / t(r, c))
            rows[v].append((shape, regime, c, min(vals), mean(vals)))
print(f"{'variant':20s} {'worst':>6s} {'mean':>6s}  picks (shape regime: column worst-ratio)")
for v in VARIANTS:
    if not rows[v]: continue
    w = min(x[3] for x in rows[v]); a = mean(x[4] for x in rows[v])
    print(f"{v:20s} {w:6.3f} {a:6.3f}  " + "; ".join(f"{s} {g}: {c} {x:.3f}" for s, g, c, x, _ in rows[v] if x < 0.985))
print("not measured:", sorted(missing, key=str)[:12], len(missing))
