"""columns.py SPEED_DIR [--base=COLUMN] SHAPE...: every column's time over the base column's (by default
`unsplit`; base min / column min, above 1 the column is faster) and cuBLAS's (cuBLAS min / column min), as
the range over the runs, fastest first."""
import glob, json, os, re, sys
d = sys.argv[1]
BASE = next((a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--base=")), "unsplit")
for shape in [a for a in sys.argv[2:] if not a.startswith("--")]:
    rs = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/{shape}_run*_*.json"))]
    t = lambda r, c: min(r["samples"][c])
    cols = sorted(rs[0]["samples"], key=lambda c: -min(t(r, BASE) / t(r, c) for r in rs))
    print(f"{shape} ({len(rs)} runs)")
    for c in cols:
        u = [t(r, BASE) / t(r, c) for r in rs]
        b = [t(r, "cuBLAS") / t(r, c) for r in rs]
        print(f"  {c:20s} vs base {min(u):.3f}-{max(u):.3f}  vs cuBLAS {min(b):.3f}-{max(b):.3f}  {min(t(r, c) for r in rs):9.2f} us")
