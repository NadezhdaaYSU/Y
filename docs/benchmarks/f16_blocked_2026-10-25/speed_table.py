"""speed_table.py DIR [BASE]: per shape, each column's time against cuBLAS's (cuBLAS min / column min) and
against BASE's (BASE min / column min; above 1 the column is faster), minimum over the interleaved rounds
of each run, as the range over the runs (two runs x int and N(0,1) timing data). Checks every run was
exact. Markdown."""
import glob, json, os, re, sys
d, base = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "Y_mma")
runs = {}
for f in sorted(glob.glob(f"{d}/*_run*_*.json")):
    shape = re.match(r"(.+)_run\d+_(int|normal)\.json", os.path.basename(f)).group(1)
    r = json.load(open(f))
    assert all(r["exact"].values()) and all(r["exact_after_graph"].values()), f
    runs.setdefault(shape, []).append(r)
rng = lambda v: f"{min(v):.3f}" if f"{min(v):.3f}" == f"{max(v):.3f}" else f"{min(v):.3f}–{max(v):.3f}"
for shape, rs in runs.items():
    cols = [c for c in rs[0]["samples"] if c != "cuBLAS"]
    util = sorted({x.split(",")[0].strip() for r in rs for x in [r["util_before"]] + r["during"]}, key=int)
    print(f"\n{shape}: {len(rs)} runs; utilisation before/during {util[0]}–{util[-1]}%")
    print(f"| column | cuBLAS / column | {base} / column | min us |")
    print("|---|---|---|---|")
    for c in cols:
        m = [min(r["samples"][c]) for r in rs]
        vc = [min(r["samples"]["cuBLAS"]) / min(r["samples"][c]) for r in rs]
        vb = [min(r["samples"][base]) / min(r["samples"][c]) for r in rs] if base in rs[0]["samples"] else [float("nan")]
        print(f"| {c} | {rng(vc)} | {rng(vb)} | {min(m):.2f}–{max(m):.2f} |")
