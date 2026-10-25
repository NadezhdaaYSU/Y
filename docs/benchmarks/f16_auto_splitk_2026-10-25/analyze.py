"""analyze.py SPEED_DIR... : per shape, what the split-K rule chose (the `auto` column) against the unsplit
kernel, the best split column and cuBLAS. Each ratio is minimum over minimum (21 interleaved rounds) as the
range over the runs (two runs x int and N(0,1) timing data); above 1 the second kernel is faster. A run's tie
band is the larger of the two columns' (median - minimum) / minimum. Every run must have been exact.
Markdown to stdout; --json OUT writes the per-shape rows."""
import glob, json, os, re, statistics, sys

args = [a for a in sys.argv[1:] if not a.startswith("--json=")]
out_json = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--json=")), None)
runs = {}
for d in args:
    for f in sorted(glob.glob(f"{d}/*_run*_*.json")):
        shape = re.match(r"(.+)_run\d+_(int|normal)\.json", os.path.basename(f)).group(1)
        r = json.load(open(f))
        assert all(r["exact"].values()) and all(r["exact_after_graph"].values()), f
        runs.setdefault(shape, []).append(r)

rng = lambda v: f"{min(v):.3f}" if f"{min(v):.3f}" == f"{max(v):.3f}" else f"{min(v):.3f}-{max(v):.3f}"
band = lambda r, a, b: max((statistics.median(r["samples"][c]) - min(r["samples"][c])) / min(r["samples"][c]) for c in (a, b))
rows = []
for shape in sorted(runs, key=lambda s: tuple(map(int, s.split("x")))):
    rs = runs[shape]
    cols = list(rs[0]["samples"])
    splits = [c for c in cols if c.startswith("s")]
    auto = rs[0]["columns"]["auto"]
    t = lambda r, c: min(r["samples"][c])
    # the split column every run ranks fastest, by the sum of its ratios to the per-run best split
    best_split = min(splits, key=lambda c: sum(t(r, c) / min(t(r, x) for x in splits) for r in rs)) if splits else None
    fastest = [min(cols, key=lambda c: t(r, c)) for r in rs]
    row = {
        "shape": shape, "runs": len(rs), "auto_splits": auto["splits"], "auto_tile": auto["tile"],
        "unsplit_over_auto": [t(r, "unsplit") / t(r, "auto") for r in rs],
        "cublas_over_unsplit": [t(r, "cuBLAS") / t(r, "unsplit") for r in rs],
        "cublas_over_auto": [t(r, "cuBLAS") / t(r, "auto") for r in rs],
        "best_split": best_split,
        "unsplit_over_best_split": [t(r, "unsplit") / t(r, best_split) for r in rs] if best_split else [],
        "auto_over_best_split": [t(r, "auto") / t(r, best_split) for r in rs] if best_split else [],
        "band_auto_unsplit": [band(r, "auto", "unsplit") for r in rs],
        "band_best_unsplit": [band(r, best_split, "unsplit") for r in rs] if best_split else [],
        "fastest_per_run": fastest,
        "us_auto": [t(r, "auto") for r in rs], "us_unsplit": [t(r, "unsplit") for r in rs],
    }
    rows.append(row)

print("| shape | rule | unsplit / rule | cuBLAS / rule | cuBLAS / unsplit | best split (every S, tile) | unsplit / best split | rule / best split | band |")
print("|---|---|---|---|---|---|---|---|---|")
for w in rows:
    rule = f"S={w['auto_splits']} {w['auto_tile']}" if w["auto_splits"] > 1 else "unsplit"
    bs = w["best_split"] or "-"
    print(f"| {w['shape']} | {rule} | {rng(w['unsplit_over_auto'])} | {rng(w['cublas_over_auto'])} | {rng(w['cublas_over_unsplit'])} | {bs} | "
          f"{rng(w['unsplit_over_best_split']) if w['best_split'] else '-'} | {rng(w['auto_over_best_split']) if w['best_split'] else '-'} | "
          f"{max(w['band_auto_unsplit'] + w['band_best_unsplit']):.3f} |")
if out_json:
    json.dump(rows, open(out_json, "w"), indent=1)
