"""llm_table.py DIR: per LLM layer shape, cuBLAS min / column min for the chain and blocked 256, and the
chain's min over blocked 128's and 256's (above 1 the blocked kernel is faster); ranges over the runs."""
import glob, json, os, re, statistics, sys
runs = {}
for f in glob.glob(f"{sys.argv[1]}/*_run*_*.json"):
    s = re.match(r"(.+)_run\d+_(int|normal)\.json", os.path.basename(f)).group(1)
    r = json.load(open(f)); assert all(r["exact"].values()) and all(r["exact_after_graph"].values()), f
    runs.setdefault(s, []).append(r)
mn = lambda r, c: min(r["samples"][c])
rg = lambda v: f"{min(v):.3f}–{max(v):.3f}"
print("| shape | tile | cuBLAS / chain | cuBLAS / blocked 256 | chain / blocked 128 | chain / blocked 256 |")
print("|---|---|---|---|---|---|")
meds = []
for s in sorted(runs, key=lambda s: tuple(int(x) for x in s.split("x"))):
    rs = runs[s]
    meds.append(statistics.median(mn(r, "Y_mma") / mn(r, "Y_blk256") for r in rs))
    print(f"| {s} | {rs[0]['columns']['Y_mma']['tile']} | {rg([mn(r,'cuBLAS')/mn(r,'Y_mma') for r in rs])} | "
          f"{rg([mn(r,'cuBLAS')/mn(r,'Y_blk256') for r in rs])} | {rg([mn(r,'Y_mma')/mn(r,'Y_blk128') for r in rs])} | "
          f"{rg([mn(r,'Y_mma')/mn(r,'Y_blk256') for r in rs])} |")
print(f"\n{len(runs)} shapes, {sum(len(v) for v in runs.values())} runs; median over shapes of chain / blocked 256: {statistics.median(meds):.3f}")
