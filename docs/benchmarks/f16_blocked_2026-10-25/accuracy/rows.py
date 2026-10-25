"""rows.py COLUMN SHAPE[,SHAPE...] acc.json...: one README table row: cuBLAS's error over COLUMN's as
rel_F / max_norm per input group (ranges over the shapes' distributions), the range of the share of
outputs where COLUMN's |E_ij| is larger, and how many cases are below cuBLAS in both metrics."""
import json, sys
GROUP = {"mixed sign": ["normal", "uniform_pm1", "wide_range", "small", "large"], "same sign": ["abs_normal", "uniform01"],
         "cancellation": ["offset", "paired"]}
col, shapes = sys.argv[1], sys.argv[2].split(",")
cases = [c for f in sys.argv[3:] for c in json.load(open(f))["cases"] if c["shape"] in shapes and col in c["columns"]]
assert {c["shape"] for c in cases} == set(shapes), {c["shape"] for c in cases}
f = lambda x: f"{x:.2g}" if x < 10 else f"{x:.0f}"
rng = lambda v, fm: (lambda a, b: a if a == b else f"{a}–{b}")(fm(min(v)), fm(max(v)))
pct = lambda x: "0" if x == 0 else (f"{100 * x:.2g}%" if x < 0.1 else f"{100 * x:.0f}%")
cells = []
for g, ds in GROUP.items():
    gc = [c for c in cases if c["dist"] in ds]
    cells.append(rng([c["columns"]["cuBLAS"]["rel_fro"] / c["columns"][col]["rel_fro"] for c in gc], f) + " / "
                 + rng([c["columns"]["cuBLAS"]["max_norm"] / c["columns"][col]["max_norm"] for c in gc], f))
both = sum(c["columns"][col]["rel_fro"] < c["columns"]["cuBLAS"]["rel_fro"] and c["columns"][col]["max_norm"] < c["columns"]["cuBLAS"]["max_norm"] for c in cases)
canc = max(c["columns"][col]["rel_fro"] for c in cases if c["dist"] == "paired")
print(" | ".join(cells) + f" | {rng([c['vs_cuBLAS'][col]['worse'] for c in cases], pct)} | below cuBLAS in both {both}/{len(cases)} | max paired rel_F {canc:.2g}")
