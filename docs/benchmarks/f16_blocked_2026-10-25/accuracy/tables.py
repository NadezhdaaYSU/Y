"""tables.py COLUMN acc.json [...]: for COLUMN, per shape and input group, cuBLAS's error over the
column's as rel_F / max_norm (above 1: the column's error is the smaller), ranges over the group's
distributions, and the share of outputs where the column's |E_ij| is larger than cuBLAS's; then
per distribution the column's rel_F range. Markdown."""
import json, sys
from collections import OrderedDict
GROUP = OrderedDict([("mixed sign", ["normal", "uniform_pm1", "wide_range", "small", "large"]),
                     ("same sign", ["abs_normal", "uniform01"]), ("cancellation", ["offset", "paired"])])
col = sys.argv[1]
cases = [c for f in sys.argv[2:] for c in json.load(open(f))["cases"] if col in c["columns"]]


def rng(v, fmt):
    lo, hi = min(v), max(v)
    a, b = fmt(lo), fmt(hi)
    return a if a == b else f"{a}–{b}"


def f(x):
    return f"{x:.2g}" if x < 10 else f"{x:.0f}"


def pct(x):
    return "0" if x == 0 else (f"{100 * x:.2g}%" if x < 0.1 else f"{100 * x:.0f}%")


print(f"| shape | " + " | ".join(GROUP) + " | outputs where Y's error is larger | cases below cuBLAS in both |")
print("|---|" + "---|" * (len(GROUP) + 2))
for shape in dict.fromkeys(c["shape"] for c in cases):
    sc = [c for c in cases if c["shape"] == shape]
    cells = []
    for g, ds in GROUP.items():
        gc = [c for c in sc if c["dist"] in ds]
        rf = [c["columns"]["cuBLAS"]["rel_fro"] / c["columns"][col]["rel_fro"] for c in gc]
        mn = [c["columns"]["cuBLAS"]["max_norm"] / c["columns"][col]["max_norm"] for c in gc]
        cells.append(f"{rng(rf, f)} / {rng(mn, f)}")
    worse = [c["vs_cuBLAS"][col]["worse"] for c in sc]
    both = sum(1 for c in sc if c["columns"][col]["rel_fro"] < c["columns"]["cuBLAS"]["rel_fro"]
               and c["columns"][col]["max_norm"] < c["columns"]["cuBLAS"]["max_norm"])
    print(f"| {shape} | " + " | ".join(cells) + f" | {rng(worse, pct)} | {both}/{len(sc)} |")
