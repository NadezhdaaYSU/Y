"""analyze.py acc.json: per shape group and column, cuBLAS rel_F / column rel_F and
cuBLAS max_norm / column max_norm (above 1: the column's error is smaller), the share
of outputs where the column's |E| is larger than cuBLAS's, and counts of cases where
the column is below cuBLAS in both metrics."""
import json, sys
from collections import defaultdict
cases = json.load(open(sys.argv[1]))["cases"]
GROUP = {"normal": "mixed", "uniform_pm1": "mixed", "wide_range": "mixed", "small": "mixed", "large": "mixed",
         "abs_normal": "same sign", "uniform01": "same sign", "offset": "cancel", "paired": "cancel"}
cols = [c for c in cases[0]["columns"] if c.startswith("Y_")]
cols.sort(key=lambda c: (c.endswith("_s4"), c.split("_")[1].startswith("blk"), int(c.split("_")[1][3:]) if c.split("_")[1].startswith("blk") else 0, c))
print(f"{len(cases)} cases")
for shape in dict.fromkeys(c["shape"] for c in cases):
    sc = [c for c in cases if c["shape"] == shape]
    print(f"\n{shape}  ({len(sc)} cases)")
    for col in cols:
        rf = [c["columns"]["cuBLAS"]["rel_fro"] / c["columns"][col]["rel_fro"] for c in sc]
        mn = [c["columns"]["cuBLAS"]["max_norm"] / c["columns"][col]["max_norm"] for c in sc]
        worse = [c["vs_cuBLAS"][col]["worse"] for c in sc]
        bits = [c["vs_cuBLAS"][col]["bitwise"] for c in sc]
        both = sum(1 for a, b in zip(rf, mn) if a > 1 and b > 1)
        eq = sum(1 for b in bits if b == 1.0)
        print(f"  {col:10s} rel_F x{min(rf):7.3f}-{max(rf):7.3f}  max_norm x{min(mn):7.3f}-{max(mn):7.3f}  "
              f"worse {min(worse):.3f}-{max(worse):.3f}  below cuBLAS in both {both}/{len(sc)}  same bits {eq}/{len(sc)}")
print("\nTotals over all cases: below cuBLAS in rel_F and max_norm / same bits as cuBLAS / neither")
for col in cols:
    both = sum(1 for c in cases if c["columns"][col]["rel_fro"] < c["columns"]["cuBLAS"]["rel_fro"] and c["columns"][col]["max_norm"] < c["columns"]["cuBLAS"]["max_norm"])
    eq = sum(1 for c in cases if c["vs_cuBLAS"][col]["bitwise"] == 1.0)
    print(f"  {col:10s} {both:3d} / {eq:3d} / {len(cases) - both - eq:3d}")
