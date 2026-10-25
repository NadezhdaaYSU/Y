"""gate.py acc.json: the step-5 gate. For every case of every shape, each Y column's rel_F and
max_norm (max |E| / (|A| @ |B|)) against cuBLAS's: PASS where both are at or below cuBLAS's. Then per
shape group and column, cuBLAS's error over the column's (above 1: the column's error is smaller) and
the share of outputs farther from float64 than cuBLAS's, as ranges over the input distributions."""
import json, sys
cases = json.load(open(sys.argv[1]))["cases"]
GROUP = {"normal": "mixed", "uniform_pm1": "mixed", "wide_range": "mixed", "small": "mixed", "large": "mixed",
         "abs_normal": "same sign", "uniform01": "same sign", "offset": "cancellation", "paired": "cancellation"}
r = lambda v: f"{min(v):.3g}-{max(v):.3g}"
print(f"{len(cases)} cases")
fails = []
for shape in dict.fromkeys(c["shape"] for c in cases):
    sc = [c for c in cases if c["shape"] == shape]
    cols = [x for x in sc[0]["columns"] if x.startswith("Y_")]
    print(f"\n{shape} ({len(sc)} cases)")
    for col in cols:
        ok = [c["columns"][col]["rel_fro"] <= c["columns"]["cuBLAS"]["rel_fro"] and c["columns"][col]["max_norm"] <= c["columns"]["cuBLAS"]["max_norm"] for c in sc]
        fails += [(shape, col, c["dist"]) for c, o in zip(sc, ok) if not o and col == "Y_auto"]
        line = f"  {col:16s} at or below cuBLAS in both: {sum(ok)}/{len(sc)} |"
        for g in ("mixed", "same sign", "cancellation"):
            gc = [c for c in sc if GROUP[c["dist"]] == g]
            rf = [c["columns"]["cuBLAS"]["rel_fro"] / c["columns"][col]["rel_fro"] for c in gc]
            mn = [c["columns"]["cuBLAS"]["max_norm"] / c["columns"][col]["max_norm"] for c in gc]
            worse = [c["vs_cuBLAS"][col]["worse"] for c in gc]
            line += f" {g}: rel_F x{r(rf)}, max_norm x{r(mn)}, worse {r(worse)} |"
        print(line)
print("\nY_auto (the rule's kernel) not at or below cuBLAS in both:", fails if fails else "none")
