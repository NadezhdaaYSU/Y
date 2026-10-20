"""ratios.py acc.json: cuBLAS's (and SGEMM's) error over each Y column's, as rel_F / max_norm ranges per
input group and K, the per-output 'farther' range against cuBLAS, and the mean signed normalized error."""
import json, sys
r = json.load(open(sys.argv[1]))
groups = {"mixed sign": ["normal", "uniform_pm1", "wide_range", "small", "large"],
          "same sign": ["abs_normal", "uniform01"], "cancellation": ["offset", "paired"]}
cols = [c for c in ["Y_f32", "Y_f32_s2", "Y_f32_s4", "Y_f32_s8", "Y_f32_s16", "Y_mma_s2", "Y_mma_s4", "Y_mma_s8"]
        if c in r["cases"][0]["columns"]]
def rng(v):
    lo, hi = min(v), max(v)
    f = lambda x: f"{x:.3g}"
    return f(lo) if f(lo) == f(hi) else f"{f(lo)}-{f(hi)}"
for base in ["cuBLAS", "SGEMM"]:
    print(f"\n### {base}'s error over Y's (rel_F / max_norm; above 1, Y's error is smaller)\n")
    print("| column | K | " + " | ".join(groups) + " |")
    print("| :--- | ---: | " + " | ".join(":---" for _ in groups) + " |")
    for col in cols:
        for k in sorted({c["k"] for c in r["cases"]}):
            cells = []
            for g, ds in groups.items():
                cs = [c for c in r["cases"] if c["k"] == k and c["dist"] in ds]
                q = lambda m: rng([c["columns"][base][m] / c["columns"][col][m] for c in cs])
                cells.append(f"{q('rel_fro')} / {q('max_norm')}")
            print(f"| {col} | {k} | " + " | ".join(cells) + " |")
print("\n### Outputs farther from R than cuBLAS's (range over the group)\n")
print("| column | K | " + " | ".join(groups) + " |")
print("| :--- | ---: | " + " | ".join(":---" for _ in groups) + " |")
for col in [c for c in cols if c in r["cases"][0]["vs_cuBLAS"]]:
    for k in sorted({c["k"] for c in r["cases"]}):
        cells = []
        for g, ds in groups.items():
            cs = [c for c in r["cases"] if c["k"] == k and c["dist"] in ds]
            cells.append(rng([100 * c["vs_cuBLAS"][col]["worse"] for c in cs]) + "%")
        print(f"| {col} | {k} | " + " | ".join(cells) + " |")
m = "mean_norm" if "mean_norm" in r["cases"][0]["columns"]["cuBLAS"] else None
if m:
    print("\n### Mean signed normalized error mean((C - R) / S), same-sign inputs\n")
    allc = cols + ["cuBLAS", "SGEMM"]
    print("| distribution | K | " + " | ".join(allc) + " |")
    print("| :--- | ---: | " + " | ".join("---:" for _ in allc) + " |")
    for c in r["cases"]:
        if c["dist"] in groups["same sign"]:
            print(f"| {c['dist']} | {c['k']} | " + " | ".join(f"{c['columns'][x][m]:.2e}" for x in allc) + " |")
