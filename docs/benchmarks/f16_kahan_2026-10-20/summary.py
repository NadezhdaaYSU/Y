"""summary.py acc.json: per shape and Y column, cuBLAS's and SGEMM's rel_F and max_norm over
Y's for the mixed-sign, same-sign and cancellation groups; the cases Y wins; same-sign rel_F
and mean_norm; and the outputs farther from R than cuBLAS's, SGEMM's and the f32 twin's."""
import json, sys
r = json.load(open(sys.argv[1]))
groups = {"mixed": ["normal", "uniform_pm1", "wide_range", "small", "large"], "same": ["abs_normal", "uniform01"],
          "canc": ["offset", "paired"]}
def rng(v):
    f = lambda x: f"{x:.2g}" if x < 10 else f"{x:.0f}"
    return f(min(v)) if f(min(v)) == f(max(v)) else f"{f(min(v))}-{f(max(v))}"
for shape in dict.fromkeys(c["shape"] for c in r["cases"]):
    cs = [c for c in r["cases"] if c["shape"] == shape]
    print(f"\n### {shape}\n")
    print("| Y | vs | mixed rel_F / max_norm | same | cancellation | wins rel_F / max_norm | farther outputs |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for y in [x for x in cs[0]["columns"] if x.startswith("Y_") and x != "Y_mma"]:
        for base in ["cuBLAS", "SGEMM"]:
            cells = []
            for ds in groups.values():
                g = [c for c in cs if c["dist"] in ds]
                cells.append(" / ".join(rng([c["columns"][base][m] / c["columns"][y][m] for c in g]) for m in ("rel_fro", "max_norm")))
            wins = [sum(c["columns"][y][m] < c["columns"][base][m] for c in cs) for m in ("rel_fro", "max_norm")]
            far = [100 * c[f"vs_{base}"][y]["worse"] for c in cs]
            print(f"| {y} | {base} | " + " | ".join(cells) + f" | {wins[0]}/{len(cs)} / {wins[1]}/{len(cs)} | {min(far):.1f}-{max(far):.1f}% |")
    for y in [x for x in cs[0]["columns"] if x.startswith("Y_kahan")]:
        far = [100 * c["vs_f32_twin"][y]["worse"] for c in cs if y in c["vs_f32_twin"]]
        same = [c for c in cs if c["dist"] in groups["same"]]
        print(f"\n{y}: farther than its f32 twin on {min(far):.1f}-{max(far):.1f}% of outputs; same-sign rel_F "
              f"{rng([c['columns'][y]['rel_fro'] for c in same])} (twin {rng([c['columns'][y.replace('Y_kahan', 'Y_f32')]['rel_fro'] for c in same])}), "
              f"mean_norm {rng([-c['columns'][y]['mean_norm'] for c in same])} (negative)")
