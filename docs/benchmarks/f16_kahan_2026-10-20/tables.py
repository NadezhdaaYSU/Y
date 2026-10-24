"""tables.py acc.json: accuracy_kahan.py's results as markdown: each metric per case, the
cases each column wins against cuBLAS and SGEMM, and the per-output fractions."""
import json, sys
r = json.load(open(sys.argv[1]))
order = ["Y_mma", "Y_f32", "Y_kahan", "Y_f32_s2", "Y_kahan_s2", "Y_f32_s4", "Y_kahan_s4", "Y_f32_s16", "Y_kahan_s16",
         "cuBLAS", "SGEMM", "RN32"]
shapes = list(dict.fromkeys(c["shape"] for c in r["cases"]))
for shape in shapes:
    cs = [c for c in r["cases"] if c["shape"] == shape]
    cols = [x for x in order if x in cs[0]["columns"]]
    print(f"\n## {shape} (cuBLAS: {', '.join(k[:60] for k in cs[0]['cublas_kernels'] if 'Memset' not in k)})\n")
    for metric, title in [("rel_fro", "rel_F"), ("max_norm", "max_norm"), ("mean_norm", "mean_norm")]:
        print(f"\n{title}\n")
        print("| distribution | " + " | ".join(cols) + " |")
        print("| :--- | " + " | ".join("---:" for _ in cols) + " |")
        for c in cs:
            print(f"| {c['dist']} | " + " | ".join(f"{c['columns'][x][metric]:.2e}" for x in cols) + " |")
    print("\nCases (of %d) with lower rel_F / max_norm than cuBLAS, then than SGEMM\n" % len(cs))
    for x in cols:
        if x.startswith("Y_"):
            w = [sum(c["columns"][x][m] < c["columns"][b][m] for c in cs) for b in ("cuBLAS", "SGEMM") for m in ("rel_fro", "max_norm")]
            print(f"- {x}: cuBLAS {w[0]}/{w[1]}, SGEMM {w[2]}/{w[3]}")
    print("\nPer output, farther from R than cuBLAS's / SGEMM's / the f32 twin's (min-max over distributions)\n")
    for x in cols:
        if x.startswith("Y_"):
            f = lambda key: [c[key][x]["worse"] for c in cs if x in c.get(key, {})]
            parts = []
            for key in ("vs_cuBLAS", "vs_SGEMM", "vs_f32_twin"):
                v = f(key)
                parts.append(f"{100*min(v):.1f}-{100*max(v):.1f}%" if v else "-")
            print(f"- {x}: " + " / ".join(parts))
