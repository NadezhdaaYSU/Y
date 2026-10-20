"""tables.py acc.json: markdown tables of accuracy_split.py's results."""
import json, sys
r = json.load(open(sys.argv[1]))
cols = [c for c in ["Y_mma", "Y_f32", "Y_f32_s2", "Y_f32_s4", "Y_f32_s8", "Y_f32_s16", "Y_mma_s2", "Y_mma_s4", "Y_mma_s8",
                     "cuBLAS", "SGEMM", "RN32"] if c in r["cases"][0]["columns"]]
for metric, title in [("rel_fro", "rel_F = ||C - R||_F / ||R||_F"), ("max_abs", "max_abs = max |C - R|"),
                      ("max_norm", "max_norm = max |C - R| / S, S = |A| @ |B|")]:
    print(f"\n### {title}\n")
    print("| distribution | K | kappa_F | " + " | ".join(cols) + " |")
    print("| :--- | ---: | ---: | " + " | ".join("---:" for _ in cols) + " |")
    for c in r["cases"]:
        print(f"| {c['dist']} | {c['k']} | {c['kappa_F']:.1e} | " + " | ".join(f"{c['columns'][x][metric]:.2e}" for x in cols) + " |")
for base, key in [("cuBLAS", "vs_cuBLAS"), ("Y_f32 (unsplit)", "vs_Y_f32")]:
    print(f"\n### Per output against {base}: fraction closer to R / equal / farther\n")
    sc = ["Y_f32_s2", "Y_f32_s4", "Y_f32_s8", "Y_f32_s16"] + (["Y_f32", "Y_mma_s8"] if key == "vs_cuBLAS" else [])
    print("| distribution | K | " + " | ".join(sc) + " |")
    print("| :--- | ---: | " + " | ".join(":---" for _ in sc) + " |")
    for c in r["cases"]:
        print(f"| {c['dist']} | {c['k']} | " + " | ".join(
            "{better:.3f} / {equal:.3f} / {worse:.3f}".format(**c[key][x]) for x in sc) + " |")
wins = {x: [0, 0] for x in cols}
for c in r["cases"]:
    for x in cols:
        for i, m in enumerate(["rel_fro", "max_norm"]):
            wins[x][i] += c["columns"][x][m] < c["columns"]["cuBLAS"][m]
print("\n### Cases (of %d) with lower rel_F / lower max_norm than cuBLAS\n" % len(r["cases"]))
print(" | ".join(f"{x} {w[0]}/{w[1]}" for x, w in wins.items() if x != "cuBLAS"))
print("\nreference check: max deviation over S", max(c["ref_check"]["max_dev_over_S"] for c in r["cases"]),
      "max deviation / column error", max(c["ref_check"]["max_dev_over_column_error"] for c in r["cases"]))
print("Y_f32_s4 == Y_f32_s4_t128 (bits) in every case:", all(c["columns"]["Y_f32_s4"]["sha256_C"] == c["columns"]["Y_f32_s4_t128"]["sha256_C"] for c in r["cases"]))
for x in [c for c in ["Y_mma_s2", "Y_mma_s4", "Y_mma_s8"] if c in cols]:
    print(x, "== cuBLAS (bits) in cases:", [f"{c['dist']}:{c['k']}" for c in r["cases"] if c["columns"][x]["sha256_C"] == c["columns"]["cuBLAS"]["sha256_C"]])
