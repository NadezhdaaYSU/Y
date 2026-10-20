"""analyze.py <acc_1.json> <acc_2.json> <tables.md>: checks the two runs agree
bit for bit (inputs and every output), checks the floor and the reference,
and writes the per-case tables and the per-distribution summary.
"""
import json
import sys

r1, r2 = (json.load(open(p)) for p in sys.argv[1:3])
out = open(sys.argv[3], "w")
cases = r1["cases"]
assert len(cases) == len(r2["cases"]) == 99
COLS = ["Y_mma", "Y_f32", "cuBLAS", "Triton", "SGEMM", "RN32"]

# 1. reproducibility
for a, b in zip(cases, r2["cases"]):
    assert (a["dist"], a["shape"], a["seed"], a["sha256_A"], a["sha256_B"]) == \
           (b["dist"], b["shape"], b["seed"], b["sha256_A"], b["sha256_B"]), a["dist"]
    for c in a["columns"]:
        assert a["columns"][c]["sha256_C"] == b["columns"][c]["sha256_C"], (a["dist"], a["shape"], c)
print("runs 1 and 2: same inputs and same output bits in every column of every case")

# 2. checks
u = 2.0 ** -24
worst_floor = max(max(x["columns"]["RN32"]["max_norm"], x["columns"]["RN32"]["rel_fro"]) for x in cases)
assert worst_floor <= u, worst_floor
zs = sum(v["zero_scale_nonzero"] for x in cases for v in x["columns"].values())
ref = max(x["ref_check"]["max_dev_over_S"] for x in cases)
refc = max(x["ref_check"]["max_dev_over_column_error"] for x in cases)
refe = max(x["ref_check"]["max_dev_over_exact"] for x in cases)
rz = sum(x["r_exact_zero"] for x in cases)
print(f"RN32 worst max_norm / rel_fro {worst_floor:.3e} <= 2^-24; zero-scale outputs nonzero: {zs}; "
      f"exactly-zero references: {rz}")
print(f"reference vs math.fsum on {sum(x['ref_check']['entries'] for x in cases)} outputs: max |dev|/S {ref:.2e}, "
      f"max |dev|/|exact| {refe:.2e}, max |dev| / a column's |error| there {refc:.2e}")
sub = max(max(x["A"]["frac_subnormal"], x["B"]["frac_subnormal"]) for x in cases if x["dist"] == "small")
print(f"small: f16 subnormal fraction of the inputs up to {sub:.3f}")
kernels = sorted({k for x in cases for k in x["cublas_kernels"]})
print("cuBLAS kernels:", *kernels, sep="\n  ")


def f(v):
    return f"{v:.2e}"


# 3. per-case tables
present = lambda x: [c for c in COLS if c in x["columns"]]  # noqa: E731
for metric, title in (("rel_fro", "Relative Frobenius error ||C - R||_F / ||R||_F"),
                      ("max_abs", "Maximum absolute error max |C - R|"),
                      ("max_norm", "Maximum normalized error max |C - R| / (|A| |B|)"),
                      ("mean_norm", "Mean signed normalized error mean (C - R) / (|A| |B|)")):
    out.write(f"### {title}\n\n| distribution | shape | seed | kappa_F | " + " | ".join(COLS) + " |\n")
    out.write("| :--- | :--- | ---: | ---: | " + " | ".join("---:" for _ in COLS) + " |\n")
    for x in cases:
        cells = [f(x["columns"][c][metric]) if c in x["columns"] else "-" for c in COLS]
        out.write(f"| {x['dist']} | {x['shape']} | {x['seed']} | {x['kappa_F']:.1e} | " + " | ".join(cells) + " |\n")
    out.write("\n")
out.write("### Per output against cuBLAS: fraction of outputs with smaller / equal / larger |C - R|\n\n")
out.write("| distribution | shape | Y_mma | Y_f32 | SGEMM | Y_f32 against SGEMM |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n")
for x in cases:
    t = lambda v: f"{v['better']:.3f} / {v['equal']:.3f} / {v['worse']:.3f}"  # noqa: E731
    out.write(f"| {x['dist']} | {x['shape']} | {t(x['vs_cuBLAS']['Y_mma'])} | {t(x['vs_cuBLAS']['Y_f32'])} | "
              f"{t(x['vs_cuBLAS']['SGEMM'])} | {t(x['Y_f32_vs_SGEMM'])} |\n")
out.write("\n")

# 4. per-distribution summary: ranges over the eleven shapes
print()
dists = list(dict.fromkeys(x["dist"] for x in cases))
hdr = ("| distribution | kappa_F | Y_f32 rel_fro | cuBLAS / Y_f32: rel_fro | max_abs | max_norm | "
       "Y_f32 worse than cuBLAS (outputs) | Y_mma = cuBLAS bits | SGEMM / Y_f32 rel_fro |")
print(hdr)
print("| :--- | " + " | ".join("---:" for _ in range(8)) + " |")
summary = {}
for d in dists:
    xs = [x for x in cases if x["dist"] == d]
    rng = lambda vals: f"{min(vals):.3g}-{max(vals):.3g}"  # noqa: E731
    ratio = lambda m, num="cuBLAS", den="Y_f32": [x["columns"][num][m] / x["columns"][den][m] for x in xs]  # noqa: E731
    row = {"kappa_F": [x["kappa_F"] for x in xs], "Y_f32_rel_fro": [x["columns"]["Y_f32"]["rel_fro"] for x in xs],
           "fro": ratio("rel_fro"), "abs": ratio("max_abs"), "norm": ratio("max_norm"),
           "worse": [x["vs_cuBLAS"]["Y_f32"]["worse"] for x in xs],
           "mma_bits": [x["vs_cuBLAS"]["Y_mma"]["bitwise"] for x in xs],
           "sgemm": ratio("rel_fro", "SGEMM", "Y_f32")}
    summary[d] = row
    print(f"| {d} | {rng(row['kappa_F'])} | {rng(row['Y_f32_rel_fro'])} | {rng(row['fro'])} | {rng(row['abs'])} | "
          f"{rng(row['norm'])} | {rng(row['worse'])} | {rng(row['mma_bits'])} | {rng(row['sgemm'])} |")
json.dump(summary, open(sys.argv[3].replace(".md", "_summary.json"), "w"), indent=1)
