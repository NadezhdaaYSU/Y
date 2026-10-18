#!/usr/bin/env python3
"""table.py: the README table from run1.json and run2.json (minimum per column)."""
import json, os, statistics
D = os.path.dirname(os.path.abspath(__file__))
runs = [json.load(open(os.path.join(D, f))) for f in ("run1.json", "run2.json")]
print("| M=N=K | Y us | cuBLAS us | Triton us | Y vs cuBLAS | Y vs Triton | Y TFLOPS | median/min Y, cuBLAS |")
print("|---|---|---|---|---|---|---|---|")
for size in runs[0]["sizes"]:
    n = int(size)
    cells = []
    for r in runs:
        s = r["sizes"][size]
        assert all(s["exact"].values()), (size, s["exact"])
        m = {c: min(v) for c, v in s["samples"].items()}
        md = {c: statistics.median(v) for c, v in s["samples"].items()}
        cells.append((m["Y"], m["cuBLAS"], m["Triton"], m["cuBLAS"] / m["Y"], m["Triton"] / m["Y"],
                      2 * n ** 3 / (m["Y"] * 1e-6) / 1e12, md["Y"] / m["Y"], md["cuBLAS"] / m["cuBLAS"]))
    f = ["%.2f", "%.2f", "%.2f", "%.3fx", "%.3fx", "%.1f"]
    print("| %d | " % n + " | ".join(" / ".join(f[i] % c[i] for c in cells) for i in range(6))
          + " | " + " / ".join("%.3f, %.3f" % (c[6], c[7]) for c in cells) + " |")
