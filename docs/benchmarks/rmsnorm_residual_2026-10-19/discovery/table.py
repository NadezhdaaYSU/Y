#!/usr/bin/env python3
"""table.py: minimum per column from nr_1.log and nr_2.log; Y's time over Triton's."""
import re, os
D = os.path.dirname(os.path.abspath(__file__))
runs = []
for f in ("nr_1.log", "nr_2.log"):
    rows, cur = {}, None
    for line in open(os.path.join(D, f)):
        m = re.match(r"== (\S+) rows (\d+)\s+(\d+) MB", line)
        if m:
            cur = (m.group(1), int(m.group(2)), int(m.group(3))); rows[cur] = {}
            continue
        m = re.match(r"\s+(Y|Triton|PyTorch)\s+min\s+([\d.]+)", line)
        if m and cur:
            rows[cur][m.group(1)] = float(m.group(2))
    runs.append(rows)
print("| kernel | rows | footprint | Y | Triton | PyTorch | Y / Triton |")
print("|---|---|---|---|---|---|---|")
for k in runs[0]:
    a, b = runs[0][k], runs[1][k]
    f = lambda c: f"{a[c]:.2f} / {b[c]:.2f}"
    print(f"| {k[0]} | {k[1]} | {k[2]} MB | {f('Y')} | {f('Triton')} | {f('PyTorch')} | {a['Y'] / a['Triton']:.3f} / {b['Y'] / b['Triton']:.3f} |")
