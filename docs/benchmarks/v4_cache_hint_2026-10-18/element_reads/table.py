#!/usr/bin/env python3
"""table.py: the minimum of each column from ca2_1.log and ca2_2.log, and `.ca`'s
time over plain's (above 1: `.ca` slower) and Triton's over plain's."""
import re, pathlib
here = pathlib.Path(__file__).parent
runs = []
for name in ("ca2_1.log", "ca2_2.log"):
    rows, cur = {}, None
    for line in (here / name).read_text().splitlines():
        m = re.match(r"== (\w+) (\d+) \((\d+) MB\)", line)
        if m:
            cur = (m.group(1), int(m.group(3))); rows[cur] = {}
            continue
        m = re.match(r"\s+(Y ca|Y plain|Triton)\s+min\s+([\d.]+)", line)
        if m and cur:
            rows[cur][m.group(1)] = float(m.group(2))
    runs.append(rows)
print("| kernel | footprint | Y plain | Y `.ca` | Triton | `.ca` / plain | Triton / plain |")
print("|---|---|---|---|---|---|---|")
for key in runs[0]:
    a, b = runs[0][key], runs[1][key]
    f = lambda c: f"{a[c]:.2f} / {b[c]:.2f}"
    r = lambda c: f"{a[c] / a['Y plain']:.3f} / {b[c] / b['Y plain']:.3f}"
    print(f"| {key[0]} | {key[1]} MB | {f('Y plain')} | {f('Y ca')} | {f('Triton')} | {r('Y ca')} | {r('Triton')} |")
