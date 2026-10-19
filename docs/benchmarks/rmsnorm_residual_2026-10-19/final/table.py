#!/usr/bin/env python3
"""table.py: minimum per column from final_1.log and final_2.log. Ratios are
times over Y after's: above 1, Y after is faster."""
import os, re
D = os.path.dirname(os.path.abspath(__file__))
runs = []
for f in ("final_1.log", "final_2.log"):
    rows, cur = {}, None
    for line in open(os.path.join(D, f)):
        m = re.match(r"== d=(\d+) rows (\d+)\s+(\d+) MB", line)
        if m:
            cur = (int(m.group(1)), int(m.group(3))); rows[cur] = {"rows": int(m.group(2))}
            continue
        m = re.match(r"\s+(Y before|Y after|Triton|FlashInfer)\s+min\s+([\d.]+)", line)
        if m and cur:
            rows[cur][m.group(1)] = float(m.group(2))
    runs.append(rows)
r2 = lambda k, c: " / ".join(f"{r[k][c] / r[k]['Y after']:.3f}" if c in r[k] else "-" for r in runs)
print("| hidden | footprint | Y after (us) | before / after | Triton / after | FlashInfer / after |")
print("|---|---|---|---|---|---|")
for k in runs[0]:
    print(f"| {k[0]} | {k[1]} MB | {runs[0][k]['Y after']:.2f} / {runs[1][k]['Y after']:.2f} | {r2(k, 'Y before')} | {r2(k, 'Triton')} | {r2(k, 'FlashInfer')} |")
