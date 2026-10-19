#!/usr/bin/env python3
"""table.py final_1.log final_2.log: each column's graph-replay minimum over
"after"'s, run 1 / run 2 (in place: FlashInfer's over after_inplace's).
Triton and Triton fast are the faster of the two configurations each run
kept. Above 1, "after" is faster."""
import re, sys

def parse(path):
    rows, cur = {}, None
    for line in open(path):
        m = re.match(r"== rope_(\d+) rows (\d+)\s+([\d.]+) MB", line)
        if m:
            cur = (int(m.group(1)), int(m.group(2)), float(m.group(3)))
            rows[cur] = {}
            continue
        m = re.match(r"   (\S+)\s+graph\s+([\d.]+) us", line)
        if m and cur:
            rows[cur][m.group(1)] = float(m.group(2))
    return rows

runs = [parse(p) for p in sys.argv[1:]]

def best(r, pre):
    v = [t for n, t in r.items() if n.startswith(pre) and (pre != "triton_r" or not n.startswith("triton_fast"))]
    return min(v)

print("| head_dim | rows | footprint | after (us) | before / after | Triton / after | Triton fast / after | FlashInfer / after | FlashInfer in place / after in place |")
print("|---|---|---|---|---|---|---|---|---|")
for key in runs[0]:
    hd, rows, mb = key
    cells = []
    rs = [r[key] for r in runs]
    cells.append(" / ".join(f"{r['after']:.2f}" for r in rs))
    for num, den in (("before", "after"), ("triton_r", "after"), ("triton_fast_", "after"),
                     ("flashinfer", "after"), ("flashinfer_inplace", "after_inplace")):
        vals = []
        for r in rs:
            n = best(r, num) if num.startswith("triton") else r[num]
            vals.append(f"{n / r[den]:.3f}")
        cells.append(" / ".join(vals))
    print(f"| {hd} | {rows} | {mb:.0f} MB | " + " | ".join(cells) + " |")
