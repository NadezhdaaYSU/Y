#!/usr/bin/env python3
"""tables.py: the README's table, from probe_cs1.log and probe_cs2.log."""
import os, re
D = os.path.dirname(os.path.abspath(__file__))
COLS = ["Y cs", "Y plain", "Y ca", "Triton"]
FOOT = {"vec_add": lambda n: 12 * n, "swiglu": lambda n: 12 * n}


def parse(f):
    d, case = {}, None
    for line in open(os.path.join(D, f)):
        m = re.match(r"== (.+?)  all exact", line)
        if m:
            case = m.group(1); d[case] = {}
            continue
        m = re.match(r"\s+(Y cs|Y plain|Y ca|Triton)\s+min\s+([\d.]+)", line)
        if m:
            d[case][m.group(1)] = float(m.group(2))
    return d


runs = [parse("probe_cs1.log"), parse("probe_cs2.log")]
print("| shape | footprint | " + " | ".join(COLS) + " | cs / plain | plain vs Triton |")
print("|---" * (len(COLS) + 4) + "|")
for case in runs[0]:
    m = re.match(r"(vec_add|swiglu) (\d+)", case)
    if m:
        mb = 12 * int(m.group(2)) / 1e6
    else:
        r, h = (int(x) for x in re.findall(r"\d+", case))
        mb = (8 * r * h + 4 * h) / 1e6
    cells = [" / ".join("%.2f" % run[case][c] for run in runs) for c in COLS]
    ratio = " / ".join("%.2fx" % (run[case]["Y cs"] / run[case]["Y plain"]) for run in runs)
    tri = " / ".join("%.2fx" % (run[case]["Triton"] / run[case]["Y plain"]) for run in runs)
    print("| %s | %.0f MB | %s | %s | %s |" % (case, mb, " | ".join(cells), ratio, tri))
