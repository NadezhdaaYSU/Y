#!/usr/bin/env python3
"""hot.py <csv> <min samples>: every SASS line with its executed count, samples and
top reasons; lines under the threshold collapse into a count, HMMA/BAR/DEPBAR/LDSM/LDGSTS always shown."""
import sys
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
from regions_lib import load
name, ins = load(sys.argv[1])
th = int(sys.argv[2])
skipped = 0
hm = 0
for k, x in enumerate(ins):
    op = x["sass"].split()
    key = any(t in x["sass"] for t in ("BAR.SYNC", "DEPBAR", "LDGDEPBAR", "LDSM", "LDGSTS", "EXIT", "BRA"))
    if "HMMA" in x["sass"]:
        hm += 1
    if x["samples"] >= th or key:
        if skipped:
            print(f"        ... {skipped} lines")
            skipped = 0
        why = ", ".join(f"{a} {b}" for a, b in sorted(x["why"].items(), key=lambda kv: -kv[1])[:3] if b)
        print(f"{k:4d} h{hm:3d} {x['exec']:6d} {x['samples']:4d}  {x['sass'][:60]:60s} {why}")
    else:
        skipped += 1
