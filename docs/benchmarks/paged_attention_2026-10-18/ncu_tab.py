#!/usr/bin/env python3
"""ncu_tab.py <report> [metric ...] : one row per profiled kernel."""
import csv, subprocess, sys
rep = sys.argv[1]
mets = sys.argv[2:] or ["gpu__time_duration.sum", "launch__grid_size", "launch__registers_per_thread",
    "sm__warps_active.avg.pct_of_peak_sustained_active", "smsp__inst_executed.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed", "lts__t_bytes.sum", "dram__bytes_read.sum"]
out = subprocess.run(["ncu", "-i", rep, "--page", "raw", "--csv"], capture_output=True, text=True).stdout
r = list(csv.reader(out.splitlines()))
h = r[0]
cols = [h.index("Kernel Name")] + [h.index(m) if m in h else None for m in mets]
print(" | ".join(["kernel"] + [m.split("__")[1][:28] for m in mets]))
print(" | ".join(["-"] + [r[1][c] if c is not None else "?" for c in cols[1:]]))
for row in r[2:]:
    print(" | ".join([row[cols[0]][:34]] + [row[c] if c is not None else "n/a" for c in cols[1:]]))
