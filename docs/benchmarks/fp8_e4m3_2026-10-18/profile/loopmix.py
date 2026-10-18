"""Per kernel in an ncu report: the K loop (the most-executed backward
branch), its SASS opcode mix, and warp-stall samples before, in and after it,
with the loop's top stall reasons.

    python3 loopmix.py report.ncu-repz
"""
import collections
import csv
import io
import subprocess
import sys


def num(x):
    try:
        return float(x.replace(",", "."))
    except ValueError:
        return 0.0


out = subprocess.run(["ncu", "--import", sys.argv[1], "--page", "source", "--print-source", "sass", "--csv"],
                     capture_output=True, text=True).stdout
sections, cur = [], None
for row in csv.reader(io.StringIO(out)):
    if row and row[0] == "Kernel Name":
        cur = [row[1], None, []]
        sections.append(cur)
    elif row and row[0] == "Address":
        cur[1] = row
    elif cur and row:
        cur[2].append(row)
for name, hdr, data in sections:
    ai, si = hdr.index("Address"), hdr.index("Source")
    wi, ei = hdr.index("Warp Stall Sampling (All Samples)"), hdr.index("Instructions Executed")
    stall_cols = [i for i, h in enumerate(hdr) if h.startswith("stall_") or "Stall Sampling" in h and "(" not in h]
    addrs = [int(r[ai], 16) for r in data]
    best = None
    for i, r in enumerate(data):
        if "BRA" in r[si] and "0x" in r[si]:
            tgt = int(r[si].split("0x")[-1].split()[0].rstrip(";"), 16)
            if tgt < addrs[i] and (best is None or num(r[ei]) > num(data[best[0]][ei])):
                best = (i, tgt)
    end, start = best[0], addrs.index(best[1])
    loop = data[start:end + 1]
    total = sum(num(r[wi]) for r in data)
    part = lambda rs: sum(num(r[wi]) for r in rs) / total
    ops = collections.Counter(r[si].split()[0] if not r[si].strip().startswith("@") else r[si].split()[1] for r in loop)
    ops = collections.Counter({k.split(".")[0]: 0 for k in ops}) + collections.Counter(
        (r[si].split()[1] if r[si].strip().startswith("@") else r[si].split()[0]).split(".")[0] for r in loop)
    print(f"{name[:60]}\n  loop {len(loop)} SASS instructions; samples before {part(data[:start]):.1%}, "
          f"in {part(loop):.1%}, after {part(data[end + 1:]):.1%}; total {total:.0f}")
    print("  mix:", ", ".join(f"{k} {v}" for k, v in ops.most_common(14)))
    reasons = collections.Counter()
    for i, h in enumerate(hdr):
        if h.startswith("Warp Stall Sampling") or not any(x in h.lower() for x in ("stall", "throttle", "scoreboard", "barrier", "wait", "selected", "dispatch", "branch", "membar", "sleeping", "math", "mio", "lg_", "tex", "drain", "imc", "misc", "no_inst")):
            continue
        reasons[h] = sum(num(r[i]) for r in loop)
    print("  loop stalls:", ", ".join(f"{k} {v:.0f}" for k, v in reasons.most_common(8)))
