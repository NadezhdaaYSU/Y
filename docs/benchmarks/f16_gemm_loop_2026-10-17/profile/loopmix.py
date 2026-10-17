"""For one kernel's exported SASS source page (ncu --page source --print-source
sass --csv): the K loop (the most-executed backward branch), its opcode mix
weighted by executions, and the instructions carrying the most samples of
the stall reasons named on the command line.

    python3 loopmix.py src_y_4096.csv stall_long_sb stall_mio stall_wait
"""
import csv
import sys
from collections import Counter


def num(x):
    try:
        return float(x.replace(".", "").replace(",", ".")) if x.count(",") == 1 or x.count(".") > 1 else float(x)
    except ValueError:
        return 0.0


rows = list(csv.reader(open(sys.argv[1])))
hdr, data = rows[0], rows[1:]
if "Address" not in hdr:
    hdr, data = rows[1], rows[2:]
col = {h: i for i, h in enumerate(hdr)}
addrs = [int(r[col["Address"]], 16) for r in data]
src = [r[col["Source"]] for r in data]
ex = [num(r[col["Instructions Executed"]]) for r in data]
best = None
for i, s in enumerate(src):
    if "BRA" in s and "0x" in s:
        tgt = int(s.split("0x")[-1].split()[0].rstrip(";"), 16)
        if tgt < addrs[i] and (best is None or ex[i] > ex[best[0]]):
            best = (i, tgt)
end, start = best[0], addrs.index(best[1])
loop = range(start, end + 1)
trips = ex[end]
print(f"loop {addrs[start]:#x}..{addrs[end]:#x}: {end - start + 1} instructions, back-branch executed {trips:.0f} times")
mix = Counter()
for i in loop:
    op = src[i].split()[0] if not src[i].startswith("@") else src[i].split()[1]
    mix[op.split(".")[0]] += ex[i] / trips
print("per iteration: " + ", ".join(f"{k} {v:.1f}" for k, v in mix.most_common()))
print(f"total per iteration {sum(mix.values()):.1f}")
total = {r: sum(num(data[i][col[r]]) for i in range(len(data))) for r in sys.argv[2:]}
for reason in sys.argv[2:]:
    print(f"\n{reason}: {total[reason]:.0f} samples in the kernel, {sum(num(data[i][col[reason]]) for i in loop):.0f} in the loop")
    top = sorted(range(len(data)), key=lambda i: -num(data[i][col[reason]]))[:8]
    for i in top:
        where = "loop" if i in loop else ("before" if i < start else "after")
        print(f"   {num(data[i][col[reason]]):7.0f}  {where:6s} {addrs[i]:#06x}  {src[i][:90]}")
