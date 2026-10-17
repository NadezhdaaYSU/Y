"""Where each kernel's warp-stall samples fall: before the K loop, in it,
after it. The loop is the backward branch executed most often.

    python3 regions.py before_1024.ncu-repz
"""
import csv
import io
import subprocess
import sys


def num(x):
    try:
        return float(x.replace(",", "."))
    except ValueError:
        return 0.0


for name, regex in (("Y", "gemm_f16"), ("Triton", "triton")):
    out = subprocess.run(["ncu", "--import", sys.argv[1], "--page", "source", "--print-source", "sass",
                          "--csv", "--kernel-name", f"regex:{regex}"], capture_output=True, text=True).stdout
    rows = list(csv.reader(io.StringIO(out)))
    hdr, data = rows[1], rows[2:]
    ai, si = hdr.index("Address"), hdr.index("Source")
    wi, ei = hdr.index("Warp Stall Sampling (All Samples)"), hdr.index("Instructions Executed")
    addrs = [int(r[ai], 16) for r in data]
    best = None
    for i, r in enumerate(data):
        if "BRA" in r[si] and "0x" in r[si]:
            tgt = int(r[si].split("0x")[-1].split()[0].rstrip(";"), 16)
            if tgt < addrs[i] and (best is None or num(r[ei]) > num(data[best[0]][ei])):
                best = (i, tgt)
    end, start = best[0], addrs.index(best[1])
    total = sum(num(r[wi]) for r in data)
    part = lambda rs: sum(num(r[wi]) for r in rs) / total
    stores = {}
    for r in data:
        if "STG" in r[si]:
            op = [t for t in r[si].split() if t.startswith("STG")][0]
            stores[op] = stores.get(op, 0) + num(r[ei])
    print(f"{name}: before the loop {part(data[:start]):.1%}, loop {part(data[start:end + 1]):.1%}, "
          f"after it {part(data[end + 1:]):.1%}; global stores executed {stores}")
