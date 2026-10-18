import csv, re, sys

def load(path):
    rows = list(csv.reader(open(path)))
    name = rows[0][1]
    hdr = rows[1]
    ix = {h: i for i, h in enumerate(hdr)}
    out = []
    for r in rows[2:]:
        if len(r) < len(hdr) or not r[0].startswith("0x"):
            continue
        out.append({"addr": int(r[0], 16), "sass": r[1].strip(),
                    "samples": int(r[ix["Warp Stall Sampling (All Samples)"]] or 0),
                    "exec": int(r[ix["Instructions Executed"]] or 0),
                    "why": {h[6:]: int(r[i] or 0) for h, i in ix.items() if h.startswith("stall_") and "Not Issued" not in h}})
    return name, out

