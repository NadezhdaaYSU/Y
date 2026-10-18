#!/usr/bin/env python3
"""regions.py <ncu source-page csv> [<csv> ...]: per kernel, its backward branches,
then instructions executed and stall samples by region (before the first loop,
each loop, after the last), with the samples by reason."""
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

for path in sys.argv[1:]:
    name, ins = load(path)
    base = ins[0]["addr"]
    for k, x in enumerate(ins):
        x["k"] = k
    at = {x["addr"]: x["k"] for x in ins}
    loops = []
    for x in ins:
        m = re.search(r"\bBRA\b.*?(?:`?\(\.L_x_\d+\)|0x([0-9a-f]+))", x["sass"])
        t = re.search(r"0x([0-9a-f]+)\s*;?$", x["sass"])
        if "BRA" in x["sass"] and t:
            tgt = int(t.group(1), 16)
            if tgt in at and at[tgt] <= x["k"]:
                loops.append((at[tgt], x["k"]))
    tot = sum(x["samples"] for x in ins)
    texe = sum(x["exec"] for x in ins)
    print(f"#### {name}: {len(ins)} SASS, {texe} warp instructions executed, {tot} samples")
    for a, b in loops:
        print(f"  backward branch: SASS {a}..{b} ({b - a + 1} instructions), body executed {ins[a]['exec']} times (warp-level)")
    cuts = sorted(set([0] + [a for a, b in loops] + [b + 1 for a, b in loops] + [len(ins)]))
    for s, e in zip(cuts, cuts[1:]):
        part = ins[s:e]
        smp = sum(x["samples"] for x in part)
        exe = sum(x["exec"] for x in part)
        why = {}
        for x in part:
            for k2, v in x["why"].items():
                why[k2] = why.get(k2, 0) + v
        top = ", ".join(f"{k2} {v}" for k2, v in sorted(why.items(), key=lambda kv: -kv[1])[:6] if v)
        kind = "loop" if any(s == a for a, b in loops) else "straight"
        print(f"  SASS {s:4d}..{e - 1:4d} {kind:8s} {e - s:4d} static, {exe:7d} executed ({100 * exe / texe:5.1f}%), "
              f"{smp:6d} samples ({100 * smp / tot:5.1f}%): {top}")
