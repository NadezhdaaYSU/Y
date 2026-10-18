#!/usr/bin/env python3
"""table.py: minimum of each column, run 1 / run 2, from cold_<r>.json (the
committed decode kernels) and m32_<r>.json (tiles at M = 32). "cold" takes
the next of 4 B copies (134 MB) each call; "warm" reuses one (33.5 MB, inside
the 48 MB L2). cuBLAS / Y above 1: Y faster."""
import json, os
D = os.path.dirname(os.path.abspath(__file__))
def load(name):
    out = {}
    for r in (1, 2):
        for shape, v in json.load(open(os.path.join(D, f"{name}_{r}.json")))["shapes"].items():
            assert all(v["exact"].values()), (name, r, shape, v["exact"])
            for c, s in v["samples"].items():
                out.setdefault(shape, {}).setdefault(c, []).append(min(s))
    return out
c = load("cold")
print("| M | Y from DRAM | cuBLAS from DRAM | cuBLAS / Y | Y from L2 | cuBLAS from L2 | cuBLAS / Y |")
print("|---|---|---|---|---|---|---|")
for shape, v in c.items():
    cell = lambda k: " / ".join(f"{x:.3f}" for x in v[k])
    rat = lambda a, b: " / ".join(f"{x / y:.3f}" for x, y in zip(v[a], v[b]))
    print(f"| {shape.split('x')[0]} | {cell('Y cold')} | {cell('cuBLAS cold')} | {rat('cuBLAS cold', 'Y cold')} | "
          f"{cell('Y warm')} | {cell('cuBLAS warm')} | {rat('cuBLAS warm', 'Y warm')} |")
m = load("m32")["32x4096x4096"]
print("\n| M = 32 tile | from DRAM | cuBLAS / tile | from L2 | cuBLAS / tile |")
print("|---|---|---|---|---|")
tiles = sorted({k.rsplit(" ", 1)[0] for k in m if k.startswith("t_")}, key=lambda t: min(m[t + " warm"]))
for t in tiles + ["cuBLAS"]:
    cell = lambda k: " / ".join(f"{x:.3f}" for x in m[k])
    rat = lambda k: " / ".join(f"{x / y:.3f}" for x, y in zip(m["cuBLAS " + k], m[t + " " + k]))
    name = t[2:].replace("_", ", ") if t != "cuBLAS" else "cuBLAS"
    print(f"| {name} | {cell(t + ' cold')} | {rat('cold')} | {cell(t + ' warm')} | {rat('warm')} |")
