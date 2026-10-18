#!/usr/bin/env python3
"""tables.py: the README's tables from the runs' JSON (minimum of each column,
run 1 / run 2). "before / after" is the committed kernel's time over the new
schedule's: above 1, the new schedule is faster."""
import json, os
D = os.path.dirname(os.path.abspath(__file__))

def mins(path):
    d = json.load(open(os.path.join(D, path)))
    out = {}
    for size, v in d["sizes"].items():
        assert all(v["exact"].values()), (path, size, v["exact"])
        out[size] = {c: min(s) for c, s in v["samples"].items()}
    return out

def two(fmt, a, b):
    return f"{fmt % a} / {fmt % b}"

print("## hand/: the committed 512 kernel against its hand-pipelined copy")
h = [mins(f"hand/bench_{r}.json")["512"] for r in (1, 2)]
print("| committed | hand-pipelined | cuBLAS | committed / hand |")
print("|---|---|---|---|")
print(f"| {two('%.3f', h[0]['base'], h[1]['base'])} | {two('%.3f', h[0]['xbar'], h[1]['xbar'])} | "
      f"{two('%.3f', h[0]['cuBLAS'], h[1]['cuBLAS'])} | {two('%.3f', h[0]['base'] / h[0]['xbar'], h[1]['base'] / h[1]['xbar'])} |")

for title, files in (("sizes/", ("sizes/bench_1.json", "sizes/bench_2.json")),
                     ("decode/", ("decode/bench_dec_1.json", "decode/bench_dec_2.json"))):
    print(f"\n## {title}")
    r = [mins(f) for f in files]
    print("| shape | before | after | cuBLAS | before / after | cuBLAS / after |")
    print("|---|---|---|---|---|---|")
    for size in r[0]:
        a, b = r[0][size], r[1][size]
        print(f"| {size} | {two('%.3f', a['off'], b['off'])} | {two('%.3f', a['on'], b['on'])} | "
              f"{two('%.3f', a['cuBLAS'], b['cuBLAS'])} | {two('%.3f', a['off'] / a['on'], b['off'] / b['on'])} | "
              f"{two('%.3f', a['cuBLAS'] / a['on'], b['cuBLAS'] / b['on'])} |")

print("\n## tiles512/")
r = [mins(f"tiles512/sw_{i}.json")["512"] for i in (1, 2)]
tiles = sorted({c[c.index("_") + 1:] for c in r[0] if c.startswith(("off_", "on_"))},
               key=lambda t: min(r[0]["off_" + t], r[1]["off_" + t]))
print("| tile (cta_m, cta_n, cta_k, warps_m, warps_n, stages) | before | after | cuBLAS / before |")
print("|---|---|---|---|")
for t in tiles:
    print(f"| {t.replace('_', ', ')} | {two('%.3f', r[0]['off_' + t], r[1]['off_' + t])} | "
          f"{two('%.3f', r[0]['on_' + t], r[1]['on_' + t])} | {two('%.3f', r[0]['cuBLAS'] / r[0]['off_' + t], r[1]['cuBLAS'] / r[1]['off_' + t])} |")
print(f"| cuBLAS | {two('%.3f', r[0]['cuBLAS'], r[1]['cuBLAS'])} | | |")
print(f"| Triton | {two('%.3f', r[0]['Triton'], r[1]['Triton'])} | | |")
