"""speed_table.py DIR: run_speed2.sh's results as markdown: decode min us from L2 (bench_ab.py)
and from DRAM (bench_cold.py), per run, and the 512 x 512 kernels' min us and cuBLAS / Y."""
import json, os, sys
D = sys.argv[1]
for kind, pre, key in [("from L2 (`bench_ab.py`)", "decode", "sizes"), ("from DRAM (`bench_cold.py`, cold)", "cold", "shapes")]:
    rs = [json.load(open(f"{D}/{pre}_{i}.json")) for i in (1, 2)]
    shapes = list(rs[0][key])
    cols = [c for c in rs[0][key][shapes[0]]["samples"] if "warm" not in c]
    print(f"\n### Decode, M x 4096 x 4096, {kind}: min us, run 1 / 2\n")
    print("| M | " + " | ".join(c.replace(" cold", "") for c in cols) + " |")
    print("| ---: | " + " | ".join("---:" for _ in cols) + " |")
    for sh in shapes:
        for r in rs:
            assert all(r[key][sh]["exact"].values()), (sh, r[key][sh]["exact"])
        print(f"| {sh.split('x')[0]} | " + " | ".join(" / ".join(f"{min(r[key][sh]['samples'][c]):.2f}" for r in rs) for c in cols) + " |")
for K in ["4096", "65536"]:
    runs = {f[len(f"k{K}_"):-5]: json.load(open(f"{D}/{f}")) for f in sorted(os.listdir(D)) if f.startswith(f"k{K}_") and f.endswith(".json")}
    print(f"\n### 512 x 512 x {K}: min us (cuBLAS / Y), " + ", ".join(runs) + "\n")
    print("| column | " + " | ".join(runs) + " |")
    print("| :--- | " + " | ".join("---:" for _ in runs) + " |")
    for c in next(iter(runs.values()))["samples"]:
        print(f"| {c} | " + " | ".join(f"{min(r['samples'][c]):.2f} ({min(r['samples']['cuBLAS']) / min(r['samples'][c]):.3f})" + ("" if r["exact"].get(c, True) and r["exact_after_graph"].get(c, True) else " NOT EXACT") for r in runs.values()) + " |")
