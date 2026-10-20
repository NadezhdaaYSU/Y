"""final_table.py DIR: the final speed table from DIR/<K>/run<i>_<data>.json (sbench.py):
per column, the minimum per-call time over the interleaved rounds, and cuBLAS's minimum over it
(above 1, Y is faster), for each run and timing data; every column's exactness before and after."""
import json, os, sys
D = sys.argv[1]
order = ["default", "f32", "s2_mma", "s4_mma", "s2_f32", "s4_f32", "s8_f32", "s16_f32"]
for K in ["4096", "16384", "65536"]:
    runs = {}
    for f in sorted(os.listdir(f"{D}/{K}")):
        if f.startswith("run") and f.endswith(".json"):
            runs[f[:-5]] = json.load(open(f"{D}/{K}/{f}"))
    if not runs:
        continue
    any_r = next(iter(runs.values()))
    cols = [c for c in order if c in any_r["samples"]]
    cub = ["cuBLAS"]
    print(f"\n### 512 x 512 x {K}\n")
    print("| column | tile | splits | " + " | ".join(runs) + " |")
    print("| :--- | :--- | ---: | " + " | ".join("---:" for _ in runs) + " |")
    for c in cols + cub:
        info = any_r["columns"].get(c, {})
        cells = []
        for name, r in runs.items():
            t = min(r["samples"][c])  # us per call
            tc = min(r["samples"][cub[0]])
            ok = r["exact"].get(c, True) and r["exact_after_graph"].get(c, True)
            cells.append(f"{t:.2f} ({tc / t:.3f})" + ("" if ok else " NOT EXACT"))
        print(f"| {c} | {info.get('tile', '')} | {info.get('splits', '')} | " + " | ".join(cells) + " |")
    print("\nutilisation before / after (%, MHz, W): " + "; ".join(f"{n} {r['util_before']} / {r['util_after']}" for n, r in runs.items()))
