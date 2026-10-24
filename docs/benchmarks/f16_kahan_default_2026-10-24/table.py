"""table.py <run dir>: minimum us per column from gemm_1.json and gemm_2.json (run.sh), as
"run 1 / run 2", with kahan's time over f32's and cuBLAS's time over kahan's (above 1, the
compensated default is faster than cuBLAS), and the SM clock range of each run."""
import json
import pathlib
import sys

d = pathlib.Path(sys.argv[1])
runs = [json.loads((d / f"gemm_{i}.json").read_text()) for i in (1, 2)]
cols = ["f32", "kahan", "mma", "cuBLAS"]
print("| M x N x K | tile | " + " | ".join(cols) + " | kahan / f32 | cuBLAS / kahan |")
print("| :--- | :--- | " + " | ".join("---:" for _ in cols) + " | ---: | ---: |")
for size in runs[0]["sizes"]:
    mins = [{c: min(r["sizes"][size]["samples"][c]) for c in cols} for r in runs]
    assert all(all(r["sizes"][size]["exact"].values()) for r in runs), size
    cells = [" / ".join(f"{m[c]:.1f}" for m in mins) for c in cols]
    kf = " / ".join(f"{m['kahan'] / m['f32']:.3f}" for m in mins)
    ck = " / ".join(f"{m['cuBLAS'] / m['kahan']:.3f}" for m in mins)
    print(f"| {size} | {runs[0]['sizes'][size]['tiles']['kahan']} | " + " | ".join(cells) + f" | {kf} | {ck} |")
for i, r in enumerate(runs, 1):
    clk = [x for s in r["sizes"].values() for v in s["sm_clock_mhz"].values() for x in v]
    print(f"\nrun {i}: SM {min(clk)}-{max(clk)} MHz, utilisation before / after {r['util_before']} / {r['util_after']}")
