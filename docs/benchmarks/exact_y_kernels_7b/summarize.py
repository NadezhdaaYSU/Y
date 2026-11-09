"""Tables for this folder's README from what run.sh left in it.

    python3 summarize.py [folder]        (default: this script's folder)

Reads, without recomputing anything:
* `b7_<arm>.<HHMMSS>.log` (exact_throughput): the arm's tok/s/seq, its
  spread over the 5 repetitions, and whether its sanity and greedy == generate
  lines were printed;
* `vllm_fp8_graphs.<HHMMSS>.json`: vLLM FP8 with CUDA graphs;
* `gpu_sm89_y_kernels_compiled.json` against the 7B CPU record
  (`../exact_cross_device_7b_2026-11-04/cpu_zen5_eager.json`);
* `profile_all.json`, if present: the all-on step's kernels.
Runs are listed in the order they ran (their HHMMSS), so a drift across the
run shows. Prints Markdown.
"""
import glob
import json
import os
import re
import sys

D = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)))
ROW = re.compile(r"^exact\s.*?\s([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s*$")


def y_runs():
    runs = []
    for p in glob.glob(os.path.join(D, "b7_*.log")):
        m = re.match(r"b7_(.+)\.(\d{6})\.log$", os.path.basename(p))
        if not m:
            continue
        text = open(p, errors="replace").read()
        rows = [ROW.match(line) for line in text.splitlines()]
        rows = [r for r in rows if r]
        runs.append({
            "arm": m.group(1), "time": m.group(2), "kind": "y",
            "tok_s": float(rows[-1].group(2)) if rows else None,
            "seconds": float(rows[-1].group(1)) if rows else None,
            "spread": float(rows[-1].group(4)) if rows else None,
            "sanity": "canary ok" in text,
            "loop": "greedy == generate" in text,
            "error": next((ln.strip() for ln in text.splitlines()
                           if "Error" in ln or ln.startswith("***")), None),
        })
    return runs


def vllm_runs():
    runs = []
    for p in glob.glob(os.path.join(D, "vllm_fp8_graphs.*.json")):
        t = re.search(r"\.(\d{6})\.json$", p).group(1)
        d = json.load(open(p))
        runs.append({"arm": "vLLM FP8, CUDA graphs", "time": t, "kind": "vllm",
                     "tok_s": d.get("tok_s_per_seq"), "seconds": d.get("best_s"),
                     "spread": None, "sanity": d.get("alone_reproduces"), "loop": None,
                     "error": None})
    return runs


def record():
    g = os.path.join(D, "gpu_sm89_y_kernels_compiled.json")
    c = os.path.join(D, "..", "exact_cross_device_7b_2026-11-04", "cpu_zen5_eager.json")
    if not (os.path.exists(g) and os.path.exists(c)):
        return None
    ga, ca = json.load(open(g))["arms"]["exact"], json.load(open(c))["arms"]["exact"]
    differ = [k for k in ca["cases"] if ga["cases"].get(k) != ca["cases"][k]]
    return {"gpu": ga["digest"], "cpu": ca["digest"],
            "weights": (ga.get("weight_fingerprint"), ca.get("weight_fingerprint")),
            "differ": differ, "cases": len(ca["cases"]), "self_check": ga.get("self_check")}


def main():
    runs = sorted(y_runs() + vllm_runs(), key=lambda r: r["time"])
    if not runs:
        sys.exit(f"no b7_*.log or vllm_fp8_graphs.*.json in {D}")
    vllm = [r["tok_s"] for r in runs if r["kind"] == "vllm" and r["tok_s"]]
    ref = sum(vllm) / len(vllm) if vllm else None
    print(f"## Runs, in order ({os.path.basename(D)})\n")
    print("| time | process | tok/s/seq | spread | checks |")
    print("|---|---|---|---|---|")
    for r in runs:
        checks = []
        if r["kind"] == "y":
            checks += ["sanity" if r["sanity"] else "NO SANITY LINE",
                       "greedy == generate" if r["loop"] else "loop not checked"]
        elif r["sanity"] is not None:
            checks.append("alone reproduces" if r["sanity"] else "ALONE DIFFERS")
        if r["error"]:
            checks.append(f"ERROR: {r['error'][:80]}")
        tok = f"{r['tok_s']:.2f}" if r["tok_s"] else "-"
        spread = f"{r['spread']:.1f}%" if r["spread"] is not None else ""
        name = r["arm"] if r["kind"] == "vllm" else f"Y, {r['arm']}"
        print(f"| {r['time'][:2]}:{r['time'][2:4]}:{r['time'][4:]} | {name} | {tok} | {spread} | "
              f"{', '.join(checks)} |")
    arms = {}
    for r in runs:
        if r["kind"] == "y" and r["tok_s"]:
            arms.setdefault(r["arm"], []).append(r["tok_s"])
    if arms and ref:
        print(f"\n## Each arm against vLLM's mean ({ref:.2f} tok/s/seq over {len(vllm)} runs: "
              f"{', '.join(f'{v:.2f}' for v in vllm)})\n")
        print("| arm | runs | mean | / vLLM |")
        print("|---|---|---|---|")
        for arm, vals in arms.items():
            mean = sum(vals) / len(vals)
            print(f"| {arm} | {', '.join(f'{v:.2f}' for v in vals)} | {mean:.2f} | {mean / ref:.3f}x |")
    rec = record()
    if rec:
        print(f"\n## CPU == GPU record\n\nGPU digest `{rec['gpu']}`, CPU `{rec['cpu']}`; "
              f"weight fingerprints {rec['weights'][0]} / {rec['weights'][1]} "
              f"({'same' if rec['weights'][0] == rec['weights'][1] else 'DIFFER'}; records "
              f"before 60fe418 hashed merged names); "
              f"{len(rec['differ'])} of {rec['cases']} cases differ"
              + (f" ({', '.join(rec['differ'])})" if rec["differ"] else "")
              + f"; same card twice: {rec['self_check']}.")
    prof = os.path.join(D, "profile_all.json")
    if os.path.exists(prof):
        p = json.load(open(prof))
        busy = p["gpu_busy_ms"]
        print(f"\n## The all-on step ({p.get('model', '?')}, {p['new_tokens']} greedy tokens)\n")
        print(f"Wall {p['wall_ms']:.1f} ms, GPU busy {busy:.1f} ms; one decode step on an idle GPU: "
              f"CPU issue {min(p['decode_step_issue_ms']):.2f} ms, latency "
              f"{min(p['decode_step_latency_ms']):.2f} ms (minima).\n")
        print("| kernel | ms | share of busy | calls |")
        print("|---|---|---|---|")
        for k in p["kernels"][:15]:
            print(f"| `{k['name'][:70]}` | {k['ms']:.2f} | {100 * k['ms'] / busy:.1f}% | {k['calls']} |")


if __name__ == "__main__":
    main()
