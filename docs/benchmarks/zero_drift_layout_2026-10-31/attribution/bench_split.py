"""The split ACC layout against the compiled one, kernel alone (ACC zeroed first), and the
unchecked fixed-point and f64 atomics the same way. The split kernel's totals are read back on the
host (low word i, high word n + i) and must equal the exact sums at 3 block sizes.
Usage: bench_split.py <ljs.ptx> <baselines.ptx> <out json> [games script]"""
import json, os, subprocess, sys, time
import numpy as np, torch
sys.argv = [sys.argv[0], "--lib"] + sys.argv[1:]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_h.py")).read().split("if sys.argv[1] == \"--stop\":")[0])
V, BASE, OUT = sys.argv[2:5]
GAMES = sys.argv[5] if len(sys.argv) > 5 else None
closed, split = load(V, "lj_closed"), load(V, "lj_closed_split")
terms_k = load(V, "lj_terms")
fixk, f64k = load(BASE, "lj_fixed"), load(BASE, "lj_f64")
res = {}
def pmon():
    out = subprocess.run(["nvidia-smi", "pmon", "-c", "3", "-s", "u"], capture_output=True, text=True).stdout
    rows = {}
    for line in out.splitlines():
        w = line.split()
        if len(w) >= 10 and not line.startswith("#") and w[3] != "-":
            rows.setdefault(f"{w[1]} {w[-1]}", []).append(int(w[3]))
    return {k: max(v) for k, v in rows.items()}
res["before"] = pmon()
for m in (32, 64):
    pos_h, pi_h, pj_h = system(m, m)
    atoms, npairs = m ** 3, len(pi_h); nt = 3 * atoms
    pos, pi, pj = (torch.from_numpy(x).to(dev) for x in (pos_h.reshape(-1), pi_h, pj_h))
    blocks = lambda b: ((npairs + b - 1) // b, 1, 1)
    terms = torch.empty(3 * npairs, device=dev)
    terms_k.launch(blocks(256), (256, 1, 1), [pos, pi, pj, terms, npairs]); torch.cuda.synchronize()
    y = terms.cpu().numpy().astype(np.float64).reshape(-1, 3) * 2.0 ** 32
    q = (np.sign(y) * np.floor(np.abs(y) + 0.5)).astype(np.int64)
    s = np.zeros((atoms, 3), np.int64); np.add.at(s, pi_h, q); np.add.at(s, pj_h, -q); s = s.reshape(-1)
    acc = torch.zeros(2 * nt + 7, dtype=torch.int64, device=dev)
    ok = []
    for b in (64, 256, 1024):
        acc.zero_(); split.launch(blocks(b), (b, 1, 1), [pos, pi, pj, acc, npairs, nt]); torch.cuda.synchronize()
        a = acc.cpu().numpy()
        lo, hi = a[:nt], a[nt:2 * nt]
        ok.append(bool(np.array_equal(lo, s) and np.array_equal(hi, np.where(s < 0, -1, 0))))
    fx, f64 = torch.zeros(nt, dtype=torch.int64, device=dev), torch.zeros(nt, dtype=torch.float64, device=dev)
    cands = {"Y compiled (16-byte totals)": lambda: (acc.zero_(), closed.launch(blocks(256), (256, 1, 1), [pos, pi, pj, acc, npairs, nt])),
             "Y split (8-byte low words)": lambda: (acc.zero_(), split.launch(blocks(256), (256, 1, 1), [pos, pi, pj, acc, npairs, nt])),
             "fixed unchecked": lambda: (fx.zero_(), fixk.launch(blocks(256), (256, 1, 1), [pos, pi, pj, fx, npairs])),
             "f64 atomics": lambda: (f64.zero_(), f64k.launch(blocks(256), (256, 1, 1), [pos, pi, pj, f64, npairs]))}
    t_end = time.time() + 1.5
    while time.time() < t_end:
        for f in cands.values(): f()
    torch.cuda.synchronize()
    times = {k: [] for k in cands}
    for _ in range(40):
        for k, f in cands.items():
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(10): f()
            e1.record(); torch.cuda.synchronize()
            times[k].append(e0.elapsed_time(e1) * 100.0)
    base = min(times["fixed unchecked"])
    res[str(atoms)] = {"split_exact_64_256_1024": ok, "us": {k: {"min": min(v), "median": float(np.median(v)), "x_fixed": min(v) / base} for k, v in times.items()}}
    print(f"atoms {atoms}: split exact at 64/256/1024: {ok}")
    for k, v in res[str(atoms)]["us"].items():
        print(f"  {k:28s} min {v['min']:8.1f} us  median {v['median']:8.1f}  x{v['x_fixed']:.2f} of unchecked")
res["after"] = pmon()
res["games"] = subprocess.run(["bash", GAMES], capture_output=True, text=True).stdout.strip() if GAMES else "not checked"
print("load before", res["before"], "after", res["after"], "games", repr(res["games"]))
json.dump(res, open(OUT, "w"), indent=1)
