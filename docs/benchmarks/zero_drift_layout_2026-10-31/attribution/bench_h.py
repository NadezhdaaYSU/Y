"""Hand-over attribution: the `lj_closed` variants of variants.py, each followed by the compiled
`lj_convert`, against the CUDA baselines. Correctness first (each exact variant must store the exact
sums at 3 block sizes and leave its buffer zero), then the timing: minimum of REPS interleaved
rounds of INNER launches, after 1.5 s of clock warm-up.
Usage: bench_h.py <ljv.ptx> <baselines.ptx> <out json> [games script]
--stop <ljv.ptx> <kernel> <case>: one stop case in this process."""
import json, os, subprocess, sys, time
import numpy as np
import torch
sys.path.insert(0, os.environ["Y_PYTHON"])
from y_lang.torch_interop import TorchKernel
dev = torch.device("cuda"); torch.zeros(1, device=dev)
SPACING, JITTER, CUTOFF = 1.1, 0.05, 2.5


def load(path, name):
    k = TorchKernel.__new__(TorchKernel)
    k.source, k.kernel_name, k.target_sm, k.ptx = None, name, "sm_89", open(path).read()
    k._module = k._function = None; k._ensure_loaded(); return k


def system(m, seed):
    g = np.random.default_rng(seed)
    ix = np.stack(np.meshgrid(np.arange(m), np.arange(m), np.arange(m), indexing="ij"), -1).reshape(-1, 3)
    pos = (ix * SPACING + g.uniform(-JITTER, JITTER, ix.shape)).astype(np.float32)
    reach = int(np.ceil((CUTOFF + 2 * JITTER) / SPACING)); rng = range(-reach, reach + 1)
    flat = lambda v: (v[:, 0] * m + v[:, 1]) * m + v[:, 2]
    pi, pj = [], []
    for o in [(a, b, c) for a in rng for b in rng for c in rng if (a, b, c) > (0, 0, 0)]:
        nb = ix + np.array(o); ok = np.all((nb >= 0) & (nb < m), axis=1)
        i, j = flat(ix[ok]), flat(nb[ok]); d = pos[i].astype(np.float64) - pos[j]
        close = (d * d).sum(1) < CUTOFF * CUTOFF
        pi.append(i[close]); pj.append(j[close])
    return pos, np.concatenate(pi).astype(np.int32), np.concatenate(pj).astype(np.int32)


if sys.argv[1] == "--stop":
    path, name, case = sys.argv[2:5]
    pos, pi, pj = system(16, 16)
    if case == "index": pj[1000] = 16 ** 3          # one atom past the last
    if case == "nan": pos[3 * 5] = np.nan
    nt = 3 * 16 ** 3
    acc = torch.zeros(2 * nt + 7, dtype=torch.int64, device=dev)
    args = [torch.from_numpy(pos.reshape(-1)).to(dev), torch.from_numpy(pi).to(dev), torch.from_numpy(pj).to(dev), acc, len(pi), nt]
    try:
        load(path, name).launch(((len(pi) + 255) // 256, 1, 1), (256, 1, 1), args); torch.cuda.synchronize()
        print("RAN")
    except Exception as e:
        print("STOPPED", type(e).__name__)
    sys.exit(0)

V, BASE, OUT = sys.argv[1:4]
GAMES = sys.argv[4] if len(sys.argv) > 4 else None
names = ["lj_closed", "lj_closed_flags", "lj_closed_defer", "lj_closed_both", "lj_closed_red"]
K = {n: load(V, n) for n in names}
conv, terms_k = load(V, "lj_convert"), load(V, "lj_terms")
f32k, f64k, fixk, tof32 = (load(BASE, n) for n in ("lj_f32", "lj_f64", "lj_fixed", "fixed_to_f32"))
res = {"stops": {}}
for n in names[1:4]:
    for case in ("index", "nan", "good"):
        r = subprocess.run([sys.executable, __file__, "--stop", V, n, case], capture_output=True, text=True, env=os.environ)
        res["stops"][f"{n} {case}"] = ([l for l in r.stdout.splitlines() if l.startswith(("RAN", "STOPPED"))] or [r.stderr[-200:]])[0]


def state():
    out = subprocess.run(["nvidia-smi", "pmon", "-c", "3", "-s", "u"], capture_output=True, text=True).stdout
    rows = {}
    for line in out.splitlines():
        w = line.split()
        if len(w) >= 10 and not line.startswith("#") and w[3] != "-":
            rows.setdefault(f"{w[1]} {w[-1]}", []).append(int(w[3]))
    g = subprocess.run(["bash", GAMES], capture_output=True, text=True).stdout.strip() if GAMES else "not checked"
    return {"games": g, "sm_by_process": {k: max(v) for k, v in rows.items()}}


res["state_before"] = state()
SMS = torch.cuda.get_device_properties(0).multi_processor_count
for m in (32, 64):
    pos_h, pi_h, pj_h = system(m, m)
    atoms, npairs = m ** 3, len(pi_h); nt = 3 * atoms
    pos, pi, pj = (torch.from_numpy(x).to(dev) for x in (pos_h.reshape(-1), pi_h, pj_h))
    blocks = lambda b: ((npairs + b - 1) // b, 1, 1)
    terms = torch.empty(3 * npairs, device=dev)
    terms_k.launch(blocks(256), (256, 1, 1), [pos, pi, pj, terms, npairs]); torch.cuda.synchronize()
    y = terms.cpu().numpy().astype(np.float64).reshape(-1, 3) * 2.0 ** 32
    q = (np.sign(y) * np.floor(np.abs(y) + 0.5)).astype(np.int64)
    s = np.zeros((atoms, 3), np.int64); np.add.at(s, pi_h, q); np.add.at(s, pj_h, -q)
    want = (s.astype(np.float64) * 2.0 ** -32).astype(np.float32).reshape(-1).view(np.int32)
    acc = torch.zeros(2 * nt + 7, dtype=torch.int64, device=dev)
    out = torch.empty(nt, device=dev)
    row = {"pairs": npairs, "exact": {}}
    for n in names[:4]:
        ok = []
        for b in (64, 256, 1024):
            out.fill_(7.0)
            K[n].launch(blocks(b), (b, 1, 1), [pos, pi, pj, acc, npairs, nt])
            conv.launch((SMS * 8, 1, 1), (256, 1, 1), [acc, out, nt]); torch.cuda.synchronize()
            ok.append(bool(np.array_equal(out.cpu().numpy().view(np.int32), want)) and acc.abs().sum().item() == 0)
        row["exact"][n] = ok
    fx, f64, f32 = torch.zeros(nt, dtype=torch.int64, device=dev), torch.zeros(nt, dtype=torch.float64, device=dev), torch.zeros(nt, device=dev)
    cands = {n: (lambda n=n: (K[n].launch(blocks(256), (256, 1, 1), [pos, pi, pj, acc, npairs, nt]),
                              conv.launch((SMS * 8, 1, 1), (256, 1, 1), [acc, out, nt]))) for n in names[:4]}
    # Kernel alone, ACC zeroed first: the only way to run `red`, whose totals `drift_convert` refuses.
    acc2 = torch.zeros(2 * nt + 7, dtype=torch.int64, device=dev)
    for n in ("lj_closed", "lj_closed_both", "lj_closed_red"):
        cands[n + " (zero + kernel)"] = (lambda n=n: (acc2.zero_(), K[n].launch(blocks(256), (256, 1, 1), [pos, pi, pj, acc2, npairs, nt])))
    cands["fixed unchecked"] = lambda: (fx.zero_(), fixk.launch(blocks(256), (256, 1, 1), [pos, pi, pj, fx, npairs]),
                                        tof32.launch(((nt + 255) // 256, 1, 1), (256, 1, 1), [fx, out, nt]))
    cands["f64 atomics"] = lambda: (f64.zero_(), f64k.launch(blocks(256), (256, 1, 1), [pos, pi, pj, f64, npairs]))
    cands["f32 atomics"] = lambda: (f32.zero_(), f32k.launch(blocks(256), (256, 1, 1), [pos, pi, pj, f32, npairs]))
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
    row["us"] = {k: {"min": min(v), "median": float(np.median(v)), "x_fixed": min(v) / base} for k, v in times.items()}
    res[str(atoms)] = row
    print(f"atoms {atoms}, pairs {npairs}, exact at 64/256/1024: {row['exact']}")
    for k, v in row["us"].items():
        print(f"  {k:18s} min {v['min']:8.1f} us  median {v['median']:8.1f}  x{v['x_fixed']:.2f} of unchecked fixed point")
res["state_after"] = state()
print("stops:", res["stops"]); print("before:", res["state_before"]); print("after:", res["state_after"])
json.dump(res, open(OUT, "w"), indent=1)
