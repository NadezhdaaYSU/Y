"""E8 stress test: denser lattices, until f64 atomics give more than one answer.

The kernels and the configurations of `bench_forces.py` (5 block sizes x 3 pair orders), on
32,768 atoms, with the lattice spacing shrinking from 1.1 to 0.7 sigma: atoms come closer, the
largest forces grow by orders of magnitude, and one component's sum mixes them with terms of
1e-6 and below. A stress test, not a typical simulation state. For each spacing:
- `lj_exact_wide` (bounds +-2e9) must store the exact sums of the same f32 terms in every
  configuration, or stop;
- f64 and f32 atomics: how many different results, and how many components differ from S_k
  (the exact sum, rounded once to f64 by `math.fsum`).
Each spacing runs in a child process: a trap leaves the device unusable for its process.

Usage: stress_forces.py <lj.ptx> <baselines.ptx> <out dir>
"""
import json, math, os, subprocess, sys
import numpy as np

LJ, BASE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
SPACINGS = [1.1, 1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
M, JITTER, CUTOFF = 32, 0.05, 2.5


def one(spacing):
    import torch
    sys.path.insert(0, os.environ.get("Y_PYTHON", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python")))
    from y_lang.torch_interop import TorchKernel
    dev = torch.device("cuda")
    torch.zeros(1, device=dev)

    def load(path, name):
        k = TorchKernel.__new__(TorchKernel)
        k.source, k.kernel_name, k.target_sm, k.ptx = None, name, "sm_89", open(path).read()
        k._module = k._function = None
        k._ensure_loaded()
        return k

    lj, terms_k = load(LJ, "lj_exact_wide"), load(LJ, "lj_terms")
    f32k, f64k = load(BASE, "lj_f32"), load(BASE, "lj_f64")
    g = np.random.default_rng(M)
    ix = np.stack(np.meshgrid(np.arange(M), np.arange(M), np.arange(M), indexing="ij"), -1).reshape(-1, 3)
    pos_h = (ix * spacing + g.uniform(-JITTER, JITTER, ix.shape)).astype(np.float32)
    reach = int(np.ceil((CUTOFF + 2 * JITTER) / spacing))
    rng = range(-reach, reach + 1)
    flat = lambda v: (v[:, 0] * M + v[:, 1]) * M + v[:, 2]
    pi, pj = [], []
    for o in [(a, b, c) for a in rng for b in rng for c in rng if (a, b, c) > (0, 0, 0)]:
        nb = ix + np.array(o)
        ok = np.all((nb >= 0) & (nb < M), axis=1)
        i, j = flat(ix[ok]), flat(nb[ok])
        d = pos_h[i].astype(np.float64) - pos_h[j]
        close = (d * d).sum(1) < CUTOFF * CUTOFF
        pi.append(i[close]); pj.append(j[close])
    pi_h, pj_h = np.concatenate(pi).astype(np.int32), np.concatenate(pj).astype(np.int32)
    npairs, atoms = len(pi_h), M ** 3
    nt = 3 * atoms
    d = pos_h[pi_h].astype(np.float64) - pos_h[pj_h]
    r_min = float(np.sqrt((d * d).sum(1)).min())
    pos = torch.from_numpy(pos_h.reshape(-1)).to(dev)
    blocks = lambda block: ((npairs + block - 1) // block, 1, 1)
    terms = torch.empty(3 * npairs, dtype=torch.float32, device=dev)
    terms_k.launch(blocks(256), (256, 1, 1), [pos, torch.from_numpy(pi_h).to(dev), torch.from_numpy(pj_h).to(dev), terms, npairs])
    torch.cuda.synchronize()
    t_h = terms.cpu().numpy()
    y = t_h.astype(np.float64).reshape(-1, 3) * 2.0 ** 32
    q = (np.sign(y) * np.floor(np.abs(y) + 0.5)).astype(np.int64)
    s = np.zeros((atoms, 3), np.int64)
    np.add.at(s, pi_h, q)
    np.add.at(s, pj_h, -q)
    assert np.abs(s).max() < 2 ** 52
    want = (s.astype(np.float64) * 2.0 ** -32).astype(np.float32).reshape(-1).view(np.int32)
    comp = np.concatenate([(3 * pi_h[:, None] + np.arange(3)).reshape(-1), (3 * pj_h[:, None] + np.arange(3)).reshape(-1)])
    vals = np.concatenate([t_h.astype(np.float64), -t_h.astype(np.float64)])
    order = np.argsort(comp, kind="stable")
    S = np.array([math.fsum(x) for x in np.split(vals[order], np.searchsorted(comp[order], np.arange(1, nt)))])
    acc = torch.zeros(2 * nt + 1, dtype=torch.int64, device=dev)
    out = torch.empty(nt, dtype=torch.float32, device=dev)
    f32 = torch.zeros(nt, dtype=torch.float32, device=dev)
    f64 = torch.zeros(nt, dtype=torch.float64, device=dev)
    perm = np.random.default_rng(1).permutation(npairs)
    orders = [(pi_h, pj_h), (pi_h[::-1].copy(), pj_h[::-1].copy()), (pi_h[perm], pj_h[perm])]
    y_ok, seen32, seen64, seen64_32, off64, off32 = 0, set(), set(), set(), [], []
    configs = 0
    for a, b in orders:
        pa, pb = torch.from_numpy(a).to(dev), torch.from_numpy(b).to(dev)
        for block in [64, 128, 256, 512, 1024]:
            configs += 1
            out.fill_(7.0)
            lj.launch(blocks(block), (block, 1, 1), [pos, pa, pb, acc, out, npairs, nt])
            f32.zero_(); f32k.launch(blocks(block), (block, 1, 1), [pos, pa, pb, f32, npairs])
            f64.zero_(); f64k.launch(blocks(block), (block, 1, 1), [pos, pa, pb, f64, npairs])
            torch.cuda.synchronize()
            y_ok += bool(np.array_equal(out.cpu().numpy().view(np.int32), want)) and acc.abs().sum().item() == 0
            g32, g64 = f32.cpu().numpy(), f64.cpu().numpy()
            seen32.add(hash(g32.tobytes())); seen64.add(hash(g64.tobytes())); seen64_32.add(hash(g64.astype(np.float32).tobytes()))
            off64.append(int((g64 != S).sum()))
            off32.append(int((g32 != S.astype(np.float32)).sum()))
    return {"spacing": spacing, "pairs": npairs, "r_min": r_min, "max_abs_S": float(np.abs(S).max()),
            "max_abs_term": float(np.abs(t_h).max()), "configs": configs, "y_exact_configs": y_ok,
            "y_differs_from_f32_of_S": int((want != S.astype(np.float32).view(np.int32)).sum()),
            "f64_distinct": len(seen64), "f64_distinct_after_rounding_to_f32": len(seen64_32),
            "f64_components_not_S_max": max(off64), "f32_distinct": len(seen32),
            "f32_components_not_f32_of_S_max": max(off32), "components": nt}


if len(sys.argv) > 4 and sys.argv[4] == "--one":
    print("ROW " + json.dumps(one(float(sys.argv[5]))))
    sys.exit(0)

rows = []
for sp in SPACINGS:
    r = subprocess.run([sys.executable, os.path.abspath(__file__), LJ, BASE, OUT, "--one", str(sp)], capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("ROW ")]
    row = json.loads(line[0][4:]) if line else {"spacing": sp, "stopped": (r.stderr.strip().splitlines() or ["?"])[-1]}
    rows.append(row)
    if "stopped" in row:
        print(f"spacing {sp}: Y STOPPED: {row['stopped']}", flush=True)
        continue
    print(f"spacing {sp}: {row['pairs']} pairs, r_min {row['r_min']:.3f}, max|term| {row['max_abs_term']:.3g}; "
          f"Y exact {row['y_exact_configs']}/{row['configs']} (!= f32(S) in {row['y_differs_from_f32_of_S']}); "
          f"f64: {row['f64_distinct']} distinct ({row['f64_distinct_after_rounding_to_f32']} as f32), up to {row['f64_components_not_S_max']} components != S; "
          f"f32: {row['f32_distinct']} distinct", flush=True)
os.makedirs(OUT, exist_ok=True)
json.dump({"atoms": M ** 3, "jitter": JITTER, "cutoff": CUTOFF, "rows": rows}, open(f"{OUT}/stress.json", "w"), indent=1)
