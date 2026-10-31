"""`drift_add_at`'s totals split in ACC (low words, then high words), before and after (2026-10-31).

Compiles `../zero_drift_forces_2026-10-28/lj.ysu` with the compiler before the change and the one
after it. For each compiler, `lj_exact` (`drift_finish`) and `lj_closed` + `lj_convert` must store
the exact sums of the same f32 terms (each term rounded to Q32.32, ties away from zero, summed in
integers, read back through f64) for 3 block sizes x 3 pair orders, at 32,768 and 262,144 atoms, and
leave ACC zero. Then everything is timed with the E8 baselines (`baselines.ptx` there): minimum of 60
interleaved rounds of 10 launches after 1.5 s of clock warm-up; each timed unit makes f32 forces
from nothing. The load per process and any game are recorded before and after.

    python3 bench.py <old Y> <new Y> <.ysu_hw_profile> <out dir> [games script]
"""
import json, os, shutil, subprocess, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FORCES = os.path.join(HERE, "..", "zero_drift_forces_2026-10-28")
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python"))
from y_lang.torch_interop import TorchKernel

OLD, NEW, PROFILE, OUT = sys.argv[1:5]
GAMES = sys.argv[5] if len(sys.argv) > 5 else None
dev = torch.device("cuda")
torch.zeros(1, device=dev)
os.makedirs(OUT, exist_ok=True)
SPACING, JITTER, CUTOFF, BLOCK = 1.1, 0.05, 2.5, 256


def build(tag, ybin):
    d = os.path.join(OUT, f"build_{tag}")
    os.makedirs(d, exist_ok=True)
    shutil.copy(PROFILE, os.path.join(d, ".ysu_hw_profile"))
    shutil.copy(os.path.join(FORCES, "lj.ysu"), d)
    subprocess.run([ybin, "lj.ysu", "--emit-ptx", "-o", "lj.ptx"], cwd=d, check=True, capture_output=True)
    shutil.copy(os.path.join(d, "lj.ptx"), os.path.join(OUT, f"lj_{tag}.ptx"))
    shutil.rmtree(d)
    return os.path.join(OUT, f"lj_{tag}.ptx")


def load(path, name):
    k = TorchKernel.__new__(TorchKernel)
    k.source, k.kernel_name, k.target_sm, k.ptx = None, name, "sm_89", open(path).read()
    k._module = k._function = None
    k._ensure_loaded()
    return k


def state():
    out = subprocess.run(["nvidia-smi", "pmon", "-c", "5", "-s", "u"], capture_output=True, text=True).stdout
    rows = {}
    for line in out.splitlines():
        w = line.split()
        if len(w) >= 10 and not line.startswith("#") and w[3] != "-":
            rows.setdefault(f"{w[1]} {w[-1]}", []).append(int(w[3]))
    smi = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm,clocks.mem", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    games = subprocess.run(["bash", GAMES], capture_output=True, text=True).stdout.strip() if GAMES else "not checked"
    return {"nvidia_smi": smi, "sm_by_process": {k: max(v) for k, v in sorted(rows.items())}, "games": games}


def system(m):
    g = np.random.default_rng(m)
    ix = np.stack(np.meshgrid(np.arange(m), np.arange(m), np.arange(m), indexing="ij"), -1).reshape(-1, 3)
    pos = (ix * SPACING + g.uniform(-JITTER, JITTER, ix.shape)).astype(np.float32)
    reach = int(np.ceil((CUTOFF + 2 * JITTER) / SPACING))
    rng = range(-reach, reach + 1)
    flat = lambda v: (v[:, 0] * m + v[:, 1]) * m + v[:, 2]
    pi, pj = [], []
    for o in [(a, b, c) for a in rng for b in rng for c in rng if (a, b, c) > (0, 0, 0)]:
        nb = ix + np.array(o)
        ok = np.all((nb >= 0) & (nb < m), axis=1)
        i, j = flat(ix[ok]), flat(nb[ok])
        d = pos[i].astype(np.float64) - pos[j]
        close = (d * d).sum(1) < CUTOFF * CUTOFF
        pi.append(i[close]); pj.append(j[close])
    return pos, np.concatenate(pi).astype(np.int32), np.concatenate(pj).astype(np.int32)


P = {"old": build("old", OLD), "new": build("new", NEW)}
K = {t: {n: load(P[t], n) for n in ("lj_exact", "lj_closed", "lj_convert")} for t in P}
terms_k = load(P["new"], "lj_terms")
BASE = os.path.join(FORCES, "baselines.ptx")
f32k, f64k, fixk, tof32 = (load(BASE, n) for n in ("lj_f32", "lj_f64", "lj_fixed", "fixed_to_f32"))
SMS = torch.cuda.get_device_properties(0).multi_processor_count
res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "state_before": state(), "sizes": {}}
warm = torch.empty(1 << 26, device=dev)
t_end = time.time() + 1.5
while time.time() < t_end:
    warm.sum()
torch.cuda.synchronize()
del warm
for m in (32, 64):
    pos_h, pi_h, pj_h = system(m)
    atoms, npairs = m ** 3, len(pi_h)
    nt = 3 * atoms
    pos = torch.from_numpy(pos_h.reshape(-1)).to(dev)
    blocks = lambda b: ((npairs + b - 1) // b, 1, 1)
    pi0, pj0 = torch.from_numpy(pi_h).to(dev), torch.from_numpy(pj_h).to(dev)
    terms = torch.empty(3 * npairs, device=dev)
    terms_k.launch(blocks(BLOCK), (BLOCK, 1, 1), [pos, pi0, pj0, terms, npairs])
    torch.cuda.synchronize()
    y = terms.cpu().numpy().astype(np.float64).reshape(-1, 3) * 2.0 ** 32
    q = (np.sign(y) * np.floor(np.abs(y) + 0.5)).astype(np.int64)
    s = np.zeros((atoms, 3), np.int64)
    np.add.at(s, pi_h, q); np.add.at(s, pj_h, -q)
    want = (s.astype(np.float64) * 2.0 ** -32).astype(np.float32).reshape(-1).view(np.int32)
    perm = np.random.default_rng(1).permutation(npairs)
    orders = [(pi_h, pj_h), (pi_h[::-1].copy(), pj_h[::-1].copy()), (pi_h[perm], pj_h[perm])]
    acc1 = torch.zeros(2 * nt + 1, dtype=torch.int64, device=dev)
    acc7 = torch.zeros(2 * nt + 7, dtype=torch.int64, device=dev)
    out = torch.empty(nt, device=dev)
    exact = {}
    for t in P:
        for form in ("finish", "close + convert"):
            ok = 0
            for a, b in orders:
                pa, pb = torch.from_numpy(a).to(dev), torch.from_numpy(b).to(dev)
                for blk in (64, 256, 1024):
                    out.fill_(7.0)
                    if form == "finish":
                        K[t]["lj_exact"].launch(blocks(blk), (blk, 1, 1), [pos, pa, pb, acc1, out, npairs, nt])
                    else:
                        K[t]["lj_closed"].launch(blocks(blk), (blk, 1, 1), [pos, pa, pb, acc7, npairs, nt])
                        K[t]["lj_convert"].launch((SMS * 8, 1, 1), (256, 1, 1), [acc7, out, nt])
                    torch.cuda.synchronize()
                    ok += bool(np.array_equal(out.cpu().numpy().view(np.int32), want)) and acc1.abs().sum().item() == 0 \
                        and acc7.abs().sum().item() == 0
            exact[f"{t} {form}"] = f"{ok}/9"
    fx = torch.zeros(nt, dtype=torch.int64, device=dev)
    f64 = torch.zeros(nt, dtype=torch.float64, device=dev)
    f32 = torch.zeros(nt, device=dev)
    cands = {}
    for t in P:
        cands[f"Y finish, {t}"] = lambda t=t: K[t]["lj_exact"].launch(blocks(BLOCK), (BLOCK, 1, 1), [pos, pi0, pj0, acc1, out, npairs, nt])
        cands[f"Y close + convert, {t}"] = lambda t=t: (K[t]["lj_closed"].launch(blocks(BLOCK), (BLOCK, 1, 1), [pos, pi0, pj0, acc7, npairs, nt]),
                                                         K[t]["lj_convert"].launch((SMS * 8, 1, 1), (256, 1, 1), [acc7, out, nt]))
    cands["fixed unchecked"] = lambda: (fx.zero_(), fixk.launch(blocks(BLOCK), (BLOCK, 1, 1), [pos, pi0, pj0, fx, npairs]),
                                        tof32.launch(((nt + 255) // 256, 1, 1), (256, 1, 1), [fx, out, nt]))
    cands["f64 atomics"] = lambda: (f64.zero_(), f64k.launch(blocks(BLOCK), (BLOCK, 1, 1), [pos, pi0, pj0, f64, npairs]))
    cands["f32 atomics"] = lambda: (f32.zero_(), f32k.launch(blocks(BLOCK), (BLOCK, 1, 1), [pos, pi0, pj0, f32, npairs]))
    for f in cands.values():
        f()
    torch.cuda.synchronize()
    times = {k: [] for k in cands}
    for _ in range(60):
        for k, f in cands.items():
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(10):
                f()
            e1.record()
            torch.cuda.synchronize()
            times[k].append(e0.elapsed_time(e1) * 100.0)
    # After all the timed launches the Y kernels' buffers are zero, and one more launch of each is exact.
    assert acc1.abs().sum().item() == 0 and acc7.abs().sum().item() == 0
    for t in P:
        for name in (f"Y finish, {t}", f"Y close + convert, {t}"):
            out.fill_(7.0)
            cands[name]()
            torch.cuda.synchronize()
            assert np.array_equal(out.cpu().numpy().view(np.int32), want), name
    ref = {k: min(times[k]) for k in ("fixed unchecked", "f64 atomics", "f32 atomics")}
    res["sizes"][str(atoms)] = {"pairs": npairs, "exact": exact,
                                "us": {k: {"min": min(v), "median": float(np.median(v)), "x_f32": min(v) / ref["f32 atomics"],
                                           "x_f64": min(v) / ref["f64 atomics"], "x_fixed": min(v) / ref["fixed unchecked"], "all": v}
                                       for k, v in times.items()}}
    print(f"{atoms} atoms, {npairs} pairs; exact (3 block sizes x 3 orders): {exact}", flush=True)
    for k, v in res["sizes"][str(atoms)]["us"].items():
        print(f"  {k:26s} min {v['min']:8.1f} us  median {v['median']:8.1f}  x{v['x_f32']:.2f} f32  x{v['x_f64']:.2f} f64  "
              f"x{v['x_fixed']:.2f} unchecked", flush=True)
res["state_after"] = state()
print("before:", res["state_before"]); print("after:", res["state_after"])
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
