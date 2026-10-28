"""E8: exact Lennard-Jones forces with `drift_add_at`, against f32, f64 and unchecked fixed-point atomics.

Atoms sit on a jittered cubic lattice (spacing 1.1 sigma); every pair closer than 2.5 sigma is one
thread. Correctness first, for every variant and size:

* `lj_terms` (the same Y arithmetic) stores each pair's three f32 terms; the host sums them exactly
  (each term rounded to Q32.32, ties away from zero, summed in integers, read back through f64).
  `lj_exact` must store those bits for every atom, for every block size and pair order (as built,
  reversed, shuffled), and leave its buffer zero.
* The CUDA baselines use Y's own instructions for the arithmetic (its fused multiply-adds and
  `div.approx.f32`): their `lj_terms_cuda` must store the same bits, so every variant sums the same
  terms. f32 and f64 atomics run the same configurations: how many different answers they give, and
  how far from S_k, the exact sum of component k's terms rounded once to f64 (`math.fsum`):
  max_abs = max_k |got_k - S_k|, rel_max = max_abs / max_k |S_k|, and the fraction of components
  whose f32 differs from f32(S_k).

Then each variant is timed with CUDA events, interleaved, REPS rounds of INNER launches back to back
(per-launch mean recorded); the minimum ranks them, the median is kept beside it. Each timed unit
produces f32 forces from nothing: the atomic baselines zero their buffer first (the Y kernel leaves
its own at zero), and the fixed-point one converts afterwards.

Usage: bench_forces.py <lj.ptx> <baselines.ptx> <out dir> [games script]
"""
import json, math, os, subprocess, sys, time
import numpy as np
import torch

LJ, BASE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
GAMES = sys.argv[4] if len(sys.argv) > 4 else None
sys.path.insert(0, os.environ.get("Y_PYTHON", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python")))
from y_lang.torch_interop import TorchKernel

dev = torch.device("cuda")
torch.zeros(1, device=dev)
REPS, INNER, BLOCK = 60, 10, 256
SPACING, JITTER, CUTOFF = 1.1, 0.05, 2.5
SIDES = [32, 64]


def load(path, name):
    k = TorchKernel.__new__(TorchKernel)
    k.source, k.kernel_name, k.target_sm, k.ptx = None, name, "sm_89", open(path).read()
    k._module = k._function = None
    k._ensure_loaded()
    return k


def util():
    return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm,clocks.mem", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


def who():
    """Per-process SM use over 5 samples (`nvidia-smi pmon`): which process loads the GPU."""
    out = subprocess.run(["nvidia-smi", "pmon", "-c", "5", "-s", "u"], capture_output=True, text=True).stdout
    rows = {}
    for line in out.splitlines():
        w = line.split()
        if len(w) >= 10 and not line.startswith("#") and w[3] != "-":
            rows.setdefault(f"{w[1]} {w[-1]}", []).append(int(w[3]))
    return {k: max(v) for k, v in sorted(rows.items())}


def games():
    if not GAMES:
        return "not checked"
    return subprocess.run(["bash", GAMES], capture_output=True, text=True).stdout.strip() or "none"


def lattice(m, seed):
    g = np.random.default_rng(seed)
    ix = np.stack(np.meshgrid(np.arange(m), np.arange(m), np.arange(m), indexing="ij"), -1).reshape(-1, 3)
    pos = ix * SPACING + g.uniform(-JITTER, JITTER, ix.shape)
    return ix, pos.astype(np.float32)


def pair_list(m, ix, pos):
    """Every pair i < j closer than the cutoff, by lattice offsets in one half-space."""
    reach = int(np.ceil((CUTOFF + 2 * JITTER) / SPACING))
    rng = range(-reach, reach + 1)
    offsets = [(a, b, c) for a in rng for b in rng for c in rng if (a, b, c) > (0, 0, 0)]
    flat = lambda v: (v[:, 0] * m + v[:, 1]) * m + v[:, 2]
    pi, pj = [], []
    for o in offsets:
        nb = ix + np.array(o)
        ok = np.all((nb >= 0) & (nb < m), axis=1)
        i, j = flat(ix[ok]), flat(nb[ok])
        d = pos[i].astype(np.float64) - pos[j]
        close = (d * d).sum(1) < CUTOFF * CUTOFF
        pi.append(i[close]); pj.append(j[close])
    return np.concatenate(pi).astype(np.int32), np.concatenate(pj).astype(np.int32)


def exact_forces(terms, pi, pj, atoms):
    """The f32 bits of each total: each term rounded to Q32.32 (ties away), summed in int64, read back."""
    y = terms.astype(np.float64).reshape(-1, 3) * 2.0 ** 32
    q = (np.sign(y) * np.floor(np.abs(y) + 0.5)).astype(np.int64)
    s = np.zeros((atoms, 3), np.int64)
    np.add.at(s, pi, q)
    np.add.at(s, pj, -q)
    assert np.abs(s).max() < 2 ** 52, "a total does not fit f64 exactly: the reference would round"
    return (s.astype(np.float64) * 2.0 ** -32).astype(np.float32).reshape(-1), s


lj_exact, lj_terms = load(LJ, "lj_exact"), load(LJ, "lj_terms")
lj_f32, lj_f64, lj_fixed, fixed_to_f32, lj_terms_cuda = (load(BASE, n) for n in ("lj_f32", "lj_f64", "lj_fixed", "fixed_to_f32", "lj_terms_cuda"))
res = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
       "nvcc": subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1],
       "driver": subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
       "block": BLOCK, "reps": REPS, "inner": INNER, "spacing": SPACING, "jitter": JITTER, "cutoff": CUTOFF,
       "games_before": games(), "util_before": util(), "sm_by_process_before": who(), "sizes": {}}
warm = torch.empty(1 << 26, device=dev)
t_warm = time.time()
while time.time() - t_warm < 1.5:
    warm.sum()
torch.cuda.synchronize()
del warm
res["clocks_after_warmup"] = util()

for m in SIDES:
    atoms = m ** 3
    nt = 3 * atoms
    ix, pos_h = lattice(m, seed=m)
    pi_h, pj_h = pair_list(m, ix, pos_h)
    npairs = len(pi_h)
    pos = torch.from_numpy(pos_h.reshape(-1)).to(dev)
    pi0, pj0 = torch.from_numpy(pi_h).to(dev), torch.from_numpy(pj_h).to(dev)
    blocks = lambda block: ((npairs + block - 1) // block, 1, 1)

    # The reference, from the same Y arithmetic.
    terms = torch.empty(3 * npairs, dtype=torch.float32, device=dev)
    lj_terms.launch(blocks(BLOCK), (BLOCK, 1, 1), [pos, pi0, pj0, terms, npairs])
    torch.cuda.synchronize()
    t_h = terms.cpu().numpy()
    want, raw = exact_forces(t_h, pi_h, pj_h, atoms)
    want_bits = want.view(np.int32)
    # The CUDA baselines' arithmetic gives the same terms, bit for bit.
    terms_cuda = torch.empty_like(terms)
    lj_terms_cuda.launch(blocks(BLOCK), (BLOCK, 1, 1), [pos, pi0, pj0, terms_cuda, npairs])
    torch.cuda.synchronize()
    terms_agree = bool(torch.equal(terms_cuda.view(torch.int32), terms.view(torch.int32)))
    assert terms_agree, "the CUDA baselines do not compute Y's terms"
    # S_k: each component's exact sum, rounded once to f64.
    comp = np.concatenate([(3 * pi_h[:, None] + np.arange(3)).reshape(-1), (3 * pj_h[:, None] + np.arange(3)).reshape(-1)])
    vals = np.concatenate([t_h.astype(np.float64), -t_h.astype(np.float64)])
    order = np.argsort(comp, kind="stable")
    cuts = np.searchsorted(comp[order], np.arange(1, nt))
    S = np.array([math.fsum(g) for g in np.split(vals[order], cuts)])
    S32 = S.astype(np.float32)
    y_vs_rounded = int((want_bits != S32.view(np.int32)).sum())

    acc = torch.zeros(2 * nt + 1, dtype=torch.int64, device=dev)
    out = torch.empty(nt, dtype=torch.float32, device=dev)
    f32 = torch.zeros(nt, dtype=torch.float32, device=dev)
    f64 = torch.zeros(nt, dtype=torch.float64, device=dev)
    fx = torch.zeros(nt, dtype=torch.int64, device=dev)
    fx_out = torch.empty(nt, dtype=torch.float32, device=dev)

    # Correctness: block sizes x pair orders.
    perm = np.random.default_rng(1).permutation(npairs)
    orders = {"built": (pi_h, pj_h), "reversed": (pi_h[::-1].copy(), pj_h[::-1].copy()), "shuffled": (pi_h[perm], pj_h[perm])}
    exact_ok, f32_seen, f64_seen, f64_as_f32_seen, fixed_seen = 0, set(), set(), set(), set()
    f32_max_abs, f32_differing, f64_not_rounded = 0.0, [], []
    configs = 0
    for oname, (a, b) in orders.items():
        pa, pb = torch.from_numpy(a).to(dev), torch.from_numpy(b).to(dev)
        for block in [64, 128, 256, 512, 1024]:
            configs += 1
            out.fill_(7.0)
            lj_exact.launch(blocks(block), (block, 1, 1), [pos, pa, pb, acc, out, npairs, nt])
            f32.zero_(); lj_f32.launch(blocks(block), (block, 1, 1), [pos, pa, pb, f32, npairs])
            f64.zero_(); lj_f64.launch(blocks(block), (block, 1, 1), [pos, pa, pb, f64, npairs])
            fx.zero_(); lj_fixed.launch(blocks(block), (block, 1, 1), [pos, pa, pb, fx, npairs])
            torch.cuda.synchronize()
            got = out.cpu().numpy().view(np.int32)
            exact_ok += bool(np.array_equal(got, want_bits)) and acc.abs().sum().item() == 0
            g32 = f32.cpu().numpy()
            f32_seen.add(hash(g32.tobytes()))
            f32_max_abs = max(f32_max_abs, float(np.abs(g32.astype(np.float64) - S).max()))
            f32_differing.append(float((g32.view(np.int32) != S32.view(np.int32)).mean()))
            g64 = f64.cpu().numpy()
            f64_seen.add(hash(g64.tobytes()))
            f64_as_f32_seen.add(hash(g64.astype(np.float32).tobytes()))
            f64_not_rounded.append(int((g64 != S).sum()))
            fixed_seen.add(hash(fx.cpu().numpy().tobytes()))
    assert exact_ok == configs, f"lj_exact matched the exact forces in {exact_ok} of {configs} configurations"
    fixed_equals_exact = bool(np.array_equal(fx.cpu().numpy(), raw.reshape(-1)))

    # Timing, interleaved.
    grid = blocks(BLOCK)
    def run_exact():
        lj_exact.launch(grid, (BLOCK, 1, 1), [pos, pi0, pj0, acc, out, npairs, nt])
    def run_tail():
        # No pairs: every block only takes its ticket, and the last converts 3N zero totals.
        lj_exact.launch(grid, (BLOCK, 1, 1), [pos, pi0, pj0, acc, out, 0, nt])
    def run_f32():
        f32.zero_(); lj_f32.launch(grid, (BLOCK, 1, 1), [pos, pi0, pj0, f32, npairs])
    def run_f32_kernel():
        lj_f32.launch(grid, (BLOCK, 1, 1), [pos, pi0, pj0, f32, npairs])
    def run_f64():
        f64.zero_(); lj_f64.launch(grid, (BLOCK, 1, 1), [pos, pi0, pj0, f64, npairs])
    def run_fixed():
        fx.zero_(); lj_fixed.launch(grid, (BLOCK, 1, 1), [pos, pi0, pj0, fx, npairs])
        fixed_to_f32.launch(((nt + 255) // 256, 1, 1), (256, 1, 1), [fx, fx_out, nt])
    variants = {"Y exact": run_exact, "Y finish only": run_tail, "f32 atomics": run_f32, "f32 atomics, kernel only": run_f32_kernel,
                "f64 atomics": run_f64, "fixed unchecked": run_fixed}
    times = {k: [] for k in variants}
    for _ in range(5):
        for fn in variants.values():
            fn()
    torch.cuda.synchronize()
    g_before = games()
    for r in range(REPS):
        for k, fn in variants.items():
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(INNER):
                fn()
            e1.record()
            torch.cuda.synchronize()
            times[k].append(e0.elapsed_time(e1) * 1000.0 / INNER)
    g_after = games()
    # every timed launch left exact forces and a zero buffer behind
    run_exact(); torch.cuda.synchronize()
    assert np.array_equal(out.cpu().numpy().view(np.int32), want_bits) and acc.abs().sum().item() == 0
    row = {"atoms": atoms, "pairs": npairs, "totals": nt, "terms_host_equals_gpu": terms_agree,
           "configs": configs, "exact_configs_matching": exact_ok,
           "y_components_differing_from_f32_of_S": y_vs_rounded, "max_abs_S": float(np.abs(S).max()),
           "f32_distinct_results": len(f32_seen), "f32_max_abs_err": f32_max_abs,
           "f32_rel_max": f32_max_abs / float(np.abs(S).max()),
           "f32_fraction_of_components_differing_from_f32_of_S_per_config": f32_differing,
           "f64_components_not_equal_to_S_per_config": f64_not_rounded,
           "f64_distinct_results": len(f64_seen), "f64_distinct_after_rounding_to_f32": len(f64_as_f32_seen),
           "fixed_distinct_results": len(fixed_seen), "fixed_raw_equals_exact_raw": fixed_equals_exact,
           "games_before_timing": g_before, "games_after_timing": g_after,
           "us": {k: {"min": min(v), "median": float(np.median(v)), "all": v} for k, v in times.items()}}
    print(f"{m}^3 = {atoms} atoms, {npairs} pairs; exact in {exact_ok}/{configs} configurations; CUDA terms == Y terms: {terms_agree}; Y != f32(S_k) in {y_vs_rounded} of {nt} components")
    print(f"   f32 atomics: {len(f32_seen)} distinct results; max_abs {f32_max_abs:.3g} (rel_max {f32_max_abs / np.abs(S).max():.3g}); components != f32(S_k): {min(f32_differing):.1%}-{max(f32_differing):.1%}")
    print(f"   f64 atomics: {len(f64_seen)} distinct ({len(f64_as_f32_seen)} after rounding to f32); components != S_k per configuration: max {max(f64_not_rounded)}; fixed: {len(fixed_seen)} distinct, raw == Y's raw sums: {fixed_equals_exact}")
    base = min(times["f32 atomics"])
    for k, v in times.items():
        print(f"   {k:>26}: min {min(v):9.1f} us  median {np.median(v):9.1f} us  ({min(v) / base:5.2f}x f32 atomics)", flush=True)
    res["sizes"][str(atoms)] = row
res["util_after"] = util()
res["sm_by_process_after"] = who()
res["games_after"] = games()
print("games before", res["games_before"], "after", res["games_after"])
print("util before", res["util_before"], "after", res["util_after"])
print("SM % by process (max of 5 samples), before", res["sm_by_process_before"], "after", res["sm_by_process_after"])
os.makedirs(OUT, exist_ok=True)
json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)
