#!/usr/bin/env python3
"""probe_cs.py <examples dir> <bench_ab dir>: Y's v4 built-ins with their
`.cs` (evict-first) hints, without them, and with `.ca` loads, against Triton,
in L2-resident and DRAM-bound sizes. Every column checked first."""
import statistics, sys
import torch, triton
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
import run_head_to_head_comparison as H
from bench_ab import kernel_from_ptx

gen = torch.Generator(device="cuda").manual_seed(11)
VARIANTS = ("cs", "plain", "ca")


def ptx(k, v):
    return "y_%s.ptx" % k if v == "cs" else "y_%s_%s.ptx" % (k, v)


def bench(name, cols, outs, ref, tol, nbytes):
    for f in cols.values():
        f()
    torch.cuda.synchronize()
    scale = ref.abs().max().item()
    for c in cols:
        o = outs[c]().double()
        ok = bool(torch.isfinite(o).all()) and ((o - ref).abs().max().item() <= tol * scale if tol else torch.equal(o, ref))
        assert ok, (name, c)
    H.warm_clocks()
    one = min(H.event_time_us(lambda f=f: [f() for _ in range(10)], 10) for f in cols.values() for _ in range(3))
    calls = max(3, int(3000 / max(one, 1)))
    samples = {c: [] for c in cols}
    clocks = []
    names = list(cols)
    for r in range(17):
        for c in names[r % len(names):] + names[:r % len(names)]:
            f = cols[c]
            t = H.event_time_us(lambda f=f: [f() for _ in range(calls)], calls)
            if r >= 2:
                samples[c].append(t)
        clocks.append(torch.cuda.clock_rate(0))
    print(f"== {name}  all exact/within tol  calls {calls}  SM {min(clocks)}-{max(clocks)} MHz", flush=True)
    best = min(min(v) for v in samples.values())
    for c in sorted(samples, key=lambda c: min(samples[c])):
        m = min(samples[c])
        print(f"   {c:9s} min {m:9.2f} median {statistics.median(samples[c]):9.2f} us  {nbytes / m / 1e3:7.1f} GB/s  {m / best:.3f}x the fastest", flush=True)


for n in (1 << 20, 1 << 22, 1 << 24):
    a, b = torch.randn(n, device="cuda", generator=gen), torch.randn(n, device="cuda", generator=gen)
    g = ((n // 4 + 127) // 128, 1, 1)
    for kname, tri, refv, tol in (("vec_add", H.triton_vector_add, (a.double() + b.double()).float().double(), 0.0),
                                  ("swiglu", H.triton_swiglu, a.double() * torch.sigmoid(a.double()) * b.double(), 1e-5)):
        o = {x: torch.full((n,), float("nan"), device="cuda") for x in VARIANTS + ("Triton",)}
        cols = {}
        for v in VARIANTS:
            k = kernel_from_ptx(ptx(kname, v), kname)
            cols["Y " + v] = (lambda k=k, v=v: k.launch(g, (128, 1, 1), [a, b, o[v], n]))
        cols["Triton"] = lambda tri=tri: tri[(triton.cdiv(n, 1024),)](a, b, o["Triton"], n, BLOCK=1024)
        outs = {"Y " + v: (lambda v=v: o[v]) for v in VARIANTS}
        outs["Triton"] = lambda: o["Triton"]
        bench(f"{kname} {n}", cols, outs, refv, tol, 12 * n)

for rows, hidden in ((4096, 1024), (16384, 1024), (4096, 4096), (16384, 4096)):
    x = torch.randn(rows, hidden, device="cuda", generator=gen)
    w = torch.randn(hidden, device="cuda", generator=gen)
    eps = 1e-5
    o = {c: torch.full_like(x, float("nan")) for c in VARIANTS + ("Triton",)}
    cols = {}
    for v in VARIANTS:
        k = kernel_from_ptx(ptx("rmsnorm", v), "rmsnorm")
        cols["Y " + v] = (lambda k=k, v=v: k.launch((rows, 1, 1), (hidden // 4, 1, 1), [x, w, o[v], hidden, eps]))
    cols["Triton"] = lambda: H.triton_rmsnorm[(rows,)](x, w, o["Triton"], eps, N=hidden)
    outs = {"Y " + v: (lambda v=v: o[v]) for v in VARIANTS}
    outs["Triton"] = lambda: o["Triton"]
    xd = x.double()
    bench(f"RMSNorm {rows} x {hidden}", cols, outs, xd * torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + eps) * w.double(), 1e-5,
          8 * rows * hidden + 4 * hidden)
