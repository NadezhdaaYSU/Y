#!/usr/bin/env python3
"""probe.py <examples dir>: the head-to-head's elementwise and RMSNorm kernels
(Y, Triton, PyTorch) at sizes where bandwidth, not launch latency, decides.
Every column checked first; CUDA events over back-to-back calls, 15
interleaved rounds after 2 discarded, minimum and median; SM clock per round."""
import statistics, sys
import torch, triton
sys.path.insert(0, sys.argv[1])
import run_head_to_head_comparison as H

target = "sm_%d%d" % torch.cuda.get_device_capability()
gen = torch.Generator(device="cuda").manual_seed(7)


def bench(name, cols, outs, ref, tol, nbytes):
    for f in cols.values():
        f()
    torch.cuda.synchronize()
    scale = ref.abs().max().item()
    ok = {c: bool(torch.isfinite(outs[c]()).all()) and ((outs[c]().double() - ref).abs().max().item() <= tol * scale if tol else torch.equal(outs[c]().double(), ref)) for c in cols}
    H.warm_clocks()
    one = min(H.event_time_us(lambda f=f: [f() for _ in range(10)], 10) for f in cols.values() for _ in range(3))
    calls = max(3, int(3000 / max(one, 1)))
    samples = {c: [] for c in cols}
    clocks = []
    H.warm_clocks()
    names = list(cols)
    for r in range(17):
        for c in names[r % len(names):] + names[:r % len(names)]:
            f = cols[c]
            t = H.event_time_us(lambda f=f: [f() for _ in range(calls)], calls)
            if r >= 2:
                samples[c].append(t)
        clocks.append(torch.cuda.clock_rate(0))
    print(f"== {name}  exact/within tol {ok}  calls {calls}  SM {min(clocks)}-{max(clocks)} MHz", flush=True)
    best = min(min(v) for v in samples.values())
    for c in sorted(samples, key=lambda c: min(samples[c])):
        m = min(samples[c])
        print(f"   {c:8s} min {m:9.2f} median {statistics.median(samples[c]):9.2f} us  {nbytes / m / 1e3:7.1f} GB/s  {m / best:.3f}x the fastest", flush=True)


for n in (1 << 22, 1 << 24):
    a, b = torch.randn(n, device="cuda", generator=gen), torch.randn(n, device="cuda", generator=gen)
    k = H.y_kernel(H.Y_VEC_ADD, "vec_add", target)
    c = {x: torch.full((n,), float("nan"), device="cuda") for x in ("Y", "Triton", "PyTorch")}
    g = ((n // 4 + 127) // 128, 1, 1)
    bench(f"vector add {n}", {"Y": lambda: k.launch(grid=g, block=(128, 1, 1), args=[a, b, c["Y"], n]),
                              "Triton": lambda: H.triton_vector_add[(triton.cdiv(n, 1024),)](a, b, c["Triton"], n, BLOCK=1024),
                              "PyTorch": lambda: torch.add(a, b, out=c["PyTorch"])},
          {x: (lambda x=x: c[x]) for x in c}, (a.double() + b.double()).float().double(), 0.0, 12 * n)
    ks = H.y_kernel(H.Y_SWIGLU, "swiglu", target)
    open("y_vec_add.ptx", "w").write(k.ptx); open("y_swiglu.ptx", "w").write(ks.ptx)
    o = {x: torch.full((n,), float("nan"), device="cuda") for x in ("Y", "Triton", "PyTorch")}
    bench(f"SwiGLU {n}", {"Y": lambda: ks.launch(grid=g, block=(128, 1, 1), args=[a, b, o["Y"], n]),
                          "Triton": lambda: H.triton_swiglu[(triton.cdiv(n, 1024),)](a, b, o["Triton"], n, BLOCK=1024),
                          "PyTorch": lambda: torch.mul(torch.nn.functional.silu(a), b, out=o["PyTorch"])},
          {x: (lambda x=x: o[x]) for x in o}, a.double() * torch.sigmoid(a.double()) * b.double(), 1e-5, 12 * n)

for rows, hidden in ((4096, 1024), (4096, 4096), (16384, 4096)):
    x = torch.randn(rows, hidden, device="cuda", generator=gen)
    w = torch.randn(hidden, device="cuda", generator=gen)
    kr = H.y_kernel(H.Y_RMSNORM, "rmsnorm", target)
    open("y_rmsnorm.ptx", "w").write(kr.ptx)
    r = {c: torch.full_like(x, float("nan")) for c in ("Y", "Triton", "PyTorch")}
    eps = 1e-5
    xd = x.double()
    bench(f"RMSNorm {rows} x {hidden}",
          {"Y": lambda: kr.launch(grid=(rows, 1, 1), block=(hidden // 4, 1, 1), args=[x, w, r["Y"], hidden, eps]),
           "Triton": lambda: H.triton_rmsnorm[(rows,)](x, w, r["Triton"], eps, N=hidden),
           "PyTorch": lambda: r["PyTorch"].copy_(x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w)},
          {c: (lambda c=c: r[c]) for c in r}, xd * torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + eps) * w.double(), 1e-5,
          8 * rows * hidden + 4 * hidden)
