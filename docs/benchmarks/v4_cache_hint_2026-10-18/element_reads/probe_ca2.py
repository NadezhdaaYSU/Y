#!/usr/bin/env python3
"""probe_ca2.py <examples dir> <bench_ab dir>: probe_ca.py at chosen footprints (decimal MB, every byte read or written once): element-read kernels compiled from
Y source (one element a thread), their loads as emitted (`ld.global.ca`) and
with the cache operator removed, against Triton. Every column checked first;
CUDA events, 17 interleaved rounds, 2 dropped, minimum and median of 15."""
import statistics, sys
import torch, triton
import triton.language as tl
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
import run_head_to_head_comparison as H
from bench_ab import kernel_from_ptx

ADD = """
kernel add1(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, N: I32) {
    let i: I32 = global_thread_id();
    if i < N {
        C[i] = A[i] + B[i];
    }
}
"""
FMA = """
kernel fma1(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, D: GlobalMemory<F32>, N: I32) {
    let i: I32 = global_thread_id();
    if i < N {
        D[i] = A[i] * B[i] + C[i];
    }
}
"""


@triton.jit
def t_fma(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(d_ptr + offs, tl.load(a_ptr + offs, mask=m) * tl.load(b_ptr + offs, mask=m) + tl.load(c_ptr + offs, mask=m), mask=m)


torch.zeros(1, device="cuda")
gen = torch.Generator(device="cuda").manual_seed(5)
kernels = {}
for src, name in ((ADD, "add1"), (FMA, "fma1")):
    k = H.y_kernel(src, name, "sm_89")
    open(f"{name}_ca.ptx", "w").write(k.ptx)
    assert k.ptx.count("ld.global.ca.") >= 2, k.ptx
    open(f"{name}_plain.ptx", "w").write(k.ptx.replace("ld.global.ca.", "ld.global."))
    kernels[name] = {v: kernel_from_ptx(f"{name}_{v}.ptx", name) for v in ("ca", "plain")}


def bench(name, cols, outs, ref, nbytes):
    for f in cols.values():
        f()
    torch.cuda.synchronize()
    for c in cols:
        o = outs[c]().double()
        assert torch.isfinite(o).all() and (o - ref).abs().max().item() <= 1e-6 * ref.abs().max().item(), (name, c)
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
    print(f"== {name}  all checked  calls {calls}  SM {min(clocks)}-{max(clocks)} MHz", flush=True)
    best = min(min(v) for v in samples.values())
    for c in sorted(samples, key=lambda c: min(samples[c])):
        m = min(samples[c])
        print(f"   {c:9s} min {m:9.2f} median {statistics.median(samples[c]):9.2f} us  {nbytes / m / 1e3:7.1f} GB/s  {m / best:.3f}x the fastest", flush=True)


def n_for(mb, per):
    return max(256, round(mb * 1e6 / per / 256) * 256)


for mb in (24, 32, 40, 96, 192):
    n = n_for(mb, 12)
    a, b, c = (torch.randn(n, device="cuda", generator=gen) for _ in range(3))
    g = ((n + 255) // 256, 1, 1)
    o = {x: torch.full((n,), float("nan"), device="cuda") for x in ("ca", "plain", "Triton")}
    cols = {"Y " + v: (lambda v=v: kernels["add1"][v].launch(g, (256, 1, 1), [a, b, o[v], n])) for v in ("ca", "plain")}
    cols["Triton"] = lambda: H.triton_vector_add[(triton.cdiv(n, 1024),)](a, b, o["Triton"], n, BLOCK=1024)
    bench(f"add1 {n} ({12 * n / 1e6:.0f} MB)", cols, {"Y ca": lambda: o["ca"], "Y plain": lambda: o["plain"], "Triton": lambda: o["Triton"]},
          (a.double() + b.double()).float().double(), 12 * n)
    n = n_for(mb, 16)
    a, b, c = (torch.randn(n, device="cuda", generator=gen) for _ in range(3))
    g = ((n + 255) // 256, 1, 1)
    o = {x: torch.full((n,), float("nan"), device="cuda") for x in ("ca", "plain", "Triton")}
    cols = {"Y " + v: (lambda v=v: kernels["fma1"][v].launch(g, (256, 1, 1), [a, b, c, o[v], n])) for v in ("ca", "plain")}
    cols["Triton"] = lambda: t_fma[(triton.cdiv(n, 1024),)](a, b, c, o["Triton"], n, BLOCK=1024)
    bench(f"fma1 {n} ({16 * n / 1e6:.0f} MB)", cols, {"Y ca": lambda: o["ca"], "Y plain": lambda: o["plain"], "Triton": lambda: o["Triton"]},
          a.double() * b.double() + c.double(), 16 * n)
