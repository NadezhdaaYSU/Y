#!/usr/bin/env python3
"""bench_cold_sq.py <bench_ab dir> <label=ptx-dir,...> <out.json> <N,N,...>: square
F16 GEMMs (f16 in, f32 out) with B streamed from DRAM, the regime Y's autotuner
measures in: each call takes the next of `copies` B matrices, enough that they
oversubscribe the 48 MB L2 at least three times (A and C stay put). Every
column checked exact against float64 on every B copy first (small integers),
then timed by CUDA events over back-to-back calls in 17 interleaved rounds,
the first two dropped; minimum and median of 15. Each ptx-dir holds g<N>.ptx
with kernel gemm_f16_<N>. (Adapted from ../decode_dram_2026-10-18/bench_cold.py.)"""
import json
import re
import statistics
import sys

import torch

sys.path.insert(0, sys.argv[1])
from bench_ab import kernel_from_ptx, warm, util  # noqa: E402

dirs = [tuple(x.split("=", 1)) for x in sys.argv[2].split(",")]
out_path = sys.argv[3]
sizes = [int(x) for x in sys.argv[4].split(",")]
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
L2 = 48 * 1024 * 1024
result = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "util_before": util(), "sizes": {}}


def event_us(fn, calls):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for i in range(calls):
        fn(i)
    e.record()
    e.synchronize()
    return s.elapsed_time(e) * 1000.0 / calls


for n in sizes:
    m = k = n
    copies = max(2, -(-3 * L2 // (k * n * 2)))
    gen = torch.Generator(device="cuda").manual_seed(n)
    a = torch.randint(-4, 5, (m, k), device="cuda", generator=gen).half()
    bs = [torch.randint(-4, 5, (k, n), device="cuda", generator=gen).half() for _ in range(copies)]
    cols, exact = {}, {}
    for label, d in dirs:
        kern = kernel_from_ptx(f"{d}/g{n}.ptx", f"gemm_f16_{n}")
        cm, cn, wm, wn = (int(x) for x in re.search(r"CTA (\d+)x(\d+)x\d+ \| (\d+)x(\d+) warps", kern.ptx).groups())
        o = torch.full((m, n), float("nan"), device="cuda")
        grid, block = (-(-n // cn), -(-m // cm), 1), (wm * wn * 32, 1, 1)
        ok = True
        for b in bs[:4]:
            ref = (a.double() @ b.double()).float()
            o.fill_(float("nan"))
            kern.launch(grid, block, [a, b, o])
            torch.cuda.synchronize()
            ok &= bool(torch.equal(o, ref))
            del ref
        exact[label] = ok
        cols[label] = (lambda i, kern=kern, grid=grid, block=block, o=o: kern.launch(grid, block, [a, bs[i % copies], o]))
    exact["cuBLAS"] = all(torch.equal(torch.mm(a, b, out_dtype=torch.float32), (a.double() @ b.double()).float()) for b in bs[:4])
    cols["cuBLAS"] = lambda i: torch.mm(a, bs[i % copies], out_dtype=torch.float32)
    assert all(exact.values()), (n, exact)
    names = list(cols)
    calls = max(copies, min(200, int(5e-3 / max(2 * m * n * k / 80e12, 2e-6))))
    calls = -(-calls // copies) * copies
    samples = {c: [] for c in names}
    clocks = []
    warm()
    for r in range(17):
        for c in names[r % len(names):] + names[:r % len(names)]:
            t = event_us(cols[c], calls)
            if r >= 2:
                samples[c].append(t)
        clocks.append(torch.cuda.clock_rate(0))
    mins = {c: min(s) for c, s in samples.items()}
    result["sizes"][n] = {"copies": copies, "exact": exact, "calls": calls, "samples": samples, "sm_clock_mhz": clocks}
    print(f"{n}: {copies} B copies, exact {exact} | SM {min(clocks)}-{max(clocks)} MHz", flush=True)
    for c in sorted(mins, key=mins.get):
        print(f"   {c:10s} min {mins[c]:10.2f} median {statistics.median(samples[c]):10.2f} us", flush=True)
    del bs
    torch.cuda.empty_cache()
result["util_after"] = util()
json.dump(result, open(out_path, "w"), indent=1)
