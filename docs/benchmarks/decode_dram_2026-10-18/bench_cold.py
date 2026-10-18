#!/usr/bin/env python3
"""bench_cold.py <bench_ab dir> <label=ptx-dir,...> <out.json> [copies] [M,M,...]: the decode
GEMMs (M x 4096 x 4096, f16 in, f32 out) with B read from DRAM: each call takes
the next of `copies` B matrices (4 x 33.5 MB = 134 MB, about 2.8x the 48 MB L2),
and, as a control, the same loop on one B (L2-resident). Every column is
checked exact against float64 on every B copy first (small integers), then
timed by CUDA events over back-to-back calls in 17 interleaved rounds, the
first two dropped; minimum and median of 15. Each directory holds
g<M>x4096x4096.ptx with kernel gemm_f16_<M>x4096x4096."""
import json, re, statistics, subprocess, sys
import torch
sys.path.insert(0, sys.argv[1])
from bench_ab import kernel_from_ptx, warm, util

dirs = [tuple(x.split("=", 1)) for x in sys.argv[2].split(",")]
out_path = sys.argv[3]
copies = int(sys.argv[4]) if len(sys.argv) > 4 else 4
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
n = k = 4096
result = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "copies": copies, "util_before": util(), "shapes": {}}
gen = torch.Generator(device="cuda").manual_seed(7)
bs = [torch.randint(-4, 5, (k, n), device="cuda", generator=gen).half() for _ in range(copies)]

def event_us(fn, calls):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for i in range(calls):
        fn(i)
    e.record(); e.synchronize()
    return s.elapsed_time(e) * 1000.0 / calls

for m in (tuple(int(x) for x in sys.argv[5].split(",")) if len(sys.argv) > 5 else (1, 4, 8, 32)):
    size = f"{m}x{n}x{k}"
    a = torch.randint(-4, 5, (m, k), device="cuda", generator=gen).half()
    refs = [(a.double() @ b.double()).float() for b in bs]
    cols, exact = {}, {}
    for label, d in dirs:
        kern = kernel_from_ptx(f"{d}/g{size}.ptx", f"gemm_f16_{size}")
        cm, cn, wm, wn = (int(x) for x in re.search(r"CTA (\d+)x(\d+)x\d+ \| (\d+)x(\d+) warps", kern.ptx).groups())
        o = torch.full((m, n), float("nan"), device="cuda")
        grid, block = (-(-n // cn), -(-m // cm), 1), (wm * wn * 32, 1, 1)
        ok = True
        for b, ref in zip(bs, refs):
            o.fill_(float("nan")); kern.launch(grid, block, [a, b, o]); torch.cuda.synchronize()
            ok &= bool(torch.equal(o, ref))
        exact[label] = ok
        for mode, nb in (("cold", copies), ("warm", 1)):
            cols[f"{label} {mode}"] = (lambda i, kern=kern, grid=grid, block=block, o=o, nb=nb: kern.launch(grid, block, [a, bs[i % nb], o]))
    ok = all(torch.equal(torch.mm(a, b, out_dtype=torch.float32), ref) for b, ref in zip(bs, refs))
    exact["cuBLAS"] = ok
    for mode, nb in (("cold", copies), ("warm", 1)):
        cols[f"cuBLAS {mode}"] = (lambda i, nb=nb: torch.mm(a, bs[i % nb], out_dtype=torch.float32))
    assert all(exact.values()), (size, exact)
    names = list(cols)
    calls = 40 * copies
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
    result["shapes"][size] = {"exact": exact, "calls": calls, "samples": samples, "sm_clock_mhz": clocks}
    print(f"{size}: exact {exact} | SM {min(clocks)}-{max(clocks)} MHz", flush=True)
    for c in sorted(mins, key=mins.get):
        gbs = (k * n * 2 + m * k * 2 + m * n * 4) / (mins[c] * 1e-6) / 1e9
        print(f"   {c:16s} min {mins[c]:8.3f} median {statistics.median(samples[c]):8.3f} us  {gbs:7.1f} GB/s of A, B and C", flush=True)
result["util_after"] = util()
json.dump(result, open(out_path, "w"), indent=1)
