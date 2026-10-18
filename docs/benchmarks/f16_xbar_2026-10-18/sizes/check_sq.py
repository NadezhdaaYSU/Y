#!/usr/bin/env python3
"""check_sq.py <bench_ab dir> <off dir> <on dir> <N> ...: the switched kernel
against the committed one, bit for bit on random normal f16 inputs, and both
exact against float64 on small integers; NaN-filled outputs."""
import sys, re
import torch
sys.path.insert(0, sys.argv[1])
from bench_ab import kernel_from_ptx
torch.zeros(1, device="cuda")
bad = 0
for n in map(int, sys.argv[4:]):
    ks = {m: kernel_from_ptx(f"{d}/g{n}.ptx", f"gemm_f16_{n}") for m, d in (("off", sys.argv[2]), ("on", sys.argv[3]))}
    cm, cn, wm, wn = (int(x) for x in re.search(r"CTA (\d+)x(\d+)x\d+ \| (\d+)x(\d+) warps", ks["off"].ptx).groups())
    def run(k, a, b):
        o = torch.full((n, n), float("nan"), device="cuda")
        k.launch((-(-n // cn), -(-n // cm), 1), (wm * wn * 32, 1, 1), [a, b, o]); torch.cuda.synchronize()
        return o
    g = torch.Generator(device="cuda").manual_seed(n)
    for trial in range(3):
        a = torch.randn(n, n, device="cuda", generator=g).half(); b = torch.randn(n, n, device="cuda", generator=g).half()
        x, y = run(ks["off"], a, b), run(ks["on"], a, b)
        same = (not x.isnan().any()) and torch.equal(x.view(torch.int32), y.view(torch.int32))
        a = torch.randint(-4, 5, (n, n), device="cuda", generator=g).half(); b = torch.randint(-4, 5, (n, n), device="cuda", generator=g).half()
        ref = (a.double() @ b.double()).float()
        ex = [torch.equal(run(ks[m], a, b), ref) for m in ("off", "on")]
        print(f"N={n} trial {trial}: random bitwise equal {same}; integer exact off {ex[0]} on {ex[1]}")
        bad += (not same) + (not ex[0]) + (not ex[1])
print("FAILURES", bad)
