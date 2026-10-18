#!/usr/bin/env python3
"""check_all.py <bench_ab dir> <committed ptx> <dir> ...: every kernel against the
committed gemm_f16_512, bit for bit on random normal f16 inputs, 3 trials."""
import sys, re, torch
sys.path.insert(0, sys.argv[1])
from bench_ab import kernel_from_ptx
torch.zeros(1, device="cuda")
n = 512
def load(p):
    k = kernel_from_ptx(p, "gemm_f16_512")
    cm, cn, wm, wn = (int(x) for x in re.search(r"CTA (\d+)x(\d+)x\d+ \| (\d+)x(\d+) warps", k.ptx).groups())
    def run(a, b):
        o = torch.full((n, n), float("nan"), device="cuda")
        k.launch((-(-n // cn), -(-n // cm), 1), (wm * wn * 32, 1, 1), [a, b, o]); torch.cuda.synchronize()
        return o
    return run
ref = load(sys.argv[2])
runs = {d.rstrip("/").split("/")[-1]: load(d + "/g512.ptx") for d in sys.argv[3:]}
g = torch.Generator(device="cuda").manual_seed(29)
bad = 0
for trial in range(3):
    a = torch.randn(n, n, device="cuda", generator=g).half(); b = torch.randn(n, n, device="cuda", generator=g).half()
    r = ref(a, b)
    assert not r.isnan().any()
    for name, f in runs.items():
        o = f(a, b)
        d = int((o.view(torch.int32) != r.view(torch.int32)).sum())
        if d:
            bad += 1; print(name, "trial", trial, d, "elements differ")
print("checked", len(runs), "kernels x 3 trials; mismatching runs:", bad)
