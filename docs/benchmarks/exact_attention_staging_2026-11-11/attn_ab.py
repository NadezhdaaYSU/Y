"""Builds of exact_decode_attention (argv: .ysu files; the first is the
baseline) in one process: the same operands, output bits compared with the
first's, then timed L2-hot (one copy) and cold (a ring of AppendLayer-like
buffers cycled past 4x the L2), without and with a mask. CUDA graphs, 15
shuffled rounds, the minimum; ratios to the first.

`path.ysu:128` rewrites the build's `.maxnreg` to 128 (the register-cap
experiment). NOMASK=1 skips the masked runs; TLIST=32,100 picks the key
counts. Run from the repository with PYTHONPATH=python."""
import math
import random
import sys
import time

import torch

from y_lang.exact import attention as A
from y_lang.exact import kv
from y_lang.torch_interop import TorchKernel

dev = torch.device("cuda")
L2 = torch.cuda.get_device_properties(0).L2_cache_size
REPS = 16
TSV = 1024


def make(B, NKV, rep, D, T, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    NH = NKV * rep
    qi = torch.randint(-127, 128, (B, NH, D), generator=g, device=dev).float()
    a = torch.rand(B, NH, generator=g, device=dev, dtype=torch.float64) * 1e-3
    ki = torch.randint(-1023, 1024, (B, NKV, T, D), generator=g, device=dev).float()
    sk = torch.rand(B, NKV, T, 1, generator=g, device=dev) * 0.01 + 1e-3
    vi = torch.randint(-127, 128, (B, NKV, T, D), generator=g, device=dev).to(torch.int8)
    sv = torch.rand(B, NKV, T, 1, generator=g, device=dev) * 0.01 + 1e-3
    kp, vp = kv.pack_k(ki, sk), kv.pack_v(vi, sv)
    kb = torch.zeros(B, NKV, TSV, kp.shape[-1], dtype=torch.int8, device=dev)
    vb = torch.zeros(B, NKV, TSV, vp.shape[-1], dtype=torch.int8, device=dev)
    kb[:, :, :T] = kp
    vb[:, :, :T] = vp
    blocked = torch.zeros(B, T, dtype=torch.int32, device=dev)
    for i in range(B):
        blocked[i, :(i * 7) % max(T // 2, 1)] = 1
    return qi, a, kb, vb, blocked


def run(kern, qi, a, kb, vb, blocked, T, rep, table):
    A._Y_ATTN[torch.cuda.current_device()] = kern
    return A._y_exact_decode_attention(qi, a, kb[:, :, :T], vb[:, :, :T], blocked, table, rep)


def bench(kerns, B, NKV, rep, D, T):
    table = A._table(dev)
    qi, a, kb, vb, blocked = make(B, NKV, rep, D, T, T + D)
    outs = {}
    for name, kern in kerns.items():
        for m in (None, blocked):
            outs[(name, m is None)] = run(kern, qi, a, kb, vb, m, T, rep, table)
    names = list(kerns)
    for nomask in (True, False):
        for nm in names[1:]:
            x, y = outs[(names[0], nomask)], outs[(nm, nomask)]
            assert torch.equal(x.view(torch.int32), y.view(torch.int32)), (nm, T, D, nomask)
    rows_bytes = B * NKV * T * (2 * (D + kv.K_PAD) + D + kv.V_PAD)
    ring = max(2, math.ceil(4 * L2 / rows_bytes))
    cold = [(kb.clone(), vb.clone()) for _ in range(ring)]
    graphs = {}
    for name, kern in kerns.items():
        for temp, bufs in (("hot", [(kb, vb)]), ("cold", cold)):
            for mname, m in MASKS(blocked):
                n = max(REPS, len(bufs))
                for k_, v_ in bufs[:2]:
                    run(kern, qi, a, k_, v_, m, T, rep, table)
                torch.cuda.synchronize()
                gph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gph):
                    for i in range(n):
                        k_, v_ = bufs[i % len(bufs)]
                        run(kern, qi, a, k_, v_, m, T, rep, table)
                graphs[(name, temp + mname)] = (gph, n)
    for gph, _ in graphs.values():
        gph.replay()
    torch.cuda.synchronize()
    times = {k: [] for k in graphs}
    rng = random.Random(T * 1000 + D)
    for r in range(15):
        order = list(graphs)
        rng.shuffle(order)
        for k in order:
            gph, n = graphs[k]
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(5):
                gph.replay()
            e1.record()
            e1.synchronize()
            times[k].append(e0.elapsed_time(e1) * 1000 / (5 * n))
    return {k: (min(t), sorted(t)[len(t) // 2]) for k, t in times.items()}


def warm():
    x = torch.randn(4096, 4096, device=dev)
    t0 = time.time()
    while time.time() - t0 < 2.0:
        x @ x
    torch.cuda.synchronize()


import os
MASKS = (lambda b: (("", None), ("+mask", b))) if not os.environ.get("NOMASK") else (lambda b: (("", None),))
TS_LIST = tuple(int(x) for x in os.environ.get("TLIST", "32,100,256,300").split(","))

if __name__ == "__main__":
    major, minor = torch.cuda.get_device_capability()
    kerns = {}
    import re
    import y_lang.torch_interop as ti
    orig = ti.compile_to_ptx
    for arg in sys.argv[1:]:
        path, _, cap = arg.partition(":")
        label = os.path.basename(path).replace(".ysu", "") + (f"@{cap}" if cap else "")
        if cap:
            ti.compile_to_ptx = lambda *a, **k: re.sub(r"(?m)^\.maxnreg \d+", f".maxnreg {cap}", orig(*a, **k))
        kerns[label] = TorchKernel(open(path).read(), kernel_name="exact_decode_attention",
                                   target_sm=f"sm_{major}{minor}")
        ti.compile_to_ptx = orig
        if cap:
            assert f".maxnreg {cap}" in kerns[label].ptx
    warm()
    shapes = {"0.5B": (32, 2, 7, 64), "7B": (32, 4, 7, 128)}
    for label, (B, NKV, rep, D) in shapes.items():
        for T in TS_LIST:
            res = bench(kerns, B, NKV, rep, D, T)
            names = list(kerns)
            conds = sorted({c for _, c in res}, key=lambda c: ("cold" in c, "mask" in c))
            for cond in conds:
                o = res[(names[0], cond)]
                parts = [f"{names[0]} {o[0]:7.2f}"]
                for nm in names[1:]:
                    n = res[(nm, cond)]
                    parts.append(f"{nm} {n[0]:7.2f} ({n[0] / o[0]:.3f})")
                print(f"{label:4s} T={T:3d} {cond:9s} " + "  ".join(parts), flush=True)
