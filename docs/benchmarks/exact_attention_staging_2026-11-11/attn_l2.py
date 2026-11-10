"""Y's exact_decode_attention alone: the cache L2-resident (one copy, read
again and again) against DRAM-resident (a ring of copies cycled past 4x the
L2), contiguous (the cat cache) and as views of AppendLayer-like buffers
(TS rows a head). CUDA graphs of 16 launches, shuffled rounds, minimum.
Prints us a launch and checks the bits equal across layouts."""
import math
import random
import sys
import time

import torch

from y_lang.exact import attention as A
from y_lang.exact import kv

dev = torch.device("cuda")
L2 = torch.cuda.get_device_properties(0).L2_cache_size
REPS = 16


def make(B, NKV, rep, D, T, TS, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    NH = NKV * rep
    qi = torch.randint(-127, 128, (B, NH, D), generator=g, device=dev).float()
    a = torch.rand(B, NH, generator=g, device=dev, dtype=torch.float64) * 1e-3
    ki = torch.randint(-1023, 1024, (B, NKV, TS, D), generator=g, device=dev).float()
    sk = torch.rand(B, NKV, TS, 1, generator=g, device=dev) * 0.01 + 1e-3
    vi = torch.randint(-127, 128, (B, NKV, TS, D), generator=g, device=dev).to(torch.int8)
    sv = torch.rand(B, NKV, TS, 1, generator=g, device=dev) * 0.01 + 1e-3
    kp, vp = kv.pack_k(ki, sk), kv.pack_v(vi, sv)
    return qi, a, kp, vp


def bench(B, NKV, rep, D, T):
    table = A._table(dev)
    TSV = 1024                                   # AppendLayer-like spare rows
    rows_bytes = B * NKV * T * (2 * (D + kv.K_PAD) + D + kv.V_PAD)
    ring = max(2, math.ceil(4 * L2 / rows_bytes))
    variants = {}
    qi, a, kp, vp = make(B, NKV, rep, D, T, T, 0)
    # The same rows, as views into buffers of TSV rows a head.
    kb = torch.zeros(B, NKV, TSV, kp.shape[-1], dtype=torch.int8, device=dev)
    vb = torch.zeros(B, NKV, TSV, vp.shape[-1], dtype=torch.int8, device=dev)
    kb[:, :, :T] = kp
    vb[:, :, :T] = vp
    kview, vview = kb[:, :, :T], vb[:, :, :T]
    ref = A._y_exact_decode_attention(qi, a, kp, vp, None, table, rep)
    got = A._y_exact_decode_attention(qi, a, kview, vview, None, table, rep)
    assert torch.equal(ref.view(torch.int32), got.view(torch.int32))
    variants["hot contiguous"] = [(kp, vp)]
    variants["hot view"] = [(kview, vview)]
    cold_c, cold_v = [], []
    for i in range(ring):
        cold_c.append((kp.clone(), vp.clone()))
        kb2, vb2 = kb.clone(), vb.clone()
        cold_v.append((kb2[:, :, :T], vb2[:, :, :T]))
    variants["cold contiguous"] = cold_c
    variants["cold view"] = cold_v
    graphs = {}
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for name, bufs in variants.items():
            for k_, v_ in bufs[:2]:
                A._y_exact_decode_attention(qi, a, k_, v_, None, table, rep)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    for name, bufs in variants.items():
        gph = torch.cuda.CUDAGraph()
        n_launch = max(REPS, len(bufs))
        with torch.cuda.graph(gph):
            for i in range(n_launch):
                k_, v_ = bufs[i % len(bufs)]
                A._y_exact_decode_attention(qi, a, k_, v_, None, table, rep)
        graphs[name] = (gph, n_launch)
    for gph, _ in graphs.values():
        gph.replay()
    torch.cuda.synchronize()
    times = {n: [] for n in graphs}
    rng = random.Random(T)
    for r in range(15):
        order = list(graphs)
        rng.shuffle(order)
        for n in order:
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            gph, n_launch = graphs[n]
            for _ in range(5):
                gph.replay()
            e1.record()
            e1.synchronize()
            times[n].append(e0.elapsed_time(e1) * 1000 / (5 * n_launch))
    out = {n: (min(t), sorted(t)[len(t) // 2]) for n, t in times.items()}
    return out, rows_bytes, ring


def warm():
    x = torch.randn(4096, 4096, device=dev)
    t0 = time.time()
    while time.time() - t0 < 2.0:
        x @ x
    torch.cuda.synchronize()


if __name__ == "__main__":
    warm()
    shapes = {"0.5B": (32, 2, 7, 64), "7B": (32, 4, 7, 128)}
    for label, (B, NKV, rep, D) in shapes.items():
        for T in (32, 100, 256):
            out, nbytes, ring = bench(B, NKV, rep, D, T)
            line = "  ".join(f"{n}: {m:7.2f} ({md:7.2f})" for n, (m, md) in out.items())
            dram_us = nbytes / 672e9 * 1e6
            print(f"{label} T={T:3d} cache {nbytes / 1e6:5.2f} MB (DRAM floor {dram_us:5.2f} us, "
                  f"ring {ring}): {line}", flush=True)
