#!/usr/bin/env python3
"""Profiling driver for the split-K paged decode attention vs FlashInfer.

  prof_attn.py <ptx> <splits> <batch> <ctx> [--fi] [--iters N] [--check]

Runs the Y split kernel (+ its _reduce entry) from <ptx>, and with --fi also
FlashInfer's decode, on the README's shape (hd 128, 32 q heads, 8 kv heads,
page 16, shuffled page table). Meant to be run under ncu; --check compares
against the f32 reference first.
"""
import argparse, os, sys
sys.path.insert(0, os.environ.get("Y_TESTS", ""))
import torch, cupy as cp
import benchmark_y_paged_decode_attention as B

ap = argparse.ArgumentParser()
ap.add_argument("ptx"); ap.add_argument("splits", type=int)
ap.add_argument("batch", type=int); ap.add_argument("ctx", type=int)
ap.add_argument("--fi", action="store_true"); ap.add_argument("--iters", type=int, default=5)
ap.add_argument("--check", action="store_true"); ap.add_argument("--warps", type=int, default=8)
a = ap.parse_args()

HD, NQH, NKVH, PS = 128, 32, 8, 16
name = "paged_decode_attention_split_%d_%d_%d_%d_%d_%d" % (HD, NQH, NKVH, PS, a.splits, a.warps)
lens = [a.ctx] * a.batch
num_pages = max(512, a.batch * ((a.ctx + PS - 1) // PS) * 2)
q, kc, vc, pt, sl, mp = B.make_inputs(HD, NQH, NKVH, PS, lens, num_pages)
out = torch.zeros(a.batch, NQH, HD, dtype=torch.float16, device=B.DEV)
y = B.make_y_split_runner(a.ptx, name, a.warps * 32, a.splits, NKVH, q, kc, vc, pt, sl, mp, out)
if a.check:
    y(); torch.cuda.synchronize()
    ref = B.reference(q, kc, vc, pt, sl, HD, NQH, NKVH, PS)
    err = ((out.float() - ref).norm() / ref.norm()).item()
    print("rel L2 %.3e" % err, "OK" if err <= B.TOL_REL_L2 else "FAIL")
runners = [y]
if a.fi:
    fi = B.import_flashinfer()
    runners.append(B.make_flashinfer_runner(fi, kc, vc, pt, sl, q, HD, NQH, NKVH, PS))
for _ in range(a.iters):
    for r in runners:
        r()
torch.cuda.synchronize()
