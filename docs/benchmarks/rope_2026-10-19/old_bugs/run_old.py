#!/usr/bin/env python3
"""run_old.py <bench_ab dir>: rope_<hd>_parent.ptx, the parent compiler's
output for rope_96.ysu and rope_32.ysu (this directory; the compiler now
refuses both), launched as their contract said (grid = rows, 32 threads),
on NaN-filled outputs: how many of each row's elements they write."""
import os, sys
import torch
sys.path.insert(0, sys.argv[1])
from bench_ab import kernel_from_ptx
D = os.path.dirname(os.path.abspath(__file__))
torch.zeros(1, device="cuda")
for hd in (96, 32):
    k = kernel_from_ptx(f"{D}/rope_{hd}_parent.ptx", f"rope_{hd}")
    rows = 1000
    x = torch.randn(rows, hd, device="cuda").half()
    pos = torch.randint(0, 4096, (rows,), device="cuda", dtype=torch.int32)
    out = torch.full_like(x, float("nan"))
    k.launch((rows, 1, 1), (32, 1, 1), [x, pos, out])
    torch.cuda.synchronize()
    written = (~torch.isnan(out)).sum(dim=1)
    cols = (~torch.isnan(out)).all(dim=0).nonzero().flatten().tolist()
    print(f"rope_{hd}: {rows} rows; elements written a row: min {written.min().item()} max {written.max().item()} of {hd};"
          f" columns written in every row: {cols[:1] + cols[-1:] if cols else []} ({len(cols)} columns)")
