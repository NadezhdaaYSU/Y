"""One launch of lj_closed (compiled), lj_closed_red and lj_fixed at 262,144 atoms, for ncu."""
import os, sys
import numpy as np, torch
sys.argv = [sys.argv[0], "--prof"] + sys.argv[1:]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_h.py")).read().split("if sys.argv[1] == \"--stop\":")[0])
V, BASE = sys.argv[2], sys.argv[3]
pos_h, pi_h, pj_h = system(64, 64)
nt, npairs = 3 * 64 ** 3, len(pi_h)
pos, pi, pj = (torch.from_numpy(x).to(dev) for x in (pos_h.reshape(-1), pi_h, pj_h))
acc = torch.zeros(2 * nt + 7, dtype=torch.int64, device=dev)
grid = ((npairs + 255) // 256, 1, 1)
for n in ("lj_closed", "lj_closed_red"):
    acc.zero_(); load(V, n).launch(grid, (256, 1, 1), [pos, pi, pj, acc, npairs, nt]); torch.cuda.synchronize()
fx = torch.zeros(nt, dtype=torch.int64, device=dev)
load(BASE, "lj_fixed").launch(grid, (256, 1, 1), [pos, pi, pj, fx, npairs]); torch.cuda.synchronize()
print("done")
