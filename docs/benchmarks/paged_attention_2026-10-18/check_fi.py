#!/usr/bin/env python3
"""check_fi.py: FlashInfer's output on the six timed shapes against the same
f32 reference the Y kernels are checked against (NaN-filled first)."""
import os, sys
sys.path.insert(0, os.environ.get("Y_TESTS", ""))
import torch
import benchmark_y_paged_decode_attention as B
HD, NQH, NKVH, PS = 128, 32, 8, 16
fi = B.import_flashinfer()
worst = 0.0
for batch, ctx in [(1, 1024), (1, 4096), (8, 1024), (8, 4096), (32, 1024), (32, 4096)]:
    lens = [ctx] * batch
    q, kc, vc, pt, sl, mp = B.make_inputs(HD, NQH, NKVH, PS, lens, max(512, batch * ((ctx + PS - 1) // PS) * 2))
    ref = B.reference(q, kc, vc, pt, sl, HD, NQH, NKVH, PS)
    run = B.make_flashinfer_runner(fi, kc, vc, pt, sl, q, HD, NQH, NKVH, PS)
    # The runner the timing uses discards the output: call the same planned
    # wrapper on the same tensors and keep it.
    cell = dict(zip(run.__code__.co_freevars, (c.cell_contents for c in run.__closure__)))
    out = cell["wrapper"].run(cell["q"], (cell["kc"], cell["vc"])); torch.cuda.synchronize()
    err = ((out.float() - ref).norm() / ref.norm()).item()
    worst = max(worst, err)
    print("b%dc%d  FlashInfer rel L2 %.3e  finite %s" % (batch, ctx, err, torch.isfinite(out).all().item()))
print("worst %.3e, tolerance %.1e: %s" % (worst, B.TOL_REL_L2, "OK" if worst <= B.TOL_REL_L2 else "FAIL"))
