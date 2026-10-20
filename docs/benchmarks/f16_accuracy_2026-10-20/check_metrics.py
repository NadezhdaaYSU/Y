"""check_metrics.py <bench_ab.py dir>: accuracy.py's metrics on crafted
outputs whose answers are known by hand (CPU only).

Four outputs: R = 1 (C one 2^-20 high); every product zero and C = 0;
R = 2^-30 against S = 1 (cancelled: C = 0 is 100% off relative to R but
2^-30 of S); every product zero and C = 2^-100 (must be counted, and kept
out of max_norm, which would otherwise be infinite).
"""
import math
import os
import sys

import torch

sys.argv = [sys.argv[0], sys.argv[1]]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import accuracy as A  # noqa: E402

r = torch.tensor([[1.0, 0.0, 2.0 ** -30, 0.0]], dtype=torch.float64)
s = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float64)
c = torch.tensor([[1.0 + 2.0 ** -20, 0.0, 0.0, 2.0 ** -100]], dtype=torch.float32)
got = A.metrics(c, r, s, s > 0, torch.linalg.vector_norm(r).item())
want = {"rel_fro": math.sqrt(2.0 ** -40 + 2.0 ** -60 + 2.0 ** -200) / math.sqrt(1.0 + 2.0 ** -60),
        "max_abs": 2.0 ** -20, "max_norm": 2.0 ** -20, "mean_norm": (2.0 ** -20 - 2.0 ** -30) / 2,
        "mean_abs_norm": (2.0 ** -20 + 2.0 ** -30) / 2, "zero_scale_nonzero": 1}
for key, w in want.items():
    assert math.isclose(got[key], w, rel_tol=1e-12), (key, got[key], w)

# the same outputs with the zero-scale one corrected: nothing counted
c2 = c.clone()
c2[0, 3] = 0.0
assert A.metrics(c2, r, s, s > 0, 1.0)["zero_scale_nonzero"] == 0

# versus: per-output |E| against a base; bitwise compares the float32 bits
base = torch.tensor([[1.0, 0.0, 2.0 ** -31, -0.0]], dtype=torch.float32)
v = A.versus(c, base, r)
# output 0: 2^-20 vs 0 worse; 1: equal (both exact); 2: 2^-30 vs 2^-31 worse;
# 3: 2^-100 vs 0 worse. bitwise: only output 1 (0.0 vs 0.0; -0.0 differs)
assert (v["better"], v["equal"], v["worse"], v["bitwise"]) == (0.0, 0.25, 0.75, 0.25), v
v = A.versus(base, c, r)
assert (v["better"], v["equal"], v["worse"]) == (0.75, 0.25, 0.0), v

# input_stats: f16 subnormals are 0 < |x| < 2^-14
import numpy as np  # noqa: E402
x = np.array([0.0, 2.0 ** -24, 2.0 ** -14, -2.0 ** -15, 1.0], dtype=np.float16)
st = A.input_stats(x)
assert (st["frac_zero"], st["frac_subnormal"], st["max_abs"], st["min_abs_nonzero"]) == (0.2, 0.4, 1.0, 2.0 ** -24), st

# every distribution is reproducible from its seed and draws what it says
for d in A.DISTS:
    a1, b1 = d(np.random.RandomState(7), 4, 6, 8)
    a2, b2 = d(np.random.RandomState(7), 4, 6, 8)
    assert a1.shape == (4, 8) and b1.shape == (8, 6) and (a1 == a2).all() and (b1 == b2).all(), d.__name__
a, b = A.paired(np.random.RandomState(1), 2, 3, 8)
assert (a[:, :4] == a[:, 4:]).all() and np.abs(b[:4].astype(np.float16).astype(float) + b[4:]).max() < 2.0 ** -6
a, b = A.offset(np.random.RandomState(1), 2, 3, 1000)
assert np.abs(b.sum(axis=0)).max() < 1e-9 and np.abs(a - 1).max() < 0.1
a, b = A.wide_range(np.random.RandomState(1), 50, 50, 1000)
assert 2.0 ** -10 <= np.abs(a).min() and np.abs(a).max() < 2.0 ** 11 and (a < 0).any() and (a > 0).any()
print("check_metrics: ok")
