#!/usr/bin/env python3
"""exhaustive.py: the pass's multiply-and-shift against n // c for EVERY u32 n,
for the divisors the committed kernels divide by that are not powers of two
(3, from three-stage pipelines) and a few more."""
import sys, time
import numpy as np
sys.path.insert(0, sys.argv[1])
from cdiv_pass import magic
for c in (3, 5, 6, 24):
    m, l = magic(c)
    t = time.time(); step = 1 << 25
    for s in range(0, 1 << 32, step):
        n = np.arange(s, s + step, dtype=np.uint64)
        q = (n * np.uint64(m)) >> np.uint64(32 + l)
        if not np.array_equal(q, n // np.uint64(c)):
            bad = n[q != n // np.uint64(c)][0]
            print(f"c={c}: WRONG at n={bad}"); sys.exit(1)
    print(f"c={c}: m={m} l={l}: all 2^32 dividends exact ({time.time() - t:.0f} s)", flush=True)
