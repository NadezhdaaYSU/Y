#!/usr/bin/env python3
"""probe_edge.py <examples dir>: Y's vector adds (vec_add_v4, and one element a
thread) against Triton's at footprints around the 48 MB L2. Checked exact
first; CUDA events, 17 interleaved rounds, 2 dropped; minimum of 15."""
import statistics, sys
import torch, triton
sys.path.insert(0, sys.argv[1])
import run_head_to_head_comparison as H
torch.zeros(1, device="cuda")
ADD1 = """
kernel add1(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, N: I32) {
    let i: I32 = global_thread_id();
    if i < N {
        C[i] = A[i] + B[i];
    }
}
"""
k4 = H.y_kernel(H.Y_VEC_ADD, "vec_add", "sm_89")
k1 = H.y_kernel(ADD1, "add1", "sm_89")
assert ".cs." not in k4.ptx and "ld.global.v4.f32" in k4.ptx
open("vec_add.ptx", "w").write(k4.ptx); open("add1.ptx", "w").write(k1.ptx)
gen = torch.Generator(device="cuda").manual_seed(3)
print("n, footprint MB, then min us: Y v4, Y scalar, Triton; median/min of each")
for mb in (16, 24, 32, 40, 44, 48, 52, 56, 64, 96):
    n = (mb * 1000 * 1000 // 12) // 1024 * 1024
    a, b = torch.randn(n, device="cuda", generator=gen), torch.randn(n, device="cuda", generator=gen)
    o = {x: torch.full((n,), float("nan"), device="cuda") for x in ("v4", "s", "T")}
    cols = {"v4": lambda: k4.launch(((n // 4 + 127) // 128, 1, 1), (128, 1, 1), [a, b, o["v4"], n]),
            "s": lambda: k1.launch(((n + 255) // 256, 1, 1), (256, 1, 1), [a, b, o["s"], n]),
            "T": lambda: H.triton_vector_add[(triton.cdiv(n, 1024),)](a, b, o["T"], n, BLOCK=1024)}
    for f in cols.values():
        f()
    torch.cuda.synchronize()
    ref = a + b
    assert all(torch.equal(o[x], ref) for x in o)
    H.warm_clocks()
    one = min(H.event_time_us(lambda f=f: [f() for _ in range(10)], 10) for f in cols.values())
    calls = max(3, int(3000 / max(one, 1)))
    s = {c: [] for c in cols}
    names = list(cols)
    for r in range(17):
        for c in names[r % 3:] + names[:r % 3]:
            t = H.event_time_us(lambda f=cols[c]: [f() for _ in range(calls)], calls)
            if r >= 2:
                s[c].append(t)
    m = {c: min(v) for c, v in s.items()}
    print("%9d %4d MB | v4 %8.2f  s %8.2f  T %8.2f | T/v4 %.3f  T/s %.3f | med/min %.3f %.3f %.3f | SM %d"
          % (n, mb, m["v4"], m["s"], m["T"], m["T"] / m["v4"], m["T"] / m["s"],
             *(statistics.median(s[c]) / m[c] for c in ("v4", "s", "T")), torch.cuda.clock_rate(0)), flush=True)
