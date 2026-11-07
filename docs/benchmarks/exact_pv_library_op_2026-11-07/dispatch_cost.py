"""Time per call of each way to reach `exact_pv`'s kernel, at a 0.5B decode shape.

    /tmp/y_exact_nospace/venv/bin/python \
        /tmp/y_exact_nospace/docs/benchmarks/exact_pv_library_op_2026-11-07/dispatch_cost.py

(through the space-free link `tools/_nospace.py` makes, so `torch.compile`
can link). Run it with `Y_EXACT_V_INT8=1`.

The route until 2026-11-07 is rebuilt here as `old_body`: int64 `p` in,
float64 out, both casts inside, launched through `ptx_bridge.launch`, which
synchronises the context, and registered with `torch.library.custom_op`.
The other rows change one thing at a time: the synchronisation, the
registration, where the casts sit, then the launcher (`y_lang`'s
`TorchKernel` in place of ctypes). The last rows are the repo's route as it
is now, and the digit split a float32 V takes.

Each variant runs `calls` times with no synchronisation in the timing loop
(the old launch synchronises by itself); the time is wall time / calls, the
minimum over `batches` batches. Every variant's output is checked against a
float64 matmul first.
"""
import ctypes
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
sys.path.insert(0, os.path.join(REPO, "tools"))
import torch  # noqa: E402

import batch_invariance_demo as D  # noqa: E402
import ptx_bridge as PB  # noqa: E402

B, KV, REP, T, DH = (int(os.environ.get(k, d)) for k, d in
                     (("B", 32), ("KV", 2), ("REP", 7), ("T", 64), ("DH", 64)))
dev = torch.device("cuda")
torch.manual_seed(0)
p = torch.randint(0, (1 << D.P_BITS) + 1, (B, KV, REP, T), dtype=torch.int64, device=dev)
v = torch.randint(-127, 128, (B, KV, T, DH), dtype=torch.int8, device=dev)
want = torch.matmul(p.double(), v.double())
D._y_exact_pv(p.to(torch.int32), v)          # loads the kernel
FN = PB.Module(open(os.path.join(REPO, "tests", "exact_pv.ptx")).read()).fn("exact_pv")
torch.cuda.synchronize()
print(f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
      f"p {tuple(p.shape)} v {tuple(v.shape)}", flush=True)


def launch_async(fn, grid, block, args, stream):
    """`ptx_bridge.launch` on `stream`, without its synchronisation."""
    holders = [ctypes.byref(a) for a in args]
    arr = (ctypes.c_void_p * len(holders))(*[ctypes.cast(h, ctypes.c_void_p) for h in holders])
    err = PB.cuda.cuLaunchKernel(fn, grid[0], grid[1], grid[2], block[0], block[1], block[2],
                                 0, ctypes.c_void_p(stream), arr, None)
    assert err == 0, err


def old_body(p, v, sync=True):
    """The body as it was: casts inside, `launch` synchronises when `sync`."""
    lead = p.shape[:-2]
    nb = 1
    for x in lead:
        nb *= x
    qn, t = p.shape[-2], p.shape[-1]
    dh = v.shape[-1]
    p32 = p.reshape(nb, qn, t).to(torch.int32).contiguous()
    v8 = v.reshape(nb, t, dh).contiguous()
    out = torch.empty(nb * qn * dh, dtype=torch.int64, device=p.device)
    args = [PB.dptr(p32), PB.dptr(v8), PB.dptr(out),
            ctypes.c_uint(t), ctypes.c_uint(dh), ctypes.c_uint(qn),
            ctypes.c_uint(nb * qn * t), ctypes.c_uint(nb * t * dh),
            ctypes.c_uint(nb * qn * dh)]
    if sync:
        PB.launch(FN, (qn, nb, 1), (dh, 1, 1), args)
    else:
        launch_async(FN, (qn, nb, 1), (dh, 1, 1), args, torch.cuda.current_stream().cuda_stream)
    return out.view(*lead, qn, dh).to(torch.float64)


def fake64(p, v):
    return p.new_empty((*p.shape[:-1], v.shape[-1]), dtype=torch.float64)


def bench(name, fn, scale, calls=300, batches=7):
    out = fn()
    torch.cuda.synchronize()
    assert out.dtype == torch.float64 and torch.equal(out, want * scale), name
    best = float("inf")
    for _ in range(batches):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(calls):
            fn()
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        best = min(best, (t1 - t0) / calls)
    print(f"{name:62s} {best * 1e6:7.2f} us/call", flush=True)


# custom_op over the old body, with and without the synchronisation
old_sync = torch.library.custom_op("y_bench::pv_old_sync", lambda p, v: old_body(p, v, True),
                                   mutates_args=(), schema="(Tensor p, Tensor v) -> Tensor")
old_sync.register_fake(fake64)
old_async = torch.library.custom_op("y_bench::pv_old_async", lambda p, v: old_body(p, v, False),
                                    mutates_args=(), schema="(Tensor p, Tensor v) -> Tensor")
old_async.register_fake(fake64)
# Library over the old-shaped body (casts inside), no synchronisation
lib = torch.library.Library("y_bench", "FRAGMENT")
lib.define("pv_lib_casts_in(Tensor p, Tensor v) -> Tensor")
lib.impl("pv_lib_casts_in", lambda p, v: old_body(p, v, False), "CUDA")
torch.library.register_fake("y_bench::pv_lib_casts_in", fake64, lib=lib)

ops = torch.ops.y_bench
bench("eager: old body (casts in, synchronising launch), direct", lambda: old_body(p, v, True), 1)
bench("eager: old body, no synchronisation, direct", lambda: old_body(p, v, False), 1)
bench("eager: custom_op, synchronising (the route until 2026-11-07)", lambda: ops.pv_old_sync(p, v), 1)
bench("eager: D.exact_pv as now (direct, TorchKernel launch)", lambda: D.exact_pv(p, v), 1)
p32 = p.to(torch.int32)
bench("eager: D._y_exact_pv alone (int32 in, int64 out) + .double()", lambda: D._y_exact_pv(p32, v).double(), 1)

variants = [
    ("custom_op, synchronising, casts in (until 2026-11-07)", lambda p, v: ops.pv_old_sync(p, v)),
    ("custom_op, no synchronisation, casts in", lambda p, v: ops.pv_old_async(p, v)),
    ("Library, no synchronisation, casts in", lambda p, v: ops.pv_lib_casts_in(p, v)),
    ("Library, casts out, TorchKernel launch (D.exact_pv now)", lambda p, v: D.exact_pv(p, v)),
]
for name, f in variants:
    torch._dynamo.reset()
    g = torch.compile(lambda p, v, f=f: f(p, v) * 0.5, dynamic=True, fullgraph=True)
    bench(f"compiled: {name} + one pointwise", lambda g=g: g(p, v), 0.5)
torch._dynamo.reset()
vf = v.to(torch.float32)
g = torch.compile(lambda p, vf: D.exact_pv(p, vf) * 0.5, dynamic=True, fullgraph=True)
bench("compiled: digit split (float32 V) + one pointwise", lambda: g(p, vf), 0.5)
