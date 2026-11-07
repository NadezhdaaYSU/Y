"""Host time per call of each way to reach an int8 GEMM, at a 0.5B decode shape.

The repo op `torch.ops.y_lang.int8_mm` is a `torch.library.Library` op; the
`custom_op` rows register the same Python body through
`torch.library.custom_op` (`y_bench::mm_custom`), as the repo did until
2026-11-07, and the `Library` rows a second `torch.library.Library` op
(`y_bench::mm`), so both registrations are measured in one process.

Each variant is called `calls` times with no synchronisation inside the
loop; the time is wall time / calls, the minimum over `batches` batches.
The kernels are a few microseconds of GPU time each, so the loop is bound by
the host while the host takes longer than that.
"""
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.join(REPO, "tools"))
import torch  # noqa: E402

from y_lang import int8_gemm  # noqa: E402

M, N, K = int(os.environ.get("M", 64)), int(os.environ.get("N", 896)), int(os.environ.get("K", 896))
dev = torch.device("cuda")
a = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=dev)
b = torch.randint(-127, 128, (N, K), dtype=torch.int8, device=dev)
want = (a.double() @ b.double().t())
torch.cuda.synchronize()


def bench(name, fn, calls=300, batches=7):
    out = fn()
    torch.cuda.synchronize()
    if isinstance(out, torch.Tensor) and out.dtype == torch.int32:
        assert torch.equal(out.double(), want * (2 if name.startswith("compiled") else 1)), name
    best = float("inf")
    for _ in range(batches):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(calls):
            fn()
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        best = min(best, (t1 - t0) / calls)
    print(f"{name:55s} {best * 1e6:6.2f} us/call (host, loop not synced)", flush=True)
    return best


# 1. the op's Python body, called directly
bench("int8_gemm._int8_mm (Python, direct)", lambda: int8_gemm._int8_mm(a, b))
# 2. the same body through torch.library.custom_op (the registration until 2026-11-07)
custom = torch.library.custom_op("y_bench::mm_custom", int8_gemm._int8_mm, mutates_args=())
custom.register_fake(lambda a, b: a.new_empty((a.shape[0], b.shape[0]), dtype=torch.int32))
bench("torch.ops.y_bench.mm_custom (custom_op)", lambda: torch.ops.y_bench.mm_custom(a, b))
# 3. a low-level Library op with only a CUDA kernel
lib = torch.library.Library("y_bench", "FRAGMENT")
lib.define("mm(Tensor a, Tensor b) -> Tensor")
lib.impl("mm", int8_gemm._int8_mm, "CUDA")
torch.library.register_fake("y_bench::mm", lambda a, b: a.new_empty((a.shape[0], b.shape[0]), dtype=torch.int32), lib=lib)
bench("torch.ops.y_bench.mm (Library.impl CUDA)", lambda: torch.ops.y_bench.mm(a, b))
# 4. the repo's op as registered now
bench("torch.ops.y_lang.int8_mm (the repo op, Library)", lambda: torch.ops.y_lang.int8_mm(a, b))
# reference points
bench("torch._int_mm (cuBLASLt, C++ op)", lambda: torch._int_mm(a, b.t()))
bench("torch.empty((M, N), int32) alone", lambda: torch.empty((M, N), dtype=torch.int32, device=dev))
int8_gemm.check_operands(a, b)
bench("int8_gemm.check_operands alone", lambda: int8_gemm.check_operands(a, b))
try:
    import w8a8_gemv
    sx = torch.ones(M, device=dev)
    sw = torch.ones(N, device=dev)
    w8t = b.t().contiguous()
    bench("w8a8_gemv.w8a8_matmul (Triton, eager)", lambda: w8a8_gemv.w8a8_matmul(a, sx, w8t, sw, None))
except Exception as e:  # noqa: BLE001
    print("Triton path unavailable:", e)

# Compiled: the op inside a graph, the way the exact model calls it
for name, op in [("custom_op", torch.ops.y_bench.mm_custom), ("Library.impl", torch.ops.y_bench.mm),
                 ("the repo op", torch.ops.y_lang.int8_mm)]:
    f = torch.compile(lambda a, b, op=op: op(a, b) * 2, dynamic=True, fullgraph=True)
    bench(f"compiled: {name} + one pointwise", lambda: f(a, b))
try:
    g = torch.compile(lambda a: w8a8_gemv.w8a8_matmul(a, sx, w8t, sw, None) * 2, dynamic=True, fullgraph=True)
    bench("compiled: Triton w8a8_matmul + one pointwise", lambda: g(a))
except Exception as e:  # noqa: BLE001
    print("compiled Triton unavailable:", e)
h = torch.compile(lambda a, b: torch._int_mm(a, b.t()) * 2, dynamic=True, fullgraph=True)
bench("compiled: torch._int_mm + one pointwise", lambda: h(a, b))
