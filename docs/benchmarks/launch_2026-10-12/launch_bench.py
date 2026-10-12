"""Per-call cost of launching a tiny kernel from Python: Y's TorchKernel,
Triton, PyTorch, on a 100,000-element f32 add (each column checked exact).

Two measures, minimum and median over 15 interleaved rounds:
- events: back-to-back calls timed with CUDA events, as the head-to-head's
  *launch* mode times them (a caller's view; it cannot go below the GPU's
  own time per kernel);
- CPU: the same calls timed on the host clock without waiting for the GPU,
  which is the launch path's own cost.

    PYTHONPATH=<checkout>/python python3 launch_bench.py <label> <results.json>
    PYTHONPATH=<checkout>/python python3 launch_bench.py profile
"""
import cProfile
import pstats
import statistics
import sys

import torch
import triton
import triton.language as tl

import y_lang

N = 100_000
Y_VEC_ADD = """
kernel vec_add(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, N: I32) {
    let idx: I32 = global_thread_id() * 4;
    if idx < N {
        vec_add_v4(A[idx], B[idx], C[idx]);
    }
}
"""


@triton.jit
def triton_add(a, b, c, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(c + i, tl.load(a + i, mask=m) + tl.load(b + i, mask=m), mask=m)


def event_us(fn, calls):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(calls):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) * 1000.0 / calls


a = torch.randn(N, device="cuda")
b = torch.randn(N, device="cuda")
cy, ct, cp = torch.zeros_like(a), torch.zeros_like(a), torch.zeros_like(a)
k = y_lang.TorchKernel(Y_VEC_ADD, kernel_name="vec_add", target_sm="sm_89")
grid, block = ((N // 4 + 127) // 128, 1, 1), (128, 1, 1)
args = [a, b, cy, N]
cols = {
    "Y": lambda: k.launch(grid, block, args),
    "Triton": lambda: triton_add[(triton.cdiv(N, 1024),)](a, b, ct, N, BLOCK=1024),
    "PyTorch": lambda: torch.add(a, b, out=cp),
}
for f in cols.values():
    f()
torch.cuda.synchronize()
assert torch.equal(cy, a + b) and torch.equal(ct, a + b) and torch.equal(cp, a + b)

if len(sys.argv) > 1 and sys.argv[1] == "profile":
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(20000):
        k.launch(grid, block, args)
    pr.disable()
    torch.cuda.synchronize()
    pstats.Stats(pr).sort_stats("tottime").print_stats(14)
    sys.exit(0)

import time


def cpu_us(fn, calls):
    # CPU time per call: the launches are asynchronous, and 200 of them fit
    # in the launch queue, so nothing here waits on the GPU.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(calls):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    return (t1 - t0) * 1e6 / calls


samples = {name: [] for name in cols}
cpu = {name: [] for name in cols}
names = list(cols)
for r in range(17):
    for name in names[r % 3:] + names[:r % 3]:
        t = event_us(cols[name], 500)
        c = cpu_us(cols[name], 200)
        if r >= 2:
            samples[name].append(t)
            cpu[name].append(c)
import json
import os
import subprocess

record = {
    "label": sys.argv[1], "y_lang": os.path.dirname(y_lang.__file__),
    "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "triton": triton.__version__,
    "python": sys.version.split()[0],
    "gpu_busy": subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True).stdout.strip(),
    "events_us": samples, "cpu_us": cpu,
}
for name in names:
    s, c = samples[name], cpu[name]
    record.setdefault("summary", {})[name] = {
        "events_min": min(s), "events_median": statistics.median(s), "cpu_min": min(c), "cpu_median": statistics.median(c)}
    print(f"{sys.argv[1]:7s} {name:8s} events: min {min(s):6.2f} median {statistics.median(s):6.2f} | CPU per call: min {min(c):6.2f} median {statistics.median(c):6.2f} us")
with open(sys.argv[2], "w") as f:
    json.dump(record, f, indent=1)
