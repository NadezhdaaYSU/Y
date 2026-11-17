"""torch._int_mm (int8 x int8 -> int32) against a float16 matmul at the
Qwen2.5-0.5B decode shapes, 32 rows: does the int8 GEMM itself cost more?

    python3 gemm_shapes.py OUT.json [M]

Three arms per shape:
  int_mm nn  the weight as a row-major [K][N] matrix, the layout the exact
             path's `torch._int_mm` fallback used up to 2026-11-17;
  int_mm tn  the same weight stored as nn.Linear stores it, [N][K]
             row-major, passed as its transpose (no copy). cuBLAS says its
             int8 tensor-core kernels need this "TN" form;
  fp16 mm    a float16 matmul in nn.Linear's layout, as stock runs it.

CUDA-event timing, 200 calls per sample, 15 samples taken in turns, median
and minimum per arm. Each int8 result is checked against a float64 product.
The kernels each arm launched are listed, from one profiled call.
"""
import json
import statistics
import sys

import torch
from torch.profiler import ProfilerActivity, profile

M = int(sys.argv[2]) if len(sys.argv) > 2 else 32
SHAPES = {"qkv 896->1152": (896, 1152), "o 896->896": (896, 896),
          "gate or up 896->4864": (896, 4864), "gate+up 896->9728": (896, 9728),
          "down 4864->896": (4864, 896), "lm_head 896->151936": (896, 151936)}
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(0)


def timed(fn, n=200):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / n          # us per call


def kernels_of(fn):
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    return sorted({e.name for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA})


rows = {}
for name, (k, n) in SHAPES.items():
    a8 = torch.randint(-127, 128, (M, k), device=dev, dtype=torch.int8, generator=g)
    w8 = torch.randint(-127, 128, (n, k), device=dev, dtype=torch.int8, generator=g)   # [N][K]
    w8_nn = w8.t().contiguous()                     # [K][N] row-major
    w8_tn = w8.t()                                  # [K][N] view of [N][K] storage
    ah, wh = torch.randn(M, k, device=dev, dtype=torch.float16), torch.randn(n, k, device=dev, dtype=torch.float16)
    ref = a8.double() @ w8.double().t()
    exact = {"int_mm nn": torch.equal(torch._int_mm(a8, w8_nn).double(), ref),
             "int_mm tn": torch.equal(torch._int_mm(a8, w8_tn).double(), ref)}
    arms = {"int_mm nn": lambda: torch._int_mm(a8, w8_nn),
            "int_mm tn": lambda: torch._int_mm(a8, w8_tn),
            "fp16 mm": lambda: ah @ wh.t()}
    for f in arms.values():
        f()
    torch.cuda.synchronize()
    samples = {a: [] for a in arms}
    for _ in range(15):
        for a, f in arms.items():
            samples[a].append(timed(f))
    rows[name] = {"exact": exact,
                  **{a: {"median_us": round(statistics.median(v), 2), "min_us": round(min(v), 2),
                         "kernels": kernels_of(arms[a])} for a, v in samples.items()}}
    r = rows[name]
    nn_, tn, fp = (r[a]["median_us"] for a in ("int_mm nn", "int_mm tn", "fp16 mm"))
    print(f"{name:<22} nn {nn_:8.2f}  tn {tn:8.2f}  fp16 {fp:8.2f} us   nn/fp16 {nn_ / fp:5.2f}x  "
          f"tn/fp16 {tn / fp:5.2f}x  exact nn={exact['int_mm nn']} tn={exact['int_mm tn']}")
    for a in arms:
        print(f"    {a:<10} {' | '.join(x[:90] for x in r[a]['kernels'])}")
json.dump({"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "M": M, "rows": rows},
          open(sys.argv[1], "w"), indent=1)
