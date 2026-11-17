"""torch._int_mm (int8 x int8 -> int32) against a float16 matmul at the
Qwen2.5-0.5B decode shapes, 32 rows: does the int8 GEMM itself cost more?

    python3 gemm_shapes.py OUT.json

CUDA-event timing, 200 calls per sample, 15 samples taken in turns, median
and minimum per shape. The int8 result is checked against a float64 product.
"""
import json
import statistics
import sys

import torch

M = 32
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


rows = {}
for name, (k, n) in SHAPES.items():
    a8 = torch.randint(-127, 128, (M, k), device=dev, dtype=torch.int8, generator=g)
    b8 = torch.randint(-127, 128, (k, n), device=dev, dtype=torch.int8, generator=g)
    ah, bh = torch.randn(M, k, device=dev, dtype=torch.float16), torch.randn(k, n, device=dev, dtype=torch.float16)
    ok = torch.equal(torch._int_mm(a8, b8).double(), a8.double() @ b8.double())
    arms = {"int_mm": lambda: torch._int_mm(a8, b8), "fp16 mm": lambda: ah @ bh}
    for f in arms.values():
        f()
    torch.cuda.synchronize()
    samples = {a: [] for a in arms}
    for _ in range(15):
        for a, f in arms.items():
            samples[a].append(timed(f))
    rows[name] = {"exact": ok, **{a: {"median_us": round(statistics.median(v), 2), "min_us": round(min(v), 2)}
                                  for a, v in samples.items()}}
    r = rows[name]
    print(f"{name:<22} int_mm {r['int_mm']['median_us']:8.2f} us  fp16 {r['fp16 mm']['median_us']:8.2f} us  "
          f"int8/fp16 {r['int_mm']['median_us'] / r['fp16 mm']['median_us']:5.2f}x  exact={ok}")
json.dump({"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "M": M, "rows": rows},
          open(sys.argv[1], "w"), indent=1)
