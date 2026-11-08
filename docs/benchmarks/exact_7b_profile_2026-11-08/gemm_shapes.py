"""The linear layers of a Qwen2.5-7B decode step at batch 32, one GEMM at a
time, from DRAM: microseconds per call and the weight bandwidth reached.

    python3 gemm_shapes.py y      <out.json>   # Y's int8_mm and torch._int_mm (cuBLASLt)
    ~/vllm_env/bin/python gemm_shapes.py vllm <out.json>   # vLLM's CUTLASS int8 and FP8

Every kernel gets an [M][K] activation and an [N][K] weight. Each call reads a
different weight copy, cycling through enough copies to exceed the L2 cache
four times, so every weight comes from DRAM as it does in decoding. Timed
with CUDA events over 200 calls, the best of 7 rounds. The outputs of each
int8 kernel are checked against a float64 product first.
"""
import json
import sys

import torch

KIND, OUT = sys.argv[1], sys.argv[2]
M = 32
SHAPES = {                      # (N, K) and calls per decode step
    "q_proj / o_proj": ((3584, 3584), 56),
    "k_proj / v_proj": ((512, 3584), 56),
    "gate / up": ((18944, 3584), 56),
    "down": ((3584, 18944), 28),
    "lm_head": ((152064, 3584), 1),
    # Merged, as vLLM merges them: q, k and v read the same input, and so do
    # gate and up (one call each per layer in place of three and two).
    "qkv merged": ((4608, 3584), 0),
    "gate+up merged": ((37888, 3584), 0),
}
L2 = torch.cuda.get_device_properties(0).L2_cache_size
dev = "cuda"
torch.manual_seed(0)


def bench(fn, weights, a):
    n = len(weights)
    for i in range(10):
        fn(a, weights[i % n])
    best = float("inf")
    for _ in range(7):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for i in range(200):
            fn(a, weights[i % n])
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) * 1000 / 200)
    return best


kernels = {}
if KIND == "y":
    sys.path.insert(0, "python")
    from y_lang.int8_gemm import int8_mm
    kernels["Y int8_mm"] = lambda a, w: int8_mm(a, w)
    kernels["cuBLASLt int8 (torch._int_mm)"] = lambda a, w: torch._int_mm(a, w.t())
else:
    from vllm import _custom_ops as ops
    kernels["CUTLASS int8 (vLLM)"] = None
    kernels["CUTLASS FP8 (vLLM)"] = None

results = {}
for name, ((N, K), per_step) in SHAPES.items():
    copies = max(2, -(-4 * L2 // (N * K)))
    w8 = [torch.randint(-127, 128, (N, K), dtype=torch.int8, device=dev) for _ in range(copies)]
    a8 = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=dev)
    ref = a8.double() @ w8[0].double().t()
    row = {}
    if KIND == "y":
        for kname, fn in kernels.items():
            got = fn(a8, w8[0])
            assert torch.equal(got.double(), ref), f"{kname} {name} is not the exact product"
            row[kname] = bench(fn, w8, a8)
    else:
        sa = torch.ones((M, 1), dtype=torch.float32, device=dev)
        sb = torch.ones((N, 1), dtype=torch.float32, device=dev)
        fn_i8 = lambda a, w: ops.cutlass_scaled_mm(a, w.t(), sa, sb, torch.bfloat16)
        small = torch.randint(-3, 4, (M, K), dtype=torch.int8, device=dev)
        small_w = torch.randint(-3, 4, (N, K), dtype=torch.int8, device=dev)
        got = fn_i8(small, small_w).double()      # bf16 output: rounded, so a tolerance
        want = small.double() @ small_w.double().t()
        assert torch.allclose(got, want, rtol=1e-2, atol=1), f"CUTLASS int8 {name} wrong"
        row["CUTLASS int8 (vLLM)"] = bench(fn_i8, w8, a8)
        wf = [w.to(torch.float16).clamp(-448, 448).to(torch.float8_e4m3fn) for w in w8]
        af = a8.to(torch.float16).to(torch.float8_e4m3fn)
        fn_f8 = lambda a, w: ops.cutlass_scaled_mm(a, w.t(), sa, sb, torch.bfloat16)
        row["CUTLASS FP8 (vLLM)"] = bench(fn_f8, wf, af)
    gb = N * K / 1e9
    results[name] = {"N": N, "K": K, "calls_per_step": per_step,
                     "us": {k: round(v, 2) for k, v in row.items()},
                     "GB_per_s": {k: round(gb / (v / 1e6), 1) for k, v in row.items()}}
    print(f"{name:<16} N={N:<6} K={K:<6}", "  ".join(f"{k}: {v:7.1f} us {gb/(v/1e6):6.0f} GB/s" for k, v in row.items()), flush=True)
step = {k: sum(r["us"][k] * r["calls_per_step"] for r in results.values()) / 1000 for k in next(iter(results.values()))["us"]}
print("per decode step (ms):", {k: round(v, 2) for k, v in step.items()})
json.dump({"M": M, "L2_bytes": L2, "shapes": results, "per_step_ms": step}, open(OUT, "w"), indent=1)
