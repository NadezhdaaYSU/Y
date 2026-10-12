#!/usr/bin/env python3
"""Y's `@tile` FP8 GEMM before and after the 2026-10-12 change, and the
head-to-head's Triton kernel doing the same work (f32 A and B quantized to
e4m3 in the kernel, e4m3 tensor cores, f32 C).

    # PTX from each compiler, one process per checkout (each loads its own liby.so)
    python3 bench.py emit <checkout> <out_dir>
    # check every column exactly, then time them in interleaved rounds
    python3 bench.py time <before_dir> <after_dir> <results.json>

Inputs are small integers times 2^-3, exact in e4m3 and in every f32 sum, so
every column must equal A @ B bit for bit before it is timed.
"""

import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time

SIZES = [256, 512, 768, 1024, 2048, 4096, 8192]


def emit(checkout, out_dir):
    sys.path.insert(0, os.path.join(checkout, "python"))
    from y_lang.compiler import compile_to_ptx
    os.makedirs(out_dir, exist_ok=True)
    for n in SIZES:
        src = (f"@tile({n}, {n}, {n})\nkernel gemm_fp8_{n}(A: GlobalMemory<F32>, B: GlobalMemory<F32>, "
               f"scale_a: F32, scale_b: F32, C: GlobalMemory<F32>) {{\n}}\n")
        with open(os.path.join(out_dir, f"gemm_fp8_{n}.ptx"), "w") as f:
            f.write(compile_to_ptx(src, target_sm="sm_89"))


def gpu_busy():
    vals = []
    for _ in range(10):
        o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True).stdout.strip()
        if o.isdigit():
            vals.append(int(o))
        time.sleep(0.1)
    return statistics.mean(vals) if vals else None


def time_all(before_dir, after_dir, out_json):
    import torch
    import triton
    import triton.language as tl
    from y_lang.torch_interop import TorchKernel

    class PtxKernel(TorchKernel):
        def __init__(self, ptx, name):
            self.source, self.kernel_name, self.target_sm, self.ptx = None, name, None, ptx
            self._module = self._function = None

    @triton.autotune(
        configs=[
            triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=8, num_stages=3),
            triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=4, num_stages=2),
            triton.Config({"BM": 64, "BN": 128, "BK": 64}, num_warps=4, num_stages=2),
            triton.Config({"BM": 128, "BN": 64, "BK": 64}, num_warps=4, num_stages=2),
            triton.Config({"BM": 64, "BN": 64, "BK": 64}, num_warps=4, num_stages=3),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def triton_fp8_gemm(a_ptr, b_ptr, c_ptr, M, N, K, inv_sa, inv_sb, s_ab,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        rm = tl.program_id(0) * BM + tl.arange(0, BM)
        rn = tl.program_id(1) * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            a = tl.load(a_ptr + rm[:, None] * K + (k0 + rk)[None, :])
            b = tl.load(b_ptr + (k0 + rk)[:, None] * N + rn[None, :])
            acc = tl.dot((a * inv_sa).to(tl.float8e4nv), (b * inv_sb).to(tl.float8e4nv), acc)
        tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc * s_ab)

    def event_us(fn, calls):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(calls):
            fn()
        e.record()
        e.synchronize()
        return s.elapsed_time(e) * 1000.0 / calls

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.zeros(1, device="cuda")
    record = {
        "gpu": torch.cuda.get_device_name(0),
        "driver": subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                 capture_output=True, text=True).stdout.strip(),
        "torch": torch.__version__, "triton": triton.__version__,
        "gpu_busy_before": gpu_busy(), "sizes": {},
    }
    for n in SIZES:
        gen = torch.Generator(device="cuda").manual_seed(n)
        a = (torch.randint(-8, 9, (n, n), device="cuda", generator=gen).float() * 0.125).contiguous()
        b = (torch.randint(-8, 9, (n, n), device="cuda", generator=gen).float() * 0.125).contiguous()
        ref = a @ b
        cols, outs, ptx_sha = {}, {}, {}
        for label, d in (("Y before", before_dir), ("Y after", after_dir)):
            ptx = open(os.path.join(d, f"gemm_fp8_{n}.ptx")).read()
            ptx_sha[label] = hashlib.sha256(ptx.encode()).hexdigest()
            g = re.search(r"Launch grid required: \((\d+), (\d+), (\d+)\) CTAs, block \((\d+),(\d+),(\d+)\)", ptx)
            v = [int(x) for x in g.groups()]
            kern = PtxKernel(ptx, f"gemm_fp8_{n}")
            outs[label] = torch.full((n, n), float("nan"), device="cuda")
            cols[label] = (lambda kern=kern, grid=tuple(v[:3]), block=tuple(v[3:]), c=outs[label]:
                           kern.launch(grid, block, [a, b, 1.0, 1.0, c]))
        outs["Triton"] = torch.full((n, n), float("nan"), device="cuda")
        cols["Triton"] = lambda: triton_fp8_gemm[lambda m: (triton.cdiv(n, m["BM"]), triton.cdiv(n, m["BN"]))](
            a, b, outs["Triton"], n, n, n, 1.0, 1.0, 1.0)
        for f in cols.values():
            f()
        torch.cuda.synchronize()
        exact = {k: bool(torch.equal(c, ref)) for k, c in outs.items()}
        calls = min(200, max(2, int(5e-3 / (2 * n ** 3 / 30e12))))   # ~5 ms a sample
        rounds = 15 if n < 8192 else 9
        names = [k for k in cols if exact[k]]
        samples = {k: [] for k in names}
        busy = gpu_busy()
        for r in range(rounds + 2):
            for k in names[r % len(names):] + names[:r % len(names)]:
                t = event_us(cols[k], calls)
                if r >= 2:
                    samples[k].append(t)
        record["sizes"][n] = {
            "exact": exact, "calls_per_sample": calls, "gpu_busy_before": busy, "ptx_sha256": ptx_sha,
            "triton_config": str(triton_fp8_gemm.best_config),
            "samples_us": samples,
            "min_us": {k: min(s) for k, s in samples.items()},
            "median_us": {k: statistics.median(s) for k, s in samples.items()},
        }
        m = record["sizes"][n]["min_us"]
        print(f"{n:>5}  " + "  ".join(f"{k} {m[k]:9.1f}" for k in m) + f"  [GPU {busy:.0f}% busy before]", flush=True)
    record["gpu_busy_after"] = gpu_busy()
    with open(out_json, "w") as f:
        json.dump(record, f, indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "emit":
        emit(sys.argv[2], sys.argv[3])
    else:
        time_all(sys.argv[2], sys.argv[3], sys.argv[4])
