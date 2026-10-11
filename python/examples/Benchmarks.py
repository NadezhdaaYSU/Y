#!/usr/bin/env venv/bin/python
"""
Y Engine — Wafer AI Production Features Empirical Benchmark Suite
Runs rigorous performance, compilation throughput, JIT latency, autotuning scaling,
and live GPU CUDA execution benchmarks on NVIDIA GeForce RTX 4070 Ti SUPER.
"""

import platform
import sys
import time
from pathlib import Path

# Add python module path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "python"))

import y_lang

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Every number the summary report states, filled in by the sections below. The
# report used to be a fixed template - its figures, its GPU and its software
# versions were literals, written out the same whatever this run measured, and
# it listed a FlashAttention kernel as "emitted" that does not compile.
RESULTS = {}
KERNELS = repo_root / "python" / "examples" / "kernels"


def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def benchmark_jit_compilation_speed():
    print_header("1. JIT Compilation Latency & Cache Throughput")

    # This used to compile `flash_attention.ysu`, which does not compile (see
    # its header), so the whole suite stopped at its first call.
    kernel_path = KERNELS / "rmsnorm.ysu"
    with open(kernel_path, "r") as f:
        # A source the disk cache has never seen, or "cold" is a lookup.
        source = f.read() + f"\n// cold-{time.time_ns()}\n"

    # Cold Compilation
    t0 = time.perf_counter()
    ptx_cold = y_lang.compile_to_ptx(source, target_sm="sm_89")
    t_cold_ms = (time.perf_counter() - t0) * 1000.0

    # Warm Compilation (Cache Hit)
    iterations = 10000
    t0 = time.perf_counter()
    for _ in range(iterations):
        ptx_warm = y_lang.compile_to_ptx(source, target_sm="sm_89")
    t_warm_total_s = time.perf_counter() - t0
    t_warm_avg_us = (t_warm_total_s / iterations) * 1e6
    ops_per_sec = iterations / t_warm_total_s
    RESULTS["jit"] = (kernel_path.name, t_cold_ms, t_warm_avg_us, ops_per_sec)

    print(f"  * Source Kernel       : {kernel_path.name} ({len(source)} bytes)")
    print(f"  * Cold JIT Latency    : {t_cold_ms:.3f} ms")
    print(f"  * Warm Cache Latency  : {t_warm_avg_us:.3f} µs per call")
    print(f"  * JIT Cache Throughput: {ops_per_sec:,.0f} compiles/sec")

def benchmark_autotuner_search_scaling():
    print_header("2. Autotuner Search Space Generation Scaling")

    matrix_dims = [
        (128, 128, 128),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]

    print(f"  {'Matrix Size':<18} | {'Candidates Generated':<20} | {'Generation Latency (µs)':<22}")
    print("-" * 68)

    RESULTS["autotune"] = []
    for m, n, k in matrix_dims:
        t0 = time.perf_counter()
        for _ in range(100):
            candidates = y_lang.generate_autotune_search_space(m, n, k)
        t_avg_us = ((time.perf_counter() - t0) / 100) * 1e6
        dim_str = f"{m}x{n}x{k}"
        RESULTS["autotune"].append((dim_str, len(candidates), t_avg_us))
        print(f"  {dim_str:<18} | {len(candidates):<20} | {t_avg_us:<22.2f}")

def benchmark_ptx_codegen_targets():
    print_header("3. PTX Codegen Throughput Across Architecture Targets")

    kernels = [
        ("FlashAttention-2", KERNELS / "flash_attention.ysu"),
        ("RMSNorm",          KERNELS / "rmsnorm.ysu"),
        ("SwiGLU",           KERNELS / "swiglu.ysu"),
    ]

    targets = ["sm_80", "sm_86", "sm_89", "sm_90a"]

    print(f"  {'Kernel':<18} | {'Target':<8} | {'PTX Size':<10} | {'Compile Time (ms)':<18}")
    print("-" * 64)

    for kname, kpath in kernels:
        with open(kpath, "r") as f:
            code = f.read()

        for sm in targets:
            # A source the disk cache has never seen, so this is a compile.
            t0 = time.perf_counter()
            try:
                ptx = y_lang.compile_to_ptx(code + f"\n// {sm} {time.time_ns()}\n", target_sm=sm)
            except ValueError as error:
                reason = str(error).strip().splitlines()[-1].strip()
                RESULTS.setdefault("codegen", []).append((kname, sm, None, reason))
                print(f"  {kname:<18} | {sm:<8} | DOES NOT COMPILE: {reason}")
                continue
            t_ms = (time.perf_counter() - t0) * 1000.0
            RESULTS.setdefault("codegen", []).append((kname, sm, t_ms, len(ptx)))
            print(f"  {kname:<18} | {sm:<8} | {len(ptx):<10} | {t_ms:<18.3f}")

def benchmark_pytorch_gpu_execution():
    print_header("4. PyTorch GPU Tensor Execution & Latency Benchmark")

    if not HAS_TORCH or not torch.cuda.is_available():
        print("  [!] PyTorch CUDA is not available. Skipping live GPU execution benchmark.")
        return

    device_name = torch.cuda.get_device_name(0)
    print(f"  * Active GPU Accelerator : {device_name}")
    print(f"  * CUDA Memory Allocator  : PyTorch CUDACachingAllocator")

    # A launch-latency probe, not a vector add: every thread runs the loop
    # below over the first elements, and nothing reads `C` back. The number
    # it measures is the cost of a launch.
    y_vecadd_source = """
    kernel vec_add(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, N: I32) {
        @invariant(i >= 0)
        for i in 0..1024 step 1 {
            let a_val: F32 = GlobalMemory::load(A);
            let b_val: F32 = GlobalMemory::load(B);
            let sum_val: F32 = a_val + b_val;
            store(C, sum_val);
        }
    }
    """

    print("\n  [+] Compiling Y launch-latency kernel to PTX for sm_89...")
    kernel = y_lang.TorchKernel(y_vecadd_source, kernel_name="vec_add", target_sm="sm_89")
    print("      -> PTX compiled and loaded into CUDA module successfully.")

    # Allocate PyTorch tensors on GPU
    N = 1024
    a_tensor = torch.randn(N, dtype=torch.float32, device="cuda")
    b_tensor = torch.randn(N, dtype=torch.float32, device="cuda")
    c_tensor = torch.zeros(N, dtype=torch.float32, device="cuda")

    # Warmup launches
    for _ in range(10):
        kernel.launch(grid=(1, 1, 1), block=(256, 1, 1), args=[a_tensor, b_tensor, c_tensor, N])
    torch.cuda.synchronize()

    # Empirical latency benchmark over 1,000 launches
    launches = 1000
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(launches):
        kernel.launch(grid=(1, 1, 1), block=(256, 1, 1), args=[a_tensor, b_tensor, c_tensor, N])
    end_event.record()
    torch.cuda.synchronize()

    total_time_ms = start_event.elapsed_time(end_event)
    avg_latency_us = (total_time_ms / launches) * 1000.0

    print(f"  * Executed Launches      : {launches:,}")
    print(f"  * Total GPU Time         : {total_time_ms:.2f} ms")
    print(f"  * Avg Launch Latency     : {avg_latency_us:.2f} µs")
    print(f"  * Kernel Launch Rate     : {launches / (total_time_ms / 1000.0):,.0f} launches/sec")
    RESULTS["launch"] = (device_name, launches, avg_latency_us, launches / (total_time_ms / 1000.0))

def generate_benchmark_summary_report():
    print_header("5. Summary Report Artifact Generation")
    report_path = repo_root / "benchmark_wafer_features_results.md"

    lines = ["# Y Engine — Wafer AI Features Benchmark Results", ""]
    gpu = RESULTS["launch"][0] if "launch" in RESULTS else "not measured (no PyTorch CUDA)"
    torch_version = torch.__version__ if HAS_TORCH else "not installed"
    lines += [
        f"**GPU**: {gpu}  ",
        f"**Execution Environment**: Python {platform.python_version()} / PyTorch {torch_version}  ",
        "**Compiler Backend**: Y C-ABI shared library (`liby.so`)",
        "",
        "Every figure below was measured by this run of `python/examples/Benchmarks.py`.",
        "",
    ]
    if "jit" in RESULTS:
        name, cold_ms, warm_us, per_sec = RESULTS["jit"]
        lines += [
            "## JIT Compilation Latency",
            f"* Cold compile of `{name}`: {cold_ms:.3f} ms",
            f"* Warm cache lookup: {warm_us:.3f} µs ({per_sec:,.0f} lookups/sec)",
            "",
        ]
    if "launch" in RESULTS:
        _, launches, latency_us, rate = RESULTS["launch"]
        lines += [
            "## Kernel Launch Latency",
            f"* {launches:,} launches of a trivial kernel: {latency_us:.2f} µs each ({rate:,.0f} launches/sec)",
            "",
        ]
    if "autotune" in RESULTS:
        lines += ["## Autotuning Candidate Generation", ""]
        lines += [f"* {dims}: {count} candidates in {us:.2f} µs" for dims, count, us in RESULTS["autotune"]]
        lines.append("")
    if "codegen" in RESULTS:
        lines += ["## PTX Emission", ""]
        for kname, sm, t_ms, detail in RESULTS["codegen"]:
            if t_ms is None:
                lines.append(f"* {kname} (`{sm}`): does not compile - {detail}")
            else:
                lines.append(f"* {kname} (`{sm}`): {t_ms:.3f} ms, {detail} bytes")
        lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  -> Benchmark report written to: {report_path.name}")

def main():
    t_start = time.perf_counter()
    benchmark_jit_compilation_speed()
    benchmark_autotuner_search_scaling()
    benchmark_ptx_codegen_targets()
    benchmark_pytorch_gpu_execution()
    generate_benchmark_summary_report()
    total_sec = time.perf_counter() - t_start
    print(f"\n[+] Full benchmark suite completed in {total_sec:.2f} seconds.")

if __name__ == "__main__":
    main()
