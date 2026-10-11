#!/usr/bin/env python3
"""3-D block pointers: Y's `block_ptr3d_load/store` against Triton's 3-D pointers.

Both kernels compute `out = 2 * x + 1` over every element of a `D0 x D1 x D2`
float32 volume through masked 3-D addressing. Both outputs must equal the
float32 reference exactly (`2 * x` is exact, so every implementation rounds
once) before anything is timed; timing is interleaved rounds of plain
launches, reported as minimum and median.

This script used to time two kernels doing different work, neither of it the
volume it named: the Y kernel's 2 x 16 grid handled one `(d0, d1)` row per
block, 32,768 of 8,388,608 elements, and Triton's covered only `d2 < 64`,
524,288 elements. It checked neither output, printed "FASTER" whatever the
ratio, and deleted `~/.ysu/cache` when imported. Manual §34.6 published its
"2.52x" until 2026-10-11.

    PYTHONPATH=python python3 python/examples/benchmark_3d_block_ptr.py
"""

import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "python"))

import torch
import triton
import triton.language as tl
import y_lang


@triton.jit
def triton_3d(in_ptr, out_ptr, D0, D1, D2, S0, S1,
              B0: tl.constexpr, B1: tl.constexpr, B2: tl.constexpr):
    o0 = tl.program_id(0) * B0 + tl.arange(0, B0)
    o1 = tl.program_id(1) * B1 + tl.arange(0, B1)
    o2 = tl.program_id(2) * B2 + tl.arange(0, B2)
    offs = o0[:, None, None] * S0 + o1[None, :, None] * S1 + o2[None, None, :]
    mask = (o0 < D0)[:, None, None] & (o1 < D1)[None, :, None] & (o2 < D2)[None, None, :]
    tl.store(out_ptr + offs, tl.load(in_ptr + offs, mask=mask, other=0.0) * 2.0 + 1.0, mask=mask)


# One block per (d0, d1) row; each thread takes four consecutive d2.
Y_3D = """
kernel bench_3d(In: GlobalMemory<F32>, Out: GlobalMemory<F32>, D0: I32, D1: I32, D2: I32, S0: I32, S1: I32) {
    let d0: I32 = block_idx_x();
    let d1: I32 = block_idx_y();
    let d2: I32 = thread_idx_x() * 4;

    let v0: F32 = block_ptr3d_load(In, d0, d1, d2, S0, S1, D0, D1, D2);
    let v1: F32 = block_ptr3d_load(In, d0, d1, d2 + 1, S0, S1, D0, D1, D2);
    let v2: F32 = block_ptr3d_load(In, d0, d1, d2 + 2, S0, S1, D0, D1, D2);
    let v3: F32 = block_ptr3d_load(In, d0, d1, d2 + 3, S0, S1, D0, D1, D2);

    block_ptr3d_store(Out, d0, d1, d2, S0, S1, D0, D1, D2, v0 * 2.0 + 1.0);
    block_ptr3d_store(Out, d0, d1, d2 + 1, S0, S1, D0, D1, D2, v1 * 2.0 + 1.0);
    block_ptr3d_store(Out, d0, d1, d2 + 2, S0, S1, D0, D1, D2, v2 * 2.0 + 1.0);
    block_ptr3d_store(Out, d0, d1, d2 + 3, S0, S1, D0, D1, D2, v3 * 2.0 + 1.0);
}
"""


def gpu_busy_percent():
    """Mean utilisation over a second, before this process loads the GPU."""
    samples = []
    for _ in range(10):
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-i", "0"],
                             capture_output=True, text=True).stdout.strip()
        if out.isdigit():
            samples.append(int(out))
        time.sleep(0.1)
    return statistics.mean(samples) if samples else None


def time_us(fn, calls):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(calls):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / calls


def main(rounds=15, calls=20):
    busy = gpu_busy_percent()
    D0, D1, D2 = 32, 256, 1024
    S0, S1 = D1 * D2, D2
    major, minor = torch.cuda.get_device_capability(0)
    y_kernel = y_lang.TorchKernel(Y_3D, kernel_name="bench_3d", target_sm=f"sm_{major}{minor}")

    x = torch.randn(D0, D1, D2, device="cuda")
    want = x * 2.0 + 1.0
    out = {"Y": torch.full_like(x, float("nan")), "Triton": torch.full_like(x, float("nan"))}
    B0, B1, B2 = 4, 4, 64
    columns = {
        "Y": lambda: y_kernel.launch(grid=(D0, D1, 1), block=(D2 // 4, 1, 1),
                                     args=[x, out["Y"], D0, D1, D2, S0, S1]),
        "Triton": lambda: triton_3d[(triton.cdiv(D0, B0), triton.cdiv(D1, B1), triton.cdiv(D2, B2))](
            x, out["Triton"], D0, D1, D2, S0, S1, B0=B0, B1=B1, B2=B2),
    }

    print(f"{torch.cuda.get_device_name(0)}, PyTorch {torch.__version__}, Triton {triton.__version__}")
    print(f"out = 2x + 1 over {D0} x {D1} x {D2} float32 ({D0 * D1 * D2:,} elements)")
    if busy is not None and busy > 5:
        print(f"CONTENDED: the GPU was {busy:.0f}% busy before this started; the timings below are not clean")
    ok = {}
    for name, fn in columns.items():
        fn()
        torch.cuda.synchronize()
        ok[name] = torch.equal(out[name], want)
        print(f"  {name:<7} output {'equals' if ok[name] else 'DIFFERS FROM'} the reference"
              + ("" if ok[name] else f" ({int((out[name] != want).sum()):,} elements)"))
    timed = [n for n in columns if ok[n]]
    if not timed:
        sys.exit(1)

    samples = {n: [] for n in timed}
    for r in range(rounds + 3):
        for name in timed[r % len(timed):] + timed[:r % len(timed)]:
            t = time_us(columns[name], calls)
            if r >= 3:
                samples[name].append(t)
    for name in timed:
        s = samples[name]
        gbps = 2 * 4 * D0 * D1 * D2 / (min(s) * 1e-6) / 1e9
        print(f"  {name:<7} min {min(s):.2f} us, median {statistics.median(s):.2f} us "
              f"({gbps:.0f} GB/s at the minimum, reading and writing every element once; "
              f"the 32 MB input can sit partly in L2, so this can exceed DRAM bandwidth)")
    if not all(ok.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
