"""FP8 GEMM from float32 A [M][K] and B [K][N] to float32 C, as the
head-to-head's FP8 row computes it, through four pipelines:

- "Y e4m3": three Y kernels - quantise A to e4m3, quantise B and transpose it
  to [N][K], then the @tile GEMM over e4m3 bytes;
- "Y fused": the @tile GEMM that quantises inside its K loop;
- "PyTorch": quantise A, quantise and transpose B, torch._scaled_mm;
- "Triton": the head-to-head's fused kernel.

    python3 bench_pipeline.py <out.json> [sizes] [gemm-ptx-dir]

Every column equals float64 A @ B first (small integers times 1/8). Then
each pipeline, and each of its parts, is timed in interleaved rounds, clocks
brought up per size and the SM clock read after every sample.
"""
import json
import re
import statistics
import subprocess
import sys

import torch
import triton
import triton.language as tl

import y_lang
from y_lang.torch_interop import TorchKernel

QUANT = """
kernel quant_e4m3(X: GlobalMemory<F32>, inv_s: F32, Q: GlobalMemory<E4M3>, n: I32) {
    let idx: I32 = global_thread_id();
    if idx < n {
        Q[idx] = X[idx] * inv_s;
    }
}
"""
QUANT_T = """
kernel quant_e4m3_t(X: GlobalMemory<F32>, inv_s: F32, Q: GlobalMemory<E4M3>, rows: I32, cols: I32) {
    let t: I32 = global_thread_id();
    let warp: I32 = t / 32;
    let lane: I32 = t - warp * 32;
    let groups: I32 = (cols + 7) / 8;
    let rg: I32 = warp / groups;
    let c: I32 = (warp - rg * groups) * 8 + lane - (lane / 8) * 8;
    let r: I32 = rg * 16 + (lane / 8) * 4;
    if r < rows {
        if c < cols {
            let v0: F32 = X[r * cols + c] * inv_s;
            let v1: F32 = X[(r + 1) * cols + c] * inv_s;
            let v2: F32 = X[(r + 2) * cols + c] * inv_s;
            let v3: F32 = X[(r + 3) * cols + c] * inv_s;
            block_ptr2d_store_v4(Q, c, r, rows, cols, rows, v0, v1, v2, v3);
        }
    }
}
"""


def gemm_e4m3_source(m, n, k):
    return f"""
@tile({m}, {n}, {k})
kernel gemm_e4m3_{m}(A: GlobalMemory<E4M3>, B: GlobalMemory<E4M3>, scale_a: F32, scale_b: F32, C: GlobalMemory<F32>) {{
}}
"""


def gemm_fused_source(n):
    return f"""
@tile({n}, {n}, {n})
kernel gemm_fp8_{n}(A: GlobalMemory<F32>, B: GlobalMemory<F32>, scale_a: F32, scale_b: F32, C: GlobalMemory<F32>) {{
}}
"""


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


def kernel_from_ptx(path, name):
    k = TorchKernel.__new__(TorchKernel)
    k.source, k.kernel_name, k.target_sm = None, name, "sm_89"
    k.ptx = open(path).read()
    k._module = k._function = None
    k._ensure_loaded()
    return k


def compiled(source, name):
    k = y_lang.TorchKernel(source, kernel_name=name, target_sm="sm_89")
    k._ensure_loaded()
    return k


def required_launch(ptx):
    m = re.search(r"Launch grid required: \((\d+), (\d+), (\d+)\) CTAs, block \((\d+),(\d+),(\d+)\)", ptx)
    v = [int(x) for x in m.groups()]
    return tuple(v[:3]), tuple(v[3:])


def event_us(fn, calls):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(calls):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) * 1000.0 / calls


def warm():
    w = torch.randn(4096, 4096, device="cuda")
    for _ in range(30):
        w @ w
    torch.cuda.synchronize()


def util():
    return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()


def main():
    out_path = sys.argv[1]
    sizes = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1024,2048,4096,8192").split(",")]
    ptx_dir = sys.argv[3] if len(sys.argv) > 3 else None
    torch.zeros(1, device="cuda")
    result = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "triton": triton.__version__,
              "util_before": util(), "sizes": {}}
    quant = compiled(QUANT, "quant_e4m3")
    quant_t = compiled(QUANT_T, "quant_e4m3_t")
    sa = sb = 0.125
    for size in sizes:
        m = n = k = size
        gen = torch.Generator(device="cuda").manual_seed(size)
        A = torch.randint(-8, 9, (m, k), device="cuda", generator=gen).float() * sa
        B = torch.randint(-8, 9, (k, n), device="cuda", generator=gen).float() * sb
        ref = A.double() @ B.double()
        if ptx_dir:
            g8 = kernel_from_ptx(f"{ptx_dir}/g{size}.ptx", f"gemm_e4m3_{size}")
        else:
            g8 = compiled(gemm_e4m3_source(m, n, k), f"gemm_e4m3_{m}")
        fused = compiled(gemm_fused_source(size), f"gemm_fp8_{size}")
        g8_launch, fused_launch = required_launch(g8.ptx), required_launch(fused.ptx)
        a8 = torch.empty(m, k, device="cuda", dtype=torch.uint8)
        b8t = torch.empty(n, k, device="cuda", dtype=torch.uint8)
        C = {c: torch.full((m, n), float("nan"), device="cuda") for c in ("Y e4m3", "Y fused", "PyTorch", "Triton")}
        sa_t, sb_t = torch.tensor(sa, device="cuda"), torch.tensor(sb, device="cuda")
        blocks = lambda total: ((total + 255) // 256, 1, 1)

        def y_quant_a():
            quant.launch(blocks(m * k), (256, 1, 1), [A, 1.0 / sa, a8, m * k])

        def y_quant_b():
            # A warp per 8 columns x 16 rows of B (see QUANT_T).
            quant_t.launch(blocks((n + 7) // 8 * ((k + 15) // 16) * 32), (256, 1, 1), [B, 1.0 / sb, b8t, k, n])

        def y_gemm():
            g8.launch(*g8_launch, [a8, b8t, sa, sb, C["Y e4m3"]])

        def y_e4m3():
            y_quant_a()
            y_quant_b()
            y_gemm()

        def y_fused():
            fused.launch(*fused_launch, [A, B, sa, sb, C["Y fused"]])

        def torch_quant_a():
            return (A * (1.0 / sa)).to(torch.float8_e4m3fn)

        def torch_quant_b():
            return (B * (1.0 / sb)).to(torch.float8_e4m3fn).t().contiguous().t()

        qa, qb = torch_quant_a(), torch_quant_b()

        def torch_gemm():
            torch._scaled_mm(qa, qb, scale_a=sa_t, scale_b=sb_t, out_dtype=torch.float32, out=C["PyTorch"])

        def torch_fp8():
            a = torch_quant_a()
            b = torch_quant_b()
            torch._scaled_mm(a, b, scale_a=sa_t, scale_b=sb_t, out_dtype=torch.float32, out=C["PyTorch"])

        def triton_fp8():
            grid = lambda meta: (m // meta["BM"], n // meta["BN"])
            triton_fp8_gemm[grid](A, B, C["Triton"], m, n, k, 1.0 / sa, 1.0 / sb, sa * sb)

        pipes = {"Y e4m3": y_e4m3, "Y fused": y_fused, "PyTorch": torch_fp8, "Triton": triton_fp8}
        for f in pipes.values():
            f()
        torch.cuda.synchronize()
        exact = {c: bool(torch.equal(C[c].double(), ref)) for c in C}
        # The parts: Y's quantised bytes are torch's, byte for byte.
        exact["Y quant A == torch"] = bool(torch.equal(a8, qa.view(torch.uint8)))
        exact["Y quant B == torch"] = bool(torch.equal(b8t, qb.t().view(torch.uint8)))
        cols = {c: f for c, f in pipes.items() if exact[c]}
        cols.update({"  Y quant A": y_quant_a, "  Y quant B (transposing)": y_quant_b, "  Y e4m3 GEMM": y_gemm,
                     "  torch quant A": torch_quant_a, "  torch quant B (transposing)": torch_quant_b,
                     "  torch _scaled_mm": torch_gemm})
        calls = max(2, min(100, int(5e-3 / max(2 * m * n * k / 120e12, 2e-6))))
        names = list(cols)
        samples = {c: [] for c in names}
        clocks = {c: [] for c in names}
        warm()
        for r in range(17):
            for c in names[r % len(names):] + names[:r % len(names)]:
                t = event_us(cols[c], calls)
                clk = torch.cuda.clock_rate(0)
                if r >= 2:
                    samples[c].append(t)
                    clocks[c].append(clk)
        mins = {c: min(s) for c, s in samples.items()}
        result["sizes"][size] = {"exact": exact, "calls": calls, "samples": samples, "sm_clock_mhz": clocks,
                                 "gemm_e4m3_tile": re.search(r"CTA \S+ \| \S+ warps \| \d+ stages", g8.ptx).group(0),
                                 "triton_config": str(getattr(triton_fp8_gemm, "best_config", None))}
        allclk = [x for v in clocks.values() for x in v]
        print(f"{size} | exact {exact} | SM {min(allclk)}-{max(allclk)} MHz | {result['sizes'][size]['gemm_e4m3_tile']}",
              flush=True)
        for c in names:
            print(f"   {c:30s} min {mins[c]:9.1f} median {statistics.median(samples[c]):9.1f} us"
                  + (f"  {mins[c] / mins['PyTorch']:.3f}x PyTorch" if not c.startswith(" ") and "PyTorch" in mins else ""),
                  flush=True)
    result["util_after"] = util()
    json.dump(result, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
