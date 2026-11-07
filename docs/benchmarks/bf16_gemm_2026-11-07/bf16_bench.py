# Y's BF16 GEMM (C[M][N] f32 = A[M][K] bf16 @ B[K][N] bf16, row-major) against
# cuBLAS's: torch.mm(A, B, out_dtype=torch.float32), the same layout, bf16
# inputs, f32 accumulation and f32 output.
#
# Correctness first, at every shape: small integers in [-8, 8], where every
# product and partial sum is exact, so Y must equal the integer product bit for
# bit (C poisoned with NaN first). Then, on normal random inputs, the share
# of outputs bit-identical to cuBLAS's, and each one's largest error against
# a float64 product in units of sum |a||b|.
#
# Timing: per-call time, the minimum over 15 rounds, the order of the two
# shuffled every round, each a batch of 3 to 500 calls sized to about 20 ms
# at 80 TFLOPS.
#
# Usage, from the repo root: python3 docs/benchmarks/bf16_gemm_2026-11-07/bf16_bench.py MxNxK ...
# (the PTX is compiled against $Y_PROFILE, default the repo's .ysu_hw_profile)
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
sys.path.insert(0, os.path.join(REPO, "python"))

import torch                                       # noqa: E402

from y_lang.torch_interop import TorchKernel       # noqa: E402

PROFILE = os.environ.get("Y_PROFILE", os.path.join(REPO, ".ysu_hw_profile"))


def build(m, n, k, workdir):
    src = (f"@tile({m}, {n}, {k})\nkernel g(A: GlobalMemory<BF16>, B: GlobalMemory<BF16>, "
           f"C: GlobalMemory<F32>) {{\n}}\n")
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        kern = TorchKernel(src, kernel_name="g", target_sm="sm_89")
    finally:
        os.chdir(cwd)
    cm, cn, wm, wn = (int(v) for v in re.search(r"CTA (\d+)x(\d+)x\d+ \| (\d+)x(\d+) warps", kern.ptx).groups())
    tile = re.search(r"CTA (\d+x\d+x\d+) \| (\d+x\d+) warps", kern.ptx)
    stages = re.search(r"(\d+) used after", kern.ptx)
    desc = f"{tile.group(1)} {tile.group(2)}w s{stages.group(1) if stages else '?'}"
    return kern, ((n + cn - 1) // cn, (m + cm - 1) // cm, 1), (wm * wn * 32, 1, 1), desc


def main():
    shapes = [tuple(map(int, s.split("x"))) for s in sys.argv[1:]]
    workdir = tempfile.mkdtemp(prefix="y_bf16_bench_")
    shutil.copy(PROFILE, os.path.join(workdir, ".ysu_hw_profile"))
    props = torch.cuda.get_device_properties(0)
    print(f"# {props.name}, {props.multi_processor_count} SMs; torch {torch.__version__}; "
          f"profile {PROFILE}", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    rng = random.Random(0)
    g = torch.Generator(device="cuda").manual_seed(0)
    for m, n, k in shapes:
        kern, grid, block, desc = build(m, n, k, workdir)
        c = torch.empty(m, n, device="cuda")

        def y(a, b, c=c):
            kern.launch(grid, block, [a, b, c])
            return c

        ai = torch.randint(-8, 9, (m, k), device="cuda", generator=g).to(torch.bfloat16)
        bi = torch.randint(-8, 9, (k, n), device="cuda", generator=g).to(torch.bfloat16)
        c.fill_(float("nan"))
        y(ai, bi)
        torch.cuda.synchronize()
        assert torch.equal(c.double(), ai.double() @ bi.double()), f"{m}x{n}x{k}: not the exact product"
        a = torch.randn((m, k), device="cuda", generator=g).bfloat16()
        b = torch.randn((k, n), device="cuda", generator=g).bfloat16()
        cy = y(a, b).clone()
        cc = torch.mm(a, b, out_dtype=torch.float32)
        same = float((cy.view(torch.int32) == cc.view(torch.int32)).float().mean())
        ref = a.double() @ b.double()
        scale = a.double().abs() @ b.double().abs()
        ey = float(((cy.double() - ref).abs() / scale).max())
        ec = float(((cc.double() - ref).abs() / scale).max())
        runs = {"cuBLAS": lambda: torch.mm(a, b, out_dtype=torch.float32), "Y": lambda: y(a, b)}
        reps = max(3, min(500, int(0.02 * 8e13 / (2 * m * n * k))))   # ~20 ms at 80 TFLOPS
        best = {name: float("inf") for name in runs}
        for _ in range(15):
            order = list(runs.items())
            rng.shuffle(order)
            for name, fn in order:
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(reps):
                    fn()
                e.record()
                e.synchronize()
                best[name] = min(best[name], s.elapsed_time(e) / reps)
        t0 = best["cuBLAS"]
        print(f"{m}x{n}x{k}  cuBLAS {t0 * 1e3:9.1f} us {2 * m * n * k / t0 / 1e9:6.1f} TFLOPS | "
              f"Y {desc} {t0 / best['Y']:.3f}x | same bits as cuBLAS {100 * same:5.1f}% | "
              f"max err/sum|a||b|: Y {ey:.1e} cuBLAS {ec:.1e} | integers exact", flush=True)
    print("# " + subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu",
                                 "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
    shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
