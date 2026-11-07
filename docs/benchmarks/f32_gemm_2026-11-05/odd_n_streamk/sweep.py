"""FP32 GEMM shapes, Y builds and cuBLAS (TF32 off), all interleaved.

usage: sweep.py [--rounds R] Y_BIN[=TAG] [Y_BIN[=TAG] ...] -- MxNxK ...

Every shape is compiled by every Y build. Each kernel's output must equal an
in-order fmaf oracle bit for bit (C poisoned with NaN first). The GPU is then
warmed for 3 s, and each round times every (shape, side) pair once, in an
order shuffled per round; each timing is a batch of back-to-back launches
(at least ~3 ms) between CUDA events. Reported: the minimum per pair, as us
per call and as cuBLAS time / that time.
"""
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time

import cupy as cp
import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
WORK = tempfile.mkdtemp(prefix="y_f32_sweep_")
shutil.copy(os.path.join(ROOT, ".ysu_hw_profile"), WORK)
ORACLE = cp.RawKernel(r'''
extern "C" __global__ void oracle(const float* A,const float* B,float* C,int M,int N,int K){
  int j=blockIdx.x*blockDim.x+threadIdx.x, i=blockIdx.y; if(j>=N) return;
  float acc=0.f; for(int k=0;k<K;k++) acc=fmaf(A[(size_t)i*K+k],B[(size_t)k*N+j],acc);
  C[(size_t)i*N+j]=acc; }''', "oracle")


def build(ybin, tag, m, n, k):
    name = f"y_{tag}_{m}_{n}_{k}"
    src = os.path.join(WORK, name + ".ysu")
    open(src, "w").write(f"@tile({m}, {n}, {k})\nkernel {name}(A: GlobalMemory<F32>, B: GlobalMemory<F32>, "
                         f"C: GlobalMemory<F32>) {{\n}}\n\nfn main() {{\n}}\n")
    out = os.path.join(WORK, name + ".ptx")
    r = subprocess.run([ybin, src, "--emit-ptx", "-o", out], cwd=WORK, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    ptx = open(out).read()
    block = int(re.search(r"Launch: block \((\d+),1,1\)", ptx).group(1))
    g = re.search(r"\((\d+), (\d+), 1\) is one tile per CTA", ptx)
    smem = int(re.search(r"Dynamic shared memory required: (\d+) bytes", ptx).group(1))
    f = cp.RawModule(path=out).get_function(name)
    f.max_dynamic_shared_size_bytes = smem
    return f, block, (int(g.group(1)), int(g.group(2)), 1), smem


def main():
    args = sys.argv[1:]
    rounds = 15
    if args[0] == "--rounds":
        rounds, args = int(args[1]), args[2:]
    cut = args.index("--")
    builds = []
    for i, spec in enumerate(args[:cut]):
        path, _, tag = spec.partition("=")
        builds.append((tag or f"y{i}", path))
    shapes = [tuple(map(int, s.split("x"))) for s in args[cut + 1:]]
    torch.manual_seed(0)
    runs = {}
    for (m, n, k) in shapes:
        a = torch.randn(m, k, device="cuda")
        b = torch.randn(k, n, device="cuda")
        oc = torch.full((m, n), float("nan"), device="cuda")
        ORACLE(((n + 127) // 128, m), (128,), (cp.asarray(a), cp.asarray(b), cp.asarray(oc),
                                               np.int32(m), np.int32(n), np.int32(k)))
        c = torch.full((m, n), float("nan"), device="cuda")
        A, B, C = cp.asarray(a), cp.asarray(b), cp.asarray(c)
        reps = max(3, int(3e-3 / (2 * m * n * k / 20e12)))
        runs[(m, n, k, "cuBLAS")] = (lambda a=a, b=b: torch.mm(a, b), reps)
        for tag, path in builds:
            f, block, grid, smem = build(path, tag, m, n, k)
            c.fill_(float("nan"))
            f(grid, (block, 1, 1), (A, B, C), shared_mem=smem)
            torch.cuda.synchronize()
            assert torch.equal(c.view(torch.int32), oc.view(torch.int32)), f"{m}x{n}x{k} {tag}: not the oracle's bits"
            runs[(m, n, k, tag)] = (lambda f=f, block=block, grid=grid, smem=smem, A=A, B=B, C=C:
                                    f(grid, (block, 1, 1), (A, B, C), shared_mem=smem), reps)
        del oc
    print(f"all {len(shapes) * len(builds)} kernels equal the in-order oracle bit for bit", flush=True)
    t_end = time.time() + 3
    while time.time() < t_end:
        for fn, reps in runs.values():
            fn()
        torch.cuda.synchronize()
    rng = random.Random(0)
    best = {key: float("inf") for key in runs}
    for _ in range(rounds):
        order = list(runs)
        rng.shuffle(order)
        for key in order:
            fn, reps = runs[key]
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(reps):
                fn()
            e1.record()
            e1.synchronize()
            best[key] = min(best[key], e0.elapsed_time(e1) / reps)
    for (m, n, k) in shapes:
        cb = best[(m, n, k, "cuBLAS")]
        cols = "  ".join(f"{tag} {best[(m, n, k, tag)] * 1e3:8.1f} us {cb / best[(m, n, k, tag)]:.3f}x"
                         for tag, _ in builds)
        print(f"{m}x{n}x{k:<6} cuBLAS {cb * 1e3:8.1f} us  {cols}", flush=True)
    print("# " + subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu",
                                 "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())


if __name__ == "__main__":
    main()
