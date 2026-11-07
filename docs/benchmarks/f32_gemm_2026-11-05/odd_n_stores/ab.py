"""Old against new FP32 GEMM kernels, and both against cuBLAS (TF32 off).

usage: ab.py Y_OLD Y_NEW MxNxK ...

Per shape: both kernels must equal an in-order fmaf oracle bit for bit (C
poisoned with NaN first), and each other. Then 15 rounds, the three in a
shuffled order each round, each a batch of back-to-back launches timed by
CUDA events; the minimum per side.
"""
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile

import cupy as cp
import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
WORK = tempfile.mkdtemp(prefix="y_f32_ab_")
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
    y_old, y_new = sys.argv[1], sys.argv[2]
    rng = random.Random(0)
    torch.manual_seed(0)
    for spec in sys.argv[3:]:
        m, n, k = map(int, spec.split("x"))
        ks = {"old": build(y_old, "old", m, n, k), "new": build(y_new, "new", m, n, k)}
        a = torch.randn(m, k, device="cuda")
        b = torch.randn(k, n, device="cuda")
        oc = torch.full((m, n), float("nan"), device="cuda")
        ORACLE(((n + 127) // 128, m), (128,), (cp.asarray(a), cp.asarray(b), cp.asarray(oc), np.int32(m), np.int32(n), np.int32(k)))
        outs = {}
        for side, (f, block, grid, smem) in ks.items():
            c = torch.full((m, n), float("nan"), device="cuda")
            f(grid, (block, 1, 1), (cp.asarray(a), cp.asarray(b), cp.asarray(c)), shared_mem=smem)
            torch.cuda.synchronize()
            assert torch.equal(c.view(torch.int32), oc.view(torch.int32)), f"{spec} {side}: not the oracle's bits"
            outs[side] = c
        same = torch.equal(outs["old"].view(torch.int32), outs["new"].view(torch.int32))
        del oc
        c = outs["new"]
        A, B, C = cp.asarray(a), cp.asarray(b), cp.asarray(c)
        runs = {"cuBLAS": lambda: torch.mm(a, b)}
        for side, (f, block, grid, smem) in ks.items():
            runs[side] = (lambda f=f, block=block, grid=grid, smem=smem: f(grid, (block, 1, 1), (A, B, C), shared_mem=smem))
        reps = min(200, max(3, int(1e11 / (2 * m * n * k))))
        best = {s: float("inf") for s in runs}
        for _ in range(15):
            order = list(runs)
            rng.shuffle(order)
            for s in order:
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(reps):
                    runs[s]()
                e1.record()
                e1.synchronize()
                best[s] = min(best[s], e0.elapsed_time(e1) / reps)
        print(f"{spec}: old==new bits {same} | old {best['cuBLAS'] / best['old']:.3f}x  new {best['cuBLAS'] / best['new']:.3f}x"
              f"  (new/old time {best['new'] / best['old']:.3f})", flush=True)
    print("# " + subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu",
                                 "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())


if __name__ == "__main__":
    main()
