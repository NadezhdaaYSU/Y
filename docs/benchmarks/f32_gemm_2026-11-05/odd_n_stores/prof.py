"""Launch Y's FP32 GEMM at the given shapes a few times each, for ncu.

usage: prof.py MxNxK ...  (run under ncu --kernel-name regex:^y_)
Each kernel is checked bit-exact against an in-order fmaf oracle first.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

import cupy as cp
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
Y = os.path.join(ROOT, "target", "release", "Y")
WORK = tempfile.mkdtemp(prefix="y_f32_prof_")
shutil.copy(os.path.join(ROOT, ".ysu_hw_profile"), WORK)
ORACLE = cp.RawKernel(r'''
extern "C" __global__ void oracle(const float* A,const float* B,float* C,int M,int N,int K){
  int j=blockIdx.x*blockDim.x+threadIdx.x, i=blockIdx.y; if(j>=N) return;
  float acc=0.f; for(int k=0;k<K;k++) acc=fmaf(A[(size_t)i*K+k],B[(size_t)k*N+j],acc);
  C[(size_t)i*N+j]=acc; }''', "oracle")


def build(m, n, k, env=None):
    name = f"y_{m}_{n}_{k}"
    src = os.path.join(WORK, name + ".ysu")
    open(src, "w").write(f"@tile({m}, {n}, {k})\nkernel {name}(A: GlobalMemory<F32>, B: GlobalMemory<F32>, "
                         f"C: GlobalMemory<F32>) {{\n}}\n\nfn main() {{\n}}\n")
    out = os.path.join(WORK, name + ".ptx")
    r = subprocess.run([Y, src, "--emit-ptx", "-o", out], cwd=WORK, capture_output=True, text=True,
                       env={**os.environ, **(env or {})})
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    ptx = open(out).read()
    block = int(re.search(r"Launch: block \((\d+),1,1\)", ptx).group(1))
    g = re.search(r"\((\d+), (\d+), 1\) is one tile per CTA", ptx)
    smem = int(re.search(r"Dynamic shared memory required: (\d+) bytes", ptx).group(1))
    f = cp.RawModule(path=out).get_function(name)
    f.max_dynamic_shared_size_bytes = smem
    return f, block, (int(g.group(1)), int(g.group(2)), 1), smem


for spec in sys.argv[1:]:
    m, n, k = map(int, spec.split("x"))
    f, block, grid, smem = build(m, n, k)
    a = torch.randn(m, k, device="cuda")
    b = torch.randn(k, n, device="cuda")
    yc = torch.full((m, n), float("nan"), device="cuda")
    oc = torch.full((m, n), float("nan"), device="cuda")
    ORACLE(((n + 127) // 128, m), (128,), (cp.asarray(a), cp.asarray(b), cp.asarray(oc), np.int32(m), np.int32(n), np.int32(k)))
    for _ in range(3):
        f(grid, (block, 1, 1), (cp.asarray(a), cp.asarray(b), cp.asarray(yc)), shared_mem=smem)
    torch.cuda.synchronize()
    assert torch.equal(yc.view(torch.int32), oc.view(torch.int32)), spec
    print(spec, "exact", flush=True)
