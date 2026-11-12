"""Y's FP32 GEMM built for sm_75 (plain loads, no cp.async) and for sm_89
(cp.async), run on this GPU (any sm_75 or later card runs sm_75 PTX): every element equal bit for bit to the in-order
fmaf oracle, and the two builds equal to each other."""
import os, re, subprocess, sys, tempfile
import cupy as cp, numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
Y = os.path.join(ROOT, "target", "release", "Y")
ORACLE = cp.RawKernel(r'''
extern "C" __global__ void oracle(const float* A,const float* B,float* C,int M,int N,int K){
  int j=blockIdx.x*blockDim.x+threadIdx.x, i=blockIdx.y; if(j>=N) return;
  float acc=0.f; for(int k=0;k<K;k++) acc=fmaf(A[(size_t)i*K+k],B[(size_t)k*N+j],acc);
  C[(size_t)i*N+j]=acc; }''', "oracle")

def build(sm, sms, m, n, k):
    d = tempfile.mkdtemp(dir=os.environ.get("SCR", "/tmp"))
    open(f"{d}/.ysu_hw_profile", "w").write(f"SM_VERSION={sm}\nCOMPUTE_CAPABILITY={sm}\nGPU_NAME=Pinned\nSM_COUNT={sms}\n")
    name = f"y_{m}_{n}_{k}"
    open(f"{d}/{name}.ysu", "w").write(f"@tile({m}, {n}, {k})\nkernel {name}(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>) {{\n}}\n\nfn main() {{\n}}\n")
    r = subprocess.run([Y, f"{name}.ysu", "--emit-ptx", "-o", f"{name}.ptx"], cwd=d, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-1500:]
    ptx = open(f"{d}/{name}.ptx").read()
    block = int(re.search(r"Launch: block \((\d+),1,1\)", ptx).group(1))
    g = re.search(r"\((\d+), (\d+), 1\) is one tile per CTA", ptx)
    smem = int(re.search(r"Dynamic shared memory required: (\d+) bytes", ptx).group(1))
    tile = re.search(r"tile `(\w+)`", ptx).group(1)
    f = cp.RawModule(path=f"{d}/{name}.ptx").get_function(name)
    f.max_dynamic_shared_size_bytes = smem
    return f, block, (int(g.group(1)), int(g.group(2)), 1), smem, tile, "cp.async" in "\n".join(l for l in ptx.splitlines() if not l.strip().startswith("//"))

shapes = [(512, 512, 512), (1024, 1024, 1024), (1023, 1025, 1027), (2048, 2048, 2048), (100, 300, 77),
          (1, 4096, 4096), (4096, 1, 512), (65, 129, 33), (2049, 2049, 2048), (64, 8192, 2048)]
rng = np.random.default_rng(0)
bad = 0
for m, n, k in shapes:
    A = cp.asarray(rng.standard_normal((m, k), dtype=np.float32))
    B = cp.asarray(rng.standard_normal((k, n), dtype=np.float32))
    want = cp.zeros((m, n), cp.float32)
    ORACLE(((n + 127) // 128, m, 1), (128, 1, 1), (A, B, want, np.int32(m), np.int32(n), np.int32(k)))
    outs = {}
    for sm, sms in (("7.5", 40), ("8.9", 66)):
        f, block, grid, smem, tile, has_cp = build(sm, sms, m, n, k)
        for gname, gd in (("header grid", grid), ("3x1 grid", (3, 1, 1))):
            C = cp.full((m, n), cp.nan, cp.float32)
            f(gd, (block, 1, 1), (A, B, C), shared_mem=smem)
            cp.cuda.Device().synchronize()
            same = bool(cp.all(C.view(cp.uint32) == want.view(cp.uint32)))
            bad += not same
            outs[(sm, gname)] = C
            print(f"{m}x{n}x{k:<5} sm_{sm.replace('.', '')} {tile:6s} cp.async {str(has_cp):5s} {gname:11s} bit-exact {same}")
    bad += not bool(cp.all(outs[("7.5", "header grid")].view(cp.uint32) == outs[("8.9", "header grid")].view(cp.uint32)))
print("ALL BIT-EXACT" if bad == 0 else f"{bad} MISMATCHES")
