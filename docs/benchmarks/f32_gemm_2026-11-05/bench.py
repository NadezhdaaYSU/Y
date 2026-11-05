"""Y's @tile FP32 GEMM against cuBLAS (TF32 off): bit-exactness, then speed.

Run from the repository root, after `cargo build --release`, with the GPU
otherwise idle:

    venv/bin/python docs/benchmarks/f32_gemm_2026-11-05/bench.py [MxNxK ... | ragged]

Each shape is compiled by target/release/Y in a temporary directory holding a
copy of the root's `.ysu_hw_profile` (the tile choice reads its SM count).
Correctness: every element equal BIT FOR BIT to an independent kernel that
folds `fmaf` over k = 0..K-1 (the order Y claims), on the header's grid and on
a 3x1 grid. Speed: 11 interleaved rounds, the minimum per side, each round
`reps` back-to-back launches (cuBLAS through torch.mm, Y through cupy).
Below ~512^3 the cupy launch itself dominates this timing; use ncu
(kernel_time_256.txt) there.
"""
import os, re, shutil, subprocess, sys, tempfile
import numpy as np, torch, cupy as cp

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
Y = os.path.join(ROOT, "target", "release", "Y")
WORK = tempfile.mkdtemp(prefix="y_f32_gemm_bench_")
if os.path.exists(os.path.join(ROOT, ".ysu_hw_profile")):
    shutil.copy(os.path.join(ROOT, ".ysu_hw_profile"), WORK)

ORACLE = cp.RawKernel(r'''
extern "C" __global__ void oracle(const float* A,const float* B,float* C,int M,int N,int K){
  int j=blockIdx.x*blockDim.x+threadIdx.x, i=blockIdx.y; if(j>=N) return;
  float acc=0.f; for(int k=0;k<K;k++) acc=fmaf(A[(size_t)i*K+k],B[(size_t)k*N+j],acc);
  C[(size_t)i*N+j]=acc; }''', "oracle")

def build(m, n, k):
    name = f"y_{m}_{n}_{k}"
    src = os.path.join(WORK, name + ".ysu")
    open(src, "w").write(f"@tile({m}, {n}, {k})\nkernel {name}(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>) {{\n}}\n\nfn main() {{\n}}\n")
    out = os.path.join(WORK, name + ".ptx")
    r = subprocess.run([Y, src, "--emit-ptx", "-o", out], cwd=WORK, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    ptx = open(out).read()
    tile = re.search(r"\[Y F32 SIMT GEMM\].*tile `(\w+)`", ptx).group(1)
    block = int(re.search(r"Launch: block \((\d+),1,1\)", ptx).group(1))
    g = re.search(r"\((\d+), (\d+), 1\) is one tile per CTA", ptx)
    smem = int(re.search(r"Dynamic shared memory required: (\d+) bytes", ptx).group(1))
    f = cp.RawModule(path=out).get_function(name)
    f.max_dynamic_shared_size_bytes = smem
    return f, tile, block, (int(g.group(1)), int(g.group(2)), 1), smem

def ev():
    return torch.cuda.Event(enable_timing=True)

DEFAULT = ["8192x8192x8192", "4096x4096x4096", "3072x3072x3072", "2048x2048x2048", "1536x1536x1536",
           "1024x1024x1024", "768x768x768", "512x512x512", "4096x4096x1024", "1024x4096x4096",
           "1024x1024x4096", "256x4096x4096", "128x4096x4096", "4096x128x4096", "64x8192x2048"]
RAGGED = ["1x4096x4096", "1x50257x768", "1000x1000x1000", "1023x1025x1027", "1024x1028x1024",
          "1500x1500x1500", "2047x2049x2051", "3000x3000x3000", "4095x4097x512", "1024x50257x768",
          "4096x50257x768", "4096x4097x4096"]
torch.manual_seed(0)
for spec in (RAGGED if sys.argv[1:] == ["ragged"] else sys.argv[1:] or DEFAULT):
    m, n, k = map(int, spec.split("x"))
    f, tile, block, grid, smem = build(m, n, k)
    a = torch.randn(m, k, device="cuda"); b = torch.randn(k, n, device="cuda")
    yc = torch.full((m, n), float("nan"), device="cuda"); oc = torch.full((m, n), float("nan"), device="cuda")
    run = lambda g=grid: f(g, (block, 1, 1), (cp.asarray(a), cp.asarray(b), cp.asarray(yc)), shared_mem=smem)
    ORACLE(((n + 127) // 128, m), (128,), (cp.asarray(a), cp.asarray(b), cp.asarray(oc), np.int32(m), np.int32(n), np.int32(k)))
    run(); torch.cuda.synchronize()
    exact = torch.equal(yc.view(torch.int32), oc.view(torch.int32))
    yc.fill_(float("nan")); run((3, 1, 1)); torch.cuda.synchronize()
    exact3 = torch.equal(yc.view(torch.int32), oc.view(torch.int32))
    del oc
    cb = torch.mm(a, b); reps = min(200, max(3, int(1e11 / (2 * m * n * k)))); best = {"cuBLAS": 1e9, "Y": 1e9}
    for _ in range(11):
        for side in best:
            s, e = ev(), ev(); s.record()
            for _ in range(reps):
                if side == "cuBLAS": torch.mm(a, b, out=cb)
                else: run()
            e.record(); e.synchronize(); best[side] = min(best[side], s.elapsed_time(e) / reps)
    tf = lambda side: 2 * m * n * k / (best[side] * 1e-3) / 1e12
    print(f"{m}x{n}x{k} tile {tile:5s}: bit-exact {exact} (grid 3x1: {exact3}) | cuBLAS {tf('cuBLAS'):5.2f} TF  Y {tf('Y'):5.2f} TF  = {best['cuBLAS'] / best['Y']:.3f}x", flush=True)
print(subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
shutil.rmtree(WORK, ignore_errors=True)
