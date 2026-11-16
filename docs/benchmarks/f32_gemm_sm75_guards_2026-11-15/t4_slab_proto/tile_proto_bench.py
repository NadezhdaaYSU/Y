"""Runs on the T4: prototype small-tile in-order FP32 GEMM kernels (CUDA C via
NVRTC) for the shapes where Y's sm_75 tiles leave most SMs idle: few columns
at 4-16 rows, few tiles with a long K, one wave of small tiles. Each output
C[i][j] is one fmaf chain over k = 0..K-1 in order, checked bit for bit
against the oracle. A CTA computes a BM x BN tile with (BM/R)*(BN/C)
threads, R x C chains each, staging BK-deep slices of A and B through
shared memory (double buffered via registers; no cp.async below sm_80); the
K tail after the full slices reads global memory directly, still in order.
3 sweeps x 15 rounds, shuffled, medians of the per-round paired ratios
(cuBLAS time / kernel time) with quartiles. Reads YCUR (gzip JSON
{name: ptx}) for Y's current kernels. QUICK=1: 1 sweep x 2 rounds."""
import base64, gzip, json, os, random, re, statistics, subprocess, time
import cupy as cp, numpy as np, torch
torch.backends.cuda.matmul.allow_tf32 = False
YCUR = json.loads(gzip.decompress(base64.b64decode(open(os.environ.get("YCUR", "/kaggle/working/ycur.b64")).read())))
QUICK = os.environ.get("QUICK") == "1"
OUT = os.environ.get("OUT", "/kaggle/working/tile_proto_t4.json")
ORACLE = cp.RawKernel(r"""
extern "C" __global__ void oracle(const float* A, const float* B, float* C, int M, int N, int K) {
  int j = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y; if (j >= N) return;
  float acc = 0.f; for (int k = 0; k < K; k++) acc = fmaf(A[(size_t)i * K + k], B[(size_t)k * N + j], acc);
  C[(size_t)i * N + j] = acc; }""", "oracle")
SRC = r"""
#define TH ((BM_ / R_) * (BN_ / C_))
#define LA ((BM_ * BK_ + TH - 1) / TH)
#define LB ((BK_ * BN_ + TH - 1) / TH)
extern "C" __global__ void __launch_bounds__(TH) tile(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ Cm) {
  __shared__ float As[2][BK_][BM_];
  __shared__ float Bs[2][BK_][BN_];
  const int tid = threadIdx.x;
  const int c0 = blockIdx.x * BN_, r0 = blockIdx.y * BM_;
  const int cg = tid % (BN_ / C_), rg = tid / (BN_ / C_);
  float acc[R_][C_];
#pragma unroll
  for (int r = 0; r < R_; r++)
#pragma unroll
    for (int c = 0; c < C_; c++) acc[r][c] = 0.f;
  float ra[LA], rb[LB];
  const int nfull = KK / BK_;
#define LOAD(k0) do { \
  _Pragma("unroll") for (int i = 0; i < LA; i++) { int e = tid + i * TH; if (e < BM_ * BK_) { \
    int row = e % BM_, kk = e / BM_, gr = r0 + row; \
    ra[i] = gr < MM ? __ldg(A + (size_t)gr * KK + (k0) + kk) : 0.f; } } \
  _Pragma("unroll") for (int i = 0; i < LB; i++) { int e = tid + i * TH; if (e < BK_ * BN_) { \
    int col = e % BN_, kk = e / BN_, gc = c0 + col; \
    rb[i] = gc < NN ? __ldg(B + (size_t)((k0) + kk) * NN + gc) : 0.f; } } } while (0)
#define STORE(s) do { \
  _Pragma("unroll") for (int i = 0; i < LA; i++) { int e = tid + i * TH; if (e < BM_ * BK_) As[s][e / BM_][e % BM_] = ra[i]; } \
  _Pragma("unroll") for (int i = 0; i < LB; i++) { int e = tid + i * TH; if (e < BK_ * BN_) Bs[s][e / BN_][e % BN_] = rb[i]; } } while (0)
  if (nfull > 0) { LOAD(0); STORE(0); __syncthreads(); }
  for (int t = 0; t < nfull; t++) {
    const int s = t & 1;
    if (t + 1 < nfull) LOAD((t + 1) * BK_);
#pragma unroll
    for (int kk = 0; kk < BK_; kk++) {
      float a[R_], b[C_];
#pragma unroll
      for (int r = 0; r < R_; r++) a[r] = As[s][kk][rg * R_ + r];
#pragma unroll
      for (int c = 0; c < C_; c++) b[c] = Bs[s][kk][cg * C_ + c];
#pragma unroll
      for (int r = 0; r < R_; r++)
#pragma unroll
        for (int c = 0; c < C_; c++) acc[r][c] = fmaf(a[r], b[c], acc[r][c]);
    }
    if (t + 1 < nfull) STORE(s ^ 1);
    __syncthreads();
  }
  for (int k = nfull * BK_; k < KK; k++) {
    float a[R_], b[C_];
#pragma unroll
    for (int r = 0; r < R_; r++) { int gr = r0 + rg * R_ + r; a[r] = gr < MM ? __ldg(A + (size_t)gr * KK + k) : 0.f; }
#pragma unroll
    for (int c = 0; c < C_; c++) { int gc = c0 + cg * C_ + c; b[c] = gc < NN ? __ldg(B + (size_t)k * NN + gc) : 0.f; }
#pragma unroll
    for (int r = 0; r < R_; r++)
#pragma unroll
      for (int c = 0; c < C_; c++) acc[r][c] = fmaf(a[r], b[c], acc[r][c]);
  }
#pragma unroll
  for (int r = 0; r < R_; r++)
#pragma unroll
    for (int c = 0; c < C_; c++) {
      int gr = r0 + rg * R_ + r, gc = c0 + cg * C_ + c;
      if (gr < MM && gc < NN) Cm[(size_t)gr * NN + gc] = acc[r][c];
    }
}
"""
# name: (BM, BN, R, C, BK)
V = {"m1n32r1c1k128": (1, 32, 1, 1, 128), "m1n16r1c1k128": (1, 16, 1, 1, 128), "m2n32r1c1k64": (2, 32, 1, 1, 64),
     "m4n16r1c1k64": (4, 16, 1, 1, 64), "m4n32r1c1k64": (4, 32, 1, 1, 64), "m8n8r1c1k64": (8, 8, 1, 1, 64),
     "m8n16r2c1k64": (8, 16, 2, 1, 64), "m8n16r2c1k128": (8, 16, 2, 1, 128), "m8n32r2c2k64": (8, 32, 2, 2, 64),
     "m8n16r1c1k64": (8, 16, 1, 1, 64), "m16n16r4c1k32": (16, 16, 4, 1, 32), "m16n16r2c1k64": (16, 16, 2, 1, 64),
     "m16n8r2c1k64": (16, 8, 2, 1, 64), "m16n32r4c2k32": (16, 32, 4, 2, 32)}
R12 = ["m1n32r1c1k128", "m1n16r1c1k128", "m2n32r1c1k64", "m4n16r1c1k64", "m8n16r2c1k64"]
R4 = ["m4n16r1c1k64", "m4n32r1c1k64", "m8n16r2c1k64", "m8n16r2c1k128", "m8n8r1c1k64"]
R8 = ["m8n8r1c1k64", "m8n16r2c1k64", "m8n16r2c1k128", "m8n32r2c2k64", "m8n16r1c1k64", "m16n16r4c1k32"]
R16 = ["m8n16r2c1k64", "m16n16r4c1k32", "m16n16r2c1k64", "m16n8r2c1k64", "m8n16r2c1k128", "m16n32r4c2k32"]
SHAPES = [((1, 1024, 1024), R12), ((1, 896, 4864), R12), ((2, 768, 3072), R12),
          ((4, 896, 4864), R4), ((4, 4096, 4096), R4),
          ((6, 1024, 1024), R8), ((8, 896, 896), R8), ((8, 1024, 1024), R8), ((7, 4098, 1027), R8),
          ((11, 777, 4097), R16), ((12, 3072, 768), R16), ((16, 896, 4864), R16), ((16, 4096, 4096), R16)]
SWEEPS, ROUNDS = (1, 2) if QUICK else (3, 15)


def timed(fn, reps):
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        fn()
    e1.record(); e1.synchronize()
    return e0.elapsed_time(e1) / reps * 1e-3


def gpu_state():
    return subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.__version__, flush=True)
x = torch.randn(4096, 4096, device="cuda"); t0 = time.time()
while time.time() - t0 < (0.5 if QUICK else 3):
    x @ x
torch.cuda.synchronize()
rng = random.Random(13)
out = []
for (m, n, k), names in SHAPES:
    g = torch.Generator(device="cuda").manual_seed(m * 31 + n * 7 + k)
    A = torch.randn(m, k, device="cuda", generator=g); B = torch.randn(k, n, device="cuda", generator=g)
    want = cp.zeros((m, n), cp.float32)
    ORACLE(((n + 127) // 128, m, 1), (128, 1, 1), (cp.asarray(A), cp.asarray(B), want, np.int32(m), np.int32(n), np.int32(k)))
    cp.cuda.Device().synchronize()
    wantt = torch.as_tensor(want, device="cuda").view(torch.int32)
    fns, exact, regs = {}, {}, {}
    for v in names:
        bm, bn, r, c, bk = V[v]
        src = SRC
        for key, val in (("BM_", bm), ("BN_", bn), ("R_", r), ("C_", c), ("BK_", bk), ("MM", m), ("NN", n), ("KK", k)):
            src = src.replace(key, str(val))
        mod = cp.RawModule(code=src, options=("--fmad=false",))
        fn = mod.get_function("tile")
        regs[v] = fn.num_regs
        C = torch.full((m, n), float("nan"), device="cuda")
        args = (cp.asarray(A), cp.asarray(B), cp.asarray(C))
        th = (bm // r) * (bn // c)
        grid = ((n + bn - 1) // bn, (m + bm - 1) // bm, 1)
        launch = (lambda fn=fn, grid=grid, th=th, args=args: fn(grid, (th, 1, 1), args))
        launch(); torch.cuda.synchronize()
        exact[v] = bool((C.view(torch.int32) == wantt).all())
        fns[v] = launch
    name = f"g{m}_{n}_{k}"
    ptx = YCUR[name]
    path = f"/tmp/{name}.ptx"; open(path, "w").write(ptx)
    yfn = cp.RawModule(path=path).get_function(re.search(r"\.entry (\w+)\(", ptx).group(1))
    block = int(re.search(r"Launch: block \((\d+),", ptx).group(1))
    grid = int((re.search(r"\((\d+), 1, 1\) is one tile per CTA", ptx) or re.search(r"Grid \((\d+), 1, 1\)", ptx)).group(1))
    smem = int(re.search(r"Dynamic shared memory required: (\d+)", ptx).group(1))
    ytile = re.search(r"tile `(\w+)`", ptx).group(1)
    if smem > 48 * 1024:
        yfn.max_dynamic_shared_size_bytes = smem
    C = torch.full((m, n), float("nan"), device="cuda")
    yargs = (cp.asarray(A), cp.asarray(B), cp.asarray(C))
    ylaunch = (lambda: yfn((grid, 1, 1), (block, 1, 1), yargs, shared_mem=smem))
    ylaunch(); torch.cuda.synchronize()
    exact["ycur"] = bool((C.view(torch.int32) == wantt).all())
    fns["ycur"] = ylaunch
    cub = torch.empty(m, n, device="cuda")
    fns["cublas"] = (lambda: torch.mm(A, B, out=cub))
    t1 = timed(fns["cublas"], 3); reps = max(3, int(0.1 / t1))
    ratios = {v: [] for v in fns if v != "cublas"}
    cubt = []
    for sw in range(SWEEPS):
        for rd in range(ROUNDS):
            order = list(fns); rng.shuffle(order)
            t = {v: timed(fns[v], reps) for v in order}
            cubt.append(t["cublas"])
            for v in ratios:
                ratios[v].append(t["cublas"] / t[v])
    flops = 2 * m * n * k
    row = {"shape": f"{m}x{n}x{k}", "cublas_tflops": flops / statistics.median(cubt) / 1e12,
           "cublas_us": statistics.median(cubt) * 1e6, "ycur_tile": ytile, "regs": regs, "gpu": gpu_state()}
    line = f"{m}x{n}x{k:<6} cuBLAS {row['cublas_us']:.1f} us"
    for v, rs in ratios.items():
        q = statistics.quantiles(rs, n=4) if len(rs) >= 2 else [rs[0]] * 3
        row[v] = {"exact": exact[v], "ratio_median": statistics.median(rs), "q25": q[0], "q75": q[2]}
        tag = f"ycur({ytile})" if v == "ycur" else v
        line += f" | {tag} {statistics.median(rs):.3f}x [{q[0]:.2f}-{q[2]:.2f}]{'' if exact[v] else ' NOT EXACT'}"
    print(line, "|", row["gpu"].replace("\n", " "), flush=True)
    out.append(row)
json.dump(out, open(OUT, "w"), indent=1)
