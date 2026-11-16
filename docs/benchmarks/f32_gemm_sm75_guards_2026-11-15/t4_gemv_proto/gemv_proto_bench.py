"""Runs on the T4: a prototype in-order FP32 GEMV-style kernel (CUDA C via
NVRTC) for 1-8 rows, against cuBLAS and Y's current PTX for the same shape.
Each output C[i][j] is one fmaf chain over k = 0..K-1 in order (checked bit
for bit against the same oracle the other benches use). 3 sweeps x 15
rounds, shuffled, medians of the per-round paired ratios (cuBLAS time /
kernel time) with quartiles. Reads /kaggle/working/ycur.b64 (gzip JSON
{name: ptx}) for Y's current kernels."""
import base64, gzip, json, random, re, statistics, subprocess, time
import cupy as cp, numpy as np, torch
torch.backends.cuda.matmul.allow_tf32 = False
YCUR = json.loads(gzip.decompress(base64.b64decode(open("/kaggle/working/ycur.b64").read())))
ORACLE = cp.RawKernel(r"""
extern "C" __global__ void oracle(const float* A, const float* B, float* C, int M, int N, int K) {
  int j = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y; if (j >= N) return;
  float acc = 0.f; for (int k = 0; k < K; k++) acc = fmaf(A[(size_t)i * K + k], B[(size_t)k * N + j], acc);
  C[(size_t)i * N + j] = acc; }""", "oracle")
SRC = r"""
extern "C" __global__ void __launch_bounds__(TT) gemv(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
  const int j = blockIdx.x * TT + threadIdx.x;
  if (j >= NN) return;
  float acc[MM];
#pragma unroll
  for (int i = 0; i < MM; i++) acc[i] = 0.f;
  const float* b = B + j;
  int k = 0;
  for (; k + UU <= KK; k += UU) {
    float bv[UU];
#pragma unroll
    for (int u = 0; u < UU; u++) bv[u] = __ldg(b + (size_t)(k + u) * NN);
#pragma unroll
    for (int u = 0; u < UU; u++) {
#pragma unroll
      for (int i = 0; i < MM; i++) acc[i] = fmaf(__ldg(A + (size_t)i * KK + k + u), bv[u], acc[i]);
    }
  }
  for (; k < KK; k++) {
    float bk = __ldg(b + (size_t)k * NN);
#pragma unroll
    for (int i = 0; i < MM; i++) acc[i] = fmaf(__ldg(A + (size_t)i * KK + k), bk, acc[i]);
  }
#pragma unroll
  for (int i = 0; i < MM; i++) C[(size_t)i * NN + j] = acc[i];
}
"""
SHAPES = [(1, 1024, 1024), (1, 896, 4864), (4, 896, 4864), (8, 896, 896), (1, 4864, 896), (1, 4096, 4096),
          (1, 50257, 768), (1, 30522, 768), (2, 4096, 11008), (8, 4096, 4096), (4, 4096, 4096), (8, 1024, 1024)]
VARIANTS = [(32, 16), (32, 32), (64, 32), (128, 16), (128, 32)]   # (threads, k unroll)
SWEEPS, ROUNDS = 3, 15


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
while time.time() - t0 < 3:
    x @ x
torch.cuda.synchronize()
rng = random.Random(11)
out = []
for (m, n, k) in SHAPES:
    g = torch.Generator(device="cuda").manual_seed(m * 31 + n * 7 + k)
    A = torch.randn(m, k, device="cuda", generator=g); B = torch.randn(k, n, device="cuda", generator=g)
    want = cp.zeros((m, n), cp.float32)
    ORACLE(((n + 127) // 128, m, 1), (128, 1, 1), (cp.asarray(A), cp.asarray(B), want, np.int32(m), np.int32(n), np.int32(k)))
    cp.cuda.Device().synchronize()
    wantt = torch.as_tensor(want, device="cuda").view(torch.int32)
    fns, exact = {}, {}
    for (tt, uu) in VARIANTS:
        src = SRC.replace("TT", str(tt)).replace("UU", str(uu)).replace("MM", str(m)).replace("NN", str(n)).replace("KK", str(k))
        fn = cp.RawModule(code=src, options=("--fmad=false",)).get_function("gemv")
        C = torch.full((m, n), float("nan"), device="cuda")
        args = (cp.asarray(A), cp.asarray(B), cp.asarray(C))
        launch = (lambda fn=fn, tt=tt, args=args: fn(((n + tt - 1) // tt, 1, 1), (tt, 1, 1), args))
        launch(); torch.cuda.synchronize()
        v = f"t{tt}u{uu}"
        exact[v] = bool((C.view(torch.int32) == wantt).all())
        fns[v] = launch
    name = f"g{m}_{n}_{k}"
    ptx = YCUR[name]
    path = f"/tmp/{name}.ptx"; open(path, "w").write(ptx)
    yfn = cp.RawModule(path=path).get_function(re.search(r"\.entry (\w+)\(", ptx).group(1))
    block = int(re.search(r"Launch: block \((\d+),", ptx).group(1))
    grid = int((re.search(r"\((\d+), 1, 1\) is one tile per CTA", ptx) or re.search(r"Grid \((\d+), 1, 1\)", ptx)).group(1))
    smem = int(re.search(r"Dynamic shared memory required: (\d+)", ptx).group(1))
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
        for r in range(ROUNDS):
            order = list(fns); rng.shuffle(order)
            t = {v: timed(fns[v], reps) for v in order}
            cubt.append(t["cublas"])
            for v in ratios:
                ratios[v].append(t["cublas"] / t[v])
    flops = 2 * m * n * k
    row = {"shape": f"{m}x{n}x{k}", "cublas_tflops": flops / statistics.median(cubt) / 1e12, "gpu": gpu_state()}
    line = f"{m}x{n}x{k:<6} cuBLAS {row['cublas_tflops']:.3f} TF"
    for v, rs in ratios.items():
        q = statistics.quantiles(rs, n=4)
        row[v] = {"exact": exact[v], "ratio_median": statistics.median(rs), "q25": q[0], "q75": q[2]}
        line += f" | {v} {statistics.median(rs):.3f}x [{q[0]:.2f}-{q[2]:.2f}]{'' if exact[v] else ' NOT EXACT'}"
    print(line, "|", row["gpu"].replace("\n", " "), flush=True)
    out.append(row)
json.dump(out, open("/kaggle/working/gemv_proto_t4.json", "w"), indent=1)
