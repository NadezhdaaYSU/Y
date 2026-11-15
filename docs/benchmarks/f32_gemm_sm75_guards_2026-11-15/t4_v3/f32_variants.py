"""The script T4 job y-f32-t4 version 3 ran (PTX variants loaded with cupy; see the README)."""

import base64, gzip, json, os, random, re, sys, time
import cupy as cp, numpy as np, torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
V = json.loads(gzip.decompress(base64.b64decode(open("/kaggle/working/variants.b64").read())))
ORACLE = cp.RawKernel(r"""
extern "C" __global__ void oracle(const float* A, const float* B, float* C, int M, int N, int K) {
  int j = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y; if (j >= N) return;
  float acc = 0.f; for (int k = 0; k < K; k++) acc = fmaf(A[(size_t)i * K + k], B[(size_t)k * N + j], acc);
  C[(size_t)i * N + j] = acc; }""", "oracle")
os.makedirs("/tmp/ptx", exist_ok=True)
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.__version__, flush=True)
x = torch.randn(4096, 4096, device="cuda"); t0 = time.time()
while time.time() - t0 < 3: x @ x
torch.cuda.synchronize()
names = sorted(V); shapes = sorted(V[names[0]], key=lambda s: [int(v) for v in s[1:].split("_")])
rng = random.Random(7); results = []
def timed(fn, reps):
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps): fn()
    e1.record(); e1.synchronize()
    return e0.elapsed_time(e1) / reps * 1e-3
for sh in shapes:
    m, n, k = map(int, sh[1:].split("_"))
    g = torch.Generator(device="cuda").manual_seed(m * 31 + n * 7 + k)
    A = torch.randn(m, k, device="cuda", generator=g); B = torch.randn(k, n, device="cuda", generator=g)
    want = cp.zeros((m, n), cp.float32)
    ORACLE(((n + 127) // 128, m, 1), (128, 1, 1), (cp.asarray(A), cp.asarray(B), want, np.int32(m), np.int32(n), np.int32(k)))
    cp.cuda.Device().synchronize()
    wantt = torch.as_tensor(want, device="cuda")
    fns, row = {}, {"shape": f"{m}x{n}x{k}"}
    for v in names:
        ptx = V[v][sh]
        path = f"/tmp/ptx/{v}_{sh}.ptx"; open(path, "w").write(ptx)
        mod = cp.RawModule(path=path)
        fn = mod.get_function(sh)
        block = int(re.search(r"Launch: block \((\d+),", ptx).group(1))
        grid = int(re.search(r"\((\d+), 1, 1\) is one tile per CTA", ptx).group(1))
        smem = int(re.search(r"Dynamic shared memory required: (\d+)", ptx).group(1))
        if smem > 48 * 1024: fn.max_dynamic_shared_size_bytes = smem
        tile = re.search(r"tile `(\w+)`", ptx).group(1)
        C = torch.full((m, n), float("nan"), device="cuda")
        args = (cp.asarray(A), cp.asarray(B), cp.asarray(C))
        launch = (lambda fn=fn, grid=grid, block=block, args=args, smem=smem:
                  fn((grid, 1, 1), (block, 1, 1), args, shared_mem=smem))
        launch(); torch.cuda.synchronize()
        exact = bool((C.view(torch.int32) == wantt.view(torch.int32)).all())
        row[v] = {"tile": tile, "smem": smem, "exact": exact}
        fns[v] = launch
    cub = torch.empty(m, n, device="cuda")
    fns["cublas"] = lambda: torch.mm(A, B, out=cub)
    flops = 2.0 * m * n * k; reps = max(3, min(200, int(2e11 / flops)))
    ts = {s: [] for s in fns}
    for r in range(11):
        order = list(fns); rng.shuffle(order)
        for s in order: ts[s].append(timed(fns[s], reps))
    c = min(ts["cublas"])
    row["cublas_tflops"] = flops / c / 1e12
    for v in names:
        row[v]["tflops"] = flops / min(ts[v]) / 1e12; row[v]["vs_cublas"] = c / min(ts[v])
    results.append(row)
    print(f"{row['shape']:16s} cuBLAS {row['cublas_tflops']:5.2f} TF | " + " | ".join(
        f"{v} {row[v]['vs_cublas']:.3f}x{'' if row[v]['exact'] else ' NOT EXACT'}" for v in names), flush=True)
json.dump(results, open("/kaggle/working/f32_variants_t4.json", "w"), indent=1)
