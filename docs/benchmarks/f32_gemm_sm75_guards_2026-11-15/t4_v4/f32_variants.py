"""Runs on the T4: sm_75 PTX variants of Y's FP32 GEMM against cuBLAS.
Reads /kaggle/working/variants.b64 (gzip JSON {variant: {name: ptx}}).
3 sweeps x 15 rounds, order shuffled each round, each sample about 0.1 s of
launches; per shape and variant: the median of the per-round ratios (cuBLAS
time / Y time) with quartiles, every element checked against an fmaf chain."""
import base64, gzip, json, os, random, re, statistics, subprocess, sys, time
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
SWEEPS, ROUNDS = 3, 15


def gpu_state():
    return subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


def timed(fn, reps):
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        fn()
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1) / reps * 1e-3


print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.__version__, flush=True)
x = torch.randn(4096, 4096, device="cuda")
t0 = time.time()
while time.time() - t0 < 3:
    x @ x
torch.cuda.synchronize()
names = sorted(V)
shapes = sorted(V[names[0]], key=lambda s: [int(v) for v in s[1:].split("_")])
rng = random.Random(7)
setup = {}
for sh in shapes:
    m, n, k = map(int, sh[1:].split("_"))
    g = torch.Generator(device="cuda").manual_seed(m * 31 + n * 7 + k)
    A = torch.randn(m, k, device="cuda", generator=g)
    B = torch.randn(k, n, device="cuda", generator=g)
    want = cp.zeros((m, n), cp.float32)
    ORACLE(((n + 127) // 128, m, 1), (128, 1, 1), (cp.asarray(A), cp.asarray(B), want, np.int32(m), np.int32(n), np.int32(k)))
    cp.cuda.Device().synchronize()
    wantt = torch.as_tensor(want, device="cuda")
    fns, info = {}, {}
    for v in names:
        ptx = V[v][sh]
        path = f"/tmp/ptx/{v}_{sh}.ptx"
        open(path, "w").write(ptx)
        fn = cp.RawModule(path=path).get_function(sh)
        block = int(re.search(r"Launch: block \((\d+),", ptx).group(1))
        grid = int(re.search(r"\((\d+), 1, 1\) is one tile per CTA", ptx).group(1))
        smem = int(re.search(r"Dynamic shared memory required: (\d+)", ptx).group(1))
        if smem > 48 * 1024:
            fn.max_dynamic_shared_size_bytes = smem
        C = torch.full((m, n), float("nan"), device="cuda")
        args = (cp.asarray(A), cp.asarray(B), cp.asarray(C))
        launch = (lambda fn=fn, grid=grid, block=block, args=args, smem=smem:
                  fn((grid, 1, 1), (block, 1, 1), args, shared_mem=smem))
        launch()
        torch.cuda.synchronize()
        info[v] = {"tile": re.search(r"tile `(\w+)`", ptx).group(1), "smem": smem,
                   "exact": bool((C.view(torch.int32) == wantt.view(torch.int32)).all())}
        fns[v] = launch
    cub = torch.empty(m, n, device="cuda")
    fns["cublas"] = (lambda A=A, B=B, cub=cub: torch.mm(A, B, out=cub))
    flops = 2.0 * m * n * k
    setup[sh] = (fns, info, flops, max(3, min(400, int(4e11 / flops))))
results = {sh: {"samples": {s: [] for s in setup[sh][0]}, "gpu": []} for sh in shapes}
for sweep in range(SWEEPS):
    for sh in shapes:
        fns, info, flops, reps = setup[sh]
        results[sh]["gpu"].append(gpu_state())
        for r in range(ROUNDS):
            order = list(fns)
            rng.shuffle(order)
            for s in order:
                results[sh]["samples"][s].append(timed(fns[s], reps))
        results[sh]["gpu"].append(gpu_state())
out = []
for sh in shapes:
    fns, info, flops, reps = setup[sh]
    S, G = results[sh]["samples"], results[sh]["gpu"]
    row = {"shape": sh[1:].replace("_", "x"), "reps": reps, "gpu": G,
           "cublas_tflops_median": flops / statistics.median(S["cublas"]) / 1e12}
    for v in names:
        ratios = sorted(c / y for c, y in zip(S["cublas"], S[v]))
        q = lambda f: ratios[int(f * (len(ratios) - 1))]
        row[v] = dict(info[v], tflops_median=flops / statistics.median(S[v]) / 1e12,
                      ratio_median=statistics.median(ratios), ratio_q25=q(0.25), ratio_q75=q(0.75))
    out.append(row)
    print(f"{row['shape']:16s} cuBLAS {row['cublas_tflops_median']:5.2f} TF | " + " | ".join(
        f"{v} {row[v]['ratio_median']:.3f}x [{row[v]['ratio_q25']:.3f}-{row[v]['ratio_q75']:.3f}] {row[v]['tflops_median']:.2f} TF"
        + ("" if row[v]["exact"] else " NOT EXACT") for v in names) + f" | {G[0]} -> {G[-1]}", flush=True)
json.dump(out, open("/kaggle/working/f32_variants_t4.json", "w"), indent=1)
