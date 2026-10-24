"""power.py N file.ptx: each of Y's kernel and cuBLAS, back to back for ~3 s per data type (small
integers, N(0,1)), with nvidia-smi sampling the SM clock and board power every 50 ms; per run the
mean clock, mean power and the per-call time from CUDA events. Two passes, interleaved."""
import re, subprocess, sys, time, statistics, torch
from y_lang.torch_interop import TorchKernel
n, path = int(sys.argv[1]), sys.argv[2]
torch.zeros(1, device="cuda")
ptx = open(path).read()
name = re.findall(r"\.visible \.entry (\w+)\(", ptx)[0]
cm, cn, wm, wn = (int(x) for x in re.search(r"CTA (\d+)x(\d+)x\d+ \| (\d+)x(\d+) warps", ptx).groups())
k = TorchKernel.__new__(TorchKernel)
k.source, k.kernel_name, k.target_sm = None, name, "sm_89"
k.ptx = ptx; k._module = k._function = None; k._ensure_loaded()
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
data = {"int": (torch.randint(-4, 5, (n, n), device="cuda").half(), torch.randint(-4, 5, (n, n), device="cuda").half()),
        "normal": (torch.randn(n, n, device="cuda").half(), torch.randn(n, n, device="cuda").half())}
c = torch.empty(n, n, device="cuda")
def y(a, b): k.launch((-(-n // cn), -(-n // cm), 1), (wm * wn * 32, 1, 1), [a, b, c])
def cub(a, b): torch.mm(a, b, out_dtype=torch.float32)
for f in (y, cub):
    f(*data["int"])
torch.cuda.synchronize()
for p in (1, 2):
    for d in ("int", "normal"):
        for label, f in (("Y", y), ("cuBLAS", cub)) if p == 1 else (("cuBLAS", cub), ("Y", y)):
            a, b = data[d]
            for _ in range(200): f(a, b)          # ~0.3 s warm-up at this load
            torch.cuda.synchronize()
            smi = subprocess.Popen(["nvidia-smi", "--query-gpu=clocks.sm,power.draw", "--format=csv,noheader,nounits", "-lms", "50"],
                                   stdout=subprocess.PIPE, text=True)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); calls = 0; t0 = time.time()
            while time.time() - t0 < 3.0:
                for _ in range(20): f(a, b)
                calls += 20
                torch.cuda.synchronize()
            e.record(); e.synchronize()
            smi.terminate(); out = smi.communicate()[0]
            rows = [tuple(float(x) for x in l.split(",")) for l in out.strip().splitlines() if l.strip()]
            rows = rows[2:-1] if len(rows) > 4 else rows
            print(f"pass {p} {d:6s} {label:6s}: {s.elapsed_time(e) * 1000 / calls:8.1f} us/call, SM clock {statistics.mean(r[0] for r in rows):6.0f} MHz "
                  f"(min {min(r[0] for r in rows):.0f}), power {statistics.mean(r[1] for r in rows):5.1f} W, {len(rows)} samples", flush=True)
