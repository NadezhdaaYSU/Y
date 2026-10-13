"""rmsnorm_v4 from two compilers: bit-identical outputs, then graph-replay
time against the head-to-head's Triton RMSNorm.

    python3 rms_old_new.py <old Y binary> <new Y binary> <python dir> [out.json]
"""
import json
import os
import statistics
import subprocess
import sys
import tempfile

old_y, new_y, pydir = sys.argv[1], sys.argv[2], sys.argv[3]
out_json = sys.argv[4] if len(sys.argv) > 4 else None
sys.path.insert(0, pydir)
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from y_lang.torch_interop import TorchKernel  # noqa: E402

KERNEL = """kernel rmsnorm(X: GlobalMemory<F32>, W: GlobalMemory<F32>, Out: GlobalMemory<F32>, hidden_dim: I32, eps: F32) {
    let row: I32 = block_idx_x();
    let col: I32 = thread_idx_x() * 4;
    let idx: I32 = row * hidden_dim + col;
    rmsnorm_v4(X[idx], W[col], Out[idx], hidden_dim, eps);
}

fn main() {}
"""
# A hardware profile to compile with (the runs used one for the RTX 4070 Ti
# SUPER); without one the compiler probes the GPU into the scratch directory.
PROFILE = os.environ.get("Y_HW_PROFILE")


class PtxKernel(TorchKernel):
    def __init__(self, ptx, name):
        self.source, self.kernel_name, self.target_sm, self.ptx = None, name, None, ptx
        self._module = self._function = None


def emit(ybin):
    d = tempfile.mkdtemp(prefix="rms_")
    if PROFILE:
        subprocess.run(["cp", PROFILE, d], check=True)
    open(os.path.join(d, "k.ysu"), "w").write(KERNEL)
    r = subprocess.run([ybin, "k.ysu", "--emit-ptx", "-o", "k.ptx"], cwd=d, capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stdout + r.stderr)
    return open(os.path.join(d, "k.ptx")).read()


@triton.jit
def triton_rmsnorm(x_ptr, w_ptr, o_ptr, eps, N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, N)
    x = tl.load(x_ptr + row * N + cols)
    inv = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    tl.store(o_ptr + row * N + cols, x * inv * tl.load(w_ptr + cols))


def util_samples():
    vals = []
    for _ in range(10):
        o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True).stdout.strip()
        vals.append(o)
        __import__("time").sleep(0.1)
    return vals


util_before = util_samples()
ptx = {"old": emit(old_y), "new": emit(new_y)}
for k, v in ptx.items():
    print(k, "bar.sync", v.count("bar.sync"), "st.shared", v.count("st.shared"))
kern = {k: PtxKernel(v, "rmsnorm") for k, v in ptx.items()}

gen = torch.Generator(device="cuda").manual_seed(7)
for rows, hidden in [(128, 1024), (64, 4096), (256, 512), (300, 128)]:
    x = torch.randn(rows, hidden, device="cuda", generator=gen) * 3
    w = torch.randn(hidden, device="cuda", generator=gen)
    outs = {}
    for k, kk in kern.items():
        o = torch.full_like(x, float("nan"))
        kk.launch((rows, 1, 1), (hidden // 4, 1, 1), [x, w, o, hidden, 1e-5])
        outs[k] = o
    torch.cuda.synchronize()
    same = torch.equal(outs["old"].view(torch.int32), outs["new"].view(torch.int32))
    print(f"{rows}x{hidden}: new == old bit for bit: {same}")
    if not same:
        sys.exit("DIFFERENT")

# Timing: the head-to-head's row, graph replays of 500 launches, interleaved.
rows, hidden, eps = 128, 1024, 1e-5
x = torch.randn(rows, hidden, device="cuda", generator=gen)
w = torch.randn(hidden, device="cuda", generator=gen)
o = {k: torch.empty_like(x) for k in ("old", "new", "Triton")}
fns = {
    "old": lambda: kern["old"].launch((rows, 1, 1), (hidden // 4, 1, 1), [x, w, o["old"], hidden, eps]),
    "new": lambda: kern["new"].launch((rows, 1, 1), (hidden // 4, 1, 1), [x, w, o["new"], hidden, eps]),
    "Triton": lambda: triton_rmsnorm[(rows,)](x, w, o["Triton"], eps, N=hidden),
}
CALLS = 500
# Bring the clocks up first: replays this short leave a resting GPU at its
# idle clock, which once measured every column 4.6x slower.
warm = torch.randn(4096, 4096, device="cuda")
for _ in range(30):
    warm @ warm
torch.cuda.synchronize()
graphs = {}
for k, fn in fns.items():
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(CALLS):
            fn()
    graphs[k] = g
samples = {k: [] for k in fns}
clocks = []
names = list(fns)
for r in range(3 + 21):
    for k in names[r % 3:] + names[:r % 3]:
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        graphs[k].replay()
        e.record()
        e.synchronize()
        if r >= 3:
            samples[k].append(s.elapsed_time(e) * 1000 / CALLS)
    clocks.append(subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout.strip())
util = util_samples()
for k, v in samples.items():
    print(f"{k:7s} min {min(v):.3f} us  median {statistics.median(v):.3f} us")
mean = lambda v: statistics.mean(float(x.split(",")[0]) for x in v)
print("SM clock per round (MHz):", " ".join(clocks))
print("GPU utilisation before %.1f%%, after %.1f%%" % (mean(util_before), mean(util)))
if out_json:
    import hashlib, platform
    json.dump({"samples_us": samples, "calls_per_sample": CALLS, "rows": rows, "hidden": hidden, "eps": eps,
               "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "triton": triton.__version__,
               "python": platform.python_version(), "old_y": old_y, "new_y": new_y,
               "ptx_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in ptx.items()},
               "gpu_util_clock_before": util_before, "sm_clock_per_round_mhz": clocks, "gpu_util_clock_after": util}, open(out_json, "w"), indent=1)
