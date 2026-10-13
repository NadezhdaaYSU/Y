#!/usr/bin/env python3
"""Head to head: kernels the Y compiler emits, against Triton and PyTorch.

Each row is one function computed three ways. Before anything is timed, every
column's output is checked against a reference; a column that is wrong is
reported as wrong and is not timed. Every column is then timed the same two
ways, in interleaved rounds:

  launch  back-to-back calls from Python, as a caller gets them;
  graph   replay of a CUDA graph holding the same calls (launch cost removed).

The report gives the minimum and median per column. It makes no speedup
claims; compare the minima, and treat differences inside a column's own
spread as ties.

What this does not time, and why:

- A prefix scan. `y_lang.ops.block_scan` is `torch.cumsum`, and Y has no scan
  kernel. This script used to time it as "Y" through a CUDA graph against the
  same `torch.cumsum` launched plainly, under a "Triton" heading.
- `y_lang.ops.fp8_gemm`. It runs hand-written CUDA C++
  (`tests/y_tensor_core_gemm.cu`, compiled by CuPy) and falls back to
  `torch.mm` on any exception. The FP8 rows here use the GEMM the compiler
  emits for `@tile(M, N, K)` (`tests/gemm_fp8_*.ysu`): float32 A and B,
  quantised to e4m3 in the kernel, float32 C. The other columns compute that
  same function, quantisation included; they used to time FP16 matmuls.

Each row's samples start from warmed clocks, and the SM clock is read after
every sample: short replays left alone run at the idle clock.

Writes `results.json` (every sample and the SM clock after it, every check,
the toolchain and the GPU's state before and after) and `report.md` to --out, by default
`build_artifacts/head_to_head/<UTC time>/`, which git ignores. A run that
starts with the GPU more than 5% busy is marked as contended.

    PYTHONPATH=python python3 python/examples/run_head_to_head_comparison.py
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "python"))

import torch
import triton
import triton.language as tl
import y_lang
from y_lang.compiler import YCompilerLib
from y_lang.inductor import y_inductor

DEVICE = "cuda"


# ----------------------------------------------------------------------
# Kernels
# ----------------------------------------------------------------------

Y_VEC_ADD = """
kernel vec_add(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, N: I32) {
    let idx: I32 = global_thread_id() * 4;
    if idx < N {
        vec_add_v4(A[idx], B[idx], C[idx]);
    }
}
"""

Y_SWIGLU = """
kernel swiglu(Gate: GlobalMemory<F32>, Up: GlobalMemory<F32>, Out: GlobalMemory<F32>, N: I32) {
    let idx: I32 = global_thread_id() * 4;
    if idx < N {
        swiglu_v4(Gate[idx], Up[idx], Out[idx]);
    }
}
"""

Y_RMSNORM = """
kernel rmsnorm(X: GlobalMemory<F32>, Weight: GlobalMemory<F32>, Out: GlobalMemory<F32>, hidden_dim: I32, eps: F32) {
    let row: I32 = block_idx_x();
    let col: I32 = thread_idx_x() * 4;
    let idx: I32 = row * hidden_dim + col;
    rmsnorm_v4(X[idx], Weight[col], Out[idx], hidden_dim, eps);
}
"""


def y_fp8_gemm_source(n):
    return f"""
@tile({n}, {n}, {n})
kernel gemm_fp8_{n}(A: GlobalMemory<F32>, B: GlobalMemory<F32>, scale_a: F32, scale_b: F32, C: GlobalMemory<F32>) {{
}}
"""


@triton.jit
def triton_vector_add(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(c_ptr + offs, tl.load(a_ptr + offs, mask=mask) + tl.load(b_ptr + offs, mask=mask), mask=mask)


@triton.jit
def triton_swiglu(g_ptr, u_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    g = tl.load(g_ptr + offs, mask=mask)
    u = tl.load(u_ptr + offs, mask=mask)
    tl.store(o_ptr + offs, g * tl.sigmoid(g) * u, mask=mask)


@triton.jit
def triton_rmsnorm(x_ptr, w_ptr, o_ptr, eps, N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, N)
    x = tl.load(x_ptr + row * N + cols)
    inv = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    tl.store(o_ptr + row * N + cols, x * inv * tl.load(w_ptr + cols))


# Triton stages the float32 tiles in shared memory, four bytes an element, so
# every config here fits the 99 KB an Ada SM gives one block.
@triton.autotune(
    configs=[
        triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=4, num_stages=2),
        triton.Config({"BM": 64, "BN": 128, "BK": 64}, num_warps=4, num_stages=2),
        triton.Config({"BM": 128, "BN": 64, "BK": 64}, num_warps=4, num_stages=2),
        triton.Config({"BM": 64, "BN": 64, "BK": 64}, num_warps=4, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def triton_fp8_gemm(a_ptr, b_ptr, c_ptr, M, N, K, inv_sa, inv_sb, s_ab,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Y's function: float32 A, B quantised to e4m3 in the kernel, float32 C."""
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptr + rm[:, None] * K + (k0 + rk)[None, :])
        b = tl.load(b_ptr + (k0 + rk)[:, None] * N + rn[None, :])
        acc = tl.dot((a * inv_sa).to(tl.float8e4nv), (b * inv_sb).to(tl.float8e4nv), acc)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc * s_ab)


def y_kernel(source, name, target):
    k = y_lang.TorchKernel(source, kernel_name=name, target_sm=target)
    k._ensure_loaded()
    return k


def required_launch(ptx):
    """The launch the FP8 GEMM's PTX states it needs, rather than a copy of
    the emitter's tile heuristic."""
    m = re.search(r"Launch grid required: \((\d+), (\d+), (\d+)\) CTAs, block \((\d+),(\d+),(\d+)\)", ptx)
    if not m:
        raise RuntimeError("the FP8 GEMM's PTX no longer states its launch geometry")
    v = [int(x) for x in m.groups()]
    return tuple(v[:3]), tuple(v[3:])


# ----------------------------------------------------------------------
# Measurement
# ----------------------------------------------------------------------

def event_time_us(fn, calls):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / calls


def sm_clock_mhz():
    """The SM clock now, from NVML; None where NVML cannot answer."""
    try:
        return torch.cuda.clock_rate(0)
    except Exception:
        return None


def warm_clocks():
    """Bring the clocks up. A row whose samples are short (500 replays of a
    1 us kernel) does not hold them up itself: such replays were measured at
    the 210 MHz idle clock, 4.6x slow."""
    warm = torch.randn(4096, 4096, device=DEVICE)
    for _ in range(30):
        warm @ warm
    torch.cuda.synchronize()


def measure(columns, rounds, warmup_rounds, target_us=2000.0):
    """Interleaved samples, in both regimes, for every column that passed.

    `columns` maps a name to a zero-argument callable. Returns
    {name: {"calls": n, "launch": [us...], "graph": [us...] or None, "graph_error": str,
            "sm_clock_mhz": [MHz read right after each launch sample, then each graph sample]}}.
    """
    plan, one = {}, {}
    for name, fn in columns.items():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        one[name] = event_time_us(fn, 1)
    # The same number of calls per sample in every column, set by the fastest.
    calls = max(1, min(500, int(target_us / max(min(one.values(), default=1.0), 0.5))))
    for name, fn in columns.items():
        def many(fn=fn, calls=calls):
            for _ in range(calls):
                fn()

        graph, graph_error = None, None
        try:
            many()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                many()
            torch.cuda.synchronize()
        except Exception as e:  # a column that cannot be captured is reported, not dropped
            graph, graph_error = None, f"{type(e).__name__}: {e}"
            torch.cuda.synchronize()
        plan[name] = {"calls": calls, "many": many, "graph_obj": graph,
                      "launch": [], "graph": [] if graph else None, "graph_error": graph_error,
                      "sm_clock_mhz": []}

    names = list(plan)
    warm_clocks()
    for r in range(warmup_rounds + rounds):
        order = names[r % len(names):] + names[:r % len(names)]
        for name in order:
            p = plan[name]
            t_launch = event_time_us(p["many"], p["calls"])
            clock_launch = sm_clock_mhz()
            t_graph = event_time_us(p["graph_obj"].replay, p["calls"]) if p["graph_obj"] else None
            clock_graph = sm_clock_mhz() if t_graph is not None else None
            if r >= warmup_rounds:
                p["launch"].append(t_launch)
                p["sm_clock_mhz"].append(clock_launch)
                if t_graph is not None:
                    p["graph"].append(t_graph)
                    p["sm_clock_mhz"].append(clock_graph)
    for p in plan.values():
        del p["many"], p["graph_obj"]
    return plan


def gpu_state():
    fields = "utilization.gpu,clocks.sm,clocks.max.sm,clocks.mem,pstate,temperature.gpu,power.draw"
    samples = []
    for _ in range(10):
        out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits", "-i", "0"],
                             capture_output=True, text=True)
        samples.append([v.strip() for v in out.stdout.strip().split(",")])
        time.sleep(0.1)
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip().splitlines()
    others = [a for a in apps if a.split(",")[0].strip() != str(os.getpid())]
    util = [float(s[0]) for s in samples if s and s[0].replace(".", "").isdigit()]
    return {"fields": fields.split(","), "samples": samples, "compute_apps": apps,
            "other_compute_apps": others, "mean_utilization": statistics.mean(util) if util else None}


def identity(target, y_ptx):
    lib = YCompilerLib.get_instance()
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", "0"],
                            capture_output=True, text=True).stdout.strip()
    return {
        "gpu": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "driver": driver,
        "cuda_runtime": torch.version.cuda,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "python": platform.python_version(),
        "y_commit": git("rev-parse", "HEAD"),
        "y_tree_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "liby_path": lib.lib_path,
        "liby_sha256": lib.compiler_identity,
        "y_target": target,
        "y_ptx_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in y_ptx.items()},
    }


# ----------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------

def rel_err(out, ref):
    """Worst absolute error, relative to the largest reference value."""
    return ((out.double() - ref).abs().max() / ref.abs().max().clamp_min(1e-300)).item()


def check_all(columns, outputs, ref, tol, exact=False):
    """Run each column once, compare its output with `ref`, return the
    columns that passed and a record of every check."""
    passed, record = {}, {}
    for name, fn in columns.items():
        try:
            fn()
            torch.cuda.synchronize()
            out = outputs[name]()
            if exact:
                ok = bool(torch.equal(out.double(), ref))
                record[name] = {"exact": ok, "max_rel_err": rel_err(out, ref)}
            else:
                err = rel_err(out, ref)
                ok = err <= tol
                record[name] = {"max_rel_err": err, "tolerance": tol}
        except Exception as e:
            ok, record[name] = False, {"error": f"{type(e).__name__}: {e}"}
        record[name]["passed"] = ok
        if ok:
            passed[name] = fn
    return passed, record


# ----------------------------------------------------------------------
# Rows
# ----------------------------------------------------------------------

def rows(target, y_ptx, gen):
    """Yield (row name, {column: fn}, {column: output getter}, reference, tolerance, exact, notes)."""
    n = 100_000
    a, b = torch.randn(n, device=DEVICE, generator=gen), torch.randn(n, device=DEVICE, generator=gen)
    k_add = y_kernel(Y_VEC_ADD, "vec_add", target)
    y_ptx["vec_add"] = k_add.ptx
    c = {name: torch.empty(n, device=DEVICE) for name in ("Y", "Triton", "PyTorch")}
    grid_y = ((n // 4 + 127) // 128, 1, 1)
    yield ("Vector add (100,000 f32)",
           {"Y": lambda: k_add.launch(grid=grid_y, block=(128, 1, 1), args=[a, b, c["Y"], n]),
            "Triton": lambda: triton_vector_add[(triton.cdiv(n, 1024),)](a, b, c["Triton"], n, BLOCK=1024),
            "PyTorch": lambda: torch.add(a, b, out=c["PyTorch"])},
           {name: (lambda name=name: c[name]) for name in c},
           (a.double() + b.double()).float().double(), 0.0, True,
           "IEEE addition: every column must equal the correctly rounded float32 sum, exactly.")

    g, u = torch.randn(n, device=DEVICE, generator=gen), torch.randn(n, device=DEVICE, generator=gen)
    k_swi = y_kernel(Y_SWIGLU, "swiglu", target)
    y_ptx["swiglu"] = k_swi.ptx
    o = {name: torch.empty(n, device=DEVICE) for name in ("Y", "Triton", "PyTorch")}
    yield ("SwiGLU (100,000 f32)",
           {"Y": lambda: k_swi.launch(grid=grid_y, block=(128, 1, 1), args=[g, u, o["Y"], n]),
            "Triton": lambda: triton_swiglu[(triton.cdiv(n, 1024),)](g, u, o["Triton"], n, BLOCK=1024),
            "PyTorch": lambda: torch.mul(torch.nn.functional.silu(g), u, out=o["PyTorch"])},
           {name: (lambda name=name: o[name]) for name in o},
           g.double() * torch.sigmoid(g.double()) * u.double(), 1e-5, False,
           "Y uses ex2.approx and rcp.approx; the tolerance is relative to the largest output.")

    rows_, hidden, eps = 128, 1024, 1e-5
    x = torch.randn(rows_, hidden, device=DEVICE, generator=gen)
    w = torch.randn(hidden, device=DEVICE, generator=gen)
    k_rms = y_kernel(Y_RMSNORM, "rmsnorm", target)
    y_ptx["rmsnorm"] = k_rms.ptx
    r = {name: torch.empty_like(x) for name in ("Y", "Triton", "PyTorch")}

    def torch_rmsnorm():
        r["PyTorch"].copy_(x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w)

    xd = x.double()
    yield (f"RMSNorm ({rows_} x {hidden} f32)",
           {"Y": lambda: k_rms.launch(grid=(rows_, 1, 1), block=(hidden // 4, 1, 1), args=[x, w, r["Y"], hidden, eps]),
            "Triton": lambda: triton_rmsnorm[(rows_,)](x, w, r["Triton"], eps, N=hidden),
            "PyTorch": torch_rmsnorm},
           {name: (lambda name=name: r[name]) for name in r},
           xd * torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + eps) * w.double(), 1e-5, False,
           "Y uses rsqrt.approx; the tolerance is relative to the largest output.")

    for size in (1024, 2048, 4096, 8192):
        # Small integers times a power-of-two scale: every implementation
        # quantises them exactly, and the products sum exactly in float32
        # (|sum| <= 64 * 8192 < 2^24), so a correct column equals the
        # float64 product to the bit.
        sa = sb = 0.125
        A = torch.randint(-8, 9, (size, size), device=DEVICE, generator=gen).float() * sa
        B = torch.randint(-8, 9, (size, size), device=DEVICE, generator=gen).float() * sb
        k = y_kernel(y_fp8_gemm_source(size), f"gemm_fp8_{size}", target)
        y_ptx[f"gemm_fp8_{size}"] = k.ptx
        grid, block = required_launch(k.ptx)
        C = {name: torch.empty(size, size, device=DEVICE) for name in ("Y", "Triton", "PyTorch")}
        sa_t = torch.tensor(sa, device=DEVICE)
        sb_t = torch.tensor(sb, device=DEVICE)

        def torch_fp8(A=A, B=B, C=C, sa_t=sa_t, sb_t=sb_t):
            a8 = (A * (1.0 / sa)).to(torch.float8_e4m3fn)
            b8 = (B * (1.0 / sb)).to(torch.float8_e4m3fn).t().contiguous().t()
            torch._scaled_mm(a8, b8, scale_a=sa_t, scale_b=sb_t, out_dtype=torch.float32, out=C["PyTorch"])

        def triton_fp8(A=A, B=B, C=C, size=size):
            grid_t = lambda meta: (size // meta["BM"], size // meta["BN"])
            triton_fp8_gemm[grid_t](A, B, C["Triton"], size, size, size, 1.0 / sa, 1.0 / sb, sa * sb)

        yield (f"FP8 GEMM {size}^3 (f32 in, e4m3 tensor cores, f32 out)",
               {"Y": lambda k=k, A=A, B=B, C=C, grid=grid, block=block:
                    k.launch(grid=grid, block=block, args=[A, B, sa, sb, C["Y"]]),
                "Triton": triton_fp8,
                "PyTorch": torch_fp8},
               {name: (lambda name=name, C=C: C[name]) for name in C},
               (A.double() @ B.double()), 0.0, True,
               "Every column quantises float32 A and B to e4m3 and returns float32; "
               "PyTorch's time includes the quantisation and B's transpose, as Y's and Triton's include theirs.")

    gate, up = torch.randn(1 << 22, device=DEVICE, generator=gen), torch.randn(1 << 22, device=DEVICE, generator=gen)

    class SwiGLU(torch.nn.Module):
        def forward(self, g, u):
            return torch.nn.functional.silu(g) * u

    module = SwiGLU().to(DEVICE)
    y_compiled = y_inductor(module, [gate, up])
    torch_compiled = torch.compile(module)
    got = {}
    yield ("SwiGLU module through a graph compiler (2^22 f32)",
           {"Y": lambda: got.__setitem__("Y", y_compiled(gate, up)),
            "Triton": lambda: got.__setitem__("Triton", torch_compiled(gate, up)),
            "PyTorch": lambda: got.__setitem__("PyTorch", module(gate, up))},
           {name: (lambda name=name: got[name]) for name in ("Y", "Triton", "PyTorch")},
           module(gate.double(), up.double()), 1e-6, False,
           # Read after the timed calls, so the report's launch counts are the run's.
           lambda: "Y column: y_inductor (silu and the multiply are one Y kernel, bit for bit eager's). "
                   "Triton column: torch.compile, whose Inductor backend generates Triton. "
                   "PyTorch column: eager. " + y_compiled.y_report.summary().replace("\n", " "))


# ----------------------------------------------------------------------
# Cold compilation
# ----------------------------------------------------------------------

COLD_CHILD = r"""
import os, sys, time, json
sys.path.insert(0, {python!r})
which = sys.argv[1]
import torch
torch.zeros(1, device="cuda"); torch.cuda.synchronize()
if which == "y":
    import y_lang
    from y_lang.compiler import YCompilerLib
    YCompilerLib.get_instance()
    src = {y_src!r} + "\n// cold-" + str(time.time_ns()) + "\n"
    t0 = time.perf_counter(); k = y_lang.TorchKernel(src, kernel_name="rmsnorm", target_sm={target!r}); t1 = time.perf_counter()
    k._ensure_loaded(); t2 = time.perf_counter()
    print(json.dumps({{"compile_to_ptx_ms": (t1 - t0) * 1e3, "module_load_ms": (t2 - t1) * 1e3}}))
else:
    import triton, triton.language as tl
    @triton.jit
    def rms(x_ptr, w_ptr, o_ptr, eps, N: tl.constexpr):
        row = tl.program_id(0); cols = tl.arange(0, N)
        x = tl.load(x_ptr + row * N + cols)
        inv = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
        tl.store(o_ptr + row * N + cols, x * inv * tl.load(w_ptr + cols))
    x = torch.randn(128, 1024, device="cuda"); w = torch.randn(1024, device="cuda"); o = torch.empty_like(x)
    t0 = time.perf_counter(); rms[(128,)](x, w, o, 1e-5, N=1024); torch.cuda.synchronize(); t1 = time.perf_counter()
    rms[(128,)](x, w, o, 1e-5, N=1024); torch.cuda.synchronize(); t2 = time.perf_counter()
    print(json.dumps({{"first_call_ms": (t1 - t0) * 1e3, "second_call_ms": (t2 - t1) * 1e3}}))
"""


def cold_compile(target, repeats=5):
    """Each sample in a fresh process, with the driver's compute cache off: Y
    with a source its caches have never seen, Triton with an empty cache
    directory. Y's figure stops at a loaded module (the driver compiles the
    PTX at load); Triton's first call compiles, assembles, loads and launches."""
    script = COLD_CHILD.format(python=str(REPO / "python"), y_src=Y_RMSNORM, target=target)
    out = {"y": [], "triton": []}
    for _ in range(repeats):
        for which in ("y", "triton"):
            with tempfile.TemporaryDirectory() as cache:
                # The driver caches PTX it has compiled; Y's PTX is the same
                # every run (the source comment does not reach it), so without
                # this its load would be a cache hit.
                env = dict(os.environ, TRITON_CACHE_DIR=cache, CUDA_CACHE_DISABLE="1")
                p = subprocess.run([sys.executable, "-c", script, which], capture_output=True, text=True, env=env)
                line = [l for l in p.stdout.splitlines() if l.startswith("{")]
                out[which].append(json.loads(line[-1]) if line else {"error": p.stderr[-2000:]})
    return out


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

def summary(samples):
    if not samples:
        return None
    return {"min": min(samples), "median": statistics.median(samples), "max": max(samples), "n": len(samples)}


def write_report(out_dir, result):
    ident, lines = result["identity"], []
    lines.append("# Head to head: Y vs Triton vs PyTorch\n")
    lines.append(f"{ident['gpu']} (sm_{ident['capability'][0]}{ident['capability'][1]}), driver {ident['driver']}, "
                 f"CUDA {ident['cuda_runtime']}, PyTorch {ident['torch']}, Triton {ident['triton']}, "
                 f"Python {ident['python']}. Y {ident['y_commit'][:10]}{' (dirty tree)' if ident['y_tree_dirty'] else ''}, "
                 f"`liby.so` sha256 {ident['liby_sha256'][:12]}, target {ident['y_target']}. {result['utc']}.\n")
    before = result["gpu_before"]
    if result["contended"]:
        lines.append("> **Contended run.** Another process was using the GPU "
                     f"(mean utilisation {before['mean_utilization']}% before starting; "
                     f"other processes on it: {before['other_compute_apps']}). "
                     "These timings are not a clean measurement.\n")
    else:
        lines.append(f"GPU utilisation before starting: {before['mean_utilization']}% (mean of 10 samples); "
                     f"other processes holding a context: {before['other_compute_apps']}.\n")
    lines.append("Microseconds per call: minimum / median over "
                 f"{result['rounds']} interleaved rounds. *launch*: back-to-back calls from Python; "
                 "*graph*: the same calls replayed from a CUDA graph. A column that failed its check is not timed.\n")
    for row in result["rows"]:
        lines.append(f"## {row['name']}\n")
        lines.append(f"{row['notes']}\n")
        lines.append("| column | check | launch min / median | graph min / median |")
        lines.append("| :--- | :--- | ---: | ---: |")
        for col in ("Y", "Triton", "PyTorch"):
            chk = row["checks"][col]
            mark = "exact" if chk.get("exact") else (
                f"err {chk['max_rel_err']:.1e}" if "max_rel_err" in chk else chk.get("error", "?"))
            if not chk["passed"]:
                lines.append(f"| {col} | **FAILED** ({mark}) | not timed | not timed |")
                continue
            t = row["timing"][col]
            fmt = lambda s: f"{s['min']:.2f} / {s['median']:.2f}" if s else f"n/a ({t['graph_error']})"
            lines.append(f"| {col} | {mark} | {fmt(t['launch_summary'])} | {fmt(t['graph_summary'])} |")
        clocks = [c for t in row["timing"].values() for c in t["sm_clock_mhz"] if c is not None]
        if clocks:
            lines.append(f"\nSM clock read after each sample: {min(clocks)}-{max(clocks)} MHz "
                         f"(median {statistics.median(clocks):.0f}).")
        lines.append("")
    cold = result["cold_compile"]
    ok_y = [s for s in cold["y"] if "error" not in s]
    ok_t = [s for s in cold["triton"] if "error" not in s]
    if ok_y or ok_t:
        lines.append("## Cold compilation (one RMSNorm kernel, each sample a fresh process)\n")
    if ok_y:
        lines.append(f"- Y `compile_to_ptx`: median {statistics.median(s['compile_to_ptx_ms'] for s in ok_y):.1f} ms; "
                     f"loading the PTX (the driver compiles it): median {statistics.median(s['module_load_ms'] for s in ok_y):.1f} ms.")
    if ok_t:
        lines.append(f"- Triton first call with an empty cache: median {statistics.median(s['first_call_ms'] for s in ok_t):.1f} ms "
                     f"(second call {statistics.median(s['second_call_ms'] for s in ok_t):.2f} ms).")
    lines.append(f"\nRaw samples: `results.json`. Written by `python/examples/run_head_to_head_comparison.py`.\n")
    (out_dir / "report.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--warmup-rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--skip-cold", action="store_true")
    args = ap.parse_args()

    utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out or REPO / "build_artifacts" / "head_to_head" / utc
    out_dir.mkdir(parents=True, exist_ok=True)

    major, minor = torch.cuda.get_device_capability(0)
    target = f"sm_{major}{minor}"
    before = gpu_state()
    gen = torch.Generator(device=DEVICE).manual_seed(args.seed)

    # Bring the clocks up before anything is measured (measure() does it
    # again before each row's samples).
    warm_clocks()

    y_ptx, results = {}, []
    for name, columns, outputs, ref, tol, exact, notes in rows(target, y_ptx, gen):
        passed, checks = check_all(columns, outputs, ref, tol, exact)
        timing = measure(passed, args.rounds, args.warmup_rounds)
        for t in timing.values():
            t["launch_summary"] = summary(t["launch"])
            t["graph_summary"] = summary(t["graph"])
            t["sm_clock_summary"] = summary([c for c in t["sm_clock_mhz"] if c is not None])
        notes = notes() if callable(notes) else notes
        results.append({"name": name, "notes": notes, "checks": checks, "timing": timing})
        cell = lambda c: (f"{timing[c]['launch_summary']['min']:.2f}/"
                          + (f"{timing[c]['graph_summary']['min']:.2f}" if timing[c]["graph_summary"] else "n/a")
                          if c in timing else "FAILED")
        clocks = [c for t in timing.values() for c in t["sm_clock_mhz"] if c is not None]
        print(f"{name:<55} " + " | ".join(f"{c}: {cell(c)}" for c in ("Y", "Triton", "PyTorch"))
              + "  (min us, launch/graph)"
              + (f"  SM clock {min(clocks)}-{max(clocks)} MHz" if clocks else "  SM clock unread"), flush=True)
        torch.cuda.empty_cache()

    after = gpu_state()
    result = {
        "utc": utc,
        "identity": identity(target, y_ptx),
        "rounds": args.rounds,
        "warmup_rounds": args.warmup_rounds,
        "seed": args.seed,
        "gpu_before": before,
        "gpu_after": after,
        # Utilisation, not the process list: a desktop's compositor always
        # holds a context, so "any other process" would mark every run.
        "contended": (before["mean_utilization"] or 0) > 5,
        "rows": results,
        "cold_compile": {"y": [], "triton": []} if args.skip_cold else cold_compile(target),
    }
    (out_dir / "results.json").write_text(json.dumps(result, indent=1))
    write_report(out_dir, result)
    print(f"\nwrote {out_dir / 'report.md'} and results.json")
    if result["contended"]:
        print("CONTENDED: another process was using the GPU; see report.md")
    failed = [(r["name"], c) for r in results for c, chk in r["checks"].items() if not chk["passed"]]
    if failed:
        print("FAILED CHECKS:", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
