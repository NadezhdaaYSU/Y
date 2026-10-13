"""SwiGLU's activation, `silu(g) * u`, through four paths: `y_inductor` before
this change (silu eager, the multiply a Y kernel), after it (one Y kernel),
`torch.compile` (Inductor, Triton) and eager PyTorch.

    python3 bench_swiglu.py <checkout> <before-commit> <out.json>

`<checkout>/python` supplies `y_lang` (the "after" `inductor.py`); the "before"
module is `git show <before-commit>:python/y_lang/inductor.py`, loaded into
the same package so both use one `torch_interop`. Every column is checked
before it is timed: both Y columns must equal eager bit for bit; the
torch.compile column's difference from eager is recorded, not required.
Timing is the head-to-head's: CUDA events around back-to-back calls
("launch") and around a CUDA-graph replay of the same calls ("graph"),
15 interleaved rounds after 3 warm-up rounds."""
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time

checkout, before_commit, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, os.path.join(checkout, "python"))
os.environ.setdefault("YSU_CACHE_DIR", tempfile.mkdtemp(prefix="bench_swiglu_cache_"))

import torch  # noqa: E402
import triton  # noqa: E402
import y_lang  # noqa: E402
from y_lang.compiler import YCompilerLib  # noqa: E402
from y_lang.inductor import y_inductor as y_after  # noqa: E402

source = subprocess.run(["git", "-C", checkout, "show", f"{before_commit}:python/y_lang/inductor.py"],
                        capture_output=True, text=True, check=True).stdout
before_file = os.path.join(tempfile.mkdtemp(prefix="bench_swiglu_before_"), "inductor_before.py")
open(before_file, "w").write(source)
spec = importlib.util.spec_from_file_location("y_lang.inductor_before", before_file)
before_mod = importlib.util.module_from_spec(spec)
sys.modules["y_lang.inductor_before"] = before_mod
spec.loader.exec_module(before_mod)
y_before = before_mod.y_inductor

ROUNDS, WARMUP = 15, 3
SIZES = [1 << 20, 1 << 22, 1 << 24]


class SwiGLU(torch.nn.Module):
    def forward(self, g, u):
        return torch.nn.functional.silu(g) * u


def gpu_state():
    q = "utilization.gpu,clocks.sm,clocks.max.sm,clocks.mem,pstate,temperature.gpu,power.draw"
    samples = []
    for _ in range(10):
        r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits", "-i", "0"],
                           capture_output=True, text=True)
        samples.append([v.strip() for v in r.stdout.strip().split(",")])
        time.sleep(0.1)
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().splitlines()
    util = [float(s[0]) for s in samples if s and s[0].replace(".", "").isdigit()]
    return {"fields": q.split(","), "samples": samples, "compute_apps": apps,
            "mean_utilization": statistics.mean(util) if util else None}


def event_us(fn, calls):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / calls


def bits(t):
    return t.contiguous().view(torch.int32)


rows = []
before_state = gpu_state()
warm = torch.randn(4096, 4096, device="cuda")
for _ in range(30):
    warm @ warm
torch.cuda.synchronize()

for n in SIZES:
    gen = torch.Generator(device="cuda").manual_seed(n)
    g = torch.randn(n, device="cuda", generator=gen)
    u = torch.randn(n, device="cuda", generator=gen)
    module = SwiGLU()
    eager_out = module(g, u)
    compiled = {
        "Y before": y_before(module, [g, u]),
        "Y after": y_after(module, [g, u]),
        "torch.compile": torch.compile(SwiGLU(), dynamic=False),
        "eager": module,
    }
    got = {}
    fns = {name: (lambda name=name, m=m: got.__setitem__(name, m(g, u))) for name, m in compiled.items()}
    checks = {}
    with torch.no_grad():
        for name, fn in fns.items():
            fn()
            torch.cuda.synchronize()
            out = got[name]
            differing = int((bits(out) != bits(eager_out)).sum())
            rel = ((out.double() - eager_out.double()).abs().max()
                   / eager_out.double().abs().max()).item()
            checks[name] = {"differing_from_eager": differing, "max_rel_diff_from_eager": rel}
        for name in ("Y before", "Y after"):
            if checks[name]["differing_from_eager"]:
                sys.exit(f"{name} differs from eager at n={n}: {checks[name]}")

        one = {}
        for name, fn in fns.items():
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            one[name] = event_us(fn, 1)
        calls = max(1, min(500, int(2000.0 / max(min(one.values()), 0.5))))
        plan = {}
        for name, fn in fns.items():
            def many(fn=fn):
                for _ in range(calls):
                    fn()
            many()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                many()
            torch.cuda.synchronize()
            plan[name] = {"many": many, "graph": graph, "launch": [], "replay": []}
        names = list(plan)
        for r in range(WARMUP + ROUNDS):
            for name in names[r % len(names):] + names[:r % len(names)]:
                p = plan[name]
                t_launch = event_us(p["many"], calls)
                t_graph = event_us(p["graph"].replay, calls)
                if r >= WARMUP:
                    p["launch"].append(t_launch)
                    p["replay"].append(t_graph)
    report = {
        "Y before": compiled["Y before"].y_report.summary(),
        "Y after": compiled["Y after"].y_report.summary(),
    }
    rows.append({
        "n": n, "calls_per_sample": calls, "checks": checks, "reports": report,
        "launch_us": {k: v["launch"] for k, v in plan.items()},
        "graph_us": {k: v["replay"] for k, v in plan.items()},
    })
    line = " | ".join(f"{k}: {min(v['launch']):.1f}/{min(v['replay']):.1f}" for k, v in plan.items())
    print(f"n=2^{n.bit_length() - 1}: {line}  (min us, launch/graph)", flush=True)
    del plan, compiled, fns, got
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

lib = YCompilerLib.get_instance()
git = lambda *a: subprocess.run(["git", "-C", checkout, *a], capture_output=True, text=True).stdout.strip()
result = {
    "gpu": torch.cuda.get_device_name(0),
    "driver": subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", "0"],
                             capture_output=True, text=True).stdout.strip(),
    "torch": torch.__version__, "triton": triton.__version__, "python": platform.python_version(),
    "y_commit": git("rev-parse", "HEAD"), "y_tree_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    "before_commit": git("rev-parse", before_commit), "liby_sha256": lib.compiler_identity,
    "rounds": ROUNDS, "warmup_rounds": WARMUP,
    "gpu_before": before_state, "gpu_after": gpu_state(), "rows": rows,
}
with open(out_path, "w") as f:
    json.dump(result, f, indent=1)
print("wrote", out_path, "| GPU utilisation before:", before_state["mean_utilization"])
