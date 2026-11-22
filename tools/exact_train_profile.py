"""Where a compiled `y_lang.exact.train_lm` step spends its GPU time.

    python3 tools/exact_train_profile.py --size medium --arm exact32 --out prof.txt

Compiles the step (`Trainer(compile=True)`, no CUDA graph, so the profiler
sees each kernel), runs a few steps, profiles `--steps` more and prints the
GPU time per step by kind (Inductor's Triton kernels, Y's int8 GEMM, cuBLAS,
the rest) and the slowest kernels, each with its sizes and the source nodes
Inductor fused into it (the names of `train_lm`'s own variables, so a kernel
can be traced to the lines it came from).
"""

import argparse
import collections
import os
import re
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from y_lang.exact import train_lm as L  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact_train_speed import SIZES, TEXT, make  # noqa: E402


def kind(name):
    n = name.lower()
    if n.startswith("triton"):
        return "triton (Inductor)"
    if n.startswith("y_int8") or "int8_mm" in n:
        return "Y int8 GEMM"
    if any(s in n for s in ("gemm", "cutlass", "sm80", "sm89", "sm90", "xmma", "cublas", "ampere", "hopper", "nvjet")):
        return "cuBLAS"
    if "memcpy" in n or "memset" in n:
        return "memcpy/memset"
    return "other: " + name[:50]


def kernel_info(sources):
    """name -> (sizes, fused source nodes) from Inductor's generated code."""
    info = {}
    src = "\n".join(sources)
    for m in re.finditer(r"def (triton_\w+)\(", src):
        body = src[m.end():m.end() + 4000]
        sizes = dict(re.findall(r"(\w+numel) = (\d+)", body[:1500]))
        info[m.group(1)] = [sizes, ""]
    for m in re.finditer(r"# Topologically Sorted Source Nodes: \[([^\]]*)\][^\n]*\n(?:[^\n]*\n){0,3}?\s*(triton_\w+)\.run\(", src):
        if m.group(2) in info and not info[m.group(2)][1]:
            info[m.group(2)][1] = m.group(1)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=SIZES, default="medium")
    ap.add_argument("--arm", default="exact32", help="exact, exact32, f32 or bf16")
    ap.add_argument("--mm", choices=("y", "torch", "f64"), default="y")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--out")
    a = ap.parse_args()
    from torch._inductor.utils import run_and_get_code
    from torch.profiler import ProfilerActivity, profile

    torch.backends.cuda.matmul.allow_tf32 = False
    cfg, B = SIZES[a.size]
    data = L.CharData(open(TEXT, encoding="utf-8").read())
    gen = torch.Generator().manual_seed(1)
    bs = [tuple(t.cuda() for t in data.batch(gen, B, cfg.ctx)) for _ in range(a.steps + 6)]
    tr = make(a.arm, "compile", "cuda", cfg, a.mm)
    _, code = run_and_get_code(tr.step, *bs[0])
    for x, y in bs[1:6]:
        tr.step(x, y)
    tr.verify()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for x, y in bs[6:]:
            tr.step(x, y)
        torch.cuda.synchronize()
    tr.verify()
    per = collections.defaultdict(lambda: [0, 0.0])
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            per[e.name][0] += 1
            per[e.name][1] += e.device_time_total
    n = a.steps
    total = sum(t for _, t in per.values()) / n
    kinds = collections.defaultdict(lambda: [0, 0.0])
    for name, (c, t) in per.items():
        kinds[kind(name)][0] += c
        kinds[kind(name)][1] += t
    info = kernel_info(code)
    lines = [f"{a.arm} {a.size} on {torch.cuda.get_device_name()}: {total / 1e3:.2f} ms of GPU time a step, "
             f"{sum(c for c, _ in per.values()) / n:.0f} kernels"]
    for k, (c, t) in sorted(kinds.items(), key=lambda kv: -kv[1][1]):
        lines.append(f"  {t / n / 1e3:8.3f} ms  {100 * t / n / total:5.1f}%  {c / n:6.1f} kernels  {k}")
    lines.append(f"slowest {a.top} kernels (us a step, calls a step, sizes, fused source nodes):")
    for name, (c, t) in sorted(per.items(), key=lambda kv: -kv[1][1])[:a.top]:
        sizes, nodes = info.get(name, ({}, ""))
        sz = " ".join(f"{k.replace('numel', '')}={v}" for k, v in sizes.items())
        lines.append(f"  {t / n:9.1f}  x{c / n:<5.1f} {sz:28s} {name}")
        if nodes:
            lines.append(f"            nodes: {nodes[:400]}")
    text = "\n".join(lines)
    print(text, flush=True)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
