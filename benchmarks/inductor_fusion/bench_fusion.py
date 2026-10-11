#!/usr/bin/env python3
"""Does fusing a multiply into the add it feeds pay, now that it is exact?

y_inductor keeps a multiply out of the kernel of the add/sub it feeds
(`_fusable`). That existed because ptxas contracted the pair; since the PTX
emitter rounds a let-bound product first, the fused kernel is bit-identical to
eager. This measured the rule's cost at commit b8f96ab: the same models compiled with
the rule (as shipped there) and without it (`_fusable` patched to always
fuse), against eager. The rule is gone since; on a later commit, where
`_fusable` does not exist, the script measures eager against the fused kernels
only.

Correctness first: every variant's outputs are compared with eager bit for bit
before any time is recorded. Timing: CUDA events, rounds interleaved across
variants, the MINIMUM per-call time over rounds (memory: rank by minimum).
"""
import json, platform, subprocess, sys
import torch
import y_lang
from y_lang import inductor

N_SIZES = [1 << 20, 1 << 22, 1 << 24]
ROUNDS, CALLS = 15, 20


class MulAdd(torch.nn.Module):          # the test's model: two outputs
    def forward(self, a, b, c):
        return a * b + c, c - a * b


class Axpy(torch.nn.Module):            # one multiply-add
    def forward(self, a, b, c):
        return a * b + c


class Chain(torch.nn.Module):           # multiply-add inside a longer chain
    def forward(self, a, b, c):
        return torch.relu(a * b + c) * b - c


def bits(t):
    return t.contiguous().view(torch.int32)


def compile_variant(model, fuse_all, args):
    """Compile `model` and make the first (compiling) call while the rule is
    set as asked - torch.compile is lazy, so the partition happens there."""
    original = getattr(inductor, "_fusable", None)
    if fuse_all and original is not None:
        inductor._fusable = lambda producer, consumer: True
    try:
        reports = []

        def backend(gm, example_inputs):
            out = inductor.y_inductor(gm, example_inputs)
            reports.append(out.y_report)
            return out

        compiled = torch.compile(model, backend=backend)
        first = compiled(*args)
        torch.cuda.synchronize()
        return compiled, reports, first
    finally:
        if original is not None:
            inductor._fusable = original


def as_tuple(x):
    return x if isinstance(x, tuple) else (x,)


def time_call(fn, args):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(CALLS):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / CALLS * 1000.0  # microseconds per call


def main():
    torch.manual_seed(7)
    results = []
    for model_cls in (MulAdd, Axpy, Chain):
        for n in N_SIZES:
            a, b, c = (torch.randn(n, device="cuda") for _ in range(3))
            model = model_cls()
            with torch.no_grad():
                ref = as_tuple(model(a, b, c))
                variants = {"eager": model}
                kernels, reports_of = {}, {}
                torch._dynamo.reset()
                names = (("rule", False), ("fused", True)) if hasattr(inductor, "_fusable") else (("fused", True),)
                for name, fuse in names:
                    fn, reports, out = compile_variant(model_cls(), fuse, (a, b, c))
                    out = as_tuple(out)
                    for o, r in zip(out, ref):
                        assert torch.equal(bits(o), bits(r)), f"{model_cls.__name__} {name} differs from eager"
                    variants[name] = fn
                    reports_of[name] = reports
                    kernels[name] = [k.ops for k in reports[-1].kernels]
                for fn in variants.values():  # warm up
                    for _ in range(3):
                        fn(a, b, c)
                torch.cuda.synchronize()
                best = {k: float("inf") for k in variants}
                for _ in range(ROUNDS):
                    for k, fn in variants.items():
                        best[k] = min(best[k], time_call(fn, (a, b, c)))
                # No variant may have been recompiled (and so re-partitioned)
                # while it was being timed.
                for name in reports_of:
                    assert len(reports_of[name]) == 1, f"{name} was recompiled while timed"
            row = {"model": model_cls.__name__, "n": n, "us": best, "kernels": kernels}
            results.append(row)
            line = f"{model_cls.__name__:7} n=2^{n.bit_length()-1}: eager {best['eager']:8.1f} us  "
            for name in kernels:
                line += f"{name} {best[name]:8.1f} us ({len(kernels[name])} kernels)  "
            print(line, flush=True)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    meta = {"gpu": gpu, "torch": torch.__version__, "python": platform.python_version(),
            "rounds": ROUNDS, "calls_per_round": CALLS, "statistic": "min over rounds"}
    with open(sys.argv[1], "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
