# ExactLinear's three GPU GEMMs, at the shapes Qwen2.5-0.5B and -7B decode and
# prefill: "triton" (tools/w8a8_gemv.py, the default), "torch" (torch._int_mm,
# cuBLASLt) and "y" (Y's proved kernel through y_lang.int8_gemm). Each mode is
# a whole ExactLinear.forward: quantise the activations, the GEMM, the scales
# and the bias, eager.
#
# Bits first: for each layer and input, the three outputs must be identical
# (all three sums are exact and the epilogue rounds the same way). Then the
# per-call time: the minimum over 15 rounds, the modes' order shuffled every
# round, CUDA events around a batch of calls sized to about 3 ms.
#
# Usage, from the repo root: python3 docs/benchmarks/exact_y_gemm_2026-11-07/layer_bench.py [--quick]
import os
import random
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
sys.path.insert(0, os.path.join(REPO, "tools"))

import torch                     # noqa: E402
import torch.nn as nn            # noqa: E402

import exact_model as EM         # noqa: E402
import w8a8_gemv                 # noqa: E402

MODES = ("triton", "torch", "y")
SHAPES = {   # (N, K): q/o, k/v, gate/up, down
    "0.5B": ((896, 896), (128, 896), (4864, 896), (896, 4864)),
    "7B": ((3584, 3584), (512, 3584), (18944, 3584), (3584, 18944)),
}
ROWS = (1, 8, 32, 256, 2048)


def layer(n, k, dev):
    torch.manual_seed(n * 7 + k)
    lin = nn.Linear(k, n, bias=True).to(dev)
    out = {}
    for mode in MODES:
        EM.GEMM = mode
        out[mode] = EM.ExactLinear(lin)
    del lin
    return out


def call(mods, mode, x):
    EM.GEMM = mode
    return mods[mode](x)


def main():
    quick = "--quick" in sys.argv
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    print(f"# {props.name}, {props.multi_processor_count} SMs; torch {torch.__version__}; "
          f"exact_model GEMM default {os.environ.get('Y_EXACT_GEMM', '(unset)')}", flush=True)
    rng = random.Random(0)
    shapes = [(model, n, k) for model, s in SHAPES.items() for n, k in s]
    w8a8_gemv.prime(sorted({(n, k) for _, n, k in shapes}), dev)
    print(f"{'model':5} {'N':>6} {'K':>6} {'M':>5} | {'triton us':>9} {'torch us':>9} {'y us':>9} | "
          f"{'Y vs triton':>11} {'Y vs torch':>10}  (above 1: Y faster)", flush=True)
    with torch.no_grad():
        for model, n, k in shapes:
            mods = layer(n, k, dev)
            for m in ROWS:
                x = torch.randn(m, k, device=dev) * 3
                outs = {mode: call(mods, mode, x) for mode in MODES}
                bits = {mode: o.contiguous().view(torch.uint8) for mode, o in outs.items()}
                if not all(torch.equal(bits["torch"], bits[mode]) for mode in MODES):
                    sys.exit(f"{model} N={n} K={k} M={m}: the modes give different bits")
                # Per-call estimate, then batches of about 3 ms.
                reps = {}
                for mode in MODES:
                    for _ in range(3):
                        call(mods, mode, x)
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    for _ in range(10):
                        call(mods, mode, x)
                    e.record()
                    e.synchronize()
                    reps[mode] = max(5, min(400, int(3.0 / max(s.elapsed_time(e) / 10, 1e-4))))
                best = {mode: float("inf") for mode in MODES}
                for _ in range(5 if quick else 15):
                    order = list(MODES)
                    rng.shuffle(order)
                    for mode in order:
                        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        s.record()
                        for _ in range(reps[mode]):
                            call(mods, mode, x)
                        e.record()
                        e.synchronize()
                        best[mode] = min(best[mode], s.elapsed_time(e) * 1e3 / reps[mode])
                t, c, y = best["triton"], best["torch"], best["y"]
                print(f"{model:5} {n:6} {k:6} {m:5} | {t:9.1f} {c:9.1f} {y:9.1f} | "
                      f"{t / y:10.2f}x {c / y:9.2f}x  bits ok", flush=True)
            del mods
            torch.cuda.empty_cache()
    import subprocess
    print("# " + subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu",
                                 "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())


if __name__ == "__main__":
    main()
