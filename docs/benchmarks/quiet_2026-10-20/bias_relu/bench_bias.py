"""bench_bias.py <bench_ab.py dir> <label=ptx-dir,...> <out.json> [N,N,...]: Y's
fused F16 GEMM + bias + ReLU (f16 A, B; f32 bias and C) against PyTorch.

Each ptx-dir holds gemm_f16_bias_relu_<N>.ptx (kernel gemm_f16_bias_relu_<N>).
Baselines, each checked first on small integers (exact):
  addmm32+relu   torch.relu(torch.addmm(bias, A, B, out_dtype=float32)):
                 cuBLAS with its bias epilogue, then a ReLU kernel; float32
                 out, the same bytes as Y.
  linear16+relu  torch.relu(F.linear(A, B^T, bias16)): the README's earlier
                 baseline (cuBLASLt, f16 out - half Y's output bytes), then a
                 ReLU kernel.
Timing as bench_ab.py: CUDA events over back-to-back calls, 17 interleaved
rounds, the first two dropped, minimum and median of 15; clocks warmed per size.
"""
import json
import re
import statistics
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, sys.argv[1])
from bench_ab import kernel_from_ptx, event_us, warm, util  # noqa: E402


def main():
    dirs = [tuple(x.split("=", 1)) for x in sys.argv[2].split(",")]
    out_path = sys.argv[3]
    sizes = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "512,1024,2048,4096,8192").split(",")]
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    result = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "util_before": util(), "sizes": {}}
    for n in sizes:
        m = k = n
        gen = torch.Generator(device="cuda").manual_seed(n)
        a = torch.randint(-4, 5, (m, k), device="cuda", generator=gen).half()
        b = torch.randint(-4, 5, (k, n), device="cuda", generator=gen).half()
        bias = torch.randint(-64, 65, (n,), device="cuda", generator=gen).float()
        bias16 = bias.half()
        bt = b.t().contiguous()
        ref = torch.relu((a.double() @ b.double()) + bias.double()).float()
        cols, outs, tiles = {}, {}, {}
        for label, d in dirs:
            kern = kernel_from_ptx(f"{d}/gemm_f16_bias_relu_{n}.ptx", f"gemm_f16_bias_relu_{n}")
            cm, cn, wm, wn = (int(x) for x in re.search(r"CTA (\d+)x(\d+)x\d+ \| (\d+)x(\d+) warps", kern.ptx).groups())
            tiles[label] = f"{cm}x{cn} {wm}x{wn}"
            outs[label] = torch.full((m, n), float("nan"), device="cuda")
            cols[label] = (lambda kern=kern, cm=cm, cn=cn, wm=wm, wn=wn, o=outs[label]:
                           kern.launch((-(-n // cn), -(-m // cm), 1), (wm * wn * 32, 1, 1), [a, b, bias, o]))
        cols["addmm32+relu"] = lambda: torch.relu(torch.addmm(bias, a, b, out_dtype=torch.float32))
        cols["linear16+relu"] = lambda: torch.relu(F.linear(a, bt, bias16))
        for name, f in cols.items():
            r = f()
            if r is not None:
                outs[name] = r
        torch.cuda.synchronize()
        exact = {}
        for name, o in outs.items():
            want = ref.half() if o.dtype == torch.float16 else ref
            exact[name] = bool(torch.equal(o, want))
        calls = max(2, min(200, int(5e-3 / max(2 * m * n * k / 80e12, 2e-6))))
        names = [c for c in cols if exact[c]]
        samples = {c: [] for c in names}
        clocks = []
        warm()
        for r in range(17):
            for c in names[r % len(names):] + names[:r % len(names)]:
                t = event_us(cols[c], calls)
                if r >= 2:
                    samples[c].append(t)
                    clocks.append(torch.cuda.clock_rate(0))
        mins = {c: min(s) for c, s in samples.items()}
        result["sizes"][n] = {"tiles": tiles, "exact": exact, "calls": calls, "samples": samples, "sm_clock_mhz": clocks}
        print(f"{n:>6d} Y {tiles} | exact {exact} | SM {min(clocks)}-{max(clocks)} MHz", flush=True)
        for c in sorted(mins, key=mins.get):
            print(f"      {c:14s} min {mins[c]:9.2f} median {statistics.median(samples[c]):9.2f} us", flush=True)
    result["util_after"] = util()
    json.dump(result, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
