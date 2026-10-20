"""bench_swiglu.py <bench_ab.py dir> <fused label=dir,...> <plain-GEMM dir> <out.json> [N,...]:
Y's fused gate/up SwiGLU GEMM, Out = SiLU(X @ Wg) * (X @ Wu) (f16 X, Wg, Wu;
f32 Out), against Y's unfused path and PyTorch's, at M = N = K.

Each fused dir holds gemm_f16_swiglu_<N>.ptx; the plain-GEMM dir holds Y's
g<N>.ptx (kernel gemm_f16_<N>). Columns:
  <label>        the fused kernel from that dir
  Y unfused      two Y GEMMs (f32 out), then F.silu(g) * u in PyTorch
  PyTorch        two torch.mm(out_dtype=float32), then F.silu(g) * u
The fused kernels must give each other's output bit for bit (a layout moves
no arithmetic); every column is compared with float64 (its SiLU uses
ex2.approx, so with a tolerance: max |error| / max |reference| below 1e-4).
Timing as bench_ab.py: CUDA events over back-to-back calls, 17 interleaved
rounds, the first two dropped, minimum and median of 15.
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
    fused = [tuple(x.split("=", 1)) for x in sys.argv[2].split(",")]
    plain_dir, out_path = sys.argv[3], sys.argv[4]
    sizes = [int(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "512,1024,2048,4096").split(",")]
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    result = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "util_before": util(), "sizes": {}}
    for n in sizes:
        m = k = n
        gen = torch.Generator(device="cuda").manual_seed(n)
        x = (torch.randn(m, k, device="cuda", generator=gen) * 0.5).half()
        wg = (torch.randn(k, n, device="cuda", generator=gen) / k ** 0.5).half()
        wu = (torch.randn(k, n, device="cuda", generator=gen) / k ** 0.5).half()
        g64, u64 = x.double() @ wg.double(), x.double() @ wu.double()
        ref = g64 * torch.sigmoid(g64) * u64
        cols, outs, tiles = {}, {}, {}
        for label, d in fused:
            kern = kernel_from_ptx(f"{d}/gemm_f16_swiglu_{n}.ptx", f"gemm_f16_swiglu_{n}")
            cm, cn, wm, wn = (int(v) for v in re.search(r"CTA (\d+)x(\d+)x\d+ \(fixed\) \| (\d+)x(\d+) warps", kern.ptx).groups())
            tiles[label] = f"{cm}x{cn} {wm}x{wn}"
            outs[label] = torch.full((m, n), float("nan"), device="cuda")
            cols[label] = (lambda kern=kern, cm=cm, cn=cn, wm=wm, wn=wn, o=outs[label]:
                           kern.launch((-(-n // cn), -(-m // cm), 1), (wm * wn * 32, 1, 1), [x, wg, wu, o]))
        plain = kernel_from_ptx(f"{plain_dir}/g{n}.ptx", f"gemm_f16_{n}")
        cm, cn, wm, wn = (int(v) for v in re.search(r"CTA (\d+)x(\d+)x\d+ \| (\d+)x(\d+) warps", plain.ptx).groups())
        tiles["Y unfused"] = f"{cm}x{cn} {wm}x{wn}"
        gbuf = torch.empty(m, n, device="cuda")
        ubuf = torch.empty(m, n, device="cuda")

        def unfused():
            plain.launch((-(-n // cn), -(-m // cm), 1), (wm * wn * 32, 1, 1), [x, wg, gbuf])
            plain.launch((-(-n // cn), -(-m // cm), 1), (wm * wn * 32, 1, 1), [x, wu, ubuf])
            return F.silu(gbuf) * ubuf

        cols["Y unfused"] = unfused
        cols["PyTorch"] = lambda: F.silu(torch.mm(x, wg, out_dtype=torch.float32)) * torch.mm(x, wu, out_dtype=torch.float32)
        for name, f in cols.items():
            r = f()
            if r is not None:
                outs[name] = r
        torch.cuda.synchronize()
        scale = ref.abs().max().item()
        err = {name: ((o.double() - ref).abs().max().item() / scale) for name, o in outs.items()}
        labels = [l for l, _ in fused]
        same = all(torch.equal(outs[l].view(torch.int32), outs[labels[0]].view(torch.int32)) for l in labels)
        ok = {name: bool(e < 1e-4) for name, e in err.items()}
        calls = max(2, min(200, int(5e-3 / max(4 * m * n * k / 80e12, 2e-6))))
        names = [c for c in cols if ok[c]]
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
        result["sizes"][n] = {"tiles": tiles, "max_rel_err": err, "fused_bitwise_equal": same, "calls": calls,
                              "samples": samples, "sm_clock_mhz": clocks}
        print(f"{n:>6d} {tiles} | fused bitwise equal {same} | max err {', '.join(f'{c} {e:.1e}' for c, e in err.items())} | SM {min(clocks)}-{max(clocks)} MHz",
              flush=True)
        for c in sorted(mins, key=mins.get):
            print(f"      {c:12s} min {mins[c]:9.2f} median {statistics.median(samples[c]):9.2f} us", flush=True)
    result["util_after"] = util()
    json.dump(result, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
