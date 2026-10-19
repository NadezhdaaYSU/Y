#!/usr/bin/env python3
"""check_all.py <ysu-dir> <off-dir> <on-dir> <name> ...: for each kernel, the
committed PTX (off) against the regenerated one (on): bit for bit on random
normal f16 inputs (the layout moves no arithmetic), and both exact on small
integers against float64 (cuBLAS's float32 result at 16384^3, exact there
because every partial sum is an integer below 2^24). Outputs NaN-filled;
bias+ReLU kernels get an integer bias and relu(A @ B + bias)."""
import re
import sys

import torch

from y_lang.torch_interop import TorchKernel


def load(path, name):
    k = TorchKernel.__new__(TorchKernel)
    k.source, k.kernel_name, k.target_sm = None, name, "sm_89"
    k.ptx = open(path).read()
    k._module = k._function = None
    k._ensure_loaded()
    return k


def main():
    ysu_dir, off_dir, on_dir = sys.argv[1:4]
    torch.zeros(1, device="cuda")
    bad = 0
    for name in sys.argv[4:]:
        src = open(f"{ysu_dir}/{name}.ysu").read()
        m, n, kk = (int(x) for x in re.search(r"@tile\((\d+),\s*(\d+),\s*(\d+)\)", src).groups())
        kname = re.search(r"kernel (\w+)\(", src).group(1)
        bias = "Bias:" in src
        ks = {mode: load(f"{d}/{name}.ptx", kname) for mode, d in (("off", off_dir), ("on", on_dir))}
        geo = {}
        for mode, k in ks.items():
            cm, cn, wm, wn = (int(x) for x in re.search(r"CTA (\d+)x(\d+)x\d+ \| (\d+)x(\d+) warps", k.ptx).groups())
            geo[mode] = ((-(-n // cn), -(-m // cm), 1), (wm * wn * 32, 1, 1))

        def run(mode, a, b, bvec):
            o = torch.full((m, n), float("nan"), device="cuda")
            args = [a, b, bvec, o] if bias else [a, b, o]
            ks[mode].launch(*geo[mode], args)
            torch.cuda.synchronize()
            return o

        g = torch.Generator(device="cuda").manual_seed(m * 7 + n * 3 + kk)
        trials = 1 if m * n * kk >= 4096 ** 3 else 2
        for trial in range(trials):
            a = torch.randn(m, kk, device="cuda", generator=g).half()
            b = torch.randn(kk, n, device="cuda", generator=g).half()
            bvec = torch.randn(n, device="cuda", generator=g)
            x, y = run("off", a, b, bvec), run("on", a, b, bvec)
            same = (not x.isnan().any()) and torch.equal(x.view(torch.int32), y.view(torch.int32))
            del x, y
            a = torch.randint(-4, 5, (m, kk), device="cuda", generator=g).half()
            b = torch.randint(-4, 5, (kk, n), device="cuda", generator=g).half()
            bvec = torch.randint(-64, 65, (n,), device="cuda", generator=g).float()
            if m * n * kk >= 16384 ** 3:
                ref = torch.mm(a, b, out_dtype=torch.float32)
            else:
                ref = (a.double() @ b.double()).float()
            if bias:
                ref = torch.relu(ref + bvec)
            ex = [torch.equal(run(mode, a, b, bvec), ref) for mode in ("off", "on")]
            del ref
            torch.cuda.empty_cache()
            print(f"{name} ({m}x{n}x{kk}) trial {trial}: random bitwise equal {same}; integer exact off {ex[0]} on {ex[1]}",
                  flush=True)
            bad += (not same) + (not ex[0]) + (not ex[1])
    print("FAILURES", bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
