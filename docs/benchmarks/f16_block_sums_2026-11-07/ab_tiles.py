"""A/B: the same CTA tile on 2x2 warps (no room for block sums, so the K sum
chains) against 4x2 warps (block sums fit, so it is blocked), against cuBLAS.

usage: ab_tiles.py ELEM MxNxK ...   (ELEM is BF16 or F16)

Per shape, before timing: both kernels give the exact integer product on
small integers; then on N(0,1) inputs the largest |C - exact| / sum|a||b|
and rel_F against float64 (on a 512 x 1024 corner for the big shapes) and
the share of outputs bit-identical to cuBLAS. Timing: per-call minimum over
15 rounds, the three kernels shuffled each round.
"""
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile

REPO = "/home/yumin/NVME files/YSU-engine-main/YSU-engine-main/src/Y_lang/Y"
sys.path.insert(0, os.path.join(REPO, "python"))
import torch  # noqa: E402
from y_lang.torch_interop import TorchKernel  # noqa: E402

ARMS = dict(a.split("=") for a in os.environ.get("AB_ARMS", "chain 2x2w=128,128,32,2,2,3;blocked 4x2w=128,128,32,4,2,3").split(";"))


def build(elem, m, n, k, tile, workdir):
    src = (f"@tile({m}, {n}, {k})\nkernel g(A: GlobalMemory<{elem}>, B: GlobalMemory<{elem}>, "
           f"C: GlobalMemory<F32>) {{\n}}\n")
    os.environ["Y_CTA_OVERRIDE"] = tile
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        kern = TorchKernel(src, kernel_name="g", target_sm="sm_89")
    finally:
        os.chdir(cwd)
        del os.environ["Y_CTA_OVERRIDE"]
    cm, cn, wm, wn = (int(v) for v in re.search(r"CTA (\d+)x(\d+)x\d+ \| (\d+)x(\d+) warps", kern.ptx).groups())
    adds = len(re.findall(r"\badd(?:\.rn)?\.f32\b", kern.ptx))
    return kern, ((n + cn - 1) // cn, (m + cm - 1) // cm, 1), (wm * wn * 32, 1, 1), adds


def main():
    elem = sys.argv[1]
    dt = {"BF16": torch.bfloat16, "F16": torch.float16}[elem]
    shapes = [tuple(map(int, s.split("x"))) for s in sys.argv[2:]]
    workdir = tempfile.mkdtemp(prefix="y_ab_tiles_")
    shutil.copy(os.path.join(REPO, ".ysu_hw_profile"), os.path.join(workdir, ".ysu_hw_profile"))
    torch.backends.cuda.matmul.allow_tf32 = False
    rng = random.Random(0)
    g = torch.Generator(device="cuda").manual_seed(0)
    for m, n, k in shapes:
        kerns = {name: build(elem, m, n, k, tile, workdir) for name, tile in ARMS.items()}
        c = {name: torch.empty(m, n, device="cuda") for name in ARMS}

        def launcher(name):
            kern, grid, block, _ = kerns[name]
            out = c[name]
            return lambda a, b: (kern.launch(grid, block, [a, b, out]), out)[1]

        ys = {name: launcher(name) for name in ARMS}
        rm, rn = min(m, 512), min(n, 1024)
        ai = torch.randint(-8, 9, (m, k), device="cuda", generator=g).to(dt)
        bi = torch.randint(-8, 9, (k, n), device="cuda", generator=g).to(dt)
        exact = (ai[:rm].double() @ bi[:, :rn].double())
        for name, y in ys.items():
            c[name].fill_(float("nan"))
            out = y(ai, bi)
            torch.cuda.synchronize()
            assert torch.equal(out[:rm, :rn].double(), exact), f"{elem} {m}x{n}x{k} {name}: not the exact product"
            assert not torch.isnan(out).any(), f"{elem} {m}x{n}x{k} {name}: NaN left in C"
        del ai, bi, exact
        a = torch.randn((m, k), device="cuda", generator=g).to(dt)
        b = torch.randn((k, n), device="cuda", generator=g).to(dt)
        cc = torch.mm(a, b, out_dtype=torch.float32)
        ref = a[:rm].double() @ b[:, :rn].double()
        scale = a[:rm].double().abs() @ b[:, :rn].double().abs()

        def errs(out):
            d = out[:rm, :rn].double() - ref
            return float((d.abs() / scale).max()), float(d.norm() / ref.norm())

        acc = {"cuBLAS": errs(cc)}
        same = {}
        for name, y in ys.items():
            out = y(a, b).clone()
            acc[name] = errs(out)
            same[name] = float((out.view(torch.int32) == cc.view(torch.int32)).float().mean())
        runs = {"cuBLAS": lambda: torch.mm(a, b, out_dtype=torch.float32)}
        runs.update({name: (lambda y=y: y(a, b)) for name, y in ys.items()})
        reps = max(3, min(500, int(0.02 * 8e13 / (2 * m * n * k))))
        best = {name: float("inf") for name in runs}
        for _ in range(15):
            order = list(runs.items())
            rng.shuffle(order)
            for name, fn in order:
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(reps):
                    fn()
                e.record()
                e.synchronize()
                best[name] = min(best[name], s.elapsed_time(e) / reps)
        t0 = best["cuBLAS"]
        line = [f"{elem} {m}x{n}x{k} cuBLAS {t0 * 1e3:.1f} us (maxn {acc['cuBLAS'][0]:.1e} relF {acc['cuBLAS'][1]:.1e})"]
        for name in ARMS:
            line.append(f"{name} [{kerns[name][3]} adds] {t0 / best[name]:.3f}x cuBLAS, "
                        f"maxn {acc[name][0]:.1e} relF {acc[name][1]:.1e}, same bits {100 * same[name]:.1f}%")
        print(" | ".join(line), flush=True)
        del a, b, cc, ref, scale, c, kerns, ys
        torch.cuda.empty_cache()
    print("# " + subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu",
                                 "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
    shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
