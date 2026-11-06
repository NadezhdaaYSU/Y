# Y's int8 GEMM (C[M][N] int32 = A[M][K] int8 . B[N][K]^T) against cuBLASLt.
#
# cuBLASLt is torch._int_mm(A, B_nk.t()): B passed as the transpose VIEW of
# the [N][K] buffer, the column-major operand its int8 kernel takes without a
# copy (a .contiguous() inside the timed loop measured 0.2x, README).
#
# Correctness first, for every Y kernel and shape: every output against
# torch._int_mm AND against a float64 product of the same int8 values (exact:
# |sum| <= K * 2^14 < 2^53), C poisoned with 0xA5 before the launch; the
# staged kernel also with K split over z = 3 into a zeroed C, and on a grid of
# 3 CTAs.  Timing: per-call time, the minimum over 15 rounds, the order of the
# kernels shuffled every round (a fixed order handicaps whichever runs last at
# the card's power cap).
#
# --cold: every call reads its own copy of A and B, rotating over enough
# copies that they span 128 MB (this card's L2 is 48 MB), so the operands
# come from DRAM as a model's layers do; C is not rotated.
#
# Kernels: "rule" is Y with the tile its rule picks; "large", "mid", "small"
# force each tile (Y_INT8_TILE); "old" is the kernel this replaced, compiled
# by the binary in $YBIN_OLD (HEAD before the change), launched as its header
# says: block (32,1,1), grid (N/tn, M/tm, 1), C zeroed (it reduced).
#
# Usage: int8_bench.py [--cold] [--kernels rule,large,mid,small,old] MxNxK ...
import torch, cupy as cp, numpy as np, subprocess, sys, re, os, random

Y = os.environ.get("YBIN", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../target/release/Y")))
Y_OLD = os.environ.get("YBIN_OLD")
TILES = {
    "large": "128,128,64,2,2,3,2,32",
    "mid": "64,128,64,2,2,3,3,16",
    "small": "64,64,64,2,2,4,4,16",
}
PROFILE = os.environ.get("Y_PROFILE")  # a .ysu_hw_profile to compile with


def build(m, n, k, kind, workdir):
    name = f"i8_{kind}_{m}_{n}_{k}"
    src = os.path.join(workdir, name + ".ysu")
    open(src, "w").write(
        f"@tile({m}, {n}, {k})\nkernel {name}(A: GlobalMemory<I8>, B: GlobalMemory<I8>, C: GlobalMemory<I32>) {{\n}}\n\n"
        "fn main() {\n}\n")
    env = dict(os.environ)
    env.pop("Y_INT8_TILE", None)
    if kind in TILES:
        env["Y_INT8_TILE"] = TILES[kind]
    binary = Y_OLD if kind == "old" else Y
    r = subprocess.run([binary, src, "--emit-ptx", "-o", src[:-4] + ".ptx"], capture_output=True, text=True,
                       env=env, cwd=workdir)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    ptx = open(src[:-4] + ".ptx").read()
    if kind == "old":
        tl = re.search(r"\((\d+), (\d+), z\) is exact", ptx)
        blk, grid, smem, tile = 32, (int(tl.group(1)), int(tl.group(2)), 1), 0, re.search(r"warp tile (\d+x\d+)", ptx).group(1)
    else:
        blk = int(re.search(r"Launch: block \((\d+),1,1\)", ptx).group(1))
        grid = (int(re.search(r"Grid: any; \((\d+), 1, z\)", ptx).group(1)), 1, 1)
        smem = int(re.search(r"Dynamic shared memory required: (\d+) bytes", ptx).group(1))
        tile = re.search(r"\| CTA (\d+x\d+)x\d+", ptx).group(1)
    f = cp.RawModule(path=src[:-4] + ".ptx").get_function(name)
    if smem:
        f.max_dynamic_shared_size_bytes = smem
    return f, blk, grid, smem, tile, f.num_regs


def main():
    args = sys.argv[1:]
    cold = bool(args) and args[0] == "--cold"
    if cold:
        args = args[1:]
    kinds = ["rule"]
    if args and args[0] == "--kernels":
        kinds = args[1].split(",")
        args = args[2:]
    if "old" in kinds and not Y_OLD:
        sys.exit("--kernels old needs YBIN_OLD, a compiler built from the commit before the change")
    shapes = [tuple(map(int, s.split("x"))) for s in args]
    workdir = os.path.abspath(os.environ.get("WORKDIR", "work"))
    os.makedirs(workdir, exist_ok=True)
    if PROFILE:
        import shutil
        shutil.copy(PROFILE, os.path.join(workdir, ".ysu_hw_profile"))
    props = torch.cuda.get_device_properties(0)
    print(f"# {'cold: operands rotated over 128 MB of copies; ' if cold else ''}{props.name}, {props.multi_processor_count} SMs; torch {torch.__version__}, cupy {cp.__version__}, "
          f"CUDA runtime {cp.cuda.runtime.runtimeGetVersion()}, driver {cp.cuda.runtime.driverGetVersion()}", flush=True)
    torch.manual_seed(0)
    rng = random.Random(0)
    for (m, n, k) in shapes:
        a = torch.randint(-128, 128, (m, k), dtype=torch.int8, device="cuda")
        b = torch.randint(-128, 128, (n, k), dtype=torch.int8, device="cuda")
        cb = torch.empty(m, n, dtype=torch.int32, device="cuda")
        torch._int_mm(a, b.t(), out=cb)
        ref = cb.clone()
        exact = a.double() @ b.double().t()
        assert torch.equal(ref.double(), exact), "cuBLASLt's int8 product is not the exact one"
        copies = min(64, -(-(128 << 20) // (a.numel() + b.numel()))) if cold else 1
        As = [a] + [a.clone() for _ in range(copies - 1)]
        Bs = [b] + [b.clone() for _ in range(copies - 1)]
        A_, B_ = cp.asarray(a), cp.asarray(b)
        Acp, Bcp = [cp.asarray(x) for x in As], [cp.asarray(x) for x in Bs]
        runs = {"cuBLASLt": lambda i=0: torch._int_mm(As[i % copies], Bs[i % copies].t(), out=cb)}
        info = {}
        for kind in kinds:
            f, blk, grid, smem, tile, regs = build(m, n, k, kind, workdir)
            c = torch.empty(m, n, dtype=torch.int32, device="cuda")
            C_ = cp.asarray(c)

            def launch(g=grid, f=f, blk=blk, smem=smem, C_=C_, i=0):
                f(g, (blk, 1, 1), (Acp[i % copies], Bcp[i % copies], C_), shared_mem=smem)
            checks = []
            if kind == "old":
                c.zero_(); launch(); torch.cuda.synchronize()
                checks.append(torch.equal(c, ref))
            else:
                c.view(torch.uint8).fill_(0xA5); launch(); torch.cuda.synchronize()
                checks.append(torch.equal(c, ref) and torch.equal(c.double(), exact))
                c.zero_(); launch((grid[0], 1, 3)); torch.cuda.synchronize()
                checks.append(torch.equal(c, ref))
                c.view(torch.uint8).fill_(0xA5); launch((3, 1, 1)); torch.cuda.synchronize()
                checks.append(torch.equal(c, ref))
            assert all(checks), f"{m}x{n}x{k} {kind}: wrong bits {checks}"
            info[kind] = (tile, blk, grid[0] * grid[1], smem, regs)
            runs[kind] = lambda i, launch=launch: launch(i=i)
        reps = max(copies, min(200, max(3, int(2e11 / (2 * m * n * k)))))
        best = {v: 1e9 for v in runs}
        for _ in range(15):
            order = list(runs.items())
            rng.shuffle(order)
            for v, fn in order:
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                for i in range(reps):
                    fn(i)
                e.record(); e.synchronize()
                best[v] = min(best[v], s.elapsed_time(e) / reps)
        t0 = best["cuBLASLt"]
        row = f"{m}x{n}x{k}  cuBLASLt {t0 * 1e3:9.1f} us {2 * m * n * k / t0 / 1e9:6.1f} TOPS |"
        for kind in kinds:
            tile, blk, ctas, smem, regs = info[kind]
            row += f" {kind} {tile} {t0 / best[kind]:.3f}x"
        print(row + "  | bits ok", flush=True)
    print("# " + subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu", "--format=csv,noheader"],
                                capture_output=True, text=True).stdout.strip(), flush=True)


if __name__ == "__main__":
    main()
