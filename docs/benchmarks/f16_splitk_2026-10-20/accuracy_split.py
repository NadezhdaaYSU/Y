"""accuracy_split.py <bench_ab.py dir> <ptx dir> <out.json> [case ...]

The accuracy of Y's split-K F16 GEMM at 512 x 512 x K against Y's unsplit
kernels, cuBLAS, a float32 control and the float32 floor, on the nine input
distributions, seeds, reference and metrics of
../f16_accuracy_2026-10-20/accuracy.py (imported from there, unchanged):
rel_fro = ||C - R||_F / ||R||_F, max_abs = max |C - R|, max_norm =
max |C - R| / S with S = |A| @ |B|, mean_norm (signed), R = A @ B in float64
checked against math.fsum.

Columns, each C an M x N float32:
  Y_mma      unsplit, chained in the tensor core's accumulator (the default)
  Y_f32      unsplit, each mma from zero added with add.rn.f32
  Y_f32_s<S> split K: S splits, each as Y_f32, then the S partials added in
             order with add.rn.f32 (the `_reduce` entry)
  Y_mma_s<S> split K with chained splits (Y_F16_ACCUMULATE=mma): the control
             that separates the split from the float32 accumulation
  cuBLAS, SGEMM, RN32  as in accuracy.py
<ptx dir>/<K>/<column>.ptx holds each Y kernel. Per output, every column
against cuBLAS and every split column against Y_f32: the fractions of
outputs with smaller, equal and larger |E_ij|.
"""
import json
import os
import re
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "f16_accuracy_2026-10-20"))
import accuracy as acc  # noqa: E402  (inserts argv[1] and imports bench_ab)

F16 = np.float16
SHAPES = ["512x512x4096", "512x512x16384", "512x512x65536"]


def launch(path, a, b, m, n):
    ptx = open(path).read()
    names = re.findall(r"\.visible \.entry (\w+)\(", ptx)
    name = [x for x in names if not x.endswith("_reduce")][0]
    cm, cn, wm, wn = (int(x) for x in re.search(r"CTA (\d+)x(\d+)x\d+ \| (\d+)x(\d+) warps", ptx).groups())
    main = acc.kernel_from_ptx(path, name)
    y = torch.full((m, n), float("nan"), device="cuda")
    if name + "_reduce" not in names:
        main.launch((-(-n // cn), -(-m // cm), 1), (wm * wn * 32, 1, 1), [a, b, y])
    else:
        s = int(re.search(r"in (\d+) splits", ptx).group(1))
        red = acc.kernel_from_ptx(path, name + "_reduce")
        p = torch.full((s, m, n), float("nan"), device="cuda")
        main.launch((-(-n // cn), -(-m // cm), s), (wm * wn * 32, 1, 1), [a, b, y, p])
        red.launch((-(-(m * n // 4) // 256), 1, 1), (256, 1, 1), [a, b, y, p])
    torch.cuda.synchronize()
    return y


def run_case(di, shape, ptx_dir):
    dist = acc.DISTS[di]
    si = acc.SHAPES.index(shape)
    m, n, k = acc.dims(shape)
    seed = acc.seed_of(di, si)
    rs = np.random.RandomState(seed)
    a64, b64 = dist(rs, m, n, k)
    a16, b16 = a64.astype(F16), b64.astype(F16)
    del a64, b64
    case = {"dist": dist.__name__, "shape": shape, "m": m, "n": n, "k": k, "seed": seed,
            "sha256_A": acc.sha(a16), "sha256_B": acc.sha(b16)}
    a, b = torch.from_numpy(a16).cuda(), torch.from_numpy(b16).cuda()
    r = a.double() @ b.double()
    s = a.double().abs() @ b.double().abs()
    pos = s > 0
    r_fro = torch.linalg.vector_norm(r).item()
    assert r_fro > 0
    case["kappa_F"] = torch.linalg.vector_norm(s).item() / r_fro
    d = f"{ptx_dir}/{k}"
    cols = {c: launch(f"{d}/{c}.ptx", a, b, m, n) for c in sorted(x[:-4] for x in os.listdir(d) if x.endswith(".ptx"))}
    cols["cuBLAS"] = torch.mm(a, b, out_dtype=torch.float32)
    af, bf = a.float(), b.float()
    cols["SGEMM"] = torch.mm(af, bf)
    cols["RN32"] = r.float()
    torch.cuda.synchronize()
    for name, c in cols.items():
        assert torch.isfinite(c).all(), (shape, dist.__name__, name)
    case["cublas_kernels"] = acc.kernels_of(lambda: torch.mm(a, b, out_dtype=torch.float32))
    case["columns"] = {name: acc.metrics(c, r, s, pos, r_fro) for name, c in cols.items()}
    for name, c in cols.items():
        case["columns"][name]["sha256_C"] = acc.sha_t(c)
    case["vs_cuBLAS"] = {name: acc.versus(c, cols["cuBLAS"], r) for name, c in cols.items() if name != "cuBLAS"}
    case["vs_Y_f32"] = {name: acc.versus(c, cols["Y_f32"], r) for name, c in cols.items() if "_s" in name}
    case["ref_check"] = acc.ref_check(a16, b16, r, s, {x: cols[x].cpu().numpy() for x in cols if x.startswith("Y_")},
                                      np.random.RandomState(seed + 500000), m, n)
    del a, b, af, bf, r, s, pos, cols
    torch.cuda.empty_cache()
    return case


def main():
    ptx_dir, out_path = sys.argv[2], sys.argv[3]
    args = sys.argv[4:]
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
              "numpy": np.__version__, "started": time.strftime("%F %T"), "cases": []}
    for di in range(len(acc.DISTS)):
        for shape in SHAPES:
            if args and f"{acc.DISTS[di].__name__}:{shape}" not in args:
                continue
            case = run_case(di, shape, ptx_dir)
            result["cases"].append(case)
            c = case["columns"]
            print(f"{case['dist']:>11s} {shape:>14s} kF {case['kappa_F']:.1e} | "
                  + " ".join(f"{x} {v['rel_fro']:.2e}" for x, v in c.items())
                  + " | vs cuBLAS better: " + " ".join(f"{x} {v['better']:.3f}" for x, v in case["vs_cuBLAS"].items() if x.startswith("Y_")),
                  flush=True)
            json.dump(result, open(out_path, "w"), indent=1)
    result["finished"] = time.strftime("%F %T")
    json.dump(result, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
