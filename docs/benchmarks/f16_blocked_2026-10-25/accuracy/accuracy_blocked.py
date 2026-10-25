"""accuracy_blocked.py <bench_ab.py dir> <ptx dir> <out.json> [dist:shape ...]

The accuracy of Y's F16 GEMM with blocked summation (Y_F16_ACCUMULATE=blocked:
the mmas chained into a block sum, added into the accumulator in float32 every
Y_F16_BLOCK_K of K) against the chain, float32 accumulation, cuBLAS, a float32
control and the float32 floor, on the nine input distributions, seeds,
reference and metrics of f16_accuracy_2026-10-20/accuracy.py ($ACC_DIR,
imported unchanged): rel_fro = ||C - R||_F / ||R||_F, max_abs = max |C - R|,
max_norm = max |C - R| / S with S = |A| @ |B|, mean_norm (signed), R = A @ B
in float64 checked against math.fsum.

<ptx dir>/<shape>/<column>.ptx holds each Y kernel, launched as
f16_splitk_2026-10-20/accuracy_split.py ($SPLIT_DIR) does. Per output, every
column against cuBLAS, against SGEMM and against the chain (Y_mma): the
fractions of outputs with smaller, equal and larger |E_ij|, and bitwise equal.
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ["ACC_DIR"])
sys.path.insert(0, os.environ["SPLIT_DIR"])
import accuracy as acc  # noqa: E402  (inserts argv[1] and imports bench_ab)
from accuracy_split import launch  # noqa: E402

F16 = np.float16
SHAPES = ["256", "1024", "4096", "8192", "512x512x1024", "512x512x4096", "512x512x16384", "512x512x65536"]
# Shapes outside f16_accuracy_2026-10-20's eleven, seeded as its case (d, s) would be with s = 100 + their index.
EXTRA = ["2048", "64x4096x4096", "1024x4096x4096", "128x14336x4096", "512x4096x14336", "2048x4096x14336"]
SHAPES += EXTRA


def run_case(di, shape, ptx_dir):
    dist = acc.DISTS[di]
    si = acc.SHAPES.index(shape) if shape in acc.SHAPES else 100 + EXTRA.index(shape)
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
    d = f"{ptx_dir}/{shape}"
    cols = {c: launch(f"{d}/{c}.ptx", a, b, m, n) for c in sorted(x[:-4] for x in os.listdir(d) if x.endswith(".ptx"))}
    cols["cuBLAS"] = torch.mm(a, b, out_dtype=torch.float32)
    af, bf = a.float(), b.float()
    cols["SGEMM"] = torch.mm(af, bf)
    cols["RN32"] = r.float()
    torch.cuda.synchronize()
    for name, c in cols.items():
        assert torch.isfinite(c).all(), (shape, dist.__name__, name)
    case["cublas_kernels"] = acc.kernels_of(lambda: torch.mm(a, b, out_dtype=torch.float32))
    case["sgemm_kernels"] = acc.kernels_of(lambda: torch.mm(af, bf))
    case["columns"] = {name: acc.metrics(c, r, s, pos, r_fro) for name, c in cols.items()}
    for name, c in cols.items():
        case["columns"][name]["sha256_C"] = acc.sha_t(c)
    for base in ["cuBLAS", "SGEMM"]:
        case[f"vs_{base}"] = {name: acc.versus(c, cols[base], r) for name, c in cols.items() if name != base}
    case["vs_chain"] = {name: acc.versus(c, cols["Y_mma"], r) for name, c in cols.items() if name != "Y_mma"}
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
    torch.zeros(1, device="cuda")
    result = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
              "numpy": np.__version__, "started": time.strftime("%F %T"), "cases": []}
    for di in range(len(acc.DISTS)):
        for shape in SHAPES:
            if args and f"{acc.DISTS[di].__name__}:{shape}" not in args:
                continue
            if not os.path.isdir(f"{ptx_dir}/{shape}"):
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
