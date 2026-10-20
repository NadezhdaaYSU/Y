"""accuracy.py <bench_ab.py dir> <ptx dir> <out.json> [--dump DIR] [case ...]

The accuracy of Y's F16 GEMM (f16 A, B -> f32 C) in both accumulation modes
against cuBLAS, Triton and a float32 control, over nine input distributions
and eleven shapes, each output compared with a float64 reference.

Columns (each C is M x N float32):
  Y_mma   <ptx dir>/mma/<kernel>.ptx: each mma chained into its accumulator
          (Y_F16_ACCUMULATE=mma; the default above 32 rows)
  Y_f32   <ptx dir>/f32/<kernel>.ptx: each mma from zero, added in float32
          with add.rn.f32 (Y_F16_ACCUMULATE=f32; the default up to 32 rows)
  cuBLAS  torch.mm(A, B, out_dtype=torch.float32), f16 in, float32 out,
          allow_fp16_reduced_precision_reduction off
  Triton  bench_ab.py's triton_gemm (tl.dot chained into a float32
          accumulator), where M, N are multiples of 256 and K of 64
  SGEMM   torch.mm(A.float(), B.float()) with TF32 off: float32 FMAs on the
          CUDA cores, the control for "accumulates in float32"
  RN32    R rounded to float32 (round to nearest even): the floor, the
          error of the best float32 output any kernel could give

Reference. A and B are f16; every product a*b of two f16 numbers is exact
in float64 (11-bit significands, exponents within float64's range), so
R = A.double() @ B.double() (cuBLAS DGEMM) is the exact sum up to float64
summation error, at most about K * 2^-53 * S. That is checked on 32 sampled
outputs a case against math.fsum of the same products (the correctly
rounded exact sum): "ref_check".

Metrics, for E = C - R and S = |A| @ |B| (float64; S[i,j] = sum_k
|a_ik| |b_kj| >= |R[i,j]|):
  rel_fro    ||E||_F / ||R||_F
  max_abs    max_ij |E_ij|
  max_norm   max_ij |E_ij| / S_ij over S_ij > 0
  mean_norm  mean_ij E_ij / S_ij (signed: the bias), and mean_abs_norm
Near-zero references: nothing is divided by |R_ij|. A float32 sum of
these products can be off by about 2^-24 S_ij even when every step
rounds correctly, so |E_ij| / |R_ij| measures how much the sum cancels
(kappa_ij = S_ij / |R_ij|), not the kernel; max_norm divides by S_ij
instead, which is zero only when every product is zero, and then R_ij
is exactly zero: those outputs are left out of max_norm and counted in
"zero_scale_nonzero" if C_ij is not exactly 0. ||R||_F is asserted
positive. Per case, "kappa_F" = ||S||_F / ||R||_F and the median of
kappa_ij describe the cancellation; an output's relative error is at
most kappa_ij times its normalized error.

Per output, against cuBLAS (and Y_f32 against SGEMM): the fractions of
outputs with smaller, equal and larger |E_ij|, and bitwise equal.

Inputs: numpy.random.RandomState(seed) (whose stream numpy keeps fixed
across versions), drawn in float64 and rounded to f16 by numpy
(round to nearest even); see DISTS for each. The SHA-256 of every
operand's f16 bytes (C order, little endian) is recorded, and of every
output. --dump DIR writes each case's A and B as .npy.
"""
import hashlib
import json
import math
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, sys.argv[1])
from bench_ab import kernel_from_ptx, triton_gemm  # noqa: E402

import triton  # noqa: E402

F16, F64 = np.float16, np.float64


def normal(rs, m, n, k):
    return rs.standard_normal((m, k)), rs.standard_normal((k, n))


def uniform_pm1(rs, m, n, k):
    return rs.uniform(-1.0, 1.0, (m, k)), rs.uniform(-1.0, 1.0, (k, n))


def abs_normal(rs, m, n, k):
    a, b = normal(rs, m, n, k)
    return np.abs(a), np.abs(b)


def uniform01(rs, m, n, k):
    return rs.random_sample((m, k)), rs.random_sample((k, n))


def offset(rs, m, n, k):
    # rows of A nearly constant, columns of B centred: sum_k a_ik b_kj
    # cancels its 1 * sum_k b_kj term (kappa grows like 80 sqrt(K))
    a = 1.0 + 0.01 * rs.standard_normal((m, k))
    b = rs.standard_normal((k, n))
    return a, b - b.mean(axis=0, keepdims=True)


def paired(rs, m, n, k):
    # A = [U | U], B = [V ; -V + 2^-10 W]: the second half of K cancels the
    # first half's sum, leaving sum_k u_k (v_k + fl16(-v_k + 2^-10 w_k))
    h = k // 2
    u = rs.standard_normal((m, h))
    v = rs.standard_normal((h, n)).astype(F16).astype(F64)
    w = rs.standard_normal((h, n))
    return np.concatenate([u, u], axis=1), np.concatenate([v, -v + 2.0 ** -10 * w], axis=0)


def _wide(rs, shape):
    sign = np.where(rs.random_sample(shape) < 0.5, -1.0, 1.0)
    mant = 1.0 + rs.random_sample(shape)
    exp = rs.randint(-10, 11, shape)  # 2^-10 .. 2^10
    return sign * mant * np.exp2(exp)


def wide_range(rs, m, n, k):
    return _wide(rs, (m, k)), _wide(rs, (k, n))


def small(rs, m, n, k):
    a, b = normal(rs, m, n, k)
    return a * 2.0 ** -12, b * 2.0 ** -12  # about a fifth of them f16 subnormals


def large(rs, m, n, k):
    a, b = normal(rs, m, n, k)
    return a * 2.0 ** 7, b * 2.0 ** 7


DISTS = [normal, uniform_pm1, abs_normal, uniform01, offset, paired, wide_range, small, large]
SHAPES = ["256", "1024", "4096", "1x4096x4096", "8x4096x4096", "32x4096x4096",
          "512x512x1024", "512x512x4096", "512x512x16384", "512x512x65536", "8192"]
KERNEL = {"256": "gemm_f16_256", "1024": "gemm_f16_1024", "4096": "gemm_f16_4096", "8192": "gemm_f16_8192",
          "1x4096x4096": "gemm_f16_decode_m1", "8x4096x4096": "gemm_f16_decode_m8",
          "32x4096x4096": "gemm_f16_decode_m32"}


def dims(shape):
    return tuple(int(x) for x in shape.split("x")) if "x" in shape else (int(shape),) * 3


def seed_of(di, si):
    return 1000 * (di + 1) + si


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def sha_t(t):
    return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()


def input_stats(x16):
    ax = np.abs(x16.astype(F64))
    return {"frac_zero": float((ax == 0).mean()),
            "frac_subnormal": float(((ax > 0) & (ax < 2.0 ** -14)).mean()),
            "max_abs": float(ax.max()), "min_abs_nonzero": float(ax[ax > 0].min())}


def metrics(c, r, s, pos, r_fro):
    e = c.double() - r
    ae = e.abs()
    en = e[pos] / s[pos]
    return {"rel_fro": torch.linalg.vector_norm(e).item() / r_fro,
            "max_abs": ae.max().item(),
            "max_norm": en.abs().max().item(),
            "mean_norm": en.mean().item(),
            "mean_abs_norm": en.abs().mean().item(),
            "zero_scale_nonzero": int(((~pos) & (c != 0)).sum().item())}


def versus(c, base, r):
    ec, eb = (c.double() - r).abs(), (base.double() - r).abs()
    return {"better": (ec < eb).double().mean().item(), "equal": (ec == eb).double().mean().item(),
            "worse": (ec > eb).double().mean().item(),
            "bitwise": (c.view(torch.int32) == base.view(torch.int32)).double().mean().item()}


def kernels_of(fn):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    return sorted({e.name[:100] for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA})


def launch_y(path, name, a, b, m, n):
    kern = kernel_from_ptx(path, name)
    cm, cn, wm, wn = (int(x) for x in re.search(r"CTA (\d+)x(\d+)x\d+ \| (\d+)x(\d+) warps", kern.ptx).groups())
    y = torch.full((m, n), float("nan"), device="cuda")
    kern.launch((-(-n // cn), -(-m // cm), 1), (wm * wn * 32, 1, 1), [a, b, y])
    torch.cuda.synchronize()
    return y


def ref_check(a16, b16, r, s, cols, rs, m, n):
    idx = [(int(rs.randint(m)), int(rs.randint(n))) for _ in range(32)]
    rc, sc = r.cpu().numpy(), s.cpu().numpy()
    dev, rel, worst_ratio = 0.0, 0.0, 0.0
    for i, j in idx:
        exact = math.fsum((a16[i, :].astype(F64) * b16[:, j].astype(F64)).tolist())
        d = abs(rc[i, j] - exact)
        dev = max(dev, d / sc[i, j] if sc[i, j] > 0 else (0.0 if d == 0 else math.inf))
        if exact != 0:
            rel = max(rel, d / abs(exact))
        # the reference's deviation against each column's own error there
        for c in cols.values():
            ec = abs(float(c[i, j]) - exact)
            if ec > 0:
                worst_ratio = max(worst_ratio, d / ec)
    return {"entries": len(idx), "max_dev_over_S": dev, "max_dev_over_exact": rel,
            "max_dev_over_column_error": worst_ratio}


def run_case(di, si, ptx_dir, dump):
    dist, shape = DISTS[di], SHAPES[si]
    m, n, k = dims(shape)
    seed = seed_of(di, si)
    rs = np.random.RandomState(seed)
    a64, b64 = dist(rs, m, n, k)
    a16, b16 = a64.astype(F16), b64.astype(F16)
    del a64, b64
    assert np.isfinite(a16).all() and np.isfinite(b16).all()
    if dump:
        np.save(f"{dump}/{dist.__name__}_{shape}_A.npy", a16)
        np.save(f"{dump}/{dist.__name__}_{shape}_B.npy", b16)
    case = {"dist": dist.__name__, "shape": shape, "m": m, "n": n, "k": k, "seed": seed,
            "sha256_A": sha(a16), "sha256_B": sha(b16), "A": input_stats(a16), "B": input_stats(b16)}
    a = torch.from_numpy(a16).cuda()
    b = torch.from_numpy(b16).cuda()
    r = a.double() @ b.double()
    s = a.double().abs() @ b.double().abs()
    pos = s > 0
    r_fro = torch.linalg.vector_norm(r).item()
    assert r_fro > 0
    kap = s / r.abs()  # inf where R is exactly 0
    case["kappa_F"] = torch.linalg.vector_norm(s).item() / r_fro
    case["kappa_median"] = kap.median().item()
    case["r_exact_zero"] = int((r == 0).sum().item())
    kname = KERNEL.get(shape, f"gemm_f16_{shape}")
    cols = {"Y_mma": launch_y(f"{ptx_dir}/mma/{kname}.ptx", kname, a, b, m, n),
            "Y_f32": launch_y(f"{ptx_dir}/f32/{kname}.ptx", kname, a, b, m, n),
            "cuBLAS": torch.mm(a, b, out_dtype=torch.float32)}
    if m % 256 == 0 and n % 256 == 0 and k % 64 == 0:  # the Triton kernel has no masks
        t = torch.full((m, n), float("nan"), device="cuda")
        triton_gemm[lambda meta: (triton.cdiv(m, meta["BM"]), triton.cdiv(n, meta["BN"]))](a, b, t, m, n, k)
        torch.cuda.synchronize()
        cols["Triton"] = t
        case["triton_config"] = str(triton_gemm.best_config)
    af, bf = a.float(), b.float()
    cols["SGEMM"] = torch.mm(af, bf)
    cols["RN32"] = r.float()
    torch.cuda.synchronize()
    for name, c in cols.items():
        assert torch.isfinite(c).all(), (shape, dist.__name__, name)
    case["cublas_kernels"] = kernels_of(lambda: torch.mm(a, b, out_dtype=torch.float32))
    case["sgemm_kernels"] = kernels_of(lambda: torch.mm(af, bf))
    case["columns"] = {name: metrics(c, r, s, pos, r_fro) for name, c in cols.items()}
    for name, c in cols.items():
        case["columns"][name]["sha256_C"] = sha_t(c)
    case["vs_cuBLAS"] = {name: versus(c, cols["cuBLAS"], r) for name, c in cols.items() if name != "cuBLAS"}
    case["Y_f32_vs_SGEMM"] = versus(cols["Y_f32"], cols["SGEMM"], r)
    case["ref_check"] = ref_check(a16, b16, r, s, {x: cols[x].cpu().numpy() for x in ("Y_mma", "Y_f32", "cuBLAS", "SGEMM")},
                                  np.random.RandomState(seed + 500000), m, n)
    del a, b, af, bf, r, s, pos, kap, cols
    torch.cuda.empty_cache()
    return case


def line(case):
    c = case["columns"]
    cells = " | ".join(f"{x} fro {v['rel_fro']:.2e} maxabs {v['max_abs']:.2e} maxn {v['max_norm']:.2e} mean {v['mean_norm']:+.1e}"
                       for x, v in c.items())
    vs = case["vs_cuBLAS"]["Y_f32"]
    return (f"{case['dist']:>11s} {case['shape']:>14s} seed {case['seed']:5d} kF {case['kappa_F']:.1e} | {cells} | "
            f"Y_f32 vs cuBLAS better {vs['better']:.3f} equal {vs['equal']:.3f} worse {vs['worse']:.3f} | "
            f"ref {case['ref_check']['max_dev_over_S']:.1e}/S {case['ref_check']['max_dev_over_column_error']:.1e}x")


def main():
    ptx_dir, out_path = sys.argv[2], sys.argv[3]
    args = sys.argv[4:]
    dump = None
    if args[:1] == ["--dump"]:
        dump, args = args[1], args[2:]
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.zeros(1, device="cuda")
    wanted = [(di, si) for di in range(len(DISTS)) for si in range(len(SHAPES))
              if not args or f"{DISTS[di].__name__}:{SHAPES[si]}" in args]
    result = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
              "triton": triton.__version__,
              "numpy": np.__version__, "started": time.strftime("%F %T"), "cases": []}
    for di, si in wanted:
        case = run_case(di, si, ptx_dir, dump)
        result["cases"].append(case)
        print(line(case), flush=True)
        json.dump(result, open(out_path, "w"), indent=1)
    result["finished"] = time.strftime("%F %T")
    json.dump(result, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
