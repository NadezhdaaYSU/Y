"""Y's int8 GEMM in its Turing form (sm_75: mma.m8n8k16, no cp.async) on a
Tesla T4, against `torch._int_mm` (cuBLASLt).

    python3 gemm_t4.py            # round 1 (the decode shapes, six tiles)
    python3 gemm_t4.py round2     # round 2: tiles near round 1's best, decode and prefill rows

1. The same bits. `y_lang.int8_gemm.int8_mm` (the default tile, M padded
   to its row bucket) against `torch._int_mm(a, b.t())` on decode, prefill
   and square shapes, with full-range int8 and the extremes planted; on the
   small ones against a CPU int64 product too, which is not cuBLAS. Then
   each tile variant below at each decode shape, the same way.
2. The time, at the decode shapes of Qwen2.5-0.5B with batch 32 (M = 32):
   weights COLD, rotated over copies that together pass the T4's 4 MB L2
   many times, as in a decode step. Variants taken in turns each round, in
   a shuffled order, 9 rounds, the minimum kept (CUDA events over 50 calls).

Variants of Y's tile (`Y_INT8_TILE` = cta_m,cta_n,cta_k,warps_m,warps_n,
stages,ctas_per_sm,group_m): the rule's default, and smaller or two-stage
tiles. Without cp.async a CTA has one K tile of loads in flight, so stages
past two only take shared memory, which a T4 has 64 KB of per SM.
"""
import os
import random
import sys

import torch

from y_lang import int8_gemm as G
from y_lang.torch_interop import TorchKernel

DECODE = [("q/k/v", 1152, 896), ("o", 896, 896), ("gate or up", 4864, 896),
          ("down", 896, 4864), ("lm_head", 151936, 896)]
VARIANTS = [None, "64,64,64,2,2,2,4,16", "32,64,64,2,2,2,8,16", "32,128,64,2,2,2,4,16",
            "32,64,128,2,2,2,4,16", "32,32,64,2,2,2,8,16"]
PEAK = 320.0      # GB/s, Tesla T4 GDDR6
# Round 2: round 1's best (32x64, cta_k 128, 2 stages) and its neighbours,
# at M = 32 and at prefill rows (M = 128 and 2048) with the rule's default.
ROUND2 = [None, "32,64,128,2,2,2,4,16", "32,64,128,1,4,2,4,16", "64,64,128,2,2,2,4,16",
          "32,128,128,2,4,2,2,16", "32,64,128,2,2,3,2,16", "64,128,128,2,2,2,2,16",
          "128,128,64,2,2,2,2,32", "128,128,128,2,2,2,1,32"]


def operands(m, n, k, seed):
    g = torch.Generator().manual_seed(seed)
    a = torch.randint(-128, 128, (m, k), dtype=torch.int8, generator=g)
    b = torch.randint(-128, 128, (n, k), dtype=torch.int8, generator=g)
    a[0, 0], b[0, 0], a[-1, -1], b[-1, -1] = -128, -128, 127, 127
    return a, b


def variant(m, n, k, tile):
    """(call(a, b) -> c, description) for Y's kernel with `tile`, or None and
    the reason the emitter refused it."""
    if tile:
        os.environ["Y_INT8_TILE"] = tile
    try:
        kern = TorchKernel(G.source(m, n, k), kernel_name=G.KERNEL_NAME, target_sm=G._target("cuda"))
    except Exception as e:                      # the emitter refuses a tile it cannot stage
        return None, f"refused: {str(e).splitlines()[0][:120]}"
    finally:
        os.environ.pop("Y_INT8_TILE", None)
    threads, ctas, cta_k = G._header(kern.ptx, m, n, k)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    z = G.split_factor(ctas, -(-k // cta_k), sms)
    head = next(l for l in kern.ptx.splitlines() if "[Y INT8 TENSOR CORE GEMM]" in l)
    cta = head.split("|")[2].strip()
    stages = head.split("|")[4].strip()

    def call(a, b):
        c = torch.zeros((m, n), dtype=torch.int32, device="cuda") if z > 1 else \
            torch.empty((m, n), dtype=torch.int32, device="cuda")
        kern.launch((ctas, 1, z), (threads, 1, 1), [a, b, c])
        return c
    return call, f"{cta}, {stages}, {ctas} CTAs x z={z}"


def same_bits():
    bad = 0
    shapes = [(32, n, k) for _, n, k in DECODE] + [
        (17, 896, 896), (48, 72, 96), (64, 256, 128), (128, 4096, 4096), (256, 1024, 2048),
        (2048, 896, 896), (2048, 4864, 896), (32, 64, 32), (16, 64, 64)]
    for i, (m, n, k) in enumerate(shapes):
        a, b = operands(m, n, k, i)
        da, db = a.cuda(), b.cuda()
        got = G.int8_mm(da, db)
        ref = torch._int_mm(da, db.t()) if m > 16 else None
        ok_t = ref is None or torch.equal(got, ref)
        ok_c = True
        if m * n * k <= 64 * 4096 * 4096:
            ok_c = torch.equal(got.cpu().to(torch.int64), a.to(torch.int64) @ b.to(torch.int64).T)
        print(f"  int8_mm {m}x{n}x{k}: == _int_mm {ok_t if ref is not None else 'n/a (M <= 16)'}, == CPU int64 {ok_c}")
        bad += (not ok_t) + (not ok_c)
    for name, n, k in DECODE:
        a, b = operands(32, n, k, 7)
        da, db = a.cuda(), b.cuda()
        ref = torch._int_mm(da, db.t())
        for tile in VARIANTS[1:]:
            call, what = variant(32, n, k, tile)
            if call is None:
                print(f"  {name} {tile}: {what}")
                continue
            ok = torch.equal(call(da, db), ref)
            bad += not ok
            if not ok:
                print(f"  *** {name} {tile} ({what}): NOT the same bits")
    print(f"same bits: {'all' if bad == 0 else f'{bad} FAILED'}")
    return bad


def timing(variants=VARIANTS, rows=(32,)):
    print(f"\nweights cold (copies rotated past the 4 MB L2), min of 9 rounds of 50 calls; peak {PEAK} GB/s")
    for (name, n, k), m in [(s, m) for m in rows for s in DECODE if not (m > 32 and s[0] == "lm_head")]:
        copies = max(2, -(-64 * 2**20 // (n * k)))
        bs = [torch.randint(-128, 128, (n, k), dtype=torch.int8, device="cuda") for _ in range(copies)]
        bts = [b.t() for b in bs]
        a = torch.randint(-128, 128, (m, k), dtype=torch.int8, device="cuda")
        arms = {"torch._int_mm": (lambda b, i: torch._int_mm(a, bts[i]), "cuBLASLt")}
        for tile in variants:
            call, what = variant(m, n, k, tile)
            if call is None:
                print(f"  {name}: Y {tile}: {what}")
                continue
            arms[f"Y {tile or 'default'}"] = ((lambda c: lambda b, i: c(a, bs[i]))(call), what)
        best = {key: float("inf") for key in arms}
        for _ in range(9):
            order = list(arms)
            random.shuffle(order)
            for key in order:
                fn = arms[key][0]
                for i in range(3):
                    fn(None, i % copies)
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                s.record()
                for i in range(50):
                    fn(None, i % copies)
                e.record()
                torch.cuda.synchronize()
                best[key] = min(best[key], s.elapsed_time(e) * 1e3 / 50)
        base = best["torch._int_mm"]
        print(f"\n  {name}  M={m} N={n} K={k}  ({n * k / 1e6:.2f} MB of weights, {copies} copies)")
        for key, us in sorted(best.items(), key=lambda x: x[1]):
            gbs = n * k / us / 1e3
            print(f"    {key:<32} {us:8.1f} us  {gbs:6.1f} GB/s  {100 * gbs / PEAK:4.0f}%  {base / us:5.2f}x _int_mm  [{arms[key][1]}]")


def main():
    cap = torch.cuda.get_device_capability()
    print(f"{torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]}, {torch.cuda.get_device_properties(0).multi_processor_count} SMs, torch {torch.__version__}")
    if sys.argv[1:] == ["round2"]:
        bad = 0
        for m in (32, 128, 2048):
            for name, n, k in DECODE[:4]:
                a, b = operands(m, n, k, 5)
                da, db = a.cuda(), b.cuda()
                ref = torch._int_mm(da, db.t())
                for tile in ROUND2[1:]:
                    call, what = variant(m, n, k, tile)
                    if call is None:
                        print(f"  M={m} {name} {tile}: {what}")
                    elif not torch.equal(call(da, db), ref):
                        bad += 1
                        print(f"  *** M={m} {name} {tile} ({what}): NOT the same bits")
        print(f"round 2, same bits: {'all' if bad == 0 else f'{bad} FAILED'}")
        if bad:
            sys.exit(1)
        timing(ROUND2, rows=(32, 128, 2048))
        return
    bad = same_bits()
    if bad:
        sys.exit(1)
    timing()


if __name__ == "__main__":
    main()
