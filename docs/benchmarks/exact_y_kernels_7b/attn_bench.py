"""Rough GPU time of one decode-step exact attention at the 7B shape: the
torch ops compiled by Inductor (exact_attention_core) against Y's kernel on
the packed rows of the same quantised operands (`_y_decode`, as the model
runs it with Y_EXACT_KV_PACK=1). Profiler device time, after 2 s of warm-up.
Indicative only: the desktop may be using the GPU."""
import math
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile

from y_lang.exact import attention as A, kv

kv.V_INT8 = True
B, NH, NKV, D = 32, 28, 4, 128
T = int(sys.argv[1]) if len(sys.argv) > 1 else 100
g = torch.Generator(device="cuda").manual_seed(0)
q = torch.randn(B, NH, 1, D, device="cuda", generator=g)
k = torch.randn(B, NKV, T, D, device="cuda", generator=g)
v = torch.randn(B, NKV, T, D, device="cuda", generator=g)
sc = 1.0 / math.sqrt(D)
with torch.no_grad():
    # quantise once, as exact_attention does
    qi, sq = None, None
    lv = 127.0
    s = q.abs().amax(-1, keepdim=True) / A.exact_model.divisor(lv, q)
    s = torch.where(s == 0, torch.ones_like(s), s)
    qi = torch.round(q / s).clamp(-lv, lv)
    sq = s.to(torch.float64)
    ki, sk, vi, sv, k_lv, v_lv = kv.quantize_kv(None, k, v, 127)
table = A._table(torch.device("cuda"))
args = (qi, sq, ki, sk, vi, sv, k_lv, v_lv, None, sc, table, torch.float32)
# The model with Y_EXACT_KV_PACK=1 hands the kernel the cache's packed rows;
# packed once here, as the cache holds them.
kp, vp = kv.pack_k(ki, sk), kv.pack_v(vi, sv)


def run_torch():
    A.ATTN = "torch"
    return A.exact_attention_core(*args)


def run_y():
    A.ATTN = "y"
    return A._y_decode(qi, sq, sc, kp, vp, None, table, torch.float32)


compiled = torch.compile(run_torch, dynamic=False)
with torch.no_grad():
    want = compiled()
    got = run_y()
    assert torch.equal(want.view(torch.int32), got.view(torch.int32)), "bits differ"
    for name, fn in (("torch compiled", compiled), ("Y kernel", run_y)):
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 2.0:
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(50):
                fn()
            torch.cuda.synchronize()
        evs = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        total = sum(e.device_time for e in evs) / 50
        print(f"T={T} {name}: {total:.1f} us GPU per call in {len(evs) // 50} kernels")
        per = {}
        for e in evs:
            per[e.name] = per.get(e.name, 0.0) + e.device_time / 50
        for n, t in sorted(per.items(), key=lambda kv: -kv[1]):
            print(f"      {t:7.1f} us  {n[:90]}")
