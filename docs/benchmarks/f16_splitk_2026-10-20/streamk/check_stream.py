"""check_stream.py DIR: correctness of the stream-K kernels in DIR (gen_one.sh,
tile 128x128x32 4x2, f32 mode), against DIR/gemm_f16_512x512x4096.ptx, the
unsplit f32-mode kernel of the same tile. Checks, each with a control:
 1. small integers: C == float64 A @ B, for P = 1, 7, 66, 200 (K=4096) and
    P = 66 (K=65536);
 2. normal inputs, P = 1: C bits == the unsplit kernel's;
 3. normal inputs, P = 7 and 66: every piece (slot j of tile t) == the
    unsplit kernel run with A's columns outside the piece's K range zeroed,
    in that tile (control: piece 0's slot against piece 1's oracle differs);
 4. C bits == the host's in-order float32 sum of each tile's pieces, from a
    Python copy of StreamKSchedule (control: reverse order differs);
 5. 20 relaunches give the same bits.
"""
import re, sys
import numpy as np
import torch
from y_lang.torch_interop import TorchKernel

D = sys.argv[1]
fails = []
torch.zeros(1, device="cuda")

def kern(ptx, name):
    k = TorchKernel.__new__(TorchKernel)
    k.source, k.kernel_name, k.target_sm = None, name, "sm_89"
    k.ptx = ptx; k._module = k._function = None; k._ensure_loaded()
    return k

def check(label, ok):
    print(f"{'ok  ' if ok else 'FAIL'} {label}", flush=True)
    if not ok:
        fails.append(label)

def beq(x, y):
    return bool(torch.equal(x.contiguous().view(torch.int32), y.contiguous().view(torch.int32)))

class Sched:  # StreamKSchedule
    def __init__(s, m, n, k, cm, cn, ck, p, kb):
        s.p, s.tn, s.kb, s.tiles = p, n // cn, kb, (m // cm) * (n // cn)
        s.iters = k // ck // kb
        s.total = s.tiles * kb * s.iters
        s.maxp = max(s.npieces(v) for v in range(s.tiles * kb))
    def begin(s, c): return c * s.total // s.p
    def owner(s, x): return -(-(x + 1) * s.p // s.total) - 1
    def npieces(s, v): return s.owner(v * s.iters + s.iters - 1) - s.owner(v * s.iters) + 1
    def pieces(s, t):  # [(slot, kstep_lo, kstep_hi)] of output tile t, in summation order
        out = []
        for kb in range(s.kb):
            v = kb * s.tiles + t
            x, end = v * s.iters, (v + 1) * s.iters
            while x < end:
                c = s.owner(x)
                e = min(end, s.begin(c + 1))
                out.append((kb * s.maxp + c - s.owner(v * s.iters), kb * s.iters + x - v * s.iters, kb * s.iters + e - v * s.iters))
                x = e
        return out

def plain(a, b, m, n):
    ptx = open(f"{D}/gemm_f16_512x512x4096.ptx").read() if a.shape[1] == 4096 else None
    k = kern(ptx, "gemm_f16_512x512x4096")
    c = torch.full((m, n), float("nan"), device="cuda")
    k.launch((n // 128, m // 128, 1), (256, 1, 1), [a, b, c])
    torch.cuda.synchronize()
    return c

def stream(name, a, b, m, n, tag=""):
    ptx = open(f"{D}/{name}{tag}.ptx").read()
    p = int(re.search(r"over (\d+) CTAs", ptx).group(1))
    assert int(re.search(r"Partial\[0\.\.(\d+)\]", ptx).group(1)) >= 1
    slots = int(re.search(r"Partial\[0\.\.(\d+)\]", ptx).group(1))
    main, red = kern(ptx, name), kern(ptx, name + "_reduce")
    c = torch.full((m, n), float("nan"), device="cuda")
    w = torch.full((slots, m, n), float("nan"), device="cuda")
    def go():
        main.launch((p, 1, 1), (256, 1, 1), [a, b, c, w])
        red.launch((m * n // 4 // 256, 1, 1), (256, 1, 1), [a, b, c, w])
    go(); torch.cuda.synchronize()
    return c, w, p, go

m = n = 512
gen = torch.Generator(device="cuda").manual_seed(7)
for kk, ps in ((4096, (1, 7, 66, 200)), (65536, (66,))):
    ai = torch.randint(-4, 5, (m, kk), device="cuda", generator=gen).half()
    bi = torch.randint(-4, 5, (kk, n), device="cuda", generator=gen).half()
    ref = (ai.double() @ bi.double()).float()
    for p in ps:
        c, *_ = stream(f"gemm_f16_512x512x{kk}_streamk{p}", ai, bi, m, n)
        check(f"K={kk} P={p} small integers == float64", bool(torch.equal(c, ref)))

kk = 4096
a = torch.randn(m, kk, device="cuda", generator=gen).half()
b = torch.randn(kk, n, device="cuda", generator=gen).half()
u = plain(a, b, m, n)
c1, *_ = stream("gemm_f16_512x512x4096_streamk1", a, b, m, n)
check("P=1 C bits == unsplit f32 kernel", beq(c1, u))
for p, kb, tag in [(7, 1, ""), (66, 1, ""), (66, 4, "_b4"), (66, 8, "_b8"), (7, 4, "_b4")]:
    c, w, p_, go = stream(f"gemm_f16_512x512x4096_streamk{p}", a, b, m, n, tag)
    sc = Sched(m, n, kk, 128, 128, 32, p, kb)
    p = f"{p} B={kb}"
    bad, n_pieces, ctrl = 0, 0, None
    host = torch.empty_like(c)
    host_rev = torch.empty_like(c)
    for t in range(16):
        tm, tn = divmod(t, sc.tn)
        rs, cs = slice(tm * 128, tm * 128 + 128), slice(tn * 128, tn * 128 + 128)
        pcs = sc.pieces(t)
        oracles = []
        for slot, kb, ke in pcs:
            am = torch.zeros_like(a)
            am[:, kb * 32:ke * 32] = a[:, kb * 32:ke * 32]
            o = plain(am, b, m, n)
            oracles.append(o[rs, cs])
            n_pieces += 1
            bad += not beq(w[slot, rs, cs], o[rs, cs])
        if ctrl is None and len(pcs) > 1:
            ctrl = beq(w[pcs[0][0], rs, cs], oracles[1])
        acc = w[pcs[0][0], rs, cs].clone()
        for slot, _, _ in pcs[1:]:
            acc = acc + w[slot, rs, cs]
        host[rs, cs] = acc
        rev = w[pcs[-1][0], rs, cs].clone()
        for slot, _, _ in reversed(pcs[:-1]):
            rev = rev + w[slot, rs, cs]
        host_rev[rs, cs] = rev
    check(f"P={p}: {n_pieces - bad} of {n_pieces} pieces == unsplit kernel on the piece's K range", bad == 0)
    check(f"  control: a piece against its neighbour's oracle differs", ctrl is False)
    check(f"P={p}: C bits == in-order float32 sum of each tile's pieces", beq(c, host))
    d = int((c.view(torch.int32) != host_rev.view(torch.int32)).sum())
    if max(len(sc.pieces(t)) for t in range(16)) >= 3:  # with two, a + b == b + a
        check(f"  control: reverse order differs in {d} outputs", d > 0)
    first, same = c.clone(), True
    for _ in range(20):
        c.fill_(float("nan")); w.fill_(float("nan")); go(); torch.cuda.synchronize()
        same &= beq(c, first)
    check(f"P={p}: 20 relaunches give the same bits", same)
    r = a.double() @ b.double()
    print(f"     rel_F P={p} {float((c.double() - r).norm() / r.norm()):.3e}, unsplit {float((u.double() - r).norm() / r.norm()):.3e}")
print("FAILED: " + "; ".join(fails) if fails else "all passed")
sys.exit(1 if fails else 0)
