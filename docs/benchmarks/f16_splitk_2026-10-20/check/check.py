"""check.py DIR: correctness of the split-K F16 GEMM kernels gen_check.sh wrote.

Every launch writes into NaN-filled buffers. Checks, each with the control
that shows it can fail:
 1. small integers (exact in f16 products and every f32 partial sum): C ==
    float64 A @ B, for every split kernel;
 2. normal inputs: `_splitk1` C bits == the unsplit f32-mode kernel's
    (same tile, same code path but the workspace);
 3. normal inputs: Partial[z] bits == the unsplit f32-mode kernel of depth K/S
    run on contiguous copies of A[:, zK/S:(z+1)K/S] and B[zK/S:(z+1)K/S]
    (control: slot z against slice z+1 differs);
 4. C bits == ((P0 + P1) + P2) + P3 summed on the host in float32
    (control: the reverse order differs somewhere);
 5. 20 more launches give C's bits every time.
"""
import math, re, sys
import numpy as np
import torch
from y_lang.torch_interop import TorchKernel

D = sys.argv[1]
fails = []

def kern(path, name):
    k = TorchKernel.__new__(TorchKernel)
    k.source, k.kernel_name, k.target_sm = None, name, "sm_89"
    k.ptx = open(path).read()
    k._module = k._function = None
    k._ensure_loaded()
    return k

def tile(ptx):
    return [int(x) for x in re.search(r"CTA (\d+)x(\d+)x\d+ \| (\d+)x(\d+) warps", ptx).groups()]

def plain(name, m, n, a, b):
    k = kern(f"{D}/{name}.ptx", name)
    cm, cn, wm, wn = tile(k.ptx)
    c = torch.full((m, n), float("nan"), device="cuda")
    k.launch((-(-n // cn), -(-m // cm), 1), (wm * wn * 32, 1, 1), [a.contiguous(), b.contiguous(), c])
    torch.cuda.synchronize()
    return c

def split(name, m, n, s, a, b, d=D):
    main, red = kern(f"{d}/{name}.ptx", name), kern(f"{d}/{name}.ptx", name + "_reduce")
    cm, cn, wm, wn = tile(main.ptx)
    c = torch.full((m, n), float("nan"), device="cuda")
    p = torch.full((s, m, n), float("nan"), device="cuda")
    def go():
        main.launch((-(-n // cn), -(-m // cm), s), (wm * wn * 32, 1, 1), [a, b, c, p])
        red.launch((-(-(m * n // 4) // 256), 1, 1), (256, 1, 1), [a, b, c, p])
    go()
    torch.cuda.synchronize()
    return c, p, go

def check(label, ok):
    print(f"{'ok  ' if ok else 'FAIL'} {label}", flush=True)
    if not ok:
        fails.append(label)

def bits_eq(x, y):
    return bool(torch.equal(x.view(torch.int32), y.view(torch.int32)))

for (m, n, k, s, base) in [(512, 512, 4096, 4, "gemm_f16_512x512x4096"), (200, 136, 4000, 4, "gemm_f16_200x136x4000")]:
    ks = k // s
    gen = torch.Generator(device="cuda").manual_seed(m + n + k)
    # 1. exact on small integers
    ai = torch.randint(-4, 5, (m, k), device="cuda", generator=gen).half()
    bi = torch.randint(-4, 5, (k, n), device="cuda", generator=gen).half()
    ref = (ai.double() @ bi.double()).float()
    for ss in (1, s):
        c, p, _ = split(f"{base}_splitk{ss}", m, n, ss, ai, bi)
        check(f"{base}_splitk{ss} small integers == float64", bool(torch.equal(c, ref)))
    c, p, _ = split(f"{base}_splitk{s}", m, n, s, ai, bi, d=f"{D}/mma") if m == 512 else (None, None, None)
    if c is not None:
        check(f"{base}_splitk{s} (Y_F16_ACCUMULATE=mma) small integers == float64", bool(torch.equal(c, ref)))
    # normal inputs
    a = torch.randn(m, k, device="cuda", generator=gen).half()
    b = torch.randn(k, n, device="cuda", generator=gen).half()
    c1, _, _ = split(f"{base}_splitk1", m, n, 1, a, b)
    u = plain(base, m, n, a, b)
    check(f"{base}_splitk1 bits == unsplit f32 mode", bits_eq(c1, u))
    c, p, go = split(f"{base}_splitk{s}", m, n, s, a, b)
    slices = [plain(f"gemm_f16_{m}x{n}x{ks}", m, n, a[:, z * ks:(z + 1) * ks], b[z * ks:(z + 1) * ks]) for z in range(s)]
    for z in range(s):
        check(f"{base}_splitk{s} Partial[{z}] bits == unsplit f32 mode on slice {z}", bits_eq(p[z], slices[z]))
    check(f"  control: Partial[0] bits != slice 1", not bits_eq(p[0], slices[1]))
    ph = p.cpu().numpy()
    t = ph[0].copy()
    for z in range(1, s):
        t = (t + ph[z]).astype(np.float32)
    check(f"{base}_splitk{s} C bits == in-order float32 sum of the slots", np.array_equal(c.cpu().numpy().view(np.int32), t.view(np.int32)))
    r = ph[s - 1].copy()
    for z in range(s - 2, -1, -1):
        r = (r + ph[z]).astype(np.float32)
    diff = int((r.view(np.int32) != t.view(np.int32)).sum())
    check(f"  control: the reverse order differs in {diff} of {m * n} outputs", diff > 0)
    first = c.clone()
    same = True
    for _ in range(20):
        c.fill_(float("nan")); p.fill_(float("nan"))
        go(); torch.cuda.synchronize()
        same &= bits_eq(c, first)
    check(f"{base}_splitk{s} 20 relaunches give the same bits", same)
    err = lambda x: float(((x.double() - (a.double() @ b.double())).norm() / (a.double() @ b.double()).norm()))
    print(f"     rel_F: split{s} {err(c):.3e}  splitk1 {err(c1):.3e}  unsplit f32 {err(u):.3e}", flush=True)

print("FAILED: " + "; ".join(fails) if fails else "all passed")
sys.exit(1 if fails else 0)
