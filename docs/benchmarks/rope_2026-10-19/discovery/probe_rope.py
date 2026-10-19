#!/usr/bin/env python3
"""probe_rope.py <examples dir> <bench_ab dir> <committed rope_<hd>.ptx> <variant dir> <hd> <rows,...>

Times the committed rope kernel (one 32-thread CTA a row), the hand-written
mappings in <variant dir> (mk_rope.py: v_<tpr>_<slots>_<unroll>.ptx, a row
count parameter) and Triton at every (ROWS, num_warps) configuration, the
fastest three of which (screened at each size) enter the interleaved rounds.

Checks first, at each size and at an odd row count (rows + 3, so the last
CTA of every variant is partial): every column within 4e-3 of float64's
largest output, every variant bit for bit equal to the committed kernel, and
the variants' outputs past the last row untouched (NaN-filled)."""
import glob, os, statistics, sys
import torch, triton
import triton.language as tl
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
import run_head_to_head_comparison as H
from bench_ab import kernel_from_ptx
committed_path, vdir, HD = sys.argv[3], sys.argv[4], int(sys.argv[5])
SIZES = [int(s) for s in sys.argv[6].split(",")]
torch.zeros(1, device="cuda")
gen = torch.Generator(device="cuda").manual_seed(13)


@triton.jit
def t_rope(x_ptr, p_ptr, o_ptr, rows, D: tl.constexpr, ROWS: tl.constexpr):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    i = tl.arange(0, D // 2)
    m = r < rows
    pos = tl.load(p_ptr + r, mask=m, other=0).to(tl.float32)
    inv = tl.exp2(-(2.0 * i.to(tl.float32) / D) * 13.287712379549449)  # log2(10000)
    th = pos[:, None] * inv[None, :]
    c, s = tl.cos(th), tl.sin(th)
    base = r[:, None] * D + 2 * i[None, :]
    mm = m[:, None]
    x0 = tl.load(x_ptr + base, mask=mm, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + base + 1, mask=mm, other=0.0).to(tl.float32)
    tl.store(o_ptr + base, (x0 * c - x1 * s).to(tl.float16), mask=mm)
    tl.store(o_ptr + base + 1, (x0 * s + x1 * c).to(tl.float16), mask=mm)


kc = kernel_from_ptx(committed_path, os.path.basename(committed_path)[:-4])
variants = {}
for p in sorted(glob.glob(f"{vdir}/v_*.ptx")):
    head = open(p).read().split("// threads ")[1].split("\n")[0].split()
    threads, per_cta = int(head[0]), int(head[2])
    variants[os.path.basename(p)[:-4]] = (kernel_from_ptx(p, "ropev"), threads, per_cta)
TCONF = [(R, w) for R in (1, 2, 4, 8, 16, 32, 64) for w in (1, 2, 4, 8)]


def inputs(rows):
    x = torch.randn(rows, HD, device="cuda", generator=gen).half()
    pos = torch.randint(0, 4096, (rows,), device="cuda", generator=gen, dtype=torch.int32)
    inv = 10000.0 ** (-torch.arange(0, HD, 2, device="cuda", dtype=torch.float64) / HD)
    th = pos.double()[:, None] * inv[None, :]
    xv = x.double().view(rows, HD // 2, 2)
    ref = torch.stack([xv[..., 0] * th.cos() - xv[..., 1] * th.sin(),
                       xv[..., 0] * th.sin() + xv[..., 1] * th.cos()], -1).reshape(rows, HD)
    return x, pos, ref


def columns(rows, x, pos, o):
    cols = {"committed": lambda: kc.launch((rows, 1, 1), (32, 1, 1), [x, pos, o["committed"]])}
    for n, (k, t, per) in variants.items():
        cols[n] = lambda k=k, t=t, per=per, n=n: k.launch(((rows + per - 1) // per, 1, 1), (t, 1, 1), [x, pos, o[n], rows])
    for R, w in TCONF:
        n = f"triton_r{R}w{w}"
        cols[n] = lambda R=R, w=w, n=n: t_rope[(triton.cdiv(rows, R),)](x, pos, o[n], rows, D=HD, ROWS=R, num_warps=w)
    return cols


def check(rows):
    x, pos, ref = inputs(rows)
    pad = 64 * 8   # rows of NaN past the end, to see a partial CTA write nothing there
    o = {}
    names = ["committed", *variants, *(f"triton_r{R}w{w}" for R, w in TCONF)]
    for n in names:
        o[n] = torch.full((rows + pad, HD), float("nan"), device="cuda", dtype=torch.half)
    cols = columns(rows, x, pos, o)
    for f in cols.values():
        f()
    torch.cuda.synchronize()
    tol = 4e-3 * ref.abs().max().item()
    for n in names:
        out = o[n][:rows]
        err = (out.double() - ref).abs().max().item()
        assert torch.isfinite(out).all() and err <= tol, (rows, n, err)
        assert torch.isnan(o[n][rows:]).all(), (rows, n, "wrote past the last row")
        if n in variants:
            assert torch.equal(out.view(torch.int16), o["committed"][:rows].view(torch.int16)), (rows, n, "bits")
    print(f"checked rows {rows}: {len(names)} columns within {tol:.2e} of float64, variants bit for bit the committed kernel", flush=True)


def bench(rows):
    x, pos, ref = inputs(rows)
    o = {}
    cols = columns(rows, x, pos, o)
    for n in cols:
        o[n] = torch.full_like(x, float("nan"))
    H.warm_clocks()
    for f in cols.values():
        f()
    torch.cuda.synchronize()
    tol = 4e-3 * ref.abs().max().item()
    for n in cols:
        assert torch.isfinite(o[n]).all() and (o[n].double() - ref).abs().max().item() <= tol, (rows, n)
    one = min(H.event_time_us(lambda: [cols["committed"]() for _ in range(10)], 10) for _ in range(3))
    calls = max(3, int(3000 / max(one, 1)))
    # Screen Triton's configurations; keep the fastest three.
    tscreen = {}
    for R, w in TCONF:
        n = f"triton_r{R}w{w}"
        tscreen[n] = min(H.event_time_us(lambda f=cols[n]: [f() for _ in range(calls)], calls) for _ in range(5))
    keep = sorted(tscreen, key=tscreen.get)[:3]
    timed = {n: f for n, f in cols.items() if not n.startswith("triton_") or n in keep}
    samples = {n: [] for n in timed}
    clocks = []
    names = list(timed)
    for r in range(17):
        for n in names[r % len(names):] + names[:r % len(names)]:
            f = timed[n]
            t = H.event_time_us(lambda f=f: [f() for _ in range(calls)], calls)
            if r >= 2:
                samples[n].append(t)
        clocks.append(torch.cuda.clock_rate(0))
    nbytes = rows * (HD * 2 * 2 + 4)
    print(f"== rope_{HD} rows {rows}  {nbytes / 1e6:.0f} MB  all checked  calls {calls}  SM {min(clocks)}-{max(clocks)} MHz"
          f"  triton screen best {', '.join(f'{n} {tscreen[n]:.2f}' for n in keep)}", flush=True)
    base = min(samples["committed"])
    for n in sorted(samples, key=lambda n: min(samples[n])):
        m = min(samples[n])
        print(f"   {n:18s} min {m:9.2f} median {statistics.median(samples[n]):9.2f} us  {nbytes / m / 1e3:7.1f} GB/s"
              f"  committed/this {base / m:.3f}", flush=True)


for rows in SIZES:
    check(rows + 3)
for rows in SIZES:
    bench(rows)
