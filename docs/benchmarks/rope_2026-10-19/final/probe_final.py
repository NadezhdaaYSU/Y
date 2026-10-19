#!/usr/bin/env python3
"""probe_final.py <examples dir> <bench_ab dir> <ptx dir>

rope_<hd> at head_dim 64, 128 and 256, at footprints of about 2, 8.5, 17 and
34 MB (inside the 48 MB L2) and 101 MB (from DRAM):

  before        <ptx dir>/rope_<hd>_before.ptx, the parent's kernel (one
                32-thread CTA a row, grid = rows);
  after         <ptx dir>/rope_<hd>_after.ptx, this change's (Rows parameter;
                block and rows a CTA read from its header);
  triton        a Triton kernel doing the same arithmetic with tl.sin/tl.cos
                (libdevice's precise sinf/cosf), and
  triton_fast   the same with libdevice.fast_sinf/fast_cosf (sin.approx, as Y),
                each at the two fastest of 28 (ROWS, num_warps) configurations
                in a graph-replay screen at that size;
  flashinfer    flashinfer.apply_rope_pos_ids(q, k, pos), interleaved, theta
                10000, with q and k the two halves of the same rows: it rotates
                both, so every column rotates every row once, out of place;
  after_inplace after with Out = X, against
  flashinfer_inplace  apply_rope_pos_ids_inplace: in place the footprint is half
                (no second buffer), so the two in-place columns are compared
                with each other.

Positions are the same for row r and row r + rows/2, as FlashInfer's q and k
share one position vector. Every column is checked against float64 first
(within 4e-3 of the largest output), before and after bit for bit alike, and
after on rows + 3 (a partial last CTA, NaN past the end left untouched).
Timing: H.measure, CUDA-graph replays and plain launches, interleaved, 15
rounds after 2; ranked by the graph minimum."""
import os, re, sys
import torch, triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
import flashinfer
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
import run_head_to_head_comparison as H
from bench_ab import kernel_from_ptx
P = sys.argv[3]
torch.zeros(1, device="cuda")
gen = torch.Generator(device="cuda").manual_seed(29)


@triton.jit
def t_rope(x_ptr, p_ptr, o_ptr, rows, D: tl.constexpr, ROWS: tl.constexpr, FAST: tl.constexpr):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    i = tl.arange(0, D // 2)
    m = r < rows
    pos = tl.load(p_ptr + r, mask=m, other=0).to(tl.float32)
    inv = tl.exp2(-(2.0 * i.to(tl.float32) / D) * 13.287712379549449)  # log2(10000)
    th = pos[:, None] * inv[None, :]
    if FAST:
        c, s = libdevice.fast_cosf(th), libdevice.fast_sinf(th)
    else:
        c, s = tl.cos(th), tl.sin(th)
    base = r[:, None] * D + 2 * i[None, :]
    mm = m[:, None]
    x0 = tl.load(x_ptr + base, mask=mm, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + base + 1, mask=mm, other=0.0).to(tl.float32)
    tl.store(o_ptr + base, (x0 * c - x1 * s).to(tl.float16), mask=mm)
    tl.store(o_ptr + base + 1, (x0 * s + x1 * c).to(tl.float16), mask=mm)


TCONF = [(R, w) for R in (1, 2, 4, 8, 16, 32, 64) for w in (1, 2, 4, 8)]
SIZES = {64: (8192, 32768, 65536, 131072, 393216),
         128: (4096, 16384, 32768, 65536, 196608),
         256: (2048, 8192, 16384, 32768, 98304)}
pos_dtype = None


def reference(x, pos, hd):
    rows = x.shape[0]
    inv = 10000.0 ** (-torch.arange(0, hd, 2, device="cuda", dtype=torch.float64) / hd)
    th = pos.double()[:, None] * inv[None, :]
    xv = x.double().view(rows, hd // 2, 2)
    return torch.stack([xv[..., 0] * th.cos() - xv[..., 1] * th.sin(),
                        xv[..., 0] * th.sin() + xv[..., 1] * th.cos()], -1).reshape(rows, hd)


def inputs(rows, hd):
    x = torch.randn(rows, hd, device="cuda", generator=gen).half()
    half = torch.randint(0, 4096, (rows // 2,), device="cuda", generator=gen, dtype=torch.int32)
    return x, torch.cat([half, half]), half


def fi_call(xfi, half, hd):
    n = xfi.shape[0] // 2
    flashinfer.apply_rope_pos_ids_inplace(xfi[:n].view(n, 1, hd), xfi[n:].view(n, 1, hd), half.to(pos_dtype),
                                          interleave=True, rope_theta=10000.0)


def fi_out(xfi, half, hd):
    n = xfi.shape[0] // 2
    return flashinfer.apply_rope_pos_ids(xfi[:n].view(n, 1, hd), xfi[n:].view(n, 1, hd), half.to(pos_dtype),
                                         interleave=True, rope_theta=10000.0)


for hd in (64, 128, 256):
    kb = kernel_from_ptx(f"{P}/rope_{hd}_before.ptx", f"rope_{hd}")
    after_ptx = f"{P}/rope_{hd}_after.ptx"
    m = re.search(r"\[Y FUSED ROPE\].*?\| (\d+) threads a CTA: (\d+) rows", open(after_ptx).read())
    threads, per_cta = int(m.group(1)), int(m.group(2))
    ka = kernel_from_ptx(after_ptx, f"rope_{hd}")
    print(f"#### head_dim {hd}: after = {threads} threads a CTA, {per_cta} rows", flush=True)

    # A partial last CTA: rows + 3, NaN past the end.
    rows = SIZES[hd][1] + 3
    x = torch.randn(rows, hd, device="cuda", generator=gen).half()
    pos = torch.randint(0, 4096, (rows,), device="cuda", generator=gen, dtype=torch.int32)
    o = torch.full((rows + 64, hd), float("nan"), device="cuda", dtype=torch.half)
    ob = torch.full_like(o[:rows], float("nan"))
    ka.launch(((rows + per_cta - 1) // per_cta, 1, 1), (threads, 1, 1), [x, pos, o, rows])
    kb.launch((rows, 1, 1), (32, 1, 1), [x, pos, ob])
    torch.cuda.synchronize()
    assert torch.isnan(o[rows:]).all(), "after wrote past Rows"
    assert torch.equal(o[:rows].view(torch.int16), ob.view(torch.int16)), "after and before differ"
    print(f"checked rows {rows}: after equals before bit for bit, nothing written past Rows", flush=True)

    for rows in SIZES[hd]:
        x, pos, half = inputs(rows, hd)
        ref = reference(x, pos, hd)
        tol = 4e-3 * ref.abs().max().item()
        outs = {}
        cols = {}

        def add(name, fn, out):
            outs[name] = out
            cols[name] = fn

        ob, oa = torch.full_like(x, float("nan")), torch.full_like(x, float("nan"))
        add("before", lambda: kb.launch((rows, 1, 1), (32, 1, 1), [x, pos, ob]), ob)
        add("after", lambda: ka.launch(((rows + per_cta - 1) // per_cta, 1, 1), (threads, 1, 1), [x, pos, oa, rows]), oa)
        tcols = {}
        for fast in (False, True):
            for R, w in TCONF:
                n = f"triton{'_fast' if fast else ''}_r{R}w{w}"
                ot = torch.full_like(x, float("nan"))
                outs[n] = ot
                tcols[n] = (lambda R=R, w=w, ot=ot, fast=fast:
                            t_rope[(triton.cdiv(rows, R),)](x, pos, ot, rows, D=hd, ROWS=R, FAST=fast, num_warps=w))
        # FlashInfer rotates in place: check on a fresh copy, time on a working one.
        if pos_dtype is None:
            for dt in (torch.int32, torch.int64):
                try:
                    pos_dtype = dt
                    fi_call(x.clone(), half, hd)
                    torch.cuda.synchronize()
                    break
                except Exception as e:
                    print(f"flashinfer pos_ids {dt}: {type(e).__name__}: {e}", flush=True)
            print(f"flashinfer pos_ids dtype {pos_dtype}", flush=True)
        ofi = x.clone()
        fi_call(ofi, half, hd)
        outs["flashinfer_inplace"] = ofi
        q_r, k_r = fi_out(x, half, hd)
        outs["flashinfer"] = torch.cat([q_r.reshape(-1, hd), k_r.reshape(-1, hd)])
        oai = x.clone()
        ka.launch(((rows + per_cta - 1) // per_cta, 1, 1), (threads, 1, 1), [oai, pos, oai, rows])
        outs["after_inplace"] = oai
        xw, xy = x.clone(), x.clone()
        for f in [*cols.values(), *tcols.values()]:
            f()
        torch.cuda.synchronize()
        if rows == SIZES[hd][0]:
            for fast in (False, True):
                ck = t_rope[(1,)](x, pos, outs["triton_r1w1"], 1, D=hd, ROWS=1, FAST=fast, num_warps=1)
                ptx = ck.asm["ptx"]
                print(f"triton FAST={fast} PTX: sin.approx {ptx.count('sin.approx')}, cos.approx {ptx.count('cos.approx')}, "
                      f"ex2.approx {ptx.count('ex2.approx')}, lines {ptx.count(chr(10))}", flush=True)
            for f in tcols.values():
                f()
            torch.cuda.synchronize()
        for n, out in outs.items():
            err = (out.double() - ref).abs().max().item()
            assert torch.isfinite(out).all() and err <= tol, (hd, rows, n, err)
        assert torch.equal(oa.view(torch.int16), ob.view(torch.int16)), (hd, rows, "after != before")
        assert torch.equal(oai.view(torch.int16), oa.view(torch.int16)), (hd, rows, "in place != out of place")

        screen = H.measure(tcols, rounds=2, warmup_rounds=1)
        keep = []
        for fast in (False, True):
            pre = "triton_fast_" if fast else "triton_r"
            cand = sorted((n for n in tcols if n.startswith(pre)), key=lambda n: min(screen[n]["graph"]))
            keep += cand[:2]
        for n in keep:
            cols[n] = tcols[n]
        cols["flashinfer"] = lambda: fi_out(x, half, hd)
        cols["flashinfer_inplace"] = lambda: fi_call(xw, half, hd)
        cols["after_inplace"] = lambda: ka.launch(((rows + per_cta - 1) // per_cta, 1, 1), (threads, 1, 1), [xy, pos, xy, rows])
        res = H.measure(cols, rounds=15, warmup_rounds=2)
        clocks = [c for r in res.values() for c in r["sm_clock_mhz"] if c]
        nbytes = rows * (hd * 2 * 2 + 4)
        print(f"== rope_{hd} rows {rows}  {nbytes / 1e6:.1f} MB  all checked  calls {res['after']['calls']}"
              f"  SM {min(clocks)}-{max(clocks)} MHz", flush=True)
        ga = min(res["after"]["graph"])
        for n in sorted(res, key=lambda n: min(res[n]["graph"]) if res[n]["graph"] else 1e9):
            r = res[n]
            if not r["graph"]:
                print(f"   {n:18s} graph capture failed: {r['graph_error']}", flush=True)
                continue
            g, l = min(r["graph"]), min(r["launch"])
            print(f"   {n:18s} graph {g:9.2f} us  launch {l:9.2f} us  {nbytes / g / 1e3:7.1f} GB/s  this/after {g / ga:.3f}", flush=True)
