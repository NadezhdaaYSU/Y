#!/usr/bin/env python3
"""mk_rope.py <out.ptx> <head_dim> <tpr> <slots> <unroll>: a hand-written
rope kernel, `ropev(X, Positions, Out, Rows)`, with the arithmetic of
ptx_emitter::emit_rope_kernel (same instructions per pair, same order) and
another thread mapping:

  tpr     threads a row (each owns head_dim/2/tpr contiguous pairs);
  slots   rows a CTA works on at once (block = tpr * slots threads);
  unroll  rows a thread handles, all loads issued before any arithmetic.

A CTA covers slots*unroll rows: row = ctaid*slots*unroll + k*slots + slot.
Rows past `Rows` are predicated off (loads and stores)."""
import math, struct, sys

out, hd, tpr, slots, unroll = sys.argv[1], *map(int, sys.argv[2:6])
pairs = hd // 2
assert pairs % tpr == 0 and tpr & (tpr - 1) == 0
P = pairs // tpr                     # pairs a thread, one 32-bit word each
threads = tpr * slots
# The emitter computes the scale in f32 Rust: (-2.0 / hd as f32) * 10000f32.log2().
import numpy as np
scale = np.float32(-2.0) / np.float32(hd) * np.float32(np.log2(np.float32(10000.0)))
hexs = "0f%08X" % struct.unpack("<I", struct.pack("<f", float(scale)))[0]
assert hd != 128 or hexs == "0fBE549A78", hexs   # tests/rope_128.ptx

L = []
r = [0]; f = [0]; d = [0]; p = [0]
def R():
    r[0] += 1; return f"%r{r[0]-1}"
def F():
    f[0] += 1; return f"%f{f[0]-1}"
def D():
    d[0] += 1; return f"%rd{d[0]-1}"
def Pr():
    p[0] += 1; return f"%p{p[0]-1}"
e = L.append

x0, p0, o0 = D(), D(), D()
e(f"    ld.param.u64 {x0}, [X_0];"); e(f"    ld.param.u64 {p0}, [Positions_1];"); e(f"    ld.param.u64 {o0}, [Out_2];")
rows = R(); e(f"    ld.param.u32 {rows}, [Rows_3];")
xg, pg, og = D(), D(), D()
e(f"    cvta.to.global.u64 {xg}, {x0};"); e(f"    cvta.to.global.u64 {pg}, {p0};"); e(f"    cvta.to.global.u64 {og}, {o0};")
cta, tid = R(), R()
e(f"    mov.u32 {cta}, %ctaid.x;"); e(f"    mov.u32 {tid}, %tid.x;")
lane, slot = R(), R()
e(f"    and.b32 {lane}, {tid}, {tpr - 1};"); e(f"    shr.u32 {slot}, {tid}, {tpr.bit_length() - 1};")
row0 = R(); e(f"    mad.lo.u32 {row0}, {cta}, {slots * unroll}, {slot};")
pstart = R(); e(f"    mul.lo.u32 {pstart}, {lane}, {P};")
# 8 f16 a 16-byte vector: P words in chunks of up to 4.
chunks = [(c, min(4, P - c)) for c in range(0, P, 4)]
loads = []
for k in range(unroll):
    row = R(); e(f"    add.u32 {row}, {row0}, {k * slots};")
    ok = Pr(); e(f"    setp.lt.u32 {ok}, {row}, {rows};")
    pb = D(); e(f"    mul.wide.u32 {pb}, {row}, 4;")
    pa = D(); e(f"    add.s64 {pa}, {pg}, {pb};")
    pos = R(); e(f"    mov.u32 {pos}, 0;"); e(f"    @{ok} ld.global.u32 {pos}, [{pa}];")
    # element index = row*hd + 2*pstart
    eg2 = R(); e(f"    mad.lo.u32 {eg2}, {row}, {hd}, 0;")
    e2 = R(); e(f"    shl.b32 {e2}, {pstart}, 1;")
    ei = R(); e(f"    add.u32 {ei}, {eg2}, {e2};")
    bb = D(); e(f"    mul.wide.u32 {bb}, {ei}, 2;")
    words = []
    for c, n in chunks:
        a = D(); e(f"    add.s64 {a}, {xg}, {bb};")
        ws = [R() for _ in range(n)]
        for w in ws:
            e(f"    mov.u32 {w}, 0;")
        off = f"+{c * 4}" if c else ""
        if n == 1:
            e(f"    @{ok} ld.global.u32 {ws[0]}, [{a}{off}];")
        else:
            e(f"    @{ok} ld.global.v{n}.u32 {{{','.join(ws)}}}, [{a}{off}];")
        words += ws
    loads.append((ok, pos, bb, words))

for ok, pos, bb, words in loads:
    pf = F(); e(f"    cvt.rn.f32.s32 {pf}, {pos};")
    outs = []
    for j, w in enumerate(words):
        a0, a1 = F(), F()
        e(f"    cvt.f32.f16 {a0}, {w};")
        sh = R(); e(f"    shr.b32 {sh}, {w}, 16;"); e(f"    cvt.f32.f16 {a1}, {sh};")
        pi = R(); e(f"    add.u32 {pi}, {pstart}, {j};")
        i_f, ex, bp, th, s, c_ = F(), F(), F(), F(), F(), F()
        e(f"    cvt.rn.f32.u32 {i_f}, {pi};"); e(f"    mul.f32 {ex}, {i_f}, {hexs};")
        e(f"    ex2.approx.f32 {bp}, {ex};"); e(f"    mul.f32 {th}, {pf}, {bp};")
        e(f"    sin.approx.f32 {s}, {th};"); e(f"    cos.approx.f32 {c_}, {th};")
        m1, n1, o_0, m2, o_1 = F(), F(), F(), F(), F()
        e(f"    mul.f32 {m1}, {a1}, {s};"); e(f"    neg.f32 {n1}, {m1};"); e(f"    fma.rn.f32 {o_0}, {a0}, {c_}, {n1};")
        e(f"    mul.f32 {m2}, {a1}, {c_};"); e(f"    fma.rn.f32 {o_1}, {a0}, {s}, {m2};")
        h0, l0, h1, s1, wv = R(), R(), R(), R(), R()
        e(f"    cvt.rn.f16.f32 {h0}, {o_0};"); e(f"    and.b32 {l0}, {h0}, 0xFFFF;")
        e(f"    cvt.rn.f16.f32 {h1}, {o_1};"); e(f"    shl.b32 {s1}, {h1}, 16;"); e(f"    or.b32 {wv}, {l0}, {s1};")
        outs.append(wv)
    for c, n in chunks:
        a = D(); e(f"    add.s64 {a}, {og}, {bb};")
        off = f"+{c * 4}" if c else ""
        ws = outs[c:c + n]
        if n == 1:
            e(f"    @{ok} st.global.u32 [{a}{off}], {ws[0]};")
        else:
            e(f"    @{ok} st.global.v{n}.u32 [{a}{off}], {{{','.join(ws)}}};")

head = f""".version 7.8
.target sm_89
.address_size 64

// threads {threads} rows_per_cta {slots * unroll} tpr {tpr} slots {slots} unroll {unroll}
.visible .entry ropev(
    .param .u64 X_0,
    .param .u64 Positions_1,
    .param .u64 Out_2,
    .param .u32 Rows_3
)
.maxntid {threads}, 1, 1
{{
    .reg .b32 %r<{r[0]}>;
    .reg .f32 %f<{f[0]}>;
    .reg .b64 %rd<{d[0]}>;
    .reg .pred %p<{max(p[0], 1)}>;
"""
open(out, "w").write(head + "\n".join(L) + "\n    ret;\n}\n")
