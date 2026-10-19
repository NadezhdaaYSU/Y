#!/usr/bin/env python3
"""mk_reg.py <out.ptx> [hidden_dim] [warps]: rmsnorm_residual_<d> written by hand
with h kept in registers. Each thread owns groups t, t + 32*warps, ...
(8 f16 a group) as the emitted kernel does; every load (X, Residual, Weight,
all groups) is issued first, NewResidual is stored as soon as h exists, and
only the cross-warp sum goes through shared memory. The arithmetic is the
emitted kernel's, in its order: one sum-of-squares chain over the groups in
order, then (h * rstd) * w."""
import sys
out = sys.argv[1]
D = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
W = int(sys.argv[3]) if len(sys.argv) > 3 else 8
T = 32 * W
G = D // 8 // T          # groups per thread
assert D % (8 * T) == 0
name = f"rmsnorm_residual_{D}"
L = []
e = L.append
e(".version 7.8\n.target sm_89\n.address_size 64\n")
e(f".visible .entry {name}(\n    .param .u64 X_0,\n    .param .u64 Residual_1,\n    .param .u64 Weight_2,\n    .param .u64 Out_3,\n    .param .u64 NewResidual_4\n)\n{{")
e(f"    .reg .b32 %r<{64 + 96 * G}>;\n    .reg .f32 %f<{64 + 64 * G}>;\n    .reg .b64 %rd<{32 + 8 * G}>;\n    .reg .pred %p<4>;")
e(f"    .shared .align 4 .b8 smem_reduce[{4 * W}];")
for i, p in enumerate(("X_0", "Residual_1", "Weight_2", "Out_3", "NewResidual_4")):
    e(f"    ld.param.u64 %rd{i}, [{p}];")
    e(f"    cvta.to.global.u64 %rd{5 + i}, %rd{i};")
e("    mov.u32 %r0, smem_reduce;")
e("    mov.u32 %r1, %ctaid.x;\n    mov.u32 %r2, %tid.x;")
e(f"    mul.lo.u32 %r3, %r1, {D};")
r = [10]; f = [1]; rd = [12]
def R():
    r[0] += 1; return f"%r{r[0]}"
def F():
    f[0] += 1; return f"%f{f[0]}"
def RD():
    rd[0] += 1; return f"%rd{rd[0]}"
xa, ra, wa, oa, na, xw, rw, ww = [], [], [], [], [], [], [], []
for g in range(G):
    grp = R(); e(f"    add.u32 {grp}, %r2, {g * T};")
    el = R(); e(f"    mad.lo.u32 {el}, {grp}, 8, %r3;")
    off = RD(); e(f"    mul.wide.u32 {off}, {el}, 2;")
    woff = RD(); e(f"    mul.wide.u32 {woff}, {grp}, 16;")
    for base, lst in ((5, xa), (6, ra), (8, oa), (9, na)):
        a = RD(); e(f"    add.s64 {a}, %rd{base}, {off};"); lst.append(a)
    a = RD(); e(f"    add.s64 {a}, %rd7, {woff};"); wa.append(a)
for g in range(G):
    for addr, words in ((xa[g], xw), (ra[g], rw), (wa[g], ww)):
        regs = [R() for _ in range(4)]
        e(f"    ld.global.v4.u32 {{{','.join(regs)}}}, [{addr}];")
        words.append(regs)
def unpack(words):
    vals = []
    for wd in words:
        lo = F(); e(f"    cvt.f32.f16 {lo}, {wd};")
        t = R(); e(f"    shr.b32 {t}, {wd}, 16;")
        hi = F(); e(f"    cvt.f32.f16 {hi}, {t};")
        vals += [lo, hi]
    return vals
def pack(vals):
    words = []
    for a, b in zip(vals[0::2], vals[1::2]):
        ha, hb = R(), R()
        e(f"    cvt.rn.f16.f32 {ha}, {a};\n    cvt.rn.f16.f32 {hb}, {b};")
        la, sb, wd = R(), R(), R()
        e(f"    and.b32 {la}, {ha}, 65535;\n    shl.b32 {sb}, {hb}, 16;\n    or.b32 {wd}, {la}, {sb};")
        words.append(wd)
    return words
e("    mov.f32 %f0, 0f00000000;")
hs = []
for g in range(G):
    xv, rv = unpack(xw[g]), unpack(rw[g])
    h = []
    for a, b in zip(xv, rv):
        s = F(); e(f"    add.f32 {s}, {a}, {b};"); e(f"    fma.rn.f32 %f0, {s}, {s}, %f0;"); h.append(s)
    hs.append(h)
    nw = pack(h)
    e(f"    st.global.v4.u32 [{na[g]}], {{{','.join(nw)}}};")
acc = "%f0"
for k in (16, 8, 4, 2, 1):
    t, s = F(), F()
    e(f"    shfl.sync.bfly.b32 {t}, {acc}, {k}, 0x1f, 0xffffffff;\n    add.f32 {s}, {acc}, {t};")
    acc = s
if W > 1:
    warp, lane, off, ad = R(), R(), R(), R()
    e(f"    shr.u32 {warp}, %r2, 5;\n    and.b32 {lane}, %r2, 31;\n    setp.eq.u32 %p1, {lane}, 0;")
    e(f"    mul.lo.u32 {off}, {warp}, 4;\n    add.u32 {ad}, %r0, {off};\n    @%p1 st.shared.f32 [{ad}], {acc};\n    bar.sync 0;")
    tot = F(); e(f"    mov.f32 {tot}, 0f00000000;")
    for w in range(W):
        v, n2 = F(), F()
        e(f"    ld.shared.f32 {v}, [%r0+{4 * w}];\n    add.f32 {n2}, {tot}, {v};")
        tot = n2
    acc = tot
ms = F(); e(f"    fma.rn.f32 {ms}, {acc}, 0f{__import__('struct').unpack('<I', __import__('struct').pack('<f', 1.0 / D))[0]:08X}, 0f3727C5AC;")
rs = F(); e(f"    rsqrt.approx.f32 {rs}, {ms};")
for g in range(G):
    wv = unpack(ww[g])
    o = []
    for hv, wvv in zip(hs[g], wv):
        a, b = F(), F()
        e(f"    mul.f32 {a}, {hv}, {rs};\n    mul.f32 {b}, {a}, {wvv};")
        o.append(b)
    ow = pack(o)
    e(f"    st.global.v4.u32 [{oa[g]}], {{{','.join(ow)}}};")
e("    ret;\n}")
open(out, "w").write("\n".join(L) + "\n")
