"""Every instruction form the CPU interpreter models, run on the GPU and on the
interpreter with the same operands, compared bit for bit.

Each form - `add.f32`, `cvt.rzi.s64.f64`, `mad.hi.cc.u32` and about 500 more -
runs as a one-instruction kernel over the cross product of its operand types'
edge values (zeros of both signs, subnormals, the largest finite values,
infinities, NaNs with payloads, integer extremes, shift counts past the width)
plus 2,000 random bit patterns. Results compare as raw bits, so NaN payloads
and signed zeros count. The .approx functions must instead lie within the
GPU's documented error of the interpreter's accurately computed value.

Run against the interpreter as first written, 137 of the 518 forms gave a
wrong answer without any error and 5 raised Python exceptions. What it had
wrong:
- float32 and float16 instructions write one canonical NaN; float64 ones pass on
  an operand's NaN, in an order each instruction fixes;
- .ftz flushes a result that is tiny after rounding, and was ignored on neg,
  abs, min and max;
- min and max put -0 below +0, and .sat writes +0 for -0;
- a float NaN converts to 0, or to 0x80... from float64 or to a 64-bit integer;
- cvt.rni.f32.f32 and its kin returned the operand unrounded;
- sub.sat.s32 and mad.hi.sat.s32 did not saturate;
- carries out of .s32 and .s64 additions were signed comparisons;
- fma.rn.f64 raised OverflowError at overflow, and cvt to s64 from float64
  raised it at 2^63;
- div.approx.f32 returns 0 for divisors above 2^126;
- sin.approx and cos.approx differed from the GPU by up to 2 for large
  arguments, until their float32 argument reduction was modelled.

The expected semantics were measured on an RTX 4070 Ti SUPER (sm_89).
"""

import itertools
import unittest
from unittest import mock

import numpy as np
import torch

from y_lang import interpreter
from y_lang.interpreter import CPUInterpreter
from y_lang.torch_interop import TorchKernel


class PtxKernel(TorchKernel):
    """A TorchKernel over PTX that did not come from Y source."""

    def __init__(self, ptx, name):
        self.source, self.kernel_name, self.target_sm, self.ptx = None, name, None, ptx
        self._module = self._function = None


NP = {"f16": np.float16, "f32": np.float32, "f64": np.float64, "s8": np.int8, "u8": np.uint8,
      "s16": np.int16, "u16": np.uint16, "s32": np.int32, "u32": np.uint32, "s64": np.int64,
      "u64": np.uint64, "b16": np.uint16, "b32": np.uint32, "b64": np.uint64, "pred": np.uint32,
      "shift": np.uint32}
WIDTH = {t: np.dtype(v).itemsize * 8 for t, v in NP.items()}
FLOATS = ("f16", "f32", "f64")


def _bits(dtype, *xs):
    return [np.array(x, NP["u%d" % (np.dtype(dtype).itemsize * 8)]).view(dtype)[()] for x in xs]


EDGE = {
    "f32": [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 3.5, 0.49999997, 2.0, 3.0, -7.25, 1e-3,
            3.4028235e38, -3.4028235e38, np.inf, -np.inf, 16777216.0, 2147483648.0, -2147483648.0,
            4294967296.0, 2147483520.0, 1e10, -1e10, 65504.0, 65520.0, 6e-8, 3.1415927, 100.75]
           + _bits(np.float32, 0x7FC00000, 0xFFC00000, 0x7F800001, 0x7FFFFFFF, 0x00800000, 0x80800000,
                   0x00000001, 0x007FFFFF, 0x80000001, 0x3F800001, 0x3F7FFFFF, 0x4B000001, 0x00400000,
                   # 0x00800001 * 0x3F7FFFFE = 2^-126 (1 - 2^-46): kept under .ftz
                   0x00800001, 0x3F7FFFFE),
    "f64": [0.0, -0.0, 1.0, -1.0, 0.5, -2.5, 3.5, 2.5, 1e300, -1e300, np.inf, -np.inf, 1.7976931348623157e308,
            3.4028235677973366e38, 3.4028234663852886e38, 1e-46, 7e-46, 1.401298464324817e-45,
            9223372036854775807.0, -9223372036854775808.0, 18446744073709551616.0, 4294967295.5, -1.5,
            1 + 2**-24, 1 + 2**-24 + 2**-50, 1 - 2**-25, 2**-149, 2**-150, 3 * 2**-151, 65504.0, 65520.0,
            65519.99, 2**-25, 2**-24 * 1.5]
           + _bits(np.float64, 0x7FF8000000000000, 0xFFF8000000000000, 0x7FF0000000000001, 0x0000000000000001,
                   0x000FFFFFFFFFFFFF, 0x0010000000000000),
    "f16": _bits(np.float16, 0x0000, 0x8000, 0x3C00, 0xBC00, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7E00, 0xFE00,
                 0x7C01, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3555, 0x4248, 0x3C01, 0x3BFF, 0x5640, 0x1400, 0x0200),
    "s32": [0, 1, -1, 2, -2, 3, 7, 8, -8, 31, 32, 33, 63, 64, 65, 100, -100, 127, 128, -128, -129, 255, 256,
            32767, 32768, -32768, -32769, 65535, 65536, 2**31 - 1, -2**31, 2**31 - 2, -2**31 + 1, 0x55555555,
            -0x55555556, 0x7FFF0000, 0x12345678, -0x12345678],
    "s64": [0, 1, -1, 2, -2, 7, 31, 32, 63, 64, 65, -8, 2**31 - 1, -2**31, 2**31, -2**31 - 1, 2**32 - 1, 2**32,
            -2**32, 2**63 - 1, -2**63, 2**63 - 2, -2**63 + 1, 0x123456789ABCDEF0, -0x123456789ABCDEF0,
            0x5555555555555555, 1 << 40, 255, -129, 65536],
    "s16": [0, 1, -1, 2, 7, 15, 16, 17, 127, 128, -128, 255, 256, 32767, -32768, 32766, -32767, 0x5555, -0x5556, 1000],
    "s8": [0, 1, -1, 2, 7, 8, 9, 100, 127, -128, 126, -127, 0x55, -0x56],
    "pred": [0, 1],
    "shift": list(range(0, 70)) + [127, 128, 255, 256, 2**31, 2**32 - 1],
}
for _w in (8, 16, 32, 64):
    EDGE[f"u{_w}"] = [x & ((1 << _w) - 1) for x in EDGE[f"s{_w}"]]
    if _w > 8:
        EDGE[f"b{_w}"] = EDGE[f"u{_w}"]
for _t in ["s16", "u16", "s32", "u32", "s64", "u64"]:     # divisors
    EDGE["nz" + _t] = [x for x in EDGE[_t] if x != 0]
    NP["nz" + _t], WIDTH["nz" + _t] = NP[_t], WIDTH[_t]


def _random(t, n, rng):
    if t in FLOATS:
        w = WIDTH[t]
        raw = rng.integers(0, 2**w, size=n, dtype=np.uint64).astype(NP["u%d" % w]).view(NP[t])
        normal = (rng.standard_normal(n) * np.exp2(rng.integers(-30, 30, size=n))).astype(NP[t])
        return np.where(rng.random(n) < 0.5, raw, normal)
    if t == "pred":
        return rng.integers(0, 2, size=n).astype(np.uint32)
    if t == "shift":
        return rng.integers(0, 80, size=n).astype(np.uint32)
    w = WIDTH[t]
    return rng.integers(1 if t.startswith("nz") else 0, 2**w, size=n, dtype=np.uint64).astype(NP["u%d" % w]).view(NP[t])


def _inputs(kinds, rng, n_random=2000, cap=30000):
    """The edge values' cross product (a random 30,000 of it for three
    operands) followed by random rows."""
    rows = list(itertools.product(*(EDGE[k] for k in kinds)))
    if len(rows) > cap:
        rows = [rows[i] for i in rng.choice(len(rows), cap, replace=False)]
    cols = []
    for j, k in enumerate(kinds):
        edge = np.array([r[j] for r in rows], dtype=NP[k] if k in FLOATS else object).astype(NP[k])
        cols.append(np.concatenate([edge, _random(k, n_random, rng).astype(NP[k])]))
    return cols


def _ptx(body, d, srcs):
    """A kernel loading one row of operands per thread into %va, %vb, %vc,
    running `body`, and storing %vd. Registers are untyped (.bN)."""
    def reg(t):
        return "pred" if t == "pred" else ("b32" if WIDTH[t] == 8 else "b%d" % WIDTH[t])

    def mem(t):
        return "u32" if t in ("shift", "pred") else "b16" if t == "f16" else t[2:] if t.startswith("nz") else t
    params = ", ".join([f".param .u64 pin{i}" for i in range(len(srcs))] + [".param .u64 pout", ".param .u32 pn"])
    out = "u32" if d == "pred" else "b16" if d == "f16" else "b32" if WIDTH[d] == 8 else "b16" if WIDTH[d] == 16 else d
    lines = [".version 7.8", ".target sm_80", ".address_size 64", f".visible .entry k({params})", "{",
             ".reg .pred %pg; .reg .b32 %gi, %gn, %gt<4>; .reg .b64 %ga, %gb, %go;",
             ".reg .b32 %t<4>; .reg .b64 %tl<4>; .reg .pred %tp<4>; .reg .b32 %va_w, %vd_w;",
             f".reg .{reg(d)} %vd;"]
    lines += [f".reg .{reg(t)} {nm};" for nm, t in zip(("%va", "%vb", "%vc"), srcs)]
    lines += ["mov.u32 %gt0, %ctaid.x;", "mov.u32 %gt1, %ntid.x;", "mov.u32 %gt2, %tid.x;",
              "mad.lo.u32 %gi, %gt0, %gt1, %gt2;", "ld.param.u32 %gn, [pn];", "setp.ge.u32 %pg, %gi, %gn;",
              "@%pg bra DONE;"]
    for i, (nm, t) in enumerate(zip(("%va", "%vb", "%vc"), srcs)):
        lines += [f"ld.param.u64 %ga, [pin{i}];", f"mul.wide.u32 %gb, %gi, {WIDTH[mem(t)] // 8};", "add.s64 %ga, %ga, %gb;"]
        lines += ([f"ld.global.u32 %va_w, [%ga];", f"setp.ne.u32 {nm}, %va_w, 0;"] if t == "pred"
                  else [f"ld.global.{mem(t)} {nm}, [%ga];"])
    lines += [s.strip() + ";" for s in body.split(";") if s.strip()]
    lines += ["ld.param.u64 %go, [pout];", f"mul.wide.u32 %gb, %gi, {WIDTH[out] // 8};", "add.s64 %go, %go, %gb;"]
    lines += (["selp.u32 %vd_w, 1, 0, %vd;", "st.global.u32 [%go], %vd_w;"] if d == "pred"
              else [f"st.global.{out} [%go], %vd;"])
    lines += ["DONE:", "ret;", "}"]
    return "\n".join(lines) + "\n", NP["u%d" % WIDTH[out]]


def run(body, d, srcs, cols):
    """The GPU's and the interpreter's results, as raw bits."""
    ptx, ut = _ptx(body, d, srcs)
    n = len(cols[0])
    args = [torch.from_numpy(np.ascontiguousarray(c).view(np.uint8).copy()) for c in cols]
    grid, block = ((n + 255) // 256, 1, 1), (256, 1, 1)
    gpu = torch.zeros(n * np.dtype(ut).itemsize, dtype=torch.uint8, device="cuda")
    PtxKernel(ptx, "k").launch(grid, block, [a.cuda() for a in args] + [gpu, n])
    torch.cuda.synchronize()
    cpu = torch.zeros_like(gpu, device="cpu")
    CPUInterpreter("k", ptx=ptx).run(grid, block, args + [cpu, n])
    return gpu.cpu().numpy().view(ut), cpu.numpy().view(ut)


def disagree(d, g, c, approx):
    """Rows where the results differ: any bit, or for an .approx function
    more than its tolerance (relative, absolute) - never in NaN-ness or
    NaN payload."""
    bad = g != c
    if approx and d in FLOATS:
        gf, cf = g.view(NP[d]).astype(np.float64), c.view(NP[d]).astype(np.float64)
        rel, absolute = approx
        floor = {"f32": 2.0 ** -148, "f16": 2.0 ** -23}.get(d, 0.0)      # two subnormal ulps
        with np.errstate(invalid="ignore"):
            close = np.abs(gf - cf) <= np.maximum(np.maximum(rel * np.abs(cf), absolute), floor)
        same = (np.isnan(gf) == np.isnan(cf)) & (np.isinf(gf) == np.isinf(cf)) & ~np.isnan(gf)
        bad &= ~(close & same)
    return bad


CASES = []


def case(body, d, *srcs, approx=None):
    CASES.append((body, d, srcs, approx))


F_CMPS = ["eq", "ne", "lt", "le", "gt", "ge", "equ", "neu", "ltu", "leu", "gtu", "geu", "num", "nan"]
INTS = ["s8", "u8", "s16", "u16", "s32", "u32", "s64", "u64"]

for m in ["", ".rn", ".ftz", ".sat", ".rn.ftz.sat"]:
    for op in ["add", "sub", "mul"]:
        case(f"{op}{m}.f32 %vd, %va, %vb", "f32", "f32", "f32")
for m in ["", ".ftz", ".sat", ".ftz.sat"]:
    case(f"fma.rn{m}.f32 %vd, %va, %vb, %vc", "f32", "f32", "f32", "f32")
case("mad.rn.f32 %vd, %va, %vb, %vc", "f32", "f32", "f32", "f32")
for m in ["", ".ftz"]:
    case(f"div.rn{m}.f32 %vd, %va, %vb", "f32", "f32", "f32")
    case(f"sqrt.rn{m}.f32 %vd, %va", "f32", "f32")
    case(f"rcp.rn{m}.f32 %vd, %va", "f32", "f32")
    case(f"neg{m}.f32 %vd, %va", "f32", "f32")
    case(f"abs{m}.f32 %vd, %va", "f32", "f32")
    for op in ["min", "max"]:
        case(f"{op}{m}.f32 %vd, %va, %vb", "f32", "f32", "f32")
        case(f"{op}{m}.NaN.f32 %vd, %va, %vb", "f32", "f32", "f32")
    for r in ["rni", "rzi", "rmi", "rpi"]:
        case(f"cvt.{r}{m}.f32.f32 %vd, %va", "f32", "f32")
    case(f"cvt{m}.sat.f32.f32 %vd, %va", "f32", "f32")
for cmp in F_CMPS:
    case(f"setp.{cmp}.f32 %vd, %va, %vb", "pred", "f32", "f32")
    case(f"setp.{cmp}.ftz.f32 %vd, %va, %vb", "pred", "f32", "f32")
case("mov.f32 %vd, %va", "f32", "f32")
case("selp.f32 %vd, %va, %vb, %vc", "f32", "f32", "f32", "pred")
# .approx: (relative, absolute) tolerance against the accurate value, from the
# PTX ISA's error bounds; lg2's is absolute near 1, sin's and cos's absolute.
for op, tol in [("ex2.approx", (2**-21, 0)), ("ex2.approx.ftz", (2**-21, 0)), ("lg2.approx", (2**-21, 2**-21)),
                ("rcp.approx", (2**-22, 0)), ("rcp.approx.ftz", (2**-22, 0)), ("rsqrt.approx", (2**-21, 0)),
                ("sqrt.approx", (2**-21, 0)), ("tanh.approx", (2**-10, 0)), ("sin.approx", (0, 2**-20)),
                ("cos.approx", (0, 2**-20))]:
    case(f"{op}.f32 %vd, %va", "f32", "f32", approx=tol)
case("div.approx.f32 %vd, %va, %vb", "f32", "f32", "f32", approx=(2**-21, 0))
case("div.full.f32 %vd, %va, %vb", "f32", "f32", "f32", approx=(2**-21, 0))

for op in ["add", "sub", "mul"]:
    case(f"{op}.rn.f64 %vd, %va, %vb", "f64", "f64", "f64")
case("fma.rn.f64 %vd, %va, %vb, %vc", "f64", "f64", "f64", "f64")
case("div.rn.f64 %vd, %va, %vb", "f64", "f64", "f64")
for op in ["sqrt.rn", "rcp.rn", "neg", "abs"]:
    case(f"{op}.f64 %vd, %va", "f64", "f64")
for op in ["min", "max"]:
    case(f"{op}.f64 %vd, %va, %vb", "f64", "f64", "f64")
for r in ["rni", "rzi", "rmi", "rpi"]:
    case(f"cvt.{r}.f64.f64 %vd, %va", "f64", "f64")
for cmp in F_CMPS:
    case(f"setp.{cmp}.f64 %vd, %va, %vb", "pred", "f64", "f64")

for op in ["add", "sub", "mul"]:
    case(f"{op}.rn.f16 %vd, %va, %vb", "f16", "f16", "f16")
case("fma.rn.f16 %vd, %va, %vb, %vc", "f16", "f16", "f16", "f16")
for op in ["neg", "abs"]:
    case(f"{op}.f16 %vd, %va", "f16", "f16")
for op in ["min", "max", "min.NaN", "max.NaN"]:
    case(f"{op}.f16 %vd, %va, %vb", "f16", "f16", "f16")

for body, d, a in [("cvt.rn.f32.f64", "f32", "f64"), ("cvt.rn.ftz.f32.f64", "f32", "f64"),
                   ("cvt.rn.sat.f32.f64", "f32", "f64"), ("cvt.f64.f32", "f64", "f32"),
                   ("cvt.ftz.f64.f32", "f64", "f32"), ("cvt.rn.f16.f32", "f16", "f32"),
                   ("cvt.rn.f16.f64", "f16", "f64"), ("cvt.f32.f16", "f32", "f16"), ("cvt.f64.f16", "f64", "f16")]:
    case(body + " %vd, %va", d, a)
for it in INTS:
    for ft in ["f32", "f64"]:
        for r in ["rzi", "rni", "rmi", "rpi"]:
            case(f"cvt.{r}.{it}.{ft} %vd, %va", it, ft)
        case(f"cvt.rn.{ft}.{it} %vd, %va", ft, it)
    case(f"cvt.rn.f16.{it} %vd, %va", "f16", it)
    case(f"cvt.rzi.{it}.f16 %vd, %va", it, "f16")
case("cvt.rzi.ftz.s32.f32 %vd, %va", "s32", "f32")
for a in INTS:
    for b in INTS:
        if a != b:
            case(f"cvt.{b}.{a} %vd, %va", b, a)
            wa, wb = WIDTH[a], WIDTH[b]
            if not ((a[0] == "u" and ((b[0] == "u" and wa <= wb) or (b[0] == "s" and wa < wb)))
                    or (a[0] == "s" and b[0] == "s" and wa <= wb)):
                case(f"cvt.sat.{b}.{a} %vd, %va", b, a)       # only where it can saturate

for t in ["s32", "u32", "s64", "u64", "s16", "u16"]:
    for op in ["add", "sub", "mul.lo", "mul.hi", "min", "max"]:
        case(f"{op}.{t} %vd, %va, %vb", t, t, t)
    for op in ["div", "rem"]:
        case(f"{op}.{t} %vd, %va, %vb", t, t, "nz" + t)
    for op in ["mad.lo", "mad.hi"]:
        case(f"{op}.{t} %vd, %va, %vb, %vc", t, t, t, t)
    for cmp in (["eq", "ne", "lt", "le", "gt", "ge"] + (["lo", "ls", "hi", "hs"] if t[0] == "u" else [])):
        case(f"setp.{cmp}.{t} %vd, %va, %vb", "pred", t, t)
    case(f"shr.{t} %vd, %va, %vb", t, t, "shift")
for t in ["s32", "s64", "s16"]:
    for op in ["neg", "abs"]:
        case(f"{op}.{t} %vd, %va", t, t)
for t, w in [("s32", "s64"), ("u32", "u64"), ("s16", "s32"), ("u16", "u32")]:
    case(f"mul.wide.{t} %vd, %va, %vb", w, t, t)
    case(f"mad.wide.{t} %vd, %va, %vb, %vc", w, t, t, w)
for op in ["add.sat", "sub.sat"]:
    case(f"{op}.s32 %vd, %va, %vb", "s32", "s32", "s32")
case("mad.hi.sat.s32 %vd, %va, %vb, %vc", "s32", "s32", "s32", "s32")
for t in ["b16", "b32", "b64"]:
    for op in ["and", "or", "xor"]:
        case(f"{op}.{t} %vd, %va, %vb", t, t, t)
    for op in ["not", "cnot"]:
        case(f"{op}.{t} %vd, %va", t, t)
    for op in ["shl", "shr"]:
        case(f"{op}.{t} %vd, %va, %vb", t, t, "shift")
    case(f"selp.{t} %vd, %va, %vb, %vc", t, t, t, "pred")
for op in ["and", "or", "xor"]:
    case(f"{op}.pred %vd, %va, %vb", "pred", "pred", "pred")
case("not.pred %vd, %va", "pred", "pred")
case("setp.lt.and.s32 %vd, %va, %vb, %vc", "pred", "s32", "s32", "pred")
case("setp.lt.or.f32 %vd, %va, %vb, %vc", "pred", "f32", "f32", "pred")
case("setp.lt.xor.u32 %vd, %va, %vb, !%vc", "pred", "u32", "u32", "pred")
case("setp.gt.s32 %tp0|%vd, %va, %vb", "pred", "s32", "s32")

for t in ["u32", "s32", "u64", "s64"]:
    x = "%t0" if t.endswith("32") else "%tl0"
    case(f"add.cc.{t} {x}, %va, %vb; addc.{t} %vd, %vc, 0", t, t, t, t)
    case(f"sub.cc.{t} {x}, %va, %vb; subc.{t} %vd, %vc, 0", t, t, t, t)
    case(f"add.cc.{t} {x}, %va, %vb; addc.cc.{t} {x}, {x}, %vc; addc.{t} %vd, 0, 0", t, t, t, t)
    case(f"sub.cc.{t} {x}, %va, %vb; subc.cc.{t} {x}, {x}, %vc; subc.{t} %vd, 0, 0", t, t, t, t)
    case(f"add.cc.{t} {x}, %va, %vb; addc.cc.{t} %vd, {x}, %vc", t, t, t, t)
    case(f"mad.lo.cc.{t} {x}, %va, %vb, %vc; madc.hi.{t} %vd, %va, %vb, 0", t, t, t, t)
    case(f"mad.hi.cc.{t} {x}, %va, %vb, %vc; madc.lo.cc.{t} {x}, %va, %vb, {x}; addc.{t} %vd, 0, 0", t, t, t, t)


def _gpu():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


@unittest.skipUnless(_gpu(), "needs a CUDA device of sm_80 or later")
class EveryInstructionMatchesTheGpu(unittest.TestCase):
    def check(self, body, d, srcs, approx, rng):
        cols = _inputs(srcs, rng)
        g, c = run(body, d, srcs, cols)
        bad = disagree(d, g, c, approx)
        if bad.any():
            rows = []
            for i in np.nonzero(bad)[0][:4]:
                ops = ", ".join(f"{col[i]!r}" for col in cols)
                rows.append(f"({ops}) -> gpu {int(g[i]):#x}, interpreter {int(c[i]):#x}")
            self.fail(f"{int(bad.sum())} of {len(g)} rows differ:\n  " + "\n  ".join(rows))
        return len(g)

    def test_every_form(self):
        rng = np.random.default_rng(12)
        rows = 0
        for body, d, srcs, approx in CASES:
            with self.subTest(form=body):
                rows += self.check(body, d, srcs, approx, rng)
        self.assertGreater(len(CASES), 500)
        self.assertGreater(rows, 2_000_000)

    def test_the_comparison_sees_nan_payloads_and_signed_zeros(self):
        # Negative controls: without the GPU's NaN rule, or with numpy's clip
        # for .sat (which keeps -0), the comparison must fail.
        rng = np.random.default_rng(1)
        with mock.patch.object(interpreter, "_gpu_nan", lambda r, nan_from=(): r):
            g, c = run("add.f32 %vd, %va, %vb", "f32", ("f32", "f32"), _inputs(("f32", "f32"), rng))
            self.assertTrue(disagree("f32", g, c, None).any())
        with mock.patch.object(interpreter, "_saturate", lambda r: np.clip(r, 0, 1).astype(r.dtype)):
            g, c = run("add.sat.f32 %vd, %va, %vb", "f32", ("f32", "f32"), _inputs(("f32", "f32"), rng))
            self.assertTrue(disagree("f32", g, c, None).any())

    def test_fma_f64_overflow_and_invalid_operations(self):
        # math.fma raises on these; the GPU writes infinities and NaN.
        rows = [(1e300, 1e300, 0.0), (-1e300, 1e300, 1.0), (1e308, 10.0, -np.inf), (0.0, np.inf, 1.0),
                (np.inf, -2.0, np.inf), (1.7976931348623157e308, 1.0, 1.7976931348623157e308)]
        cols = [np.array([r[j] for r in rows], np.float64) for j in range(3)]
        g, c = run("fma.rn.f64 %vd, %va, %vb, %vc", "f64", ("f64",) * 3, cols)
        self.assertEqual([hex(int(x)) for x in g], [hex(int(x)) for x in c])
        self.assertEqual(list(c.view(np.float64)[:3]), [np.inf, -np.inf, -np.inf])


if __name__ == "__main__":
    unittest.main()
