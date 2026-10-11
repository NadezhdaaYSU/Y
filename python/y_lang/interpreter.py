"""Run the PTX a Y kernel compiles to on the CPU, checking every access.

`Y_INTERPRETER=1` (or `Y_DEBUG=1`) makes `TorchKernel.launch` run here instead
of on the GPU, and `debug_interpret` runs one launch directly. The interpreter
executes the same PTX the GPU would, several thread blocks at a time in SIMT
lockstep - the lowest program counter goes first, so divergent paths
reconverge where the GPU's do - and stops with a `CPUInterpreterError` naming
the thread, the buffer and the PTX line when a kernel

- reads or writes outside every buffer it was given, or past a buffer's end;
- makes a misaligned access;
- reads or writes a shared variable past its end;
- reaches `trap` (a failed `@bounds` check, an out-of-range temperature...);
- reaches `bar.sync` with part of its block elsewhere, or `shfl.sync` with
  part of the warp it names elsewhere;
- writes two different values to one address in one instruction;
- divides an integer by zero; or
- runs an instruction, or a modifier of one, that this interpreter does not
  model - the tensor-core instructions, `.relu`, `bar.arrive` among them.
  Those are refused by name, never approximated.

Every instruction form it models gives the GPU's result bit for bit -
compared on an RTX 4070 Ti SUPER over edge values and random bits by
tests/test_interpreter_ops.py - including the GPU's NaN payloads, its
flush-to-zero under `.ftz`, signed zeros, saturation and every conversion.
The `.approx` functions (`ex2`, `lg2`, `rcp`, `rsqrt`, `tanh`, `sin`, ...)
are the exception: they are computed accurately and rounded, so they differ
from the GPU's approximations within its documented error, not bit for bit;
`sin` and `cos` first reduce their argument as the GPU does. Atomics apply in
thread order.

Until 2026-10-11 this module interpreted nothing: `run` printed the arguments'
shapes and returned "SUCCESS", and `TorchKernel.launch` in debug mode printed
"Executed safely under software bounds validation" and left every output
untouched.
"""

import math
import re
import time
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class CPUInterpreterError(Exception):
    """A kernel did something the GPU would do wrongly, or that this
    interpreter does not model."""


# ----------------------------------------------------------------------
# Types. Registers hold raw bits (uint64); each instruction reads them as
# the type it names.
# ----------------------------------------------------------------------

_U = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}
_S = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}
_F = {16: np.float16, 32: np.float32, 64: np.float64}


def _kind_width(t: str) -> Tuple[str, int]:
    if t == "pred":
        return "p", 1
    k, w = t[0], int(t[1:]) if t[1:].isdigit() else -1
    if k not in "usbf" or w not in (8, 16, 32, 64) or (k == "f" and w == 8):
        raise CPUInterpreterError(f"type .{t} is not modelled by the CPU interpreter")
    return ("u" if k == "b" else k), w


def _dtype(t: str):
    k, w = _kind_width(t)
    return {"u": _U, "s": _S, "f": _F}[k][w] if k != "p" else np.bool_


def _from_bits(bits: np.ndarray, t: str) -> np.ndarray:
    k, w = _kind_width(t)
    if k == "p":
        return bits != 0
    raw = bits.astype(_U[w]) if w < 64 else bits
    if k == "u":
        return raw
    return raw.view(_S[w] if k == "s" else _F[w])


def _to_bits(v: np.ndarray, t: str) -> np.ndarray:
    k, w = _kind_width(t)
    if k == "p":
        return np.asarray(v, dtype=bool).astype(np.uint64)
    arr = np.asarray(v)
    want = _dtype(t)
    if arr.dtype != want:
        arr = arr.astype(want)
    return arr.view(_U[w]).astype(np.uint64)


def _extend(v: np.ndarray, t: str) -> np.ndarray:
    """Bits of an integer value sign- or zero-extended to 64."""
    k, _ = _kind_width(t)
    if k == "s":
        return v.astype(np.int64).view(np.uint64)
    return v.astype(np.uint64)


class _Imm:
    """An immediate operand: an int, a float, or raw float bits (`0f...`)."""

    __slots__ = ("value", "raw_bits")

    def __init__(self, value, raw_bits=None):
        self.value = value
        self.raw_bits = raw_bits

    def bits(self, t: str) -> int:
        k, w = _kind_width(t)
        if self.raw_bits is not None:
            bw, b = self.raw_bits
            if k == "f" and bw != w:
                return int(np.asarray(np.frombuffer(np.uint64(b).tobytes()[: bw // 8], _F[bw])[0], _F[w]).view(_U[w]))
            return b & ((1 << w) - 1)
        if k == "f":
            return int(np.asarray(self.value, _F[w]).view(_U[w]))
        if k == "p":
            return 1 if self.value else 0
        if isinstance(self.value, float):
            raise CPUInterpreterError(f"a float immediate {self.value} used as .{t}")
        return int(self.value) & ((1 << w) - 1)


def _parse_imm(s: str) -> Optional[_Imm]:
    s = s.strip()
    m = re.fullmatch(r"([+-]?)0[fF]([0-9a-fA-F]{8})", s)
    if m:
        b = int(m.group(2), 16) ^ (0x80000000 if m.group(1) == "-" else 0)
        return _Imm(None, (32, b))
    m = re.fullmatch(r"([+-]?)0[dD]([0-9a-fA-F]{16})", s)
    if m:
        b = int(m.group(2), 16) ^ ((1 << 63) if m.group(1) == "-" else 0)
        return _Imm(None, (64, b))
    if re.fullmatch(r"[+-]?0[xX][0-9a-fA-F]+[uU]?", s):
        return _Imm(int(s.rstrip("uU"), 16))
    if re.fullmatch(r"[+-]?\d+[uU]?", s):
        return _Imm(int(s.rstrip("uU")))
    if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?", s):
        return _Imm(float(s))
    return None


# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------

class _Instr:
    __slots__ = ("guard", "neg", "op", "parts", "args", "line", "text")

    def __init__(self, guard, neg, op, args, line, text):
        self.guard, self.neg, self.op = guard, neg, op
        self.parts = op.split(".")
        self.args, self.line, self.text = args, line, text


class _Var:
    __slots__ = ("name", "space", "size", "align", "init")

    def __init__(self, name, space, size, align, init=None):
        self.name, self.space, self.size, self.align, self.init = name, space, size, align, init


class _Func:
    def __init__(self, name: str, kind: str):
        self.name, self.kind = name, kind
        self.params: List[Tuple[str, str, str]] = []   # (space, type, name)
        self.rets: List[Tuple[str, str, str]] = []
        self.regs: Dict[str, str] = {}
        self.instrs: List[_Instr] = []
        self.labels: Dict[str, int] = {}
        self.shared: List[_Var] = []
        self.local: List[_Var] = []


def _strip_comments(text: str) -> str:
    return re.sub(r"//[^\n]*|/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)


def _split_top(s: str) -> List[str]:
    out, depth, cur = [], 0, []
    for c in s:
        if c in "[{(":
            depth += 1
        elif c in "]})":
            depth -= 1
        if c == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def _decl_list(s: str) -> List[Tuple[str, str, str]]:
    """`.param .u64 p0, .reg .b32 %x` -> [(space, type, name)]."""
    out = []
    for item in _split_top(s):
        toks = item.split()
        space = toks[0].lstrip(".")
        types = [t.lstrip(".") for t in toks[1:-1] if t.startswith(".") and not t.startswith(".align")]
        name = toks[-1]
        if "[" in name or not types:
            raise CPUInterpreterError(f"parameter `{item}` is not modelled by the CPU interpreter")
        out.append((space, types[-1], name))
    return out


def _parse_var(decl: str, space: str) -> _Var:
    m = re.match(r"(?:\.align\s+(\d+)\s+)?\.(\w+)\s+([\w$]+)\s*(?:\[\s*(\d*)\s*\])?\s*(?:=\s*(.*))?$", decl.strip(), re.S)
    if not m:
        raise CPUInterpreterError(f"declaration `.{space} {decl.strip()}` is not modelled by the CPU interpreter")
    align, t, name, count, init = m.groups()
    _, w = _kind_width(t)
    n = int(count) if count else (0 if count == "" else 1)
    data = None
    if init is not None:
        vals = [v for v in re.split(r"[\s,{}]+", init) if v]
        data = np.zeros(n * w // 8, np.uint8)
        for i, v in enumerate(vals):
            imm = _parse_imm(v)
            if imm is None:
                raise CPUInterpreterError(f"initialiser `{v}` of `{name}` is not modelled")
            data[i * w // 8:(i + 1) * w // 8] = np.frombuffer(int(imm.bits(t)).to_bytes(w // 8, "little"), np.uint8)
    return _Var(name, space, n * w // 8, int(align) if align else w // 8, data)


class _Module:
    def __init__(self, text: str):
        self.funcs: Dict[str, _Func] = {}
        self.vars: List[_Var] = []
        text = _strip_comments(text)
        # These three take no `;`, so they would fuse with the next item.
        text = re.sub(r"^[ \t]*\.(version|target|address_size)\b[^\n]*", "", text, flags=re.M)
        for start, item in self._items(text):
            line = text.count("\n", 0, start) + 1
            s = item.strip()
            if re.search(r"\.(entry|func)\b", s.split("{", 1)[0]) and "{" in s:
                self._function(s, line + item[: len(item) - len(item.lstrip())].count("\n"))
            elif re.match(r"(\.(visible|extern|weak)\s+)*\.(global|const|shared)\b", s):
                m = re.match(r"(?:\.(?:visible|extern|weak)\s+)*\.(global|const|shared)\s+(.*);$", s, re.S)
                self.vars.append(_parse_var(m.group(2), m.group(1)))
            # .version/.target/.address_size and extern declarations need nothing

    @staticmethod
    def _items(text: str):
        start, depth_b, depth_p = 0, 0, 0
        for i, c in enumerate(text):
            if c == "(":
                depth_p += 1
            elif c == ")":
                depth_p -= 1
            elif c == "{":
                depth_b += 1
            elif c == "}":
                depth_b -= 1
                if depth_b == 0 and depth_p == 0:
                    head = text[start:i].split("{", 1)[0]
                    if re.search(r"\.(entry|func)\b", head) and "=" not in head:
                        yield start, text[start:i + 1]
                        start = i + 1
            elif c == ";" and depth_b == 0 and depth_p == 0:
                yield start, text[start:i + 1]
                start = i + 1

    def _function(self, s: str, line: int):
        head, body = s.split("{", 1)
        body = body.rsplit("}", 1)[0]
        body_line = line + head.count("\n")
        m = re.search(r"\.(entry|func)\s*(?:\(([^)]*)\))?\s*([\w$]+)\s*(?:\(([^)]*)\))?", head, re.S)
        kind, rets, name, params = m.groups()
        f = _Func(name, kind)
        f.rets = _decl_list(rets) if rets and rets.strip() else []
        f.params = _decl_list(params) if params and params.strip() else []
        for space, t, pname in f.params + f.rets:
            if space == "reg":
                f.regs[pname] = t
        pos = 0
        for chunk in body.split(";"):
            chunk_at = pos
            pos += len(chunk) + 1
            stmt = chunk.strip()
            while True:
                if stmt[:1] in ("{", "}"):
                    stmt = stmt[1:].strip()
                    continue
                lm = re.match(r"([A-Za-z_$][\w$]*)\s*:(?!:)", stmt)
                if lm:
                    f.labels[lm.group(1)] = len(f.instrs)
                    stmt = stmt[lm.end():].strip()
                    continue
                break
            if not stmt:
                continue
            cline = body_line + body.count("\n", 0, chunk_at + max(chunk.find(stmt.split()[0]), 0))
            if stmt.startswith("."):
                self._directive(f, stmt)
                continue
            gm = re.match(r"@(!?)(%[\w$.]+)\s+", stmt)
            guard, neg = (gm.group(2), gm.group(1) == "!") if gm else (None, False)
            if gm:
                stmt = stmt[gm.end():]
            op, _, rest = stmt.partition(" ")
            if "\t" in op:
                op, _, more = op.partition("\t")
                rest = more + " " + rest
            f.instrs.append(_Instr(guard, neg, op.strip(), _split_top(rest.strip()) if rest.strip() else [],
                                   cline, " ".join(stmt.split())))
        self.funcs[name] = f

    @staticmethod
    def _directive(f: _Func, stmt: str):
        head = stmt.split()[0]
        if head == ".reg":
            toks = stmt[4:].strip()
            types = re.findall(r"\.(\w+)", toks.split("%")[0])
            if not types or types[0] in ("v2", "v4"):
                raise CPUInterpreterError(f"`{stmt}` is not modelled by the CPU interpreter")
            t = types[-1]
            for item in _split_top(toks[toks.index("%"):]):
                rm = re.fullmatch(r"(%[\w$]+)<(\d+)>", item)
                if rm:
                    for i in range(int(rm.group(2))):
                        f.regs[f"{rm.group(1)}{i}"] = t
                else:
                    f.regs[item] = t
        elif head in (".shared", ".local"):
            v = _parse_var(stmt[len(head):], head[1:])
            (f.shared if head == ".shared" else f.local).append(v)
        elif head in (".loc", ".file", ".pragma", ".maxntid", ".reqntid", ".minnctapersm", ".maxnreg"):
            pass
        else:
            raise CPUInterpreterError(f"directive `{stmt}` is not modelled by the CPU interpreter")


# ----------------------------------------------------------------------
# Memory
# ----------------------------------------------------------------------

_ALLOC_STRIDE = 1 << 40   # far apart, so an out-of-bounds address lands in no buffer


class _Alloc:
    __slots__ = ("base", "size", "data", "name", "space")

    def __init__(self, base, size, data, name, space):
        self.base, self.size, self.data, self.name, self.space = base, size, data, name, space


# ----------------------------------------------------------------------
# Floating point. A float32 or float16 result is computed in float64 and
# rounded once to its type; for +, -, *, / and sqrt that equals rounding the
# exact result, and fma carries its float64 rounding error along. The NaN,
# flush-to-zero and signed-zero rules below are the GPU's, measured on an
# RTX 4070 Ti SUPER (sm_89) by python/y_lang/tests/test_interpreter_ops.py.
# ----------------------------------------------------------------------

def _two_sum(p, c):
    """s = p + c rounded to float64, and its error e: s + e == p + c exactly."""
    s = p + c
    bp = s - c
    return s, (p - bp) + (c - (s - bp))


# Each format's largest finite value and the magnitude from which a value
# rounds to infinity.
_OVERFLOW = {np.dtype(np.float32): (float(np.finfo(np.float32).max), 2.0 ** 128 - 2.0 ** 103),
             np.dtype(np.float16): (65504.0, 65520.0)}


def _round_once(s, e, dt):
    """s + e (s a float64 rounding, e its error, or None when s is exact)
    rounded once to float32 or float16. Rounding s alone is right unless s lies
    exactly on a midpoint of `dt`, or on the overflow threshold, that the exact
    value is not on; s is moved toward the exact value first in that case."""
    r = s.astype(dt)
    if e is None:
        return r
    rd = r.astype(np.float64)
    toward = np.where(s > rd, np.inf, -np.inf).astype(dt)
    mid = (rd + np.nextafter(r, toward).astype(np.float64)) / 2
    fix = np.isfinite(e) & (e != 0) & ((s == mid) | (np.abs(s) == _OVERFLOW[np.dtype(dt)][1]))
    if fix.any():
        s = s.copy()
        s[fix] = np.nextafter(s[fix], np.where(e[fix] > 0, np.inf, -np.inf))
        r = s.astype(dt)
    return r


# Under .ftz the GPU writes a signed zero for a float32 result that, rounded
# to 24 bits with an unbounded exponent, is below 2^-126: tininess after
# rounding. 2^-126 * (1 - 2^-46) rounds up to 2^-126 and is kept;
# 2^-126 * (1 - 2^-24) is flushed, though subnormal rounding gives 2^-126.
_TINY32 = 2.0 ** -126 * (1 - 2.0 ** -25)


def _flush_tiny(r, s, e):
    a = np.abs(s)
    tiny = a < _TINY32
    if e is not None:
        tiny |= (a == _TINY32) & (e * np.sign(s) < 0)
    if tiny.any():
        r = np.where(tiny, np.copysign(0.0, s), r).astype(np.float32)
    return r


def _saturate(r):
    """.sat: clamped to [+0, 1]; NaN and every negative, -0 included, give +0."""
    return np.where(r > 0, np.minimum(r, 1), 0).astype(r.dtype)


def _ftz(x):
    """A float32 operand under .ftz: subnormals read as zero of their sign."""
    if x.dtype == np.float32:
        tiny = np.abs(x) < np.float32(1.1754943508222875e-38)
        if tiny.any():
            x = np.where(tiny & (x != 0), np.copysign(np.float32(0), x), x).astype(np.float32)
    return x


_CANONICAL_NAN = {np.dtype(np.float32): 0x7FFFFFFF, np.dtype(np.float16): 0x7FFF}
_QUIET64 = np.uint64(1 << 51)
_DEFAULT_NAN64 = np.uint64(0xFFF8000000000000)


def _gpu_nan(r, nan_from=()):
    """NaN results as the GPU writes them. A float32 or float16 instruction
    writes one canonical NaN (0x7FFFFFFF, 0x7FFF) whatever its operands, `neg`
    and `abs` included. A float64 one passes on the first NaN among
    `nan_from` - the operand order that instruction prefers - quieted and with
    its sign, and writes 0xFFF8000000000000 when no operand is NaN."""
    nan = np.isnan(r)
    if not nan.any():
        return r
    bits = r.view(_U[r.dtype.itemsize * 8]).copy()
    if r.dtype != np.float64:
        bits[nan] = _CANONICAL_NAN[r.dtype]
        return bits.view(r.dtype)
    left = nan.copy()
    for o in nan_from:
        o = np.asarray(o, np.float64)
        pick = left & np.isnan(o)
        bits[pick] = o.view(np.uint64)[pick] | _QUIET64
        left &= ~pick
    bits[left] = _DEFAULT_NAN64
    return bits.view(np.float64)


def _fma64(a, b, c):
    """fma.rn.f64 rounded once (math.fma), with IEEE's infinities: math.fma
    raises on overflow and on 0 * inf where the GPU writes inf and NaN."""
    if not hasattr(math, "fma"):
        raise CPUInterpreterError("fma.f64 needs Python 3.13's math.fma")
    out = np.empty(a.shape, np.float64)
    for i, (x, y, z) in enumerate(zip(a.tolist(), b.tolist(), c.tolist())):
        if not (math.isfinite(x) and math.isfinite(y)):
            out[i] = x * y + z           # an infinite or NaN factor decides
        elif not math.isfinite(z):
            out[i] = z                   # x*y is finite, however large
        else:
            try:
                out[i] = math.fma(x, y, z)
            except OverflowError:
                out[i] = math.inf if Fraction(x) * Fraction(y) + Fraction(z) > 0 else -math.inf
    return out


def _div_approx(a, b):
    """div.approx.f32 multiplies by an approximate reciprocal, which
    underflows for 2^126 < |b| < 2^128: the GPU then writes 0, or NaN when `a`
    is infinite or NaN (PTX ISA, and measured)."""
    big = np.isfinite(b) & (np.abs(b) > 2.0 ** 126)
    return np.where(big, np.where(np.isfinite(a), np.copysign(0.0, a) * np.copysign(1.0, b), np.nan), a / b)


_ROUND_TO_INT = {"rni": np.rint, "rzi": np.trunc, "rmi": np.floor, "rpi": np.ceil}

_INV_2PI_F32 = float(np.uint32(0x3E22F983).view(np.float32))


def _turns(x):
    """The GPU's argument reduction for sin.approx and cos.approx: x times
    float32(1/2pi), rounded toward zero to float32, so a large argument keeps
    no fraction of a turn. Modelled, the result was within 3.3e-7 of the GPU's
    on 8,000 arguments up to the largest float32; the true sine and cosine
    differ from the GPU's by up to 2."""
    p = x * _INV_2PI_F32
    y = p.astype(np.float32)
    y = np.where(np.abs(y.astype(np.float64)) > np.abs(p), np.nextafter(y, np.float32(0)), y)
    return 2 * np.pi * np.fmod(y.astype(np.float64), 1.0)


# The modifiers each instruction may carry besides its types. Any other
# modifier - .relu, .xorsign, .satfinite, .arrive, .shared::cluster ... -
# changes what the instruction does, so the kernel is refused rather than run
# as if it were absent. None: every modifier is accepted (fences and
# prefetches do nothing here).
_TYPE_NAMES = {f"{k}{w}" for k in "usb" for w in (8, 16, 32, 64)} | {"f16", "f32", "f64", "pred"}
_ROUND = {"rn", "rz", "rm", "rp"}
_ORDER = {"relaxed", "acquire", "release", "acq_rel", "weak", "volatile", "cta", "gpu", "sys"}
_ATOM_OPS = {"add", "min", "max", "inc", "dec", "and", "or", "xor", "exch", "cas"}
_MODIFIERS = {
    "bra": {"uni"}, "ret": {"uni"}, "call": {"uni"}, "exit": set(), "trap": set(), "brkpt": set(),
    "bar": {"sync", "cta"}, "barrier": {"sync", "aligned", "cta"},
    "membar": None, "fence": None, "createpolicy": None, "prefetch": None, "prefetchu": None,
    "mov": set(), "cvta": {"to", "global", "shared", "const", "local"},
    "add": _ROUND | {"ftz", "sat", "cc"}, "sub": _ROUND | {"ftz", "sat", "cc"},
    "addc": {"cc"}, "subc": {"cc"},
    "mul": _ROUND | {"lo", "hi", "wide", "ftz", "sat"},
    "mad": _ROUND | {"lo", "hi", "wide", "ftz", "sat", "cc"}, "madc": {"lo", "hi", "cc"},
    "fma": _ROUND | {"ftz", "sat"}, "div": _ROUND | {"ftz", "approx", "full"}, "rem": set(),
    "min": {"ftz", "NaN"}, "max": {"ftz", "NaN"}, "neg": {"ftz"}, "abs": {"ftz"},
    "not": set(), "cnot": set(), "and": set(), "or": set(), "xor": set(), "shl": set(), "shr": set(),
    "ex2": {"approx", "ftz"}, "lg2": {"approx", "ftz"}, "sin": {"approx", "ftz"}, "cos": {"approx", "ftz"},
    "tanh": {"approx"}, "rsqrt": {"approx", "ftz"},
    "rcp": _ROUND | {"approx", "ftz"}, "sqrt": _ROUND | {"approx", "ftz"},
    "setp": {"eq", "ne", "lt", "le", "gt", "ge", "lo", "ls", "hi", "hs", "equ", "neu", "ltu", "leu",
             "gtu", "geu", "num", "nan", "and", "or", "xor", "ftz"},
    "selp": set(),
    "cvt": _ROUND | {"rni", "rzi", "rmi", "rpi", "ftz", "sat"},
    "ld": {"global", "shared", "const", "param", "local", "v2", "v4", "nc", "ca", "cg", "cs", "lu", "cv"} | _ORDER,
    "ldu": {"global", "v2", "v4"},
    "st": {"global", "shared", "local", "v2", "v4", "wb", "cg", "cs", "wt"} | _ORDER,
    "atom": {"global", "shared", "noftz"} | _ATOM_OPS | _ORDER,
    "red": {"global", "shared", "noftz"} | _ATOM_OPS | _ORDER,
    "cp": {"async", "ca", "cg", "shared", "global", "commit_group", "wait_group", "wait_all"},
    "shfl": {"sync", "bfly", "idx", "up", "down"},
}
_HINTS = ("L1::", "L2::")      # cache hints: performance only


def _unmodelled_modifier(ins) -> Optional[str]:
    allowed = _MODIFIERS.get(ins.parts[0], set())
    if allowed is None:
        return None
    for p in ins.parts[1:]:
        q = "shared" if p == "shared::cta" else p
        if q in allowed or q in _TYPE_NAMES or (p.startswith(_HINTS) and ins.parts[0] in ("ld", "st", "atom", "red", "cp")):
            continue
        return p
    return None


class _Frame:
    """One function's registers for a set of lanes of a block batch."""

    def __init__(self, func: _Func, lanes: np.ndarray):
        self.func = func
        self.lanes = lanes                      # indices into the batch's lane arrays
        n = lanes.size
        self.regs = {name: np.zeros(n, np.uint64) for name in func.regs}
        self.cc = np.zeros(n, np.uint64)        # the carry flag, CC.CF
        self.local = None


class CPUInterpreter:
    """Runs a PTX kernel on the CPU. See the module docstring."""

    def __init__(self, kernel_name: str = "y_debug_kernel", ptx: Optional[str] = None,
                 max_steps: int = 50_000_000, lanes_per_batch: int = 65536):
        self.kernel_name = kernel_name
        self.ptx = ptx
        self.max_steps = max_steps
        self.lanes_per_batch = lanes_per_batch
        self.access_logs: List[str] = []
        self.bounds_checks_passed = 0
        self.bounds_checks_failed = 0
        self.instructions = 0

    # ------------------------------------------------------------------
    # The old helper, kept: checks an index tuple against a tensor's shape.
    def validate_tensor_bounds(self, tensor: Any, indices: Tuple[int, ...], name: str = "tensor") -> None:
        shape = tuple(tensor.shape) if hasattr(tensor, "shape") else ()
        if len(indices) != len(shape):
            self.bounds_checks_failed += 1
            raise CPUInterpreterError(
                f"[Y CPU Interpreter] Rank mismatch for '{name}': Expected {len(shape)} dimensions, got index tuple {indices}.")
        for dim, (idx, dim_size) in enumerate(zip(indices, shape)):
            if idx < 0 or idx >= dim_size:
                self.bounds_checks_failed += 1
                raise CPUInterpreterError(
                    f"[Y CPU Interpreter] Out-of-Bounds Memory Access in '{name}' at axis {dim}: index {idx} out of range [0, {dim_size}).")
        self.bounds_checks_passed += 1

    # ------------------------------------------------------------------
    def run(self, grid: Tuple[int, int, int], block: Tuple[int, int, int], args: List[Any],
            kernel_func: Callable = None, shared_bytes: int = 0) -> Dict[str, Any]:
        """Run one launch. With `kernel_func` and no PTX, call that Python
        function instead (it is not Y; it gets this interpreter for
        `validate_tensor_bounds`)."""
        start = time.perf_counter()
        if self.ptx is None:
            if kernel_func is None or not callable(kernel_func):
                raise CPUInterpreterError(
                    f"nothing to run for `{self.kernel_name}`: give the interpreter the kernel's PTX "
                    "(CPUInterpreter(name, ptx=...)), a TorchKernel or Y source to debug_interpret, or a kernel_func")
            try:
                kernel_func(grid, block, args, self)
            except CPUInterpreterError:
                raise
            except Exception as e:
                raise CPUInterpreterError(f"Kernel execution failure in debug interpreter: {e}") from e
        else:
            self._launch(tuple(grid) + (1,) * (3 - len(grid)), tuple(block) + (1,) * (3 - len(block)),
                         list(args), shared_bytes)
        return {
            "kernel_name": self.kernel_name,
            "status": "SUCCESS",
            "elapsed_ms": (time.perf_counter() - start) * 1000.0,
            "instructions": self.instructions,
            "bounds_checks_passed": self.bounds_checks_passed,
            "bounds_checks_failed": self.bounds_checks_failed,
        }

    # ------------------------------------------------------------------
    def _launch(self, grid, block, args, shared_bytes):
        import torch

        self.module = _Module(self.ptx)
        entry = self.module.funcs.get(self.kernel_name)
        if entry is None or entry.kind != "entry":
            names = [f.name for f in self.module.funcs.values() if f.kind == "entry"]
            raise CPUInterpreterError(f"the PTX has no kernel `{self.kernel_name}` (it has {names})")
        if len(args) != len(entry.params):
            raise CPUInterpreterError(
                f"`{self.kernel_name}` takes {len(entry.params)} arguments and was given {len(args)}")

        # Buffers: each tensor argument and each module variable, far apart.
        self.allocs: List[_Alloc] = []
        writeback = []
        self.params: Dict[str, Tuple[bytes, str]] = {}
        for (space, t, pname), arg in zip(entry.params, args):
            _, w = _kind_width(t)
            if hasattr(arg, "data_ptr") and callable(arg.data_ptr):
                if not arg.is_contiguous():
                    raise CPUInterpreterError(f"argument `{pname}` is not contiguous; pass `.contiguous()`")
                host = arg.detach().cpu() if arg.is_cuda else arg.detach()
                data = host.reshape(-1).view(torch.uint8).numpy() if host.numel() else np.zeros(0, np.uint8)
                base = (len(self.allocs) + 1) * _ALLOC_STRIDE
                self.allocs.append(_Alloc(base, data.size, data, _pretty(pname), "global"))
                if arg.is_cuda:
                    writeback.append((arg, host))
                value = base
            elif isinstance(arg, bool):
                value = int(arg)
            elif isinstance(arg, int):
                value = arg & ((1 << w) - 1)
            elif isinstance(arg, float):
                # As TorchKernel.launch packs it: a float becomes float32 bits.
                value = int(np.asarray(arg, np.float32).view(np.uint32))
            else:
                raise CPUInterpreterError(f"argument `{pname}` has unsupported type {type(arg).__name__}")
            self.params[pname] = (int(value & ((1 << w) - 1)).to_bytes(w // 8, "little"), t)
        for v in self.module.vars:
            if v.space == "shared":
                continue
            data = v.init.copy() if v.init is not None else np.zeros(v.size, np.uint8)
            base = (len(self.allocs) + 1) * _ALLOC_STRIDE
            self.allocs.append(_Alloc(base, v.size, data, v.name, v.space))
        self.symbols = {a.name: a.base for a in self.allocs if a.space in ("global", "const")}
        self.bases = np.array([a.base for a in self.allocs], np.uint64)
        self.sizes = np.array([a.size for a in self.allocs], np.uint64)

        # Shared memory: module-scope and the entry's own, laid out in order.
        self.shared_vars: List[Tuple[int, int, str]] = []
        off = 0
        for v in [v for v in self.module.vars if v.space == "shared"] + entry.shared:
            off = -(-off // v.align) * v.align
            size = v.size if v.size else shared_bytes
            self.shared_vars.append((off, size, v.name))
            off += size
        self.shared_size = off
        self.shared_offs = np.array([s[0] for s in self.shared_vars], np.int64)
        self.shared_ends = np.array([s[0] + s[1] for s in self.shared_vars], np.int64)
        self.shared_symbols = {name: o for o, _, name in self.shared_vars}

        nthreads = block[0] * block[1] * block[2]
        if nthreads == 0 or nthreads > 1024:
            raise CPUInterpreterError(f"a block of {nthreads} threads cannot be launched")
        blocks = [(x, y, z) for z in range(grid[2]) for y in range(grid[1]) for x in range(grid[0])]
        per = max(1, self.lanes_per_batch // nthreads)
        with np.errstate(all="ignore"):
            for i in range(0, len(blocks), per):
                self._run_batch(entry, blocks[i:i + per], block, grid)
        for dev, host in writeback:
            dev.copy_(host)

    def _run_batch(self, entry, ctaids, block, grid):
        nt = block[0] * block[1] * block[2]
        nb = len(ctaids)
        n = nb * nt
        lane = np.arange(n)
        t = lane % nt
        self.blk = lane // nt
        self.special = {
            "%tid.x": t % block[0], "%tid.y": (t // block[0]) % block[1], "%tid.z": t // (block[0] * block[1]),
            "%ntid.x": np.full(n, block[0]), "%ntid.y": np.full(n, block[1]), "%ntid.z": np.full(n, block[2]),
            "%ctaid.x": np.array([c[0] for c in ctaids])[self.blk], "%ctaid.y": np.array([c[1] for c in ctaids])[self.blk],
            "%ctaid.z": np.array([c[2] for c in ctaids])[self.blk],
            "%nctaid.x": np.full(n, grid[0]), "%nctaid.y": np.full(n, grid[1]), "%nctaid.z": np.full(n, grid[2]),
            "%laneid": t % 32, "%warpid": t // 32, "%nwarpid": np.full(n, -(-nt // 32)),
            "%smid": np.zeros(n), "%gridid": np.zeros(n),
        }
        self.thread_in_block = t
        self.warp_key = self.blk * 64 + t // 32
        self.shared = np.zeros((nb, max(self.shared_size, 1)), np.uint8)
        self.ctaids = ctaids
        self.block_dim = block
        frame = _Frame(entry, lane)
        for space, ty, pname in entry.params:
            if space == "reg":
                raise CPUInterpreterError("a kernel taking .reg parameters is not modelled")
        self._run_frame(frame, top=True)

    # ------------------------------------------------------------------
    # Diagnostics
    def _where(self, frame: _Frame, lane: int, ins: _Instr) -> str:
        g = frame.lanes[lane]
        tid = tuple(int(self.special[f"%tid.{a}"][g]) for a in "xyz")
        cta = self.ctaids[self.blk[g]]
        return (f"`{self.kernel_name}`, thread {tid} of block {cta}"
                f"{'' if frame.func.kind == 'entry' else f' in `{frame.func.name}`'}: "
                f"PTX line {ins.line}: {ins.text}")

    def _fail(self, frame, lane, ins, what):
        self.bounds_checks_failed += 1
        raise CPUInterpreterError(f"{what}\n  at {self._where(frame, lane, ins)}")

    # ------------------------------------------------------------------
    # Operands
    def _read(self, frame: _Frame, opnd: str, t: str, lanes: np.ndarray) -> np.ndarray:
        if opnd.startswith("!"):
            return ~self._read(frame, opnd[1:], t, lanes)
        if opnd.startswith("%"):
            if opnd in frame.regs:
                return _from_bits(frame.regs[opnd][lanes], t)
            if opnd in self.special:
                v = self.special[opnd][frame.lanes[lanes]].astype(np.int64)
                return _from_bits(v.view(np.uint64), t)
            if opnd in ("%clock", "%clock64", "%globaltimer"):
                return _from_bits(np.full(lanes.size, self.instructions, np.uint64), t)
            raise CPUInterpreterError(f"register `{opnd}` is not declared")
        imm = _parse_imm(opnd)
        if imm is not None:
            return _from_bits(np.full(lanes.size, imm.bits(t), np.uint64), t)
        addr = self._symbol(frame, opnd)
        if addr is not None:
            return _from_bits(np.full(lanes.size, addr, np.uint64), t)
        raise CPUInterpreterError(f"operand `{opnd}` is not modelled by the CPU interpreter")

    def _symbol(self, frame: _Frame, name: str) -> Optional[int]:
        base, _, off = name.partition("+")
        o = int(off, 0) if off else 0
        if base in self.symbols:
            return self.symbols[base] + o
        if base in self.shared_symbols:
            return self.shared_symbols[base] + o
        for i, v in enumerate(frame.func.local):
            if v.name == base:
                return sum(x.size for x in frame.func.local[:i]) + o
        return None

    def _write(self, frame: _Frame, dest: str, t: str, lanes: np.ndarray, value: np.ndarray):
        if dest == "_":
            return
        if dest not in frame.regs:
            raise CPUInterpreterError(f"register `{dest}` is not declared")
        frame.regs[dest][lanes] = _to_bits(value, t)

    def _write_bits(self, frame, dest, lanes, bits):
        if dest == "_":
            return
        if dest not in frame.regs:
            raise CPUInterpreterError(f"register `{dest}` is not declared")
        frame.regs[dest][lanes] = bits

    # ------------------------------------------------------------------
    # The scheduler
    def _run_frame(self, frame: _Frame, top: bool):
        func = frame.func
        n = frame.lanes.size
        pc = np.zeros(n, np.int64)
        alive = np.ones(n, bool)
        end = len(func.instrs)
        steps = 0
        while True:
            live = np.nonzero(alive)[0]
            if live.size == 0:
                return
            p = pc[live]
            pmin = int(p.min())
            if pmin >= end:
                alive[live] = False
                return
            group = live[p == pmin]
            ins = func.instrs[pmin]
            steps += 1
            self.instructions += 1
            if steps > self.max_steps:
                self._fail(frame, int(group[0]), ins,
                           f"more than {self.max_steps:,} instructions in one batch of blocks - a loop that does not end?")
            run = group
            if ins.guard is not None:
                g = self._read(frame, ins.guard, "pred", group)
                if ins.neg:
                    g = ~g
                pc[group[~g]] = pmin + 1
                run = group[g]
                if run.size == 0:
                    continue
            self._exec(frame, ins, run, pmin, pc, alive, live, group)

    # ------------------------------------------------------------------
    def _exec(self, frame, ins, lanes, here, pc, alive, live, group):
        op = ins.parts[0]
        handler = getattr(self, "_op_" + op, None)
        if handler is None:
            self._fail(frame, int(lanes[0]), ins,
                       f"`{ins.op}` is not modelled by the CPU interpreter; it refuses the kernel rather than guess")
        odd = _unmodelled_modifier(ins)
        if odd is not None:
            self._fail(frame, int(lanes[0]), ins,
                       f"the modifier .{odd} of `{ins.op}` is not modelled by the CPU interpreter; "
                       "it refuses the kernel rather than run the instruction without it")
        nxt = handler(frame, ins, lanes, here, alive, live, group)
        if nxt is None:
            pc[lanes] = here + 1
        else:
            pc[lanes] = nxt

    # Control ----------------------------------------------------------
    def _op_bra(self, frame, ins, lanes, here, *_):
        target = ins.args[0]
        if target not in frame.func.labels:
            raise CPUInterpreterError(f"label `{target}` is not defined")
        return frame.func.labels[target]

    def _op_ret(self, frame, ins, lanes, here, alive, *_):
        alive[lanes] = False
        return len(frame.func.instrs)

    def _op_exit(self, frame, ins, lanes, here, alive, *_):
        if frame.func.kind != "entry":
            self._fail(frame, int(lanes[0]), ins, "`exit` inside a called function is not modelled")
        return self._op_ret(frame, ins, lanes, here, alive)

    def _op_trap(self, frame, ins, lanes, *_):
        self._fail(frame, int(lanes[0]), ins,
                   f"the kernel executed `trap` ({lanes.size} thread(s) reached it); on the GPU the launch fails")

    def _op_brkpt(self, frame, ins, lanes, *_):
        self._fail(frame, int(lanes[0]), ins, "the kernel executed `brkpt`")

    def _barrier(self, frame, ins, lanes, live, group, what):
        if frame.func.kind != "entry":
            self._fail(frame, int(lanes[0]), ins, f"`{what}` inside a called function is not modelled")
        if lanes.size != group.size:
            self._fail(frame, int(lanes[0]), ins, f"a predicated `{what}` is not modelled")
        blocks_here = np.unique(self.blk[frame.lanes[group]])
        others = np.setdiff1d(live, group)
        stray = others[np.isin(self.blk[frame.lanes[others]], blocks_here)]
        if stray.size:
            self._fail(frame, int(stray[0]), ins,
                       f"`{what}` reached by only part of a block: this thread of the same block is at "
                       "another instruction, so on the GPU the block hangs or the barrier is undefined")

    def _op_bar(self, frame, ins, lanes, here, alive, live, group):
        # bar.sync is barrier.sync.aligned: every thread of the block must reach
        # this same instruction. barrier.sync without .aligned lets threads meet
        # at different instructions, which the check below would misreport.
        if ins.parts[0] == "barrier" and "aligned" not in ins.parts:
            self._fail(frame, int(lanes[0]), ins, f"`{ins.op}` without .aligned is not modelled")
        if len(ins.args) > 1:
            self._fail(frame, int(lanes[0]), ins, "a `bar` with a thread count is not modelled")
        self._barrier(frame, ins, lanes, live, group, ins.op)

    _op_barrier = _op_bar

    def _op_membar(self, *_):
        return None

    _op_fence = _op_membar

    def _op_call(self, frame, ins, lanes, here, *_):
        args = list(ins.args)
        rets: List[str] = []
        if args and args[0].startswith("("):
            rets = _split_top(args.pop(0)[1:-1])
        fname = args.pop(0)
        params = _split_top(args.pop(0)[1:-1]) if args and args[0].startswith("(") else []
        callee = self.module.funcs.get(fname)
        if callee is None:
            self._fail(frame, int(lanes[0]), ins, f"`{fname}` is not defined in this module")
        if any(s != "reg" for s, _, _ in callee.params + callee.rets):
            self._fail(frame, int(lanes[0]), ins, f"`{fname}` passes arguments through .param space, which is not modelled")
        sub = _Frame(callee, frame.lanes[lanes])
        for (_, t, pname), a in zip(callee.params, params):
            sub.regs[pname] = _to_bits(self._read(frame, a, t, lanes), t)
        self._run_frame(sub, top=False)
        for (_, t, rname), d in zip(callee.rets, rets):
            self._write(frame, d, t, lanes, _from_bits(sub.regs[rname], t))

    # Moves ------------------------------------------------------------
    def _op_mov(self, frame, ins, lanes, *_):
        t = ins.parts[-1]
        d, a = ins.args
        if d.startswith("{"):
            parts = _split_top(d[1:-1])
            _, w = _kind_width(t)
            src = _from_bits(self._read(frame, a, t, lanes).view(_U[w]).astype(np.uint64), "u64")
            pw = w // len(parts)
            for i, p in enumerate(parts):
                self._write_bits(frame, p, lanes, (src >> np.uint64(i * pw)) & np.uint64((1 << pw) - 1))
            return
        if a.startswith("{"):
            parts = _split_top(a[1:-1])
            _, w = _kind_width(t)
            pw = w // len(parts)
            acc = np.zeros(lanes.size, np.uint64)
            for i, p in enumerate(parts):
                acc |= (_from_bits(self._read(frame, p, f"b{pw}", lanes), f"b{pw}").astype(np.uint64)) << np.uint64(i * pw)
            self._write_bits(frame, d, lanes, acc)
            return
        self._write(frame, d, t, lanes, self._read(frame, a, t, lanes))

    def _op_cvta(self, frame, ins, lanes, *_):
        t = ins.parts[-1]
        if "shared" in ins.parts and "to" not in ins.parts:
            self._fail(frame, int(lanes[0]), ins, "generic addresses of shared memory are not modelled")
        d, a = ins.args
        self._write(frame, d, t, lanes, self._read(frame, a, t, lanes))

    def _op_createpolicy(self, frame, ins, lanes, *_):
        self._write(frame, ins.args[0], "b64", lanes, np.zeros(lanes.size, np.uint64))

    def _op_prefetch(self, *_):
        return None

    _op_prefetchu = _op_prefetch

    # Integer and float arithmetic -------------------------------------
    def _binary(self, frame, ins, lanes):
        t = ins.parts[-1]
        d, a, b = ins.args[:3]
        return t, d, self._read(frame, a, t, lanes), self._read(frame, b, t, lanes)

    def _float_pre(self, ins, *xs):
        if "ftz" in ins.parts:
            return tuple(_ftz(x) for x in xs)
        return xs

    def _check_rounding(self, frame, ins, lanes):
        for m in ("rz", "rm", "rp"):
            if m in ins.parts:
                self._fail(frame, int(lanes[0]), ins, f"rounding mode .{m} is not modelled")

    def _float_result(self, ins, dt, s, e=None, nan_from=()):
        """A float instruction's result from its value `s` in float64 (exact,
        or with rounding error `e`): rounded once to `dt`, flushed under .ftz
        when tiny, clamped under .sat, and NaN as the GPU writes it."""
        if dt == np.float64:
            r = s
        else:
            r = _round_once(s, e, dt)
            if "ftz" in ins.parts and dt == np.float32:
                r = _flush_tiny(r, s, e)
        if "sat" in ins.parts:
            r = _saturate(r)
        return _gpu_nan(r, nan_from)

    def _float_op(self, frame, ins, lanes, f, xs, nan_from):
        """f over float operands, in float64, rounded once to their type.
        `nan_from` orders the operands whose NaN a float64 result passes on."""
        self._check_rounding(frame, ins, lanes)
        xs = self._float_pre(ins, *xs)
        s = f(*(x.astype(np.float64) for x in xs))
        return self._float_result(ins, xs[0].dtype, s, None, [xs[i] for i in nan_from])

    def _fused(self, frame, ins, lanes, a, b, c):
        """a*b+c rounded once: fma, and mad on floats. A float64 result passes
        on b's NaN, then c's, then a's (measured)."""
        self._check_rounding(frame, ins, lanes)
        a, b, c = self._float_pre(ins, a, b, c)
        if a.dtype == np.float64:
            return self._float_result(ins, a.dtype, _fma64(a, b, c), None, (b, c, a))
        s, e = _two_sum(a.astype(np.float64) * b.astype(np.float64), c.astype(np.float64))
        return self._float_result(ins, a.dtype, s, e)

    def _op_add(self, frame, ins, lanes, *_):
        t, d, a, b = self._binary(frame, ins, lanes)
        k, w = _kind_width(t)
        if k == "f":
            r = self._float_op(frame, ins, lanes, np.add, (a, b), (1, 0))
        elif "cc" in ins.parts:
            # The carry out of the unsigned sum, for .s32 and .s64 too.
            ua, ub = a.view(_U[w]), b.view(_U[w])
            r = ua + ub
            frame.cc[lanes] = (r < ua).astype(np.uint64)
        elif "sat" in ins.parts:
            r = np.clip(a.astype(np.int64) + b.astype(np.int64), -2**31, 2**31 - 1)
        else:
            r = a + b
        self._write(frame, d, t, lanes, r)

    def _op_addc(self, frame, ins, lanes, *_):
        t, d, a, b = self._binary(frame, ins, lanes)
        _, w = _kind_width(t)
        a, b = a.view(_U[w]), b.view(_U[w])
        s1 = a + b
        s2 = s1 + frame.cc[lanes].astype(_U[w])
        if "cc" in ins.parts:
            frame.cc[lanes] = ((s1 < a) | (s2 < s1)).astype(np.uint64)
        self._write_bits(frame, d, lanes, s2.astype(np.uint64))

    def _op_sub(self, frame, ins, lanes, *_):
        t, d, a, b = self._binary(frame, ins, lanes)
        k, w = _kind_width(t)
        if k == "f":
            # A float64 result passes on b's NaN as it is, not negated.
            r = self._float_op(frame, ins, lanes, np.subtract, (a, b), (1, 0))
        elif "cc" in ins.parts:
            ua, ub = a.view(_U[w]), b.view(_U[w])
            r = ua - ub
            frame.cc[lanes] = (ua < ub).astype(np.uint64)
        elif "sat" in ins.parts:
            r = np.clip(a.astype(np.int64) - b.astype(np.int64), -2**31, 2**31 - 1)
        else:
            r = a - b
        self._write(frame, d, t, lanes, r)

    def _op_subc(self, frame, ins, lanes, *_):
        t, d, a, b = self._binary(frame, ins, lanes)
        _, w = _kind_width(t)
        a, b = a.view(_U[w]), b.view(_U[w])
        cin = frame.cc[lanes].astype(_U[w])
        r = a - b - cin
        if "cc" in ins.parts:
            frame.cc[lanes] = ((a < b) | ((a - b) < cin)).astype(np.uint64)
        self._write_bits(frame, d, lanes, r.astype(np.uint64))

    def _mul_parts(self, frame, ins, lanes, a, b, t):
        """(low half, high half) of a*b at the operand width."""
        k, w = _kind_width(t)
        if w <= 32:
            p = a.astype(np.int64) * b.astype(np.int64) if k == "s" else a.astype(np.uint64) * b.astype(np.uint64)
            pu = p.view(np.uint64) if k == "s" else p
            lo = pu & np.uint64((1 << w) - 1)
            hi = (p >> w).view(np.uint64) & np.uint64((1 << w) - 1) if k == "s" else (pu >> np.uint64(w))
            return lo, hi
        prod = [x * y for x, y in zip(a.tolist(), b.tolist())]
        lo = np.array([q & ((1 << 64) - 1) for q in prod], np.uint64)
        hi = np.array([(q >> 64) & ((1 << 64) - 1) for q in prod], np.uint64)
        return lo, hi

    def _op_mul(self, frame, ins, lanes, *_):
        t, d, a, b = self._binary(frame, ins, lanes)
        k, w = _kind_width(t)
        if k == "f":
            self._write(frame, d, t, lanes, self._float_op(frame, ins, lanes, np.multiply, (a, b), (1, 0)))
            return
        if "wide" in ins.parts:
            p = a.astype(np.int64) * b.astype(np.int64) if k == "s" else a.astype(np.uint64) * b.astype(np.uint64)
            self._write(frame, d, ("s" if k == "s" else "u") + str(2 * w), lanes, p)
            return
        lo, hi = self._mul_parts(frame, ins, lanes, a, b, t)
        self._write_bits(frame, d, lanes, hi if "hi" in ins.parts else lo)

    def _op_mad(self, frame, ins, lanes, *_):
        t = ins.parts[-1]
        d, a, b, c = ins.args
        k, w = _kind_width(t)
        a, b = self._read(frame, a, t, lanes), self._read(frame, b, t, lanes)
        if k == "f":
            self._write(frame, d, t, lanes, self._fused(frame, ins, lanes, a, b, self._read(frame, c, t, lanes)))
            return
        if "wide" in ins.parts:
            wt = ("s" if k == "s" else "u") + str(2 * w)
            p = a.astype(np.int64) * b.astype(np.int64) if k == "s" else a.astype(np.uint64) * b.astype(np.uint64)
            self._write(frame, d, wt, lanes, p + self._read(frame, c, wt, lanes))
            return
        lo, hi = self._mul_parts(frame, ins, lanes, a, b, t)
        part = (hi if "hi" in ins.parts else lo).astype(_U[w])
        cv = self._read(frame, c, "u" + str(w), lanes)
        s1 = part + cv
        if "cc" in ins.parts:
            frame.cc[lanes] = (s1 < part).astype(np.uint64)
        if "sat" in ins.parts:
            # mad.hi.sat.s32: the high half plus c, saturated.
            s1 = np.clip(_from_bits(part.astype(np.uint64), t).astype(np.int64)
                         + self._read(frame, c, t, lanes).astype(np.int64), -2**31, 2**31 - 1)
            self._write(frame, d, t, lanes, s1)
            return
        self._write(frame, d, "u" + str(w), lanes, s1)

    def _op_madc(self, frame, ins, lanes, *_):
        t = ins.parts[-1]
        d, a, b, c = ins.args
        _, w = _kind_width(t)
        a, b = self._read(frame, a, t, lanes), self._read(frame, b, t, lanes)
        lo, hi = self._mul_parts(frame, ins, lanes, a, b, t)
        part = (hi if "hi" in ins.parts else lo).astype(_U[w])
        cv = self._read(frame, c, "u" + str(w), lanes)
        cin = frame.cc[lanes].astype(_U[w])
        s1 = part + cv
        s2 = s1 + cin
        if "cc" in ins.parts:
            frame.cc[lanes] = ((s1 < part) | (s2 < s1)).astype(np.uint64)
        self._write(frame, d, "u" + str(w), lanes, s2)

    def _op_fma(self, frame, ins, lanes, *_):
        t = ins.parts[-1]
        d, a, b, c = ins.args
        a, b, c = (self._read(frame, x, t, lanes) for x in (a, b, c))
        self._write(frame, d, t, lanes, self._fused(frame, ins, lanes, a, b, c))

    def _int_div(self, frame, ins, lanes, a, b, rem):
        zero = b == 0
        if zero.any():
            self._fail(frame, int(lanes[np.argmax(zero)]), ins, "integer division by zero")
        if a.dtype.kind == "u":
            return a % b if rem else a // b
        q = np.floor_divide(a, b)
        fix = (np.remainder(a, b) != 0) & ((a < 0) != (b < 0))
        q = q + fix.astype(q.dtype)
        return a - q * b if rem else q

    def _op_div(self, frame, ins, lanes, *_):
        t, d, a, b = self._binary(frame, ins, lanes)
        if _kind_width(t)[0] != "f":
            self._write(frame, d, t, lanes, self._int_div(frame, ins, lanes, a, b, False))
            return
        f = _div_approx if ("approx" in ins.parts and t == "f32") else np.divide
        # A float64 quotient passes on a's NaN before b's (measured).
        self._write(frame, d, t, lanes, self._float_op(frame, ins, lanes, f, (a, b), (0, 1)))

    def _op_rem(self, frame, ins, lanes, *_):
        t, d, a, b = self._binary(frame, ins, lanes)
        self._write(frame, d, t, lanes, self._int_div(frame, ins, lanes, a, b, True))

    def _minmax(self, frame, ins, lanes, smaller):
        t, d, a, b = self._binary(frame, ins, lanes)
        if _kind_width(t)[0] != "f":
            self._write(frame, d, t, lanes, (np.minimum if smaller else np.maximum)(a, b))
            return
        a, b = self._float_pre(ins, a, b)
        r = np.where(a < b, a, b) if smaller else np.where(a > b, a, b)
        zeros = (a == 0) & (b == 0)
        if zeros.any():
            # -0 is below +0 for the GPU, in either operand order.
            neg = (np.signbit(a) | np.signbit(b)) if smaller else (np.signbit(a) & np.signbit(b))
            r = np.where(zeros, np.where(neg, -0.0, 0.0).astype(a.dtype), r)
        na, nb = np.isnan(a), np.isnan(b)
        if "NaN" in ins.parts:
            r = np.where(na | nb, np.nan, r).astype(a.dtype)
        else:
            r = np.where(na & ~nb, b, np.where(nb & ~na, a, r))
        self._write(frame, d, t, lanes, _gpu_nan(r.astype(a.dtype), (b, a)))

    def _op_min(self, frame, ins, lanes, *_):
        self._minmax(frame, ins, lanes, True)

    def _op_max(self, frame, ins, lanes, *_):
        self._minmax(frame, ins, lanes, False)

    def _unary(self, frame, ins, lanes):
        t = ins.parts[-1]
        d, a = ins.args
        return t, d, self._read(frame, a, t, lanes)

    def _sign_op(self, frame, ins, lanes, f):
        t, d, a = self._unary(frame, ins, lanes)
        if _kind_width(t)[0] == "f":
            # A float64 NaN keeps its sign; float32 and float16 write the canonical NaN.
            (a,) = self._float_pre(ins, a)
            r = _gpu_nan(np.where(np.isnan(a), a, f(a)), (a,))
        else:
            r = f(a)
        self._write(frame, d, t, lanes, r)

    def _op_neg(self, frame, ins, lanes, *_):
        self._sign_op(frame, ins, lanes, np.negative)

    def _op_abs(self, frame, ins, lanes, *_):
        self._sign_op(frame, ins, lanes, np.abs)

    def _op_not(self, frame, ins, lanes, *_):
        t, d, a = self._unary(frame, ins, lanes)
        self._write(frame, d, t, lanes, ~a)

    def _op_cnot(self, frame, ins, lanes, *_):
        t, d, a = self._unary(frame, ins, lanes)
        self._write(frame, d, t, lanes, (a == 0).astype(a.dtype))

    def _logic(self, frame, ins, lanes, f):
        t = ins.parts[-1]
        d, a, b = ins.args
        self._write(frame, d, t, lanes, f(self._read(frame, a, t, lanes), self._read(frame, b, t, lanes)))

    def _op_and(self, frame, ins, lanes, *_):
        self._logic(frame, ins, lanes, np.bitwise_and)

    def _op_or(self, frame, ins, lanes, *_):
        self._logic(frame, ins, lanes, np.bitwise_or)

    def _op_xor(self, frame, ins, lanes, *_):
        self._logic(frame, ins, lanes, np.bitwise_xor)

    def _op_shl(self, frame, ins, lanes, *_):
        t = ins.parts[-1]
        d, a, b = ins.args
        _, w = _kind_width(t)
        av = self._read(frame, a, "u" + str(w), lanes).astype(np.uint64)
        n = self._read(frame, b, "u32", lanes).astype(np.uint64)
        r = np.where(n >= w, np.uint64(0), av << np.minimum(n, np.uint64(w - 1)))
        self._write_bits(frame, d, lanes, r & np.uint64((1 << w) - 1) if w < 64 else r)

    def _op_shr(self, frame, ins, lanes, *_):
        t = ins.parts[-1]
        d, a, b = ins.args
        k, w = _kind_width(t)
        n = self._read(frame, b, "u32", lanes).astype(np.int64)
        if k == "s":
            av = self._read(frame, a, t, lanes).astype(np.int64)
            self._write(frame, d, t, lanes, av >> np.minimum(n, 63))
        else:
            av = self._read(frame, a, "u" + str(w), lanes).astype(np.uint64)
            self._write_bits(frame, d, lanes, np.where(n >= w, np.uint64(0), av >> np.minimum(n, w - 1).astype(np.uint64)))

    # Special functions -------------------------------------------------
    # Computed accurately and rounded to the result type, except sin and
    # cos, which reduce their argument as the GPU does (`_turns`). The GPU's
    # .approx results differ from these within its documented error bounds.
    def _special(self, frame, ins, lanes, f):
        t = ins.parts[-1]
        d, a = ins.args
        x = self._read(frame, a, t, lanes)
        self._write(frame, d, t, lanes, self._float_op(frame, ins, lanes, f, (x,), (0,)))

    def _op_ex2(self, frame, ins, lanes, *_):
        self._special(frame, ins, lanes, np.exp2)

    def _op_lg2(self, frame, ins, lanes, *_):
        self._special(frame, ins, lanes, np.log2)

    def _op_sin(self, frame, ins, lanes, *_):
        self._special(frame, ins, lanes, lambda v: np.sin(_turns(v)))

    def _op_cos(self, frame, ins, lanes, *_):
        self._special(frame, ins, lanes, lambda v: np.cos(_turns(v)))

    def _op_tanh(self, frame, ins, lanes, *_):
        self._special(frame, ins, lanes, np.tanh)

    def _op_rcp(self, frame, ins, lanes, *_):
        self._special(frame, ins, lanes, lambda v: 1.0 / v)

    def _op_sqrt(self, frame, ins, lanes, *_):
        self._special(frame, ins, lanes, np.sqrt)

    def _op_rsqrt(self, frame, ins, lanes, *_):
        self._special(frame, ins, lanes, lambda v: 1.0 / np.sqrt(v))

    # Comparison and selection -------------------------------------------
    @staticmethod
    def _compare(cmp, a, b, k):
        if k == "f":
            na, nb = np.isnan(a), np.isnan(b)
            either = na | nb
            ordered = {"eq": a == b, "ne": a != b, "lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b}
            if cmp in ordered:
                return ordered[cmp] & ~either
            if cmp.endswith("u") and cmp[:-1] in ordered:
                return ordered[cmp[:-1]] | either
            if cmp == "num":
                return ~either
            if cmp == "nan":
                return either
        else:
            m = {"eq": a == b, "ne": a != b, "lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b,
                 "lo": a < b, "ls": a <= b, "hi": a > b, "hs": a >= b}
            if cmp in m:
                return m[cmp]
        raise CPUInterpreterError(f"comparison .{cmp} is not modelled")

    def _op_setp(self, frame, ins, lanes, *_):
        t = ins.parts[-1]
        k, _ = _kind_width(t)
        cmp = ins.parts[1]
        if k in ("u", "s") and cmp in ("lo", "ls", "hi", "hs") and k == "s":
            raise CPUInterpreterError(f"`{ins.op}` mixes an unsigned comparison with a signed type")
        dest, a, b = ins.args[:3]
        av, bv = self._read(frame, a, t, lanes), self._read(frame, b, t, lanes)
        if "ftz" in ins.parts:
            av, bv = _ftz(av), _ftz(bv)
        r = self._compare(cmp, av, bv, k)
        if len(ins.args) == 4:
            c = self._read(frame, ins.args[3], "pred", lanes)
            boolop = ins.parts[2]
            r = {"and": r & c, "or": r | c, "xor": r ^ c}[boolop]
        p, _, q = dest.partition("|")
        self._write(frame, p, "pred", lanes, r)
        if q:
            self._write(frame, q, "pred", lanes, ~r)

    def _op_selp(self, frame, ins, lanes, *_):
        t = ins.parts[-1]
        d, a, b, c = ins.args
        self._write(frame, d, t, lanes, np.where(self._read(frame, c, "pred", lanes),
                                                 self._read(frame, a, t, lanes), self._read(frame, b, t, lanes)))

    # Conversion ---------------------------------------------------------
    def _op_cvt(self, frame, ins, lanes, *_):
        dt, at = ins.parts[-2], ins.parts[-1]
        mods = set(ins.parts[1:-2])
        if len(ins.args) != 2 or dt.endswith("x2") or at.endswith("x2"):
            self._fail(frame, int(lanes[0]), ins, f"`{ins.op}` is not modelled by the CPU interpreter")
        d, a = ins.args
        dk, dw = _kind_width(dt)
        ak, aw = _kind_width(at)
        x = self._read(frame, a, at, lanes)
        if "ftz" in mods and at == "f32":
            # A float32 source goes through the GPU's float32 path first, so
            # cvt.ftz.f64.f32 widens the canonical NaN, not the operand's.
            x = _gpu_nan(_ftz(x))
        if ak != "f" and dk != "f":
            v = x.astype(np.int64) if not (ak == "u" and aw == 64) else x
            if "sat" in mods:
                lo, hi = (-(1 << (dw - 1)), (1 << (dw - 1)) - 1) if dk == "s" else (0, (1 << dw) - 1)
                v = np.array([min(max(int(q), lo), hi) for q in v], dtype=object)
                out = np.array([q & ((1 << dw) - 1) for q in v], np.uint64)
                self._write_bits(frame, d, lanes, out if dk == "u" else _extend(out.astype(_U[dw]).view(_S[dw]), dt))
                return
            out = v.astype(_dtype(dt))
            self._write_bits(frame, d, lanes, _extend(out, dt))
            return
        irnd = [_ROUND_TO_INT[m] for m in ("rni", "rzi", "rmi", "rpi") if m in mods]
        if ak == "f" and dk != "f":
            # Rounded as the modifier says (PTX requires one), then clamped to
            # the destination's range. NaN becomes 0 from float16 or float32 to
            # an integer of up to 32 bits, and the sign-bit pattern (0x80...)
            # from float64 or to a 64-bit integer (measured).
            xd = (irnd[0] if irnd else np.trunc)(x.astype(np.float64))
            lo, hi = (-(1 << (dw - 1)), (1 << (dw - 1)) - 1) if dk == "s" else (0, (1 << dw) - 1)
            nan_value = (lo if dk == "s" else 1 << (dw - 1)) if 64 in (aw, dw) else 0
            vals = [nan_value if math.isnan(q) else (hi if q > 0 else lo) if math.isinf(q) else min(max(int(q), lo), hi)
                    for q in xd.tolist()]
            out = np.array([v & ((1 << 64) - 1) for v in vals], np.uint64)
            self._write_bits(frame, d, lanes, out & np.uint64((1 << dw) - 1) if dk == "u" and dw < 64 else out)
            return
        if mods & {"rz", "rm", "rp"}:
            self._fail(frame, int(lanes[0]), ins, "rounding modes other than .rn are not modelled for cvt")
        if ak != "f":
            r = x.astype(_F[dw])        # to nearest even, as .rn rounds
            if "sat" in mods:
                r = _saturate(r)
            self._write(frame, d, dt, lanes, r)
            return
        if irnd:
            # To an integral value in the same type: cvt.rni.f32.f32 and kin.
            if dt != at:
                self._fail(frame, int(lanes[0]), ins, "integer rounding between two float types is not modelled")
            self._write(frame, d, dt, lanes, self._float_result(ins, x.dtype, irnd[0](x.astype(np.float64)), None, (x,)))
            return
        self._write(frame, d, dt, lanes, self._float_convert(ins, x, np.dtype(_F[dw])))

    def _float_convert(self, ins, x, to):
        """Between float types, to nearest even. A NaN keeps its sign and the
        top of its payload, quieted, between float64 and float32 or float16;
        float16 <-> float32 write the canonical NaN (measured)."""
        if to == x.dtype:
            return self._float_result(ins, to, x.astype(np.float64), None, (x,))
        r = x.astype(to)
        if to == np.float32 and "ftz" in ins.parts:
            r = _flush_tiny(r, x.astype(np.float64), None)
        if "sat" in ins.parts:
            r = _saturate(r)
        nan = np.isnan(r)
        if not nan.any() or {x.dtype, to} == {np.dtype(np.float32), np.dtype(np.float64)}:
            return r                    # numpy converts these NaNs as the GPU does
        if to != np.float64 and x.dtype != np.float64:
            return _gpu_nan(r)          # float16 <-> float32
        bits = r.view(_U[to.itemsize * 8]).copy()
        if to == np.float64:            # from float16: numpy leaves it signalling
            bits[nan] |= _QUIET64
        else:                           # float64 -> float16
            xb = x.view(np.uint64)
            bits[nan] = (((xb >> np.uint64(48)) & np.uint64(0x8000)) | np.uint64(0x7E00)
                         | ((xb >> np.uint64(42)) & np.uint64(0x1FF)))[nan].astype(np.uint16)
        return bits.view(to)

    # Memory -------------------------------------------------------------
    @staticmethod
    def _space(parts) -> str:
        """The state space; generic addresses are global ones here, since
        `cvta` to a generic shared or local address is refused."""
        for s in ("global", "shared", "const", "param", "local"):
            if s in parts or (s == "shared" and "shared::cta" in parts):
                return s
        return "global"

    @staticmethod
    def _vec(parts) -> int:
        for p in parts:
            if p in ("v2", "v4", "v8"):
                return int(p[1:])
        return 1

    def _address(self, frame, opnd, lanes):
        m = re.fullmatch(r"\[\s*([^\]+\-\s]+)\s*(?:([+-])\s*([0-9a-fA-Fx]+))?\s*\]", opnd)
        if not m:
            raise CPUInterpreterError(f"address `{opnd}` is not modelled")
        base, sign, off = m.groups()
        if base.startswith("%"):
            # At the register's own width: a 32-bit register can hold
            # sign-extended bits above its 32 (cvt and ld write them).
            rt = frame.func.regs.get(base)
            b = self._read(frame, base, "u64" if rt is None else "u%d" % _kind_width(rt)[1], lanes).astype(np.uint64)
        else:
            sym = self._symbol(frame, base)
            if sym is None:
                return base, int(off, 0) * (-1 if sign == "-" else 1) if off else 0
            b = np.full(lanes.size, sym, np.uint64)
        o = int(off, 0) * (-1 if sign == "-" else 1) if off else 0
        return None, b + np.uint64(o & ((1 << 64) - 1))

    def _locate_global(self, frame, ins, lanes, addr, width, write, align=None):
        if not self.allocs:
            self._fail(frame, int(lanes[0]), ins, self._describe(int(addr[0]), width, write))
        idx = np.searchsorted(self.bases, addr, side="right") - 1
        safe = np.clip(idx, 0, None)
        off = addr - self.bases[safe]
        ok = (idx >= 0) & (off + np.uint64(width) <= self.sizes[safe]) & (addr >= self.bases[safe])
        if not ok.all():
            bad = int(np.argmin(ok))
            self._fail(frame, int(lanes[bad]), ins, self._describe(int(addr[bad]), width, write))
        width = align or width
        mis = (addr % np.uint64(width)) != 0
        if mis.any():
            bad = int(np.argmax(mis))
            a = self.allocs[int(safe[bad])]
            self._fail(frame, int(lanes[bad]), ins,
                       f"misaligned {width}-byte {'store to' if write else 'load from'} `{a.name}` at byte "
                       f"{int(off[bad])}: the GPU faults on an access not aligned to its size")
        self.bounds_checks_passed += int(lanes.size)
        return safe, off

    def _describe(self, addr, width, write):
        verb = "store to" if write else "load from"
        best = None
        for a in self.allocs:
            if a.base <= addr < a.base + _ALLOC_STRIDE:
                best = a
        if best is None or addr < best.base:
            return f"{width}-byte {verb} address {addr:#x}, which is in no buffer the kernel was given"
        past = addr - best.base
        unit = ""
        return (f"out-of-bounds {width}-byte {verb} `{best.name}`: byte {past} of a {best.size}-byte buffer"
                f" ({past - best.size + width} byte(s) past its end){unit}")

    def _locate_shared(self, frame, ins, lanes, addr, width, write):
        a = addr.astype(np.int64)
        v = np.searchsorted(self.shared_offs, a, side="right") - 1
        safe = np.clip(v, 0, None)
        ok = (v >= 0) & (a + width <= self.shared_ends[safe]) if len(self.shared_vars) else np.zeros(a.size, bool)
        if not ok.all():
            bad = int(np.argmin(ok))
            if len(self.shared_vars) == 0:
                what = "this kernel declares no shared memory"
            else:
                o, size, name = self.shared_vars[int(safe[bad])]
                what = f"`{name}` is {size} bytes and this touches bytes {int(a[bad]) - o}..{int(a[bad]) - o + width - 1}"
            self._fail(frame, int(lanes[bad]), ins,
                       f"out-of-bounds {width}-byte shared {'store' if write else 'load'}: {what}")
        if ((a % width) != 0).any():
            bad = int(np.argmax((a % width) != 0))
            self._fail(frame, int(lanes[bad]), ins, f"misaligned {width}-byte shared access at byte {int(a[bad])}")
        self.bounds_checks_passed += int(lanes.size)
        return a

    def _load_bytes(self, frame, ins, lanes, space, opnd, width, align=None):
        """Bytes of a `width`-byte access per lane, as (lanes, width) uint8;
        the address must be a multiple of `align` (default `width`)."""
        if space == "param":
            m = re.fullmatch(r"\[\s*([\w$]+)\s*(?:\+\s*(\d+))?\s*\]", opnd)
            if not m or m.group(1) not in self.params:
                self._fail(frame, int(lanes[0]), ins, f"parameter `{opnd}` is not a kernel parameter")
                return None
            raw = np.frombuffer(self.params[m.group(1)][0], np.uint8)
            o = int(m.group(2) or 0)
            return np.tile(raw[o:o + width], (lanes.size, 1))
        _, addr = self._address(frame, opnd, lanes)
        if space == "shared":
            a = self._locate_shared(frame, ins, lanes, addr, width, False)
            return self.shared[self.blk[frame.lanes[lanes]][:, None], a[:, None] + np.arange(width)]
        if space == "local":
            return self._local(frame)[lanes[:, None], self._local_off(frame, ins, lanes, addr, width)]
        safe, off = self._locate_global(frame, ins, lanes, addr, width, False, align)
        out = np.empty((lanes.size, width), np.uint8)
        for u in np.unique(safe):
            sel = safe == u
            out[sel] = self.allocs[int(u)].data[off[sel].astype(np.int64)[:, None] + np.arange(width)]
        return out

    def _store_bytes(self, frame, ins, lanes, space, opnd, data):
        width = data.shape[1]
        _, addr = self._address(frame, opnd, lanes)
        if space == "shared":
            a = self._locate_shared(frame, ins, lanes, addr, width, True)
            rows = self.blk[frame.lanes[lanes]]
            self._race(frame, ins, lanes, rows.astype(np.int64) * (1 << 40) + a, data)
            self.shared[rows[:, None], a[:, None] + np.arange(width)] = data
            return
        if space == "local":
            self._local(frame)[lanes[:, None], self._local_off(frame, ins, lanes, addr, width)] = data
            return
        if space in ("const", "param"):
            self._fail(frame, int(lanes[0]), ins, f"a store to .{space} space")
        safe, off = self._locate_global(frame, ins, lanes, addr, width, True)
        for u in np.unique(safe):
            alloc = self.allocs[int(u)]
            if alloc.space == "const":
                self._fail(frame, int(lanes[0]), ins, f"a store to the constant `{alloc.name}`")
            sel = safe == u
            o = off[sel].astype(np.int64)
            self._race(frame, ins, lanes[sel], o, data[sel])
            alloc.data[o[:, None] + np.arange(width)] = data[sel]

    def _race(self, frame, ins, lanes, addr, data):
        if addr.size < 2:
            return
        order = np.argsort(addr, kind="stable")
        sa, sd = addr[order], data[order]
        same = sa[1:] == sa[:-1]
        if same.any():
            differ = same & (sd[1:] != sd[:-1]).any(axis=1)
            if differ.any():
                i = int(np.argmax(differ))
                other = frame.lanes[int(lanes[order[i]])]
                tid = tuple(int(self.special[f"%tid.{x}"][other]) for x in "xyz")
                self._fail(frame, int(lanes[order[i + 1]]), ins,
                           f"two threads store different values to one address in one instruction "
                           f"(this one and thread {tid} of block {self.ctaids[self.blk[other]]}): "
                           "which value survives on the GPU is unspecified")

    def _local(self, frame):
        if frame.local is None:
            size = sum(v.size for v in frame.func.local)
            frame.local = np.zeros((frame.lanes.size, max(size, 1)), np.uint8)
        return frame.local

    def _local_off(self, frame, ins, lanes, addr, width):
        size = sum(v.size for v in frame.func.local)
        a = addr.astype(np.int64)
        if ((a < 0) | (a + width > size)).any():
            bad = int(np.argmax((a < 0) | (a + width > size)))
            self._fail(frame, int(lanes[bad]), ins, f"out-of-bounds local access at byte {int(a[bad])} of {size}")
        return a[:, None] + np.arange(width)

    def _op_ld(self, frame, ins, lanes, *_):
        t = ins.parts[-1]
        space = self._space(ins.parts)
        n = self._vec(ins.parts)
        _, w = _kind_width(t)
        dests = _split_top(ins.args[0][1:-1]) if ins.args[0].startswith("{") else [ins.args[0]]
        raw = self._load_bytes(frame, ins, lanes, space, ins.args[1], n * w // 8)
        for i, dreg in enumerate(dests):
            chunk = np.ascontiguousarray(raw[:, i * w // 8:(i + 1) * w // 8]).view(_U[w]).reshape(-1)
            k, _ = _kind_width(t)
            if k in ("u", "s") and w < 64:
                # Into a wider register: sign-extended for .s types, as
                # `ld.global.s32 %rd` extends to 64 bits.
                self._write_bits(frame, dreg, lanes, _extend(_from_bits(chunk.astype(np.uint64), t), t))
            else:
                self._write_bits(frame, dreg, lanes, chunk.astype(np.uint64))

    _op_ldu = _op_ld

    def _op_st(self, frame, ins, lanes, *_):
        t = ins.parts[-1]
        space = self._space(ins.parts)
        _, w = _kind_width(t)
        srcs = _split_top(ins.args[1][1:-1]) if ins.args[1].startswith("{") else [ins.args[1]]
        cols = [_to_bits(self._read(frame, s, t, lanes), t).astype(_U[w]).view(np.uint8).reshape(lanes.size, w // 8)
                for s in srcs]
        self._store_bytes(frame, ins, lanes, space, ins.args[0], np.concatenate(cols, axis=1))

    def _atomic(self, frame, ins, lanes, dest, addr_opnd, srcs):
        parts = ins.parts
        space = self._space(parts)
        t = parts[-1]
        _, w = _kind_width(t)
        opname = next((p for p in parts[1:-1] if p in ("add", "min", "max", "inc", "dec", "and", "or", "xor", "exch", "cas")), None)
        if opname is None or space not in ("global", "shared"):
            self._fail(frame, int(lanes[0]), ins, f"`{ins.op}` is not modelled by the CPU interpreter")
        olds = np.zeros(lanes.size, np.uint64)
        vals = [self._read(frame, s, t, lanes) for s in srcs]
        for i, lane in enumerate(lanes):
            one = np.array([lane])
            raw = self._load_bytes(frame, ins, one, space, addr_opnd, w // 8)
            old_bits = np.ascontiguousarray(raw).view(_U[w]).reshape(-1).astype(np.uint64)
            old = _from_bits(old_bits, t)[0]
            v = vals[0][i]
            if opname == "add":
                new = old + v
            elif opname == "min":
                new = min(old, v)
            elif opname == "max":
                new = max(old, v)
            elif opname == "and":
                new = old & v
            elif opname == "or":
                new = old | v
            elif opname == "xor":
                new = old ^ v
            elif opname == "exch":
                new = v
            elif opname == "cas":
                new = vals[1][i] if old == v else old
            elif opname == "inc":
                new = 0 if old >= v else old + 1
            else:
                new = v if (old == 0 or old > v) else old - 1
            nb = _to_bits(np.array([new], _dtype(t)), t).astype(_U[w]).view(np.uint8).reshape(1, w // 8)
            self._store_bytes_unchecked(frame, ins, one, space, addr_opnd, nb)
            olds[i] = old_bits[0]
        if dest is not None:
            self._write_bits(frame, dest, lanes, olds)

    def _store_bytes_unchecked(self, frame, ins, lanes, space, opnd, data):
        width = data.shape[1]
        _, addr = self._address(frame, opnd, lanes)
        if space == "shared":
            a = self._locate_shared(frame, ins, lanes, addr, width, True)
            self.shared[self.blk[frame.lanes[lanes]][:, None], a[:, None] + np.arange(width)] = data
        else:
            safe, off = self._locate_global(frame, ins, lanes, addr, width, True)
            self.allocs[int(safe[0])].data[off.astype(np.int64)[:, None] + np.arange(width)] = data

    def _op_atom(self, frame, ins, lanes, *_):
        self._atomic(frame, ins, lanes, ins.args[0], ins.args[1], ins.args[2:])

    def _op_red(self, frame, ins, lanes, *_):
        self._atomic(frame, ins, lanes, None, ins.args[0], ins.args[1:])

    def _op_cp(self, frame, ins, lanes, *_):
        """cp.async.{ca,cg}.shared.global [dst], [src], cp-size{, src-size |
        ignore-src}{, cache-policy}: copied at once. Only src-size bytes are
        read - per thread, from a register or an immediate - and the rest of
        cp-size is zero-filled; a true ignore-src predicate reads nothing.
        Y's GEMM zero-fills tiles past the matrix edge this way."""
        if ins.parts[1] != "async" or "bulk" in ins.parts:
            self._fail(frame, int(lanes[0]), ins, f"`{ins.op}` is not modelled by the CPU interpreter")
        if "commit_group" in ins.parts or "wait_group" in ins.parts or "wait_all" in ins.parts:
            return
        dst, src, size = ins.args[:3]
        n = int(size, 0)
        extra = ins.args[3:]
        if extra and any(p.startswith("L2::cache_hint") for p in ins.parts):
            extra = extra[:-1]
        if len(extra) > 1:
            self._fail(frame, int(lanes[0]), ins, f"`{ins.op}` with operands {extra} is not modelled")
        src_size = np.full(lanes.size, n, np.int64)
        if extra:
            if frame.func.regs.get(extra[0].lstrip("!")) == "pred":
                src_size = np.where(self._read(frame, extra[0], "pred", lanes), 0, n)
            else:
                src_size = self._read(frame, extra[0], "u32", lanes).astype(np.int64)
        if (src_size > n).any():
            bad = int(np.argmax(src_size > n))
            self._fail(frame, int(lanes[bad]), ins, f"src-size {int(src_size[bad])} exceeds the {n}-byte copy")
        data = np.zeros((lanes.size, n), np.uint8)
        for s in np.unique(src_size):
            if s:
                sel = src_size == s
                data[sel, :s] = self._load_bytes(frame, ins, lanes[sel], "global", src, int(s), align=n)
        self._store_bytes(frame, ins, lanes, "shared", dst, data)

    def _op_shfl(self, frame, ins, lanes, here, alive, live, group):
        if "sync" not in ins.parts:
            self._fail(frame, int(lanes[0]), ins, "`shfl` without `.sync` is not modelled")
        mode = next(p for p in ins.parts if p in ("bfly", "idx", "up", "down"))
        dest, a, b, c, mask_op = ins.args
        dreg, _, preg = dest.partition("|")
        val = self._read(frame, a, "b32", lanes).astype(np.uint64)
        bv = self._read(frame, b, "u32", lanes).astype(np.int64) & 31
        cv = self._read(frame, c, "u32", lanes).astype(np.int64)
        membermask = self._read(frame, mask_op, "u32", lanes).astype(np.int64)
        glanes = frame.lanes[lanes]
        lane = self.thread_in_block[glanes] % 32
        clamp, segmask = cv & 31, (cv >> 8) & 31
        max_lane = (lane & segmask) | (clamp & ~segmask)
        min_lane = lane & segmask
        if mode == "up":
            j = lane - bv
            pval = j >= max_lane
        elif mode == "down":
            j = lane + bv
            pval = j <= max_lane
        elif mode == "bfly":
            j = lane ^ bv
            pval = j <= max_lane
        else:
            j = min_lane | (bv & ~segmask)
            pval = j <= max_lane
        j = np.where(pval, j, lane)
        # Every lane the mask names must be here, at this instruction.
        key = self.warp_key[glanes]
        present = {}
        for i, (k, ln) in enumerate(zip(key, lane)):
            present[(int(k), int(ln))] = i
        for i, (k, m) in enumerate(zip(key, membermask)):
            for bit in range(32):
                if (m >> bit) & 1 and (int(k), bit) not in present:
                    tid_in_warp = bit
                    self._fail(frame, int(lanes[i]), ins,
                               f"`shfl.sync` with lane {tid_in_warp} of this warp named in the mask but not at this "
                               "instruction (diverged or exited): undefined on the GPU")
        src = np.array([present.get((int(k), int(jj)), -1) for k, jj in zip(key, j)])
        if (src < 0).any():
            bad = int(np.argmax(src < 0))
            self._fail(frame, int(lanes[bad]), ins, "`shfl.sync` reads a lane that is not taking part")
        self._write_bits(frame, dreg, lanes, val[src])
        if preg:
            self._write(frame, preg, "pred", lanes, pval)

    def _op_vote(self, frame, ins, lanes, *_):
        self._fail(frame, int(lanes[0]), ins, f"`{ins.op}` is not modelled by the CPU interpreter")


def _pretty(param_name: str) -> str:
    """`X_0` (Y's parameter naming) -> `X`."""
    m = re.fullmatch(r"(.+)_(\d+)", param_name)
    return m.group(1) if m else param_name


def debug_interpret(kernel, grid: Tuple[int, int, int], block: Tuple[int, int, int], args: List[Any],
                    fn: Callable = None, kernel_name: Optional[str] = None, target_sm: str = "auto") -> Dict[str, Any]:
    """Run one launch of a Y kernel on the CPU.

    `kernel` is a `TorchKernel`, or Y source text (compiled here; name the
    kernel with `kernel_name` when the source has more than one). Outputs are
    written into the tensors in `args`, as a GPU launch would. With `fn`, the
    old behaviour: call that Python function with this interpreter instead.
    """
    if fn is not None:
        name = kernel if isinstance(kernel, str) and "{" not in kernel else (kernel_name or "y_debug_kernel")
        return CPUInterpreter(kernel_name=name).run(grid, block, args, kernel_func=fn)
    if hasattr(kernel, "ptx") and hasattr(kernel, "kernel_name"):
        return CPUInterpreter(kernel.kernel_name, ptx=kernel.ptx).run(grid, block, args)
    if isinstance(kernel, str) and re.search(r"\bkernel\s+[\w$]+\s*\(", kernel):
        from .compiler import compile_to_ptx
        name = kernel_name or re.search(r"\bkernel\s+([\w$]+)\s*\(", kernel).group(1)
        return CPUInterpreter(name, ptx=compile_to_ptx(kernel, target_sm=target_sm)).run(grid, block, args)
    raise CPUInterpreterError(
        f"debug_interpret needs a kernel to run - a TorchKernel or Y source - not just the name {kernel!r}. "
        "It used to return SUCCESS for a name alone without running anything.")
