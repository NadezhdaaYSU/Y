#!/usr/bin/env python3
"""cdiv_pass.py <in.ptx> <out.ptx>: an independent Python port of
ptx_emitter::strength_reduce_constant_division, to check the compiler's
output against (new emission must equal this pass over the old emission)."""
import sys


def magic(c):
    if c < 2:
        return None
    for l in range(32):
        p = 1 << (32 + l)
        m = -(-p // c)
        if m >= 1 << 32:
            return None
        if m * c - p <= 1 << l:
            return m, l
    return None


def is_reg(s):
    return s.startswith("%") and len(s) > 1 and all(ch.isalnum() or ch == "_" for ch in s[1:])


def line_pass(line):
    body = line.lstrip()
    indent = line[:len(line) - len(body)]
    guard = ""
    if body.startswith("@"):
        sp = next((i for i, ch in enumerate(body) if ch.isspace()), None)
        if sp is None:
            return None
        guard, body = body[:sp] + " ", body[sp:].lstrip()
    parts = body.split(None, 1)
    if len(parts) != 2:
        return None
    op, operands = parts
    kinds = {"div.u32": (True, False), "rem.u32": (False, False), "div.u64": (True, True), "rem.u64": (False, True)}
    if op not in kinds:
        return None
    is_div, wide = kinds[op]
    operands = operands.strip()
    if not operands.endswith(";"):
        return None
    ops = [o.strip() for o in operands[:-1].split(",")]
    if len(ops) != 3:
        return None
    d, n, c = ops
    if not is_reg(d) or not is_reg(n) or not c.isdigit():
        return None
    c = int(c)
    if c == 0 or (not wide and c > 0xFFFFFFFF):
        return None
    bits, ty = ("b64", "u64") if wide else ("b32", "u32")
    if c == 1:
        seq = [f"mov.{bits} {d}, {n};" if is_div else f"mov.{bits} {d}, 0;"]
    elif c & (c - 1) == 0:
        seq = [f"shr.{ty} {d}, {n}, {c.bit_length() - 1};" if is_div else f"and.{bits} {d}, {n}, {c - 1};"]
    else:
        if wide:
            return None
        mg = magic(c)
        if mg is None:
            return None
        m, l = mg
        seq = [f"mul.hi.u32 {d}, {n}, {m};"]
        if l > 0:
            seq.append(f"shr.u32 {d}, {d}, {l};")
        if not is_div:
            if d == n:
                return None
            seq += [f"mul.lo.u32 {d}, {d}, {c};", f"sub.u32 {d}, {n}, {d};"]
    return [f"{indent}{guard}{s}" for s in seq]


def run(text):
    out = []
    for line in text.split("\n"):
        r = line_pass(line)
        out.extend(r if r is not None else [line])
    return "\n".join(out)


if __name__ == "__main__":
    src = open(sys.argv[1]).read()
    new = run(src)
    open(sys.argv[2], "w").write(new)
    print(sum(1 for a in src.split("\n") if line_pass(a) is not None))
