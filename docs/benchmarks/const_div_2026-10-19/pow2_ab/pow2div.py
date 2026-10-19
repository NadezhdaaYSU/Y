#!/usr/bin/env python3
"""pow2div.py <in.ptx> <out.ptx>: every `div.u32 d, a, 2^k` written as
`shr.u32 d, a, k` and every `rem.u32 d, a, 2^k` as `and.b32 d, a, 2^k-1`
(exact for every unsigned a). ptxas expands both, by an immediate, into a
reciprocal division of ~15 instructions, and does so after constant
propagation, so even a constant dividend is divided at run time."""
import re, sys
pat = re.compile(r"^(\s*)(@!?%p\d+\s+)?(div|rem)\.u32\s+(%r\d+),\s*(%r\d+),\s*(\d+);(.*)$")
out, n = [], 0
for line in open(sys.argv[1]).read().split("\n"):
    m = pat.match(line)
    if m:
        ind, pred, op, d, a, c, rest = m.groups()
        c = int(c)
        if c > 0 and c & (c - 1) == 0:
            k = c.bit_length() - 1
            new = f"shr.u32 {d}, {a}, {k};" if op == "div" else f"and.b32 {d}, {a}, {c - 1};"
            line = f"{ind}{pred or ''}{new}{rest}"
            n += 1
    out.append(line)
open(sys.argv[2], "w").write("\n".join(out))
print(f"rewrote {n}")
