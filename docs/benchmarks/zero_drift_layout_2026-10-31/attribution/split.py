"""A copy of the compiled `lj_closed` with ACC split (attribution only): total i's low word at
word i and its high word at word n + i, instead of words 2i and 2i + 1. The header stays at 2n.
Usage: split.py <lj.ptx> <out.ptx>"""
import re, sys
src = open(sys.argv[1]).read()
k = re.search(r"\.visible \.entry lj_closed\(.*?\n\}\n", src, re.S).group(0)
pat = re.compile(r"    setp\.ge\.s64 %p\d+, (%rd\d+), (%rd\d+);\n((?:    .*\n){2})    shl\.b64 (%rd\d+), \1, 4;\n    add\.u64 (%rd\d+), (%rd\d+), \4;\n"
                 r"((?:    .*\n)*?)    @(%p\d+) red\.global\.add\.u64 \[\5\+8\], (%rd\d+);\n")
n = [0]
def fix(m):
    i = n[0]; n[0] += 1
    idx, nreg, mid, off, addr, base, tail, p, v = m.groups()
    return (f"    setp.ge.s64 %pS{i}, {idx}, {nreg};\n".replace(f"%pS{i}", m.group(0).split()[1].rstrip(",")) + mid
            + f"    shl.b64 {off}, {idx}, 3;\n    add.u64 {addr}, {base}, {off};\n" + tail
            + f"    shl.b64 %hn{i}, {nreg}, 3;\n    add.u64 %ha{i}, {addr}, %hn{i};\n    @{p} red.global.add.u64 [%ha{i}], {v};\n")
k2, cnt = pat.subn(fix, k)
assert cnt == 6, cnt
k2 = k2.replace(".entry lj_closed(", ".entry lj_closed_split(")
d = k2.index("\n", k2.index(".reg .pred %p<"))
k2 = k2[:d] + "\n    .reg .b64 " + ", ".join(f"%hn{i}, %ha{i}" for i in range(6)) + ";" + k2[d:]
open(sys.argv[2], "w").write(src + "\n" + k2)
