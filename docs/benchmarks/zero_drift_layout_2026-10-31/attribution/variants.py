"""Hand-edited copies of `lj_closed` (attribution only, not compiler output), appended to lj.ptx.

- flags:  every trap in the hand-overs (term and index tests) sets %bad instead; an atomic (and its
          high-word add) runs only when its index is in range; one trap on %bad before the close.
- defer:  checks kept; each hand-over saves its partial, and the carry into the high word is
          worked out after the sixth atomic, so the six round trips overlap.
- both:   flags and defer.
- red:    flags, and every atomic without a return and no carry at all (NOT exact when a carry
          or a negative partial occurs: a timing bound only).
Usage: variants.py <lj.ptx> <out.ptx>
"""
import re, sys

src = open(sys.argv[1]).read()
k = re.search(r"\.visible \.entry lj_closed\(.*?\n\}\n", src, re.S).group(0)
CLOSE = "    // [Y ZERO DRIFT] the grid's exact totals are complete once every block has handed its partials over\n"
assert k.count(CLOSE) == 1
body, close = k.split(CLOSE)
HAND = "    // [Y ZERO DRIFT] a: this thread's partial joins one of the grid's exact totals\n"
ATOM = re.compile(r"    atom\.global\.add\.u64 (%rd\d+), \[(%rd\d+)\], (%rd\d+);\n")
TAIL = re.compile(r"    add\.u64 %rd\d+, %rd\d+, %rd\d+;\n    setp\.lt\.u64 .*\n    selp\.u64 .*\n    add\.u64 %rd\d+, (%rd\d+), %rd\d+;\n"
                  r"    setp\.ne\.u64 (%p\d+), %rd\d+, 0;\n    @%p\d+ red\.global\.add\.u64 \[%rd\d+\+8\], %rd\d+;\n")


def make(name, flags, defer, red):
    b = body
    extra_regs = []
    if flags:
        # Index tests: `or.pred %pI, %pA, %pB;` then `@%pI trap;` right after the hand-over comment.
        idx = re.findall(re.escape(HAND) + r"(?:    .*\n){2}    or\.pred (%p\d+), %p\d+, %p\d+;\n    @%p\d+ trap;\n", b)
        assert len(idx) == 6, len(idx)
        parts = b.split(HAND)
        out = [parts[0]]
        for n, (part, ip) in enumerate(zip(parts[1:], idx)):
            ok = f"%ok{n}"
            extra_regs.append(ok)
            part = part.replace(f"    @{ip} trap;\n", f"    or.pred %bad, %bad, {ip};\n    not.pred {ok}, {ip};\n", 1)
            part = ATOM.sub(lambda m: f"    @{ok} atom.global.add.u64 {m.group(1)}, [{m.group(2)}], {m.group(3)};\n", part, count=1)
            part = re.sub(r"    @(%p\d+) red\.global", lambda m: f"    and.pred {m.group(1)}, {m.group(1)}, {ok};\n    @{m.group(1)} red.global", part, count=1)
            out.append(part)
        b = HAND.join(out)
        b = re.sub(r"    @(%p\d+) trap;\n", r"    or.pred %bad, %bad, \1;\n", b)
    if red:
        assert flags
        b, n_atom = re.subn(r"    (@%ok\d+ )atom\.global\.add\.u64 %rd\d+, (\[%rd\d+\]), (%rd\d+);\n", r"    \1red.global.add.u64 \2, \3;\n", b)
        b, n_tail = re.subn(r"    add\.u64 %rd\d+, %rd\d+, %rd28;\n    setp\.lt\.u64 .*\n    selp\.u64 .*\n    add\.u64 .*\n"
                            r"    setp\.ne\.u64 .*\n    and\.pred .*\n    @%p\d+ red\.global\.add\.u64 \[%rd\d+\+8\], %rd\d+;\n", "", b)
        assert n_atom == 6 and n_tail == 6, (n_atom, n_tail)
    if defer:
        tails = []
        n = [0]

        def save(m):
            i = n[0]; n[0] += 1
            lo, hi = f"%sl{i}", f"%sh{i}"
            extra_regs.append((lo, hi))
            atom, tail = m.group(1), m.group(2)
            tail = tail.replace("%rd28", lo).replace("%rd29", hi)
            tails.append(tail)
            return atom + f"    mov.u64 {lo}, %rd28;\n    mov.u64 {hi}, %rd29;\n"
        b, cnt = re.subn(r"(    (?:@%ok\d+ )?atom\.global\.add\.u64 .*\n)((?:    (?:add|setp|selp|and)\.\S+ .*\n)*    @%p\d+ red\.global\.add\.u64 \[%rd\d+\+8\], %rd\d+;\n)", save, b)
        assert cnt == 6, cnt
        # the carries go just before the close, inside the `if p < NP` region
        end_if = b.rindex("    $IF_END_1:")
        b = b[:end_if] + "".join(tails) + b[end_if:]
    regs = []
    if flags:
        regs.append("    .reg .pred %bad;")
        regs += [f"    .reg .pred {r};" for r in extra_regs if isinstance(r, str)]
    if defer:
        regs += [f"    .reg .b64 {lo}, {hi};" for r in extra_regs if isinstance(r, tuple) for lo, hi in [r]]
    out = b + (("    @%bad trap;\n") if flags else "") + CLOSE + close
    out = out.replace(".entry lj_closed(", f".entry lj_closed_{name}(")
    decl_end = out.index("\n", out.index(".reg .pred %p<"))
    init = "\n    setp.ne.u32 %bad, 0, 0;" if flags else ""
    first = out.index("\n    ld.param")
    out = out[:first] + init + out[first:]
    out = out[:decl_end] + "\n" + "\n".join(regs) + out[decl_end:]
    return out


variants = [make("flags", True, False, False), make("defer", False, True, False), make("both", True, True, False),
            make("red", True, False, True)]
open(sys.argv[2], "w").write(src + "\n" + "\n".join(variants))
for v in variants:
    name = re.search(r"\.entry (\w+)\(", v).group(1)
    print(name, "traps", v.count("trap;"), "atom", v.count("atom.global"), "red", v.count("red.global"), "or.pred %bad", v.count("or.pred %bad"))
