"""drop_final.py IN.ptx OUT.ptx: the Kahan kernel without its final correction: each
`sub.rn.f32 r, r, c` after the last mma whose r is an accumulator (the s of Kahan's
`add.rn.f32 s', s, y` / `sub.rn.f32 t, s', s`) removed."""
import re, sys
text = open(sys.argv[1]).read()
accs = set(re.findall(r"add\.rn\.f32 (%f\d+), (%f\d+), %f\d+;\n\s*sub\.rn\.f32 %f\d+, \1, \2;", text))
accs = {s for _, s in accs}
lines = text.split("\n")
last = max(i for i, l in enumerate(lines) if l.strip().startswith("mma.sync"))
pat = re.compile(r"\s*sub\.rn\.f32 (%f\d+), \1, %f\d+;")
drop = [i for i, l in enumerate(lines) if i > last and (m := pat.fullmatch(l)) and m.group(1) in accs]
print(f"{sys.argv[1]}: {len(accs)} accumulators, removed {len(drop)} lines")
open(sys.argv[2], "w").write("\n".join(l for i, l in enumerate(lines) if i not in set(drop)))
