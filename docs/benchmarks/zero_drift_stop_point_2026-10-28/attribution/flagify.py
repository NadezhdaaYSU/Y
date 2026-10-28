"""Append to a PTX file a copy of kernel K whose in-loop traps only set a flag, trapped on once after the loop."""
import re, sys
src, k, out = open(sys.argv[1]).read(), sys.argv[2], sys.argv[3]
body = re.search(r"\.visible \.entry %s\(.*?\n\}\n" % k, src, re.S).group(0)
start, end = re.search(r"\n(\s*)\$LOOP_START_\d+:\n", body), re.search(r"\n\s*\$LOOP_END_\d+:\n", body)
loop = body[start.end():end.start()]
n = len(re.findall(r"@%p\d+ trap;", loop))
assert n >= 1, "no trap in the loop"
loop = re.sub(r"@(%p\d+) trap;", r"or.pred %bad, %bad, \1;", loop)
new = (body[:start.start()] + "\n    setp.ne.u32 %bad, 0, 0;" + body[start.start():start.end()] + loop
       + body[end.start():end.end()] + "    @%bad trap;\n" + body[end.end():])
new = re.sub(r"(\.reg \.pred %p<\d+>;)", r"\1\n    .reg .pred %bad;", new, count=1).replace(f".entry {k}(", f".entry {k}_flag(")
open(out, "w").write(src + "\n" + new)
print(f"{k}: {n} in-loop trap(s) deferred")
