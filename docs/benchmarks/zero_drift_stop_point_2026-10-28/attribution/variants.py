"""Hand-edited variants of rowsum_zd (attribution only, not compiler output)."""
import re, sys
src = open(sys.argv[1]).read()
zd = re.search(r"\.visible \.entry rowsum_zd\(.*?\n\}\n", src, re.S).group(0)

def rename(k, name):
    return k.replace(".entry rowsum_zd(", f".entry {name}(")

# flag: the two in-loop traps become or.pred into %bad; one trap after the loop
flag = zd.replace(".reg .pred %p<13>;", ".reg .pred %p<13>;\n    .reg .pred %bad;")
flag = flag.replace("    // for j in ...\n", "    // for j in ...\n    setp.ne.u32 %bad, 0, 0;\n")
assert flag.count("@%p9 trap;") == 1 and flag.count("@%p12 trap;") == 1
flag = flag.replace("@%p9 trap;", "or.pred %bad, %bad, %p9;").replace("@%p12 trap;", "or.pred %bad, %bad, %p12;")
flag = flag.replace("    $LOOP_END_3:\n", "    $LOOP_END_3:\n    @%bad trap;\n")
# nocheck: the two in-loop traps removed (reference; not fail-closed)
noch = zd.replace("@%p9 trap;", "").replace("@%p12 trap;", "")
head = src[:src.index(".visible .entry")]
open(sys.argv[2], "w").write(head + rename(flag, "rowsum_zd_flag") + "\n" + rename(noch, "rowsum_zd_nocheck") + "\n")
