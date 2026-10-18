#!/usr/bin/env python3
"""sass_loop.py <ptx> <entry>: opcode census of the KV loop (the first backward branch whose body holds an LDG),
and the size of the lazy-rescale block a VOTE guards, if there is one."""
import os, re, subprocess, sys, tempfile
from collections import Counter
ptx, entry = sys.argv[1], sys.argv[2]
with tempfile.TemporaryDirectory() as d:
    cubin = os.path.join(d, "k.cubin")
    subprocess.run(["ptxas", "-arch=sm_89", "-O3", ptx, "-o", cubin], check=True)
    txt = subprocess.run(["cuobjdump", "-sass", "-fun", entry, cubin], capture_output=True, text=True, check=True).stdout
ins = []
for l in txt.split("\n"):
    m = re.match(r"\s*/\*([0-9a-f]{4,})\*/\s*(.*?);", l)
    if m:
        ins.append((int(m.group(1), 16), m.group(2).strip()))
for addr, op in ins:
    b = re.search(r"BRA (0x[0-9a-f]+)", op)
    if b and int(b.group(1), 16) < addr and "DIV" not in op:
        body = [o for a, o in ins if int(b.group(1), 16) <= a <= addr]
        if any("LDG" in o for o in body):
            c = Counter(re.sub(r"^@!?U?P\w+\s+", "", o).split(" ")[0].split(".")[0] for o in body)
            print("%s: loop 0x%x-0x%x, %d instructions:" % (entry, int(b.group(1), 16), addr, len(body)),
                  " ".join("%s=%d" % kv for kv in sorted(c.items(), key=lambda t: -t[1])))
            # The lazy rescale: the forward branch after the VOTE skips the
            # rescale block; count what it skips.
            loop = [(a, o) for a, o in ins if int(b.group(1), 16) <= a <= addr]
            vote = [i for i, (a, o) in enumerate(loop) if o.startswith("VOTE")]
            if vote:
                for a, o in loop[vote[0]:]:
                    f = re.match(r"@!?P\w+ BRA (0x[0-9a-f]+)", o)
                    if f and int(f.group(1), 16) > a:
                        skipped = [o2 for a2, o2 in loop if a < a2 < int(f.group(1), 16)]
                        print("  rescale block 0x%x-0x%x: %d instructions, so %d when it is skipped"
                              % (a + 16, int(f.group(1), 16) - 16, len(skipped), len(body) - len(skipped)))
                        break
            break
