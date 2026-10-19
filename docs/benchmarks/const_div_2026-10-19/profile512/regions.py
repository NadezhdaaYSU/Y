#!/usr/bin/env python3
"""regions.py sass_0.csv sass_1.csv: each kernel's executed warp instructions
and stall samples, by opcode, whole and split at the first and last HMMA
(before / the HMMA span / after)."""
import csv, sys, collections, re
for path in sys.argv[1:]:
    rows = list(csv.reader(open(path))); hdr = rows[1]; name = rows[0][1]
    iS, iE, iSt = hdr.index("Source"), hdr.index("Instructions Executed"), hdr.index("Warp Stall Sampling (All Samples)")
    L = [(r[iS].strip(), int(r[iE] or 0), int(r[iSt] or 0)) for r in rows[2:] if len(r) > iE]
    hm = [i for i, (s, e, st) in enumerate(L) if "HMMA" in s]
    print(f"== {name}: {len(L)} SASS lines, {sum(e for _, e, _ in L)} warp instructions executed, {sum(s for _, _, s in L)} stall samples")
    for k, seg in {"whole": L, "before": L[:hm[0]], "HMMA span": L[hm[0]:hm[-1] + 1], "after": L[hm[-1] + 1:]}.items():
        ops = collections.Counter()
        for s, e, _ in seg:
            ops[re.sub(r"^@!?U?P\w+\s+", "", s).split()[0].split(".")[0]] += e
        print(f"  {k:10s} {sum(e for _, e, _ in seg):7d} | " + " ".join(f"{o}:{c}" for o, c in ops.most_common(12)))
    print(f"  dead '@!PT LDS RZ, [RZ]' executed: {sum(e for s, e, _ in L if s.startswith('@!PT') and 'LDS RZ' in s)}")
