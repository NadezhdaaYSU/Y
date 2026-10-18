#!/usr/bin/env python3
"""tables.py: the README's tables, read from cupti_1.log, cupti_2.log and the
ncu_*.txt instruction counts in this directory (nothing typed by hand)."""
import os, re

D = os.path.dirname(os.path.abspath(__file__))
CASES = ["b1c1024", "b1c4096", "b8c1024", "b8c4096", "b32c1024", "b32c4096"]
VARIANTS = ["old_16_8/16x8w", "old_4_8/4x8w", "new_16_8/16x8w", "new_4_8/4x8w", "new_4_4/4x4w", "new_32_4/32x4w", "flashinfer"]


def parse(name):
    """{(case, variant): (span, main kernel, second kernel)} in us."""
    d, case = {}, None
    for line in open(os.path.join(D, name)):
        m = re.match(r"== (\w+)", line)
        if m:
            case = m.group(1)
            continue
        m = re.match(r"\s+(\S+)\s+span/iter\s+([\d.]+) us \| (.*)", line)
        if m:
            ks = {k: float(v) for k, v in (x.split("=") for x in m.group(3).split("  ") if "=" in x)}
            main = [v for k, v in ks.items() if "reduce" not in k and "int>" not in k]
            other = [v for k, v in ks.items() if "reduce" in k or "int>" in k]
            assert len(main) == 1 and len(other) == 1, line
            d[(case, m.group(1))] = (float(m.group(2)), main[0], other[0])
    return d


runs = [parse("cupti_1.log"), parse("cupti_2.log")]
short = {v: v.split("/")[1].replace("w", " warps").replace("x", " splits x ") if v != "flashinfer" else "FlashInfer" for v in VARIANTS}

print("Main kernel, device time in us, run 1 / run 2:\n")
print("| case | " + " | ".join(("old " if v.startswith("old") else "new " if v.startswith("new") else "") + short[v] for v in VARIANTS) + " |")
print("|---" * (len(VARIANTS) + 1) + "|")
for c in CASES:
    print("| %s | " % c + " | ".join("%.2f / %.2f" % (runs[0][(c, v)][1], runs[1][(c, v)][1]) for v in VARIANTS) + " |")

# The configurations the README's benchmark compiles: 16 splits x 8 warps at
# batch 1, 4 x 8 at batch 8 and 32 (tests/paged_decode_attention_split_*).
print("\nWhole call (`span`: first kernel start to last kernel end), the README's configurations "
      "(16 splits at batch 1, 4 at batch 8 and 32, 8 warps), and 4 x 4 warps; FlashInfer over Y, >1 is Y faster:\n")
print("| case | old | new | new 4x4 | FlashInfer | FI / old | FI / new | FI / new 4x4 |")
print("|---|---|---|---|---|---|---|---|")
for c in CASES:
    cfg = "16_8/16x8w" if c.startswith("b1c") else "4_8/4x8w"
    row = []
    for r in runs:
        o, n, n44, fi = r[(c, "old_" + cfg)][0], r[(c, "new_" + cfg)][0], r[(c, "new_4_4/4x4w")][0], r[(c, "flashinfer")][0]
        row.append((o, n, n44, fi, fi / o, fi / n, fi / n44))
    print("| %s | " % c + " | ".join("%.2f / %.2f" % (row[0][i], row[1][i]) if i < 4 else "%.2fx / %.2fx" % (row[0][i], row[1][i]) for i in range(7)) + " |")

print("\nInstructions (Nsight Compute `smsp__inst_executed.sum`, main kernel) against run 1's device time, "
      "at the SM clock of 2760 MHz most of its rounds ran at; 66 SMs x 4 schedulers:\n")
print("| case | kernel | warp instructions | registers | device time | instructions / cycle / scheduler |")
print("|---|---|---|---|---|---|")
for case, label, f, v in [("b1c4096", "old 16x8", "ncu_old_b1c4096.txt", "old_16_8/16x8w"),
                          ("b1c4096", "new 16x8", "ncu_new_b1c4096.txt", "new_16_8/16x8w"),
                          ("b1c4096", "FlashInfer", "ncu_new_b1c4096.txt", "flashinfer"),
                          ("b8c1024", "old 4x8", "ncu_old_b8c1024.txt", "old_4_8/4x8w"),
                          ("b8c1024", "new 4x8", "ncu_new48_b8c1024.txt", "new_4_8/4x8w"),
                          ("b8c1024", "new 4x4", "ncu_new44_b8c1024.txt", "new_4_4/4x4w"),
                          ("b8c1024", "FlashInfer", "ncu_new44_b8c1024.txt", "flashinfer")]:
    rows = [l.split(" | ") for l in open(os.path.join(D, f)) if l.startswith(("paged", "void flashinfer::BatchDecode"))]
    want = "void flashinfer::BatchDecode" if v == "flashinfer" else "paged"
    # The main kernel, not its reduce: the one of its kind executing the most.
    row = max((r for r in rows if r[0].startswith(want)), key=lambda r: float(r[1]))
    inst, regs = float(row[1]), int(row[2])
    t = runs[0][(case, v)][1]
    print("| %s | %s | %.2f M | %d | %.2f us | %.3f |" % (case, label, inst / 1e6, regs, t, inst / 264 / (t * 2760)))
