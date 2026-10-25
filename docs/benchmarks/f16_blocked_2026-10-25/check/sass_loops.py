"""sass_loops.py PTX...: each kernel assembled for sm_89; its K loop in SASS - the innermost backward
branch around the first HMMA - with its instruction count, how many are predicated, its HMMAs and
FADDs, and the kernel's total instructions."""
import re, subprocess, sys
for p in sys.argv[1:]:
    subprocess.run(["ptxas", "-arch=sm_89", p, "-o", p + ".cubin"], check=True)
    L = [l for l in subprocess.run(["cuobjdump", "-sass", p + ".cubin"], capture_output=True, text=True, check=True).stdout.splitlines()
         if re.match(r"\s+/\*[0-9a-f]+\*/", l)]
    ad = lambda l: int(re.search(r"/\*([0-9a-f]+)\*/", l).group(1), 16)
    first_h = next(ad(l) for l in L if "HMMA" in l)
    loops = [(int(m.group(1), 16), ad(l)) for l in L for m in [re.search(r"BRA (?:!?P\d, )?0x([0-9a-f]+)", l)] if m and int(m.group(1), 16) < ad(l)]
    top, bot = min(((t, b) for t, b in loops if t <= first_h <= b), key=lambda x: x[1] - x[0])
    body = [l for l in L if top <= ad(l) <= bot]
    print(f"{p}: K loop {len(body)} instructions, {sum(1 for l in body if re.search(r'[*]/[ ]+@!?P[0-9]', l))} predicated, "
          f"HMMA {sum('HMMA' in l for l in body)}, FADD {sum('FADD' in l for l in body)}; kernel {len(L)} instructions, "
          f"FADD {sum('FADD' in l for l in L)}")
