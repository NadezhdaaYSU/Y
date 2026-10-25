"""fused_table.py DIR: the fused SwiGLU (4096) and bias + ReLU (8192) runs: each column's minimum against
PyTorch's (SwiGLU) or addmm + ReLU's (bias), and against the chain's, per run."""
import json, sys
for name, size, ref in [("swiglu", "4096", "PyTorch"), ("bias", "8192", "addmm32+relu")]:
    print(f"\n{name} {size}^3: reference / column and chain / column, minimum of 15 rounds, per run")
    for i in (1, 2):
        d = json.load(open(f"{sys.argv[1]}/{name}_{i}.json"))["sizes"][size]
        m = {c: min(v) for c, v in d["samples"].items()}
        print(f"  run {i}: " + "; ".join(f"{c} {m[ref]/v:.3f} / {m['mma']/v:.3f}" for c, v in m.items() if c != ref)
              + (f" | max rel err {', '.join(f'{c} {e:.1e}' for c, e in d['max_rel_err'].items())}" if 'max_rel_err' in d else ""))
