"""inf_check.py <bench_ab.py dir> <ptx dir>: what one +Inf (and one NaN) in A does to each
column of <ptx dir>/8x4096x4096 (Y_f32, Y_kahan, Y_mma) and to cuBLAS: the values of the
row holding it, which should be +Inf (resp. NaN) where B's column is positive."""
import os, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "f16_splitk_2026-10-20"))
sys.argv = [sys.argv[0], sys.argv[1], *sys.argv[2:]]
import accuracy_split  # noqa: E402  (imports ../f16_accuracy_2026-10-20/accuracy.py with argv[1])
torch.zeros(1, device="cuda")
m, n, k = 8, 4096, 4096
d = f"{sys.argv[2]}/8x4096x4096"
g = torch.Generator(device="cuda").manual_seed(1)
b = (torch.rand(k, n, device="cuda", generator=g) + 0.5).half()          # positive
for special in ["inf", "nan"]:
    a = (torch.rand(m, k, device="cuda", generator=g) + 0.5).half()
    a[3, 100] = float(special)
    for col in ["Y_f32", "Y_kahan", "Y_mma"]:
        c = accuracy_split.launch(f"{d}/{col}.ptx", a, b, m, n)
        row, other = c[3], torch.cat([c[:3], c[4:]])
        print(f"A[3,100] = {special}: {col:8s} row 3: {int(torch.isposinf(row).sum())} +Inf, {int(torch.isnan(row).sum())} NaN of {n}; other rows finite: {bool(torch.isfinite(other).all())}")
    c = torch.mm(a, b, out_dtype=torch.float32)
    print(f"A[3,100] = {special}: cuBLAS   row 3: {int(torch.isposinf(c[3]).sum())} +Inf, {int(torch.isnan(c[3]).sum())} NaN of {n}")
