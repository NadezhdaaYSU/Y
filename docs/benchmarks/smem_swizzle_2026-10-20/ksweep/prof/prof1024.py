"""prof1024.py <label=ptx-dir,...>: each Y kernel at 512x512x1024 twice, then
cuBLAS's torch.mm (f32 out) twice, each output checked exact, for ncu."""
import re
import sys

import torch

from y_lang.torch_interop import TorchKernel

m = n = 512
kk = 1024
name = f"512x512x{kk}"
gen = torch.Generator(device="cuda").manual_seed(kk)
a = torch.randint(-4, 5, (m, kk), device="cuda", generator=gen).half()
b = torch.randint(-4, 5, (kk, n), device="cuda", generator=gen).half()
ref = (a.double() @ b.double()).float()
for label, d in (x.split("=", 1) for x in sys.argv[1].split(",")):
    k = TorchKernel.__new__(TorchKernel)
    k.source, k.kernel_name, k.target_sm = None, f"gemm_f16_{name}", "sm_89"
    k.ptx = open(f"{d}/g{name}.ptx").read()
    k._module = k._function = None
    k._ensure_loaded()
    cm, cn, wm, wn = (int(x) for x in re.search(r"CTA (\d+)x(\d+)x\d+ \| (\d+)x(\d+) warps", k.ptx).groups())
    for _ in range(2):
        o = torch.full((m, n), float("nan"), device="cuda")
        k.launch((-(-n // cn), -(-m // cm), 1), (wm * wn * 32, 1, 1), [a, b, o])
        torch.cuda.synchronize()
        print(label, "exact", bool(torch.equal(o, ref)), flush=True)
for _ in range(2):
    o = torch.mm(a, b, out_dtype=torch.float32)
    torch.cuda.synchronize()
    print("cuBLAS exact", bool(torch.equal(o, ref)), flush=True)
