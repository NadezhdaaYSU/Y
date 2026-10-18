"""ncu --profile-from-start off -o out python3 prof_e4m3.py <ptx> <name> <n>

One launch of a Y e4m3-input GEMM from PTX and one of torch._scaled_mm on
the same bytes, both checked exact first."""
import re
import sys

import torch
from y_lang.torch_interop import TorchKernel

path, name, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
torch.zeros(1, device="cuda")  # the context the module loads into
k = TorchKernel.__new__(TorchKernel)
k.source, k.kernel_name, k.target_sm = None, name, "sm_89"
k.ptx = open(path).read()
k._module = k._function = None
k._ensure_loaded()
m = re.search(r"Launch grid required: \((\d+), (\d+), (\d+)\) CTAs, block \((\d+),(\d+),(\d+)\)", k.ptx)
grid, block = tuple(int(x) for x in m.groups()[:3]), tuple(int(x) for x in m.groups()[3:])
gen = torch.Generator(device="cuda").manual_seed(n)
ai = torch.randint(-8, 9, (n, n), device="cuda", generator=gen)
bi = torch.randint(-8, 9, (n, n), device="cuda", generator=gen)
a8, b8 = ai.float().to(torch.float8_e4m3fn), bi.float().to(torch.float8_e4m3fn)
C, Ct = torch.empty(n, n, device="cuda"), torch.empty(n, n, device="cuda")
s = torch.tensor(1.0, device="cuda")
for _ in range(2):
    k.launch(grid, block, [a8, b8, 1.0, 1.0, C])
    torch._scaled_mm(a8, b8.t(), scale_a=s, scale_b=s, out_dtype=torch.float32, out=Ct)
torch.cuda.synchronize()
ref = (ai.double() @ bi.double().t()).float()
assert torch.equal(C, ref) and torch.equal(Ct, ref), "a column is not exact"
torch.cuda.cudart().cudaProfilerStart()
k.launch(grid, block, [a8, b8, 1.0, 1.0, C])
torch._scaled_mm(a8, b8.t(), scale_a=s, scale_b=s, out_dtype=torch.float32, out=Ct)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print("profiled", n, "grid", grid, "block", block)
