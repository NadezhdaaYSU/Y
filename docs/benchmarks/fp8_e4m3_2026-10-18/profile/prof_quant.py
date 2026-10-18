"""ncu --profile-from-start off python3 prof_quant.py <quant.ysu> <n>: one launch each of
the quantise kernels in the file (named quant_e4m3 / quant_e4m3_t*) at n x n."""
import sys
import torch
import y_lang
src, n = open(sys.argv[1]).read(), int(sys.argv[2])
torch.zeros(1, device="cuda")
X = torch.randint(-8, 9, (n, n), device="cuda").float()
Q = torch.empty(n, n, device="cuda", dtype=torch.uint8)
ks = {}
import re
for name in re.findall(r"kernel (quant\w*)\(", src):
    ks[name] = y_lang.TorchKernel(src, kernel_name=name, target_sm="sm_89")
    ks[name]._ensure_loaded()
def launch(name):
    k = ks[name]
    g = re.search(r"// grid: (\w+)", k.source)
    if name.startswith("quant_e4m3_t"):
        k.launch(*LAUNCH[name](n), [X, 1.0, Q, n, n])
    else:
        k.launch(((n * n + 255) // 256, 1, 1), (256, 1, 1), [X, 1.0, Q, n * n])
LAUNCH = {}
exec(open(sys.argv[1].replace(".ysu", "_launch.py")).read()) if len(sys.argv) > 3 else None
LAUNCH.setdefault("quant_e4m3_t", lambda n: (((n * n + 255) // 256, 1, 1), (256, 1, 1)))
for name in ks:
    launch(name)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
for name in ks:
    launch(name)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
