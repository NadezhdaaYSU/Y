"""launch_mnk.py M N K [file.ptx]: one warm-up and one profiled launch of the Y kernel (if given) and of
torch.mm(a, b, out_dtype=float32), N(0,1) inputs, for ncu."""
import re, sys, torch
from y_lang.torch_interop import TorchKernel
torch.zeros(1, device="cuda")
m, n, k = (int(x) for x in sys.argv[1:4])
a = torch.randn(m, k, device="cuda").half(); b = torch.randn(k, n, device="cuda").half(); c = torch.empty(m, n, device="cuda")
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
if len(sys.argv) > 4:
    ptx = open(sys.argv[4]).read()
    name = re.findall(r"\.visible \.entry (\w+)\(", ptx)[0]
    cm, cn, wm, wn = (int(x) for x in re.search(r"CTA (\d+)x(\d+)x\d+ \| (\d+)x(\d+) warps", ptx).groups())
    kk = TorchKernel.__new__(TorchKernel)
    kk.source, kk.kernel_name, kk.target_sm = None, name, "sm_89"
    kk.ptx = ptx; kk._module = kk._function = None; kk._ensure_loaded()
    for _ in range(2): kk.launch((-(-n // cn), -(-m // cm), 1), (wm * wn * 32, 1, 1), [a, b, c])
for _ in range(2): torch.mm(a, b, out_dtype=torch.float32)
torch.cuda.synchronize()
