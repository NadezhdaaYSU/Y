"""launch_sq.py N [file.ptx ...]: for ncu, one warm-up and one profiled launch of each Y kernel
(N^3, f16 in, f32 out) and of torch.mm(a, b, out_dtype=float32), on random-normal inputs."""
import re, sys, torch
from y_lang.torch_interop import TorchKernel
n = int(sys.argv[1])
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
a = torch.randn(n, n, device="cuda").half(); b = torch.randn(n, n, device="cuda").half()
c = torch.empty(n, n, device="cuda")
for path in sys.argv[2:]:
    ptx = open(path).read()
    name = re.findall(r"\.visible \.entry (\w+)\(", ptx)[0]
    cm, cn, wm, wn = (int(x) for x in re.search(r"CTA (\d+)x(\d+)x\d+ \| (\d+)x(\d+) warps", ptx).groups())
    k = TorchKernel.__new__(TorchKernel)
    k.source, k.kernel_name, k.target_sm = None, name, "sm_89"
    k.ptx = ptx; k._module = k._function = None; k._ensure_loaded()
    for _ in range(2):
        k.launch((-(-n // cn), -(-n // cm), 1), (wm * wn * 32, 1, 1), [a, b, c])
    torch.cuda.synchronize()
for _ in range(2):
    torch.mm(a, b, out_dtype=torch.float32)
torch.cuda.synchronize()
