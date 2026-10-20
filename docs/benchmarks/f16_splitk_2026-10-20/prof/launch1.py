"""launch1.py file.ptx ...: launch each kernel's main entry once (512x512xK inputs) for ncu."""
import re, sys, torch
from y_lang.torch_interop import TorchKernel
torch.zeros(1, device="cuda")
for path in sys.argv[1:]:
    ptx = open(path).read()
    names = re.findall(r"\.visible \.entry (\w+)\(", ptx)
    name = [x for x in names if not x.endswith("_reduce")][0]
    m, n, k = (int(x) for x in re.search(r"M=(\d+) N=(\d+) K=\d+ \|", ptx).groups()), None, None
    m, n = 512, 512
    k = int(re.search(r"x(\d+)(_(split|stream)k\d+)?$", name).group(1))
    cm, cn, wm, wn = (int(x) for x in re.search(r"CTA (\d+)x(\d+)x\d+ \| (\d+)x(\d+) warps", ptx).groups())
    sk = re.search(r"over (\d+) CTAs .* Partial\[0\.\.(\d+)\]", ptx)
    s = int(re.search(r"in (\d+) splits", ptx).group(1)) if name + "_reduce" in names and not sk else 1
    kern = TorchKernel.__new__(TorchKernel)
    kern.source, kern.kernel_name, kern.target_sm = None, name, "sm_89"
    kern.ptx = ptx; kern._module = kern._function = None; kern._ensure_loaded()
    a = torch.randn(m, k, device="cuda").half(); b = torch.randn(k, n, device="cuda").half()
    c = torch.empty(m, n, device="cuda"); p = torch.empty(int(sk.group(2)) if sk else s, m, n, device="cuda")
    args = [a, b, c, p] if s > 1 or name + "_reduce" in names else [a, b, c]
    grid = (int(sk.group(1)), 1, 1) if sk else (-(-n // cn), -(-m // cm), s)
    kern.launch(grid, (wm * wn * 32, 1, 1), args)
    torch.cuda.synchronize()
