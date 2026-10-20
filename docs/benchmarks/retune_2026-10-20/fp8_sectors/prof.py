"""prof.py <ptx> <n>: the committed FP8 GEMM (f32 A, B quantized on the fly) at n^3, twice, for ncu's per-instruction L2 sectors."""
import re, sys, torch
from y_lang.torch_interop import TorchKernel
path, n = sys.argv[1], int(sys.argv[2])
torch.zeros(1, device="cuda")  # a context before cuModuleLoadData
k = TorchKernel.__new__(TorchKernel)
k.source, k.kernel_name, k.target_sm = None, f"gemm_fp8_{n}", "sm_89"
k.ptx = open(path).read(); k._module = k._function = None; k._ensure_loaded()
cm, cn, wm, wn = (int(x) for x in re.search(r"CTA (\d+)x(\d+)x\d+ \(padded\) \| (\d+)x(\d+) warps", k.ptx).groups())
a = torch.randn(n, n, device="cuda"); b = torch.randn(n, n, device="cuda")
sa = (a.abs().max() / 448.0).item(); sb = (b.abs().max() / 448.0).item()
for _ in range(2):
    c = torch.full((n, n), float("nan"), device="cuda")
    k.launch((-(-n // cn), -(-n // cm), 1), (wm * wn * 32, 1, 1), [a, b, sa, sb, c]); torch.cuda.synchronize()
    print("finite", bool(c.isfinite().all()))
