"""One launch each of the kernels to profile (ncu): grid sum at 2^26 and row sums at 4,096 rows."""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.environ["Y_PYTHON"])
from y_lang.torch_interop import TorchKernel
dev = torch.device("cuda"); torch.zeros(1, device=dev)
def kernel(path, name):
    k = TorchKernel.__new__(TorchKernel)
    k.source, k.kernel_name, k.target_sm, k.ptx = None, name, "sm_89", open(path).read()
    k._module = k._function = None; k._ensure_loaded(); return k
GS, RS, RV = sys.argv[1:4]
n, G, B = 1 << 26, 66 * 8, 256
x = (torch.randn(n, generator=torch.Generator().manual_seed(7)) * 0.3).float().to(dev)
acc = torch.zeros(3, dtype=torch.int64, device=dev); out = torch.empty(G * B, device=dev)
p = torch.tensor([(n + G * B - 1) // (G * B), n], dtype=torch.int32, device=dev)
for name in ("zd_sum", "zd_sum_flag", "f32_partials"):
    kernel(GS, name).launch((G, 1, 1), (B, 1, 1), [x, acc, out, p])
torch.cuda.synchronize()
N, R = 978, 4096
Z = np.clip(np.random.default_rng(2027).standard_normal((R, N)) * 3.0, -10, 10).astype(np.float32)
zt = torch.from_numpy(np.ascontiguousarray(Z.T)).to(dev); perm = torch.arange(N, dtype=torch.int32, device=dev)
o = torch.empty(R, device=dev)
for path, name in ((RS, "rowsum_f32"), (RS, "rowsum_zd"), (RV, "rowsum_zd_flag"), (RV, "rowsum_zd_nocheck")):
    kernel(path, name).launch((16, 1, 1), (256, 1, 1), [zt, perm, o, R, N])
torch.cuda.synchronize()
print("done")
