"""Launch the E8 kernels a few times for compute-sanitizer: partial warps, several blocks, many totals."""
import sys, numpy as np, torch
sys.path.insert(0, sys.argv[2])
from y_lang.torch_interop import TorchKernel
torch.zeros(1, device="cuda")
src = open(sys.argv[1]).read()
k = TorchKernel.__new__(TorchKernel)
k.source, k.kernel_name, k.target_sm, k.ptx = None, "lj_exact", "sm_89", src
k._module = k._function = None
k._ensure_loaded()
dev = torch.device("cuda")
m = 8
atoms = m ** 3
ix = np.stack(np.meshgrid(np.arange(m), np.arange(m), np.arange(m), indexing="ij"), -1).reshape(-1, 3)
pos_h = (ix * 1.1 + np.random.default_rng(0).uniform(-0.05, 0.05, ix.shape)).astype(np.float32)
i, j = np.triu_indices(atoms, 1)
d = pos_h[i] - pos_h[j]
close = (d * d).sum(1) < 2.5 ** 2
pi_h, pj_h = i[close].astype(np.int32), j[close].astype(np.int32)
npairs = len(pi_h)
pos = torch.from_numpy(pos_h.reshape(-1)).to(dev)
pi, pj = torch.from_numpy(pi_h).to(dev), torch.from_numpy(pj_h).to(dev)
nt = 3 * atoms
acc = torch.zeros(2 * nt + 1, dtype=torch.int64, device=dev)
out = torch.empty(nt, dtype=torch.float32, device=dev)
for block in [33, 64, 256, 1024]:
    k.launch(((npairs + block - 1) // block, 1, 1), (block, 1, 1), [pos, pi, pj, acc, out, npairs, nt])
torch.cuda.synchronize()
print("acc zero:", acc.abs().sum().item() == 0)
