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
def load(name):
    m = TorchKernel.__new__(TorchKernel)
    m.source, m.kernel_name, m.target_sm, m.ptx = None, name, "sm_89", src
    m._module = m._function = None
    m._ensure_loaded()
    return m
closed, convert = load("lj_closed"), load("lj_convert")
acc2 = torch.zeros(2 * nt + 7, dtype=torch.int64, device=dev)
out2 = torch.empty(nt, dtype=torch.float32, device=dev)
for block, (cg, cb) in zip([33, 64, 256, 1024], [(1, 1024), (7, 33), (12, 128), (2, 64)]):
    closed.launch(((npairs + block - 1) // block, 1, 1), (block, 1, 1), [pos, pi, pj, acc2, npairs, nt])
    convert.launch((cg, 1, 1), (cb, 1, 1), [acc2, out2, nt])
torch.cuda.synchronize()
print("acc zero:", acc.abs().sum().item() == 0, "close/convert acc zero:", acc2.abs().sum().item() == 0,
      "same forces:", torch.equal(out.view(torch.int32), out2.view(torch.int32)))
