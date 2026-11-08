"""compile_forward's choices: the norm boundary becomes one persistent kernel,
bits equal eager, and Y_EXACT_PERSISTENT_ROWS=0 restores the looped kernel."""
import sys, torch
import y_lang.exact.model as em
from torch.profiler import ProfilerActivity, profile
H = 3584
g = torch.Generator(device="cuda").manual_seed(0)
h = torch.randn(32, H, device="cuda", generator=g) * 0.5
h[:, ::97] *= 60.0
class N(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight = torch.nn.Parameter(torch.rand(H, device="cuda", generator=g) + 0.5); self.variance_epsilon = 1e-6
class M(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.norm = em.ExactRMSNorm(N()).cuda()
    def forward(self, x, d):
        x2 = x + d
        q, s = em._q_per_row(self.norm(x2), 127)
        return x2, q, s
m = M()
d = torch.randn(32, H, device="cuda", generator=g) * 0.2
with torch.no_grad():
    want = m(h, d)
    em.compile_forward(m)
    got = m(h, d)
    assert all(torch.equal(a, b) for a, b in zip(want, got)), "compiled != eager"
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        m(h, d); torch.cuda.synchronize()
names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
print(sys.argv[1], "kernels:", [n[:22] for n in names], "bits == eager")
