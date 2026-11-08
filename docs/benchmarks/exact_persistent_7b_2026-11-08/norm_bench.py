"""The 7B layer boundary alone: residual add + ExactRMSNorm + activation
quantisation into two digits, at decode shape (32 rows x 3584), compiled the
way the model is. Times variants and checks each one's outputs bit for bit
against the current code.

    python3 norm_bench.py <variant>    variant: base | persistent | twostage
"""
import sys
import torch
import torch._inductor.config as ic
from torch._inductor import choices as ichoices
from torch._inductor.virtualized import V

import y_lang.exact.model as em

VARIANT = sys.argv[1]
ROWS, H = 32, 3584
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(0)
# hidden states with a realistic spread: mostly O(1), a few large channels
h = torch.randn(ROWS, H, device=dev, generator=g) * 0.5
h[:, ::97] *= 60.0
delta = torch.randn(ROWS, H, device=dev, generator=g) * 0.2


class FakeNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(H, device=dev, generator=g) + 0.5)
        self.variance_epsilon = 1e-6


norm = em.ExactRMSNorm(FakeNorm()).to(dev)
LV = em.act_levels_for(H)


def f32_norm(x):            # timing reference only: NOT exact
    var = x.pow(2).mean(-1, keepdim=True)
    return (x / torch.sqrt(var + 1e-6)) * norm.weight


def mul64_norm(x):          # timing reference only: NOT exact (reciprocal multiply)
    xd = x.to(torch.float64)
    var = xd.pow(2).sum(dim=-1, keepdim=True) / em.divisor(xd.shape[-1], xd)
    y = xd * (1.0 / torch.sqrt(var + norm.eps))
    return (y.to(torch.float32) * norm.weight).to(x.dtype)


def boundary(h, delta):
    h2 = h + delta
    if VARIANT.startswith("mul64"):
        y = mul64_norm(h2)
    else:
        y = f32_norm(h2) if VARIANT == "f32" else norm(h2)
    q, s = em._q_per_row(y, LV)
    hi, lo = em.split_digits(q.to(torch.int32), LV)
    return h2, hi, lo, s


with torch.no_grad():
    boundary(h, delta)      # creates the divisors, as `convert` does for a model


if VARIANT.startswith("coop"):
    ic.triton.cooperative_reductions = True
    if VARIANT == "coop_force":
        ic.triton.force_cooperative_reductions = True
if VARIANT in ("persistent", "coop_persist", "mul64_persist"):
    class Persist(ichoices.InductorChoices):
        @staticmethod
        def should_use_persistent_reduction(features, cooperative_reduction):
            from torch._inductor.runtime.hints import ReductionHint
            if features.get_reduction_hint() == ReductionHint.INNER and \
                    V.graph.sizevars.statically_known_leq(features.reduction_numel, 4096):
                return True
            return ichoices.InductorChoices.should_use_persistent_reduction(
                features, cooperative_reduction)
    V.set_choices_handler(Persist())

fn = torch.compile(boundary, dynamic=False)
ref = None
if VARIANT not in ("base", "f32") and not VARIANT.startswith("mul64"):
    torch._dynamo.reset()
    V.set_choices_handler(ichoices.InductorChoices())
    base = torch.compile(boundary, dynamic=False)
    with torch.no_grad():
        ref = base(h, delta)
    torch._dynamo.reset()
    if VARIANT in ("persistent", "coop_persist", "mul64_persist"):
        V.set_choices_handler(Persist())
    fn = torch.compile(boundary, dynamic=False)

with torch.no_grad():
    out = fn(h, delta)
    eager = boundary(h, delta)
    for a, b, n in zip(out, eager, ("h2", "hi", "lo", "s")):
        assert VARIANT == "f32" or VARIANT.startswith("mul64") or torch.equal(a, b), f"{VARIANT}: {n} differs from eager"
    if ref is not None:
        for a, b, n in zip(out, ref, ("h2", "hi", "lo", "s")):
            assert torch.equal(a, b), f"{VARIANT}: {n} differs from base"
    # Hold the GPU busy for ~2 s first: from idle clocks a short burst is
    # timed while the clock is still ramping (40 -> 217 us for one kernel).
    import time
    a = torch.randn(4096, 4096, device=dev)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 2.0:
        for _ in range(20):
            a @ a
        for _ in range(50):
            fn(h, delta)
        torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile
    N = 200
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(N):
            fn(h, delta)
        torch.cuda.synchronize()
    ks = {}
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            k = ks.setdefault(e.name, [0.0, 0])
            k[0] += e.device_time
            k[1] += 1
total = sum(v[0] for v in ks.values()) / N
print(f"{VARIANT}: GPU {total:.1f} us/call in {sum(v[1] for v in ks.values()) // N} kernels, "
      f"outputs == eager{' == base' if ref is not None else ''}, a_levels {LV}")
for n, v in sorted(ks.items(), key=lambda kv: -kv[1][0]):
    print(f"   {v[0] / N:7.2f} us  {n[:100]}")
