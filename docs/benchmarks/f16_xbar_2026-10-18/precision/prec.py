"""prec.py <tests dir>: every pinned tile of test_f16_gemm at 256 x 256 x 768 on same-sign
f16 inputs against float64, bit for bit against the first tile, and the controls."""
import sys, torch
sys.path.insert(0, sys.argv[1])
from test_f16_gemm import f16_gemm, TILES
torch.zeros(1, device="cuda")
m, n, k = 256, 256, 768
g = torch.Generator(device="cuda").manual_seed(3)
# same sign: |N(0,1)| f16, so no cancellation hides a short accumulator
a = torch.randn(m, k, device="cuda", generator=g).abs().half()
b = torch.randn(k, n, device="cuda", generator=g).abs().half()
ref = a.double() @ b.double()
outs = {}
for tile in TILES:
    kern, grid, block, _ = f16_gemm(m, n, k, tile)
    c = torch.full((m, n), float("nan"), device="cuda")
    kern.launch(grid, block, [a, b, c]); torch.cuda.synchronize()
    outs[tile] = c
    e = (c.double() - ref) / ref
    print(f"{tile:20s} rel err max {e.abs().max().item():.3e} mean {e.abs().mean().item():.3e} signed mean {e.mean().item():+.3e}  bitwise == first tile: {torch.equal(c.view(torch.int32), outs[TILES[0]].view(torch.int32))}")
seq = torch.zeros(m, n, device="cuda", dtype=torch.float32)
for kk in range(0, k, 16):
    seq += (a[:, kk:kk+16].double() @ b[kk:kk+16, :].double()).float()
print("f32 sum of exact k16 partials, max rel err", ((seq.double() - ref).abs() / ref).max().item())
h = (a @ b)  # half output: half-precision rounding of the result at least
print("torch half matmul (control), max rel err", ((h.double() - ref).abs() / ref).max().item())
hs = torch.zeros(m, n, device="cuda", dtype=torch.float16)
for kk in range(0, k, 16):
    hs += (a[:, kk:kk+16].float() @ b[kk:kk+16, :].float()).half()
print("f16 accumulation of k16 partials (control), max rel err", ((hs.double() - ref).abs() / ref).max().item())
