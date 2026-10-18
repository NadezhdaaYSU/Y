import torch
torch.zeros(1, device="cuda")
for k in (768, 4096):
    m = n = 256
    g = torch.Generator(device="cuda").manual_seed(3)
    a = torch.randn(m, k, device="cuda", generator=g).abs().half()
    b = torch.randn(k, n, device="cuda", generator=g).abs().half()
    ref = a.double() @ b.double()
    c = torch.mm(a, b, out_dtype=torch.float32)
    seq = torch.zeros(m, n, device="cuda", dtype=torch.float32)
    for kk in range(0, k, 16):
        seq += (a[:, kk:kk+16].double() @ b[kk:kk+16, :].double()).float()
    r = lambda x: ((x.double() - ref).abs() / ref).max().item()
    print(f"K={k}: cuBLAS f32-out max rel err {r(c):.3e}; f32 sum of exact k16 partials {r(seq):.3e}")
