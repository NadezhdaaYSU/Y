import sys, torch
sys.path.insert(0, sys.argv[1])
from test_f16_gemm import f16_gemm
torch.zeros(1, device="cuda")
for (m, n, k) in ((256, 256, 768), (512, 512, 512)):
    g = torch.Generator(device="cuda").manual_seed(3)
    for kind in ("same-sign", "normal"):
        a = torch.randn(m, k, device="cuda", generator=g).half(); b = torch.randn(k, n, device="cuda", generator=g).half()
        if kind == "same-sign":
            a, b = a.abs(), b.abs()
        kern, grid, block, _ = f16_gemm(m, n, k, "64,64,64,4,2,3")
        c = torch.full((m, n), float("nan"), device="cuda")
        kern.launch(grid, block, [a, b, c]); torch.cuda.synchronize()
        cb = torch.mm(a, b, out_dtype=torch.float32)
        diff = (c.view(torch.int32) != cb.view(torch.int32)).sum().item()
        print(f"{(m, n, k)} {kind}: elements whose bits differ from cuBLAS: {diff} of {m * n}")
