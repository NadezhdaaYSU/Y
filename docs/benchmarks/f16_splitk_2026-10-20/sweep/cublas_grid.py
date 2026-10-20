import json, sys, torch
from torch.profiler import profile, ProfilerActivity
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
out = {}
for K in [1024, 4096, 16384, 65536]:
    a = torch.randn(512, K, device="cuda").half(); b = torch.randn(K, 512, device="cuda").half()
    o = torch.empty(512, 512, device="cuda")
    for _ in range(3): torch.mm(a, b, out_dtype=torch.float32, out=o)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        torch.mm(a, b, out_dtype=torch.float32, out=o); torch.cuda.synchronize()
    p.export_chrome_trace(f"trace_{K}.json")
    tr = json.load(open(f"trace_{K}.json"))
    ks = [(e["name"][:110], e["args"].get("grid"), e["args"].get("block"), e["args"].get("shared memory"), e["args"].get("registers per thread"), e.get("dur")) for e in tr["traceEvents"] if e.get("cat") == "kernel"]
    print(K, ks, flush=True)
