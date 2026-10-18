"""Every float32 bit pattern through Y's E4M3 store, every byte through its
load, and f64 values beside e4m3 ties through the round-to-odd path.

    Y_LIB_PATH=... PYTHONPATH=python python3 sweep_e4m3.py
"""
import sys

import torch
import y_lang

SRC = """
kernel quant_e4m3(X: GlobalMemory<F32>, inv_s: F32, Q: GlobalMemory<E4M3>, n: I32) {
    let idx: I32 = global_thread_id();
    if idx < n {
        Q[idx] = X[idx] * inv_s;
    }
}
"""
DEQ = """
kernel dequant_e4m3(Q: GlobalMemory<E4M3>, X: GlobalMemory<F32>, n: I32) {
    let idx: I32 = global_thread_id();
    if idx < n {
        X[idx] = Q[idx];
    }
}
"""
F64 = """
kernel store_f64(V: GlobalMemory<F64>, Q: GlobalMemory<E4M3>, n: I32) {
    let idx: I32 = global_thread_id();
    if idx < n {
        let v: F64 = V[idx];
        Q[idx] = v;
    }
}
"""


def reference(x64):
    """Round half to even to e4m3 (4 significant bits, exponent >= -6),
    saturating at 448; NaN stays NaN. float64 throughout, exact here."""
    a = x64.abs()
    e = torch.floor(torch.log2(a)).clamp(min=-6, max=8)
    q = torch.exp2(e - 3)
    r = torch.round(a / q) * q
    r = torch.where(a >= 448, torch.full_like(a, 448.0), r)
    r = torch.where(torch.isnan(a), a, r)
    return torch.copysign(r, x64)


def same(got, want):
    nan = torch.isnan(want)
    eq = (got == want) & (torch.signbit(got) == torch.signbit(want))
    return torch.where(nan, torch.isnan(got), eq)


def main():
    q = y_lang.TorchKernel(SRC, kernel_name="quant_e4m3", target_sm="sm_89")
    chunk = 1 << 27
    bad = 0
    for c in range((1 << 32) // chunk):
        bits = torch.arange(c * chunk, (c + 1) * chunk, device="cuda", dtype=torch.int64)
        x = (bits - (1 << 32) * (bits >= (1 << 31))).to(torch.int32).view(torch.float32)
        out = torch.empty(chunk, device="cuda", dtype=torch.uint8)
        q.launch(((chunk + 255) // 256, 1, 1), (256, 1, 1), [x, 1.0, out, chunk])
        got = out.view(torch.float8_e4m3fn).float()
        want = x.to(torch.float8_e4m3fn).float()
        ok = same(got, want)
        nbad = int((~ok).sum())
        if nbad:
            i = int((~ok).nonzero()[0])
            print(f"chunk {c}: {nbad} differ from torch, first {x[i].item()!r} -> {got[i].item()} vs {want[i].item()}")
        bad += nbad
        # torch itself against the definition, on every 64th value of the chunk
        xs = x[::64].double()
        tbad = int((~same(x[::64].to(torch.float8_e4m3fn).double(), reference(xs))).sum())
        if tbad:
            print(f"chunk {c}: torch differs from the reference on {tbad} sampled values")
            bad += tbad
    print("store: all 2^32 float32 inputs checked;", "OK" if bad == 0 else f"{bad} DIFFER")

    d = y_lang.TorchKernel(DEQ, kernel_name="dequant_e4m3", target_sm="sm_89")
    allb = torch.arange(256, device="cuda", dtype=torch.int32).to(torch.uint8)
    xo = torch.empty(256, device="cuda")
    d.launch((1, 1, 1), (256, 1, 1), [allb, xo, 256])
    want = allb.view(torch.float8_e4m3fn).float()
    lbad = int((~same(xo, want)).sum())
    print("load: all 256 bytes;", "OK" if lbad == 0 else f"{lbad} DIFFER")

    f = y_lang.TorchKernel(F64, kernel_name="store_f64", target_sm="sm_89")
    # Every e4m3 tie (the midpoints between neighbours), each exactly and
    # 2^-40 relative above and below - inside one float32 ulp, where rounding
    # to nearest float32 first would land on the tie.
    vals = torch.arange(0, 127, device="cuda", dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).double()
    mids = (vals[:-1] + vals[1:]) / 2
    v = torch.cat([mids, mids * (1 + 2.0 ** -40), mids * (1 - 2.0 ** -40), -mids * (1 + 2.0 ** -40),
                   torch.tensor([1e300, -1e300, float("inf"), 1e-300, -1e-300, 0.0, -0.0, float("nan")],
                                device="cuda", dtype=torch.float64)])
    out = torch.empty(v.numel(), device="cuda", dtype=torch.uint8)
    f.launch(((v.numel() + 255) // 256, 1, 1), (256, 1, 1), [v, out, v.numel()])
    got = out.view(torch.float8_e4m3fn).double()
    fbad = ~same(got, reference(v))
    for i in fbad.nonzero().flatten().tolist()[:10]:
        print(f"  f64 {v[i].item()!r} -> {got[i].item()} want {reference(v)[i].item()}")
    # The control: the same values rounded to float32 first, then to e4m3.
    twice = v.float().to(torch.float8_e4m3fn).double()
    print("f64 store:", v.numel(), "values;", "OK" if int(fbad.sum()) == 0 else f"{int(fbad.sum())} DIFFER",
          f"(rounding through float32 first differs on {int((~same(twice, reference(v))).sum())})")
    sys.exit(1 if bad or lbad or int(fbad.sum()) else 0)


if __name__ == "__main__":
    main()
