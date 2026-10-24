"""mma_k8_vs_k16.py: m16n8k16 against two m16n8k8 (same products) on the tensor core, f16 in,
f32 accumulate: (1) the same bits? one warp, random fragments, chained over 64 steps, every
lane's accumulators stored; (2) throughput: 66*8 warps, 8 independent chains each, ITER steps,
every accumulator stored, timed by CUDA events, interleaved rounds, minimum."""
import sys, torch
from y_lang.torch_interop import TorchKernel
torch.zeros(1, device="cuda")

def kernel(name, k8, chains, iters):
    regs_c = [[f"%f{c * 4 + i}" for i in range(4)] for c in range(chains)]
    body = []
    for c in range(chains):
        d = ",".join(regs_c[c])
        if k8:
            body.append(f"    mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {{{d}}}, {{%ra0,%ra1}}, {{%rb0}}, {{{d}}};")
            body.append(f"    mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {{{d}}}, {{%ra2,%ra3}}, {{%rb1}}, {{{d}}};")
        else:
            body.append(f"    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{{d}}}, {{%ra0,%ra1,%ra2,%ra3}}, {{%rb0,%rb1}}, {{{d}}};")
    stores = []
    for c in range(chains):
        for i in range(4):
            stores.append(f"    st.global.f32 [%rd2+{(c * 4 + i) * 4}], {regs_c[c][i]};")
    nf = chains * 4
    return f""".version 7.8
.target sm_89
.address_size 64
.visible .entry {name}(.param .u64 pin, .param .u64 pout, .param .u32 piters)
{{
    .reg .b32 %ra<4>; .reg .b32 %rb<2>; .reg .f32 %f<{nf}>; .reg .b64 %rd<4>; .reg .b32 %r<8>; .reg .pred %p<2>;
    ld.param.u64 %rd0, [pin]; ld.param.u64 %rd1, [pout]; ld.param.u32 %r5, [piters];
    mov.u32 %r0, %tid.x; mov.u32 %r1, %ctaid.x; mov.u32 %r2, %ntid.x;
    mad.lo.u32 %r3, %r1, %r2, %r0;
    and.b32 %r4, %r0, 31;
    mul.wide.u32 %rd3, %r4, 24;
    add.s64 %rd0, %rd0, %rd3;
    ld.global.b32 %ra0, [%rd0]; ld.global.b32 %ra1, [%rd0+4]; ld.global.b32 %ra2, [%rd0+8]; ld.global.b32 %ra3, [%rd0+12];
    ld.global.b32 %rb0, [%rd0+16]; ld.global.b32 %rb1, [%rd0+20];
    mul.wide.u32 %rd3, %r3, {nf * 4};
    add.s64 %rd2, %rd1, %rd3;
""" + "\n".join(f"    mov.f32 %f{i}, 0f00000000;" for i in range(nf)) + f"""
    mov.u32 %r6, 0;
LOOP:
    setp.ge.u32 %p0, %r6, %r5;
    @%p0 bra DONE;
""" + "\n".join(body) + """
    add.u32 %r6, %r6, 1;
    bra LOOP;
DONE:
""" + "\n".join(stores) + "\n    ret;\n}\n"

def load(src, name):
    k = TorchKernel.__new__(TorchKernel)
    k.source, k.kernel_name, k.target_sm = None, name, "sm_89"
    k.ptx = src; k._module = k._function = None; k._ensure_loaded()
    return k

# (1) bits: one warp, one chain, 64 steps, random f16 fragments (the same fragments every step)
g = torch.Generator(device="cuda").manual_seed(5)
for trial, scale in enumerate([1.0, 1.0, 64.0, 1.0 / 64]):
    frag = (torch.randn(32 * 12, device="cuda", generator=g) * scale).half()
    frag_abs = frag.abs() if trial == 1 else frag
    outs = {}
    for k8 in (False, True):
        name = f"bits_{int(k8)}"
        k = load(kernel(name, k8, 1, 64), name)
        out = torch.zeros(32 * 4, device="cuda")
        k.launch((1, 1, 1), (32, 1, 1), [frag_abs, out, 64])
        torch.cuda.synchronize()
        outs[k8] = out.clone()
    same = torch.equal(outs[False].view(torch.int32), outs[True].view(torch.int32))
    diff = (outs[False] - outs[True]).abs().max().item()
    print(f"bits trial {trial} ({'same-sign' if trial == 1 else 'mixed'} x{scale}): k16 == 2 x k8 bit for bit: {same}, max |diff| {diff:.3g}")

# (2) throughput
chains, iters, warps = 8, 4096, 66 * 8
frag = torch.randn(32 * 12, device="cuda").half()
out = torch.empty(warps * 32 * chains * 4, device="cuda")
ks = {}
for k8 in (False, True):
    name = f"tp_{int(k8)}"
    ks["k8x2" if k8 else "k16"] = load(kernel(name, k8, chains, iters), name)
macs = warps * chains * iters * 16 * 8 * 16
res = {k: [] for k in ks}
for k in ks.values():
    k.launch((warps // 8, 1, 1), (256, 1, 1), [frag, out, iters])
torch.cuda.synchronize()
for r in range(9):
    for name in (list(ks) if r % 2 == 0 else list(ks)[::-1]):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(5):
            ks[name].launch((warps // 8, 1, 1), (256, 1, 1), [frag, out, iters])
        e.record(); e.synchronize()
        res[name].append(s.elapsed_time(e) / 5)
for name, v in res.items():
    print(f"{name:5s}: min {min(v):.3f} ms, {macs / (min(v) * 1e-3) / 1e12:.2f} T MAC/s ({2 * macs / (min(v) * 1e-3) / 1e12:.1f} TFLOPS)")
