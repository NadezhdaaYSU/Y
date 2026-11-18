"""The int8 `mma.m8n8k16` fragment layout on a Turing GPU (sm_75), checked
on the device before any emitter code uses it.

    python3 probe_m8n8k16.py            # needs a CUDA GPU of sm_75 or later

Y's int8 GEMM issues `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`,
which needs sm_80. A Tesla T4 has `mma.m8n8k16` for int8 (sm_75). From the
PTX ISA's fragment tables (g = laneid >> 2, t = laneid & 3):

  m16n8k32  A: a0 = (row g,   k 4t..4t+3)   a1 = (row g+8, k 4t..)
               a2 = (row g,   k 16+4t..)    a3 = (row g+8, k 16+4t..)
            B: b0 = (k 4t.., n g)           b1 = (k 16+4t.., n g)
            D: d0,d1 = (row g, n 2t, 2t+1)  d2,d3 = (row g+8, n 2t, 2t+1)
  m8n8k16   A: a  = (row g, k 4t..4t+3)     B: b = (k 4t.., n g)
            D: d0,d1 = (row g, n 2t, 2t+1)

So one m16n8k32 is four m8n8k16 on the same registers:
  {d0,d1} += a0.b0;  {d0,d1} += a2.b1;  {d2,d3} += a1.b0;  {d2,d3} += a3.b1
(an integer sum is exact in any order). This checks, on the GPU, against a
CPU integer matmul, bit for bit:

  1. one m8n8k16 with the layout above;
  2. a 16x8x32 product as those four m8n8k16, its fragments loaded in the
     m16n8k32 register order Y's kernel uses;
  3. a control: (2) with a1 and a2 swapped, which must give wrong numbers.

The operands are full-range int8 (-128..127), with both extremes planted.
A wrong layout assembles, launches and returns plausible numbers, so only a
product against a reference settles it (tests/ptx_int8_mma_layout.rs did
the same for m16n8k32 before that kernel was written).
"""
import sys

import torch

PTX_HEAD = """
.version 6.5
.target sm_75
.address_size 64
"""

# One warp: D[8][8] = A[8][16] . B[8][16]^T  (B is [n][k], K-contiguous).
ONE = PTX_HEAD + """
.visible .entry probe_m8n8k16(.param .u64 pA, .param .u64 pB, .param .u64 pD)
{
    .reg .b64 %rd<16>;
    .reg .b32 %r<24>;
    ld.param.u64 %rd1, [pA];
    ld.param.u64 %rd2, [pB];
    ld.param.u64 %rd3, [pD];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    cvta.to.global.u64 %rd3, %rd3;
    mov.u32 %r1, %laneid;
    shr.u32 %r2, %r1, 2;            // g
    and.b32 %r3, %r1, 3;            // t
    // A (row g, k 4t..4t+3) at g*16 + 4t; B (n g, k 4t..) the same offset.
    shl.b32 %r4, %r2, 4;
    shl.b32 %r5, %r3, 2;
    add.u32 %r6, %r4, %r5;
    cvt.u64.u32 %rd4, %r6;
    add.u64 %rd5, %rd1, %rd4;
    add.u64 %rd6, %rd2, %rd4;
    ld.global.u32 %r7, [%rd5];
    ld.global.u32 %r8, [%rd6];
    mov.b32 %r9, 0;
    mov.b32 %r10, 0;
    mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%r9, %r10}, {%r7}, {%r8}, {%r9, %r10};
    // D (row g, n 2t) at (g*8 + 2t) * 4 bytes.
    shl.b32 %r11, %r2, 3;
    shl.b32 %r12, %r3, 1;
    add.u32 %r13, %r11, %r12;
    mul.wide.u32 %rd7, %r13, 4;
    add.u64 %rd8, %rd3, %rd7;
    st.global.v2.s32 [%rd8], {%r9, %r10};
    ret;
}
"""


def four(swap_a1_a2=False):
    """One warp: D[16][8] = A[16][32] . B[8][32]^T as four m8n8k16, with A
    and B loaded in the m16n8k32 fragment order."""
    a1, a2 = ("%r12", "%r11") if swap_a1_a2 else ("%r11", "%r12")
    return PTX_HEAD + f"""
.visible .entry probe_four(.param .u64 pA, .param .u64 pB, .param .u64 pD)
{{
    .reg .b64 %rd<24>;
    .reg .b32 %r<40>;
    ld.param.u64 %rd1, [pA];
    ld.param.u64 %rd2, [pB];
    ld.param.u64 %rd3, [pD];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    cvta.to.global.u64 %rd3, %rd3;
    mov.u32 %r1, %laneid;
    shr.u32 %r2, %r1, 2;            // g
    and.b32 %r3, %r1, 3;            // t
    // A is [16][32]: a0 at g*32 + 4t, a1 at (g+8)*32 + 4t = +256,
    // a2 at g*32 + 16 + 4t = +16, a3 at +272.
    shl.b32 %r4, %r2, 5;
    shl.b32 %r5, %r3, 2;
    add.u32 %r6, %r4, %r5;
    cvt.u64.u32 %rd4, %r6;
    add.u64 %rd5, %rd1, %rd4;
    ld.global.u32 %r10, [%rd5];
    ld.global.u32 %r11, [%rd5+256];
    ld.global.u32 %r12, [%rd5+16];
    ld.global.u32 %r13, [%rd5+272];
    // B is [8][32] (n-major): b0 at g*32 + 4t, b1 at +16.
    add.u64 %rd6, %rd2, %rd4;
    ld.global.u32 %r14, [%rd6];
    ld.global.u32 %r15, [%rd6+16];
    mov.b32 %r20, 0;
    mov.b32 %r21, 0;
    mov.b32 %r22, 0;
    mov.b32 %r23, 0;
    // {{d0,d1}} += a0.b0 + a2.b1;  {{d2,d3}} += a1.b0 + a3.b1
    mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {{%r20, %r21}}, {{%r10}}, {{%r14}}, {{%r20, %r21}};
    mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {{%r20, %r21}}, {{{a2}}}, {{%r15}}, {{%r20, %r21}};
    mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {{%r22, %r23}}, {{{a1}}}, {{%r14}}, {{%r22, %r23}};
    mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {{%r22, %r23}}, {{%r13}}, {{%r15}}, {{%r22, %r23}};
    // D is [16][8] int32: d0,d1 at (g*8 + 2t)*4, d2,d3 at +8 rows = +256 bytes.
    shl.b32 %r30, %r2, 3;
    shl.b32 %r31, %r3, 1;
    add.u32 %r32, %r30, %r31;
    mul.wide.u32 %rd7, %r32, 4;
    add.u64 %rd8, %rd3, %rd7;
    st.global.v2.s32 [%rd8], {{%r20, %r21}};
    st.global.v2.s32 [%rd8+256], {{%r22, %r23}};
    ret;
}}
"""


def kernel(ptx, name):
    """A `y_lang.torch_interop.TorchKernel` around hand-written PTX: its
    loader and launcher, without compiling Y source."""
    from y_lang.torch_interop import TorchKernel
    k = TorchKernel.__new__(TorchKernel)
    k.source, k.kernel_name, k.target_sm, k.ptx = None, name, "sm_75", ptx
    k._module = k._function = None
    return k


def operands(m, n, kk, seed):
    g = torch.Generator().manual_seed(seed)
    a = torch.randint(-128, 128, (m, kk), dtype=torch.int8, generator=g)
    b = torch.randint(-128, 128, (n, kk), dtype=torch.int8, generator=g)
    a[0, 0], a[-1, -1], b[0, 0], b[-1, -1] = -128, 127, -128, 127
    return a, b


def run(ptx, name, m, n, kk, seed):
    a, b = operands(m, n, kk, seed)
    want = a.to(torch.int64) @ b.to(torch.int64).T
    da, db = a.cuda(), b.cuda()
    dd = torch.full((m, n), -7, dtype=torch.int32, device="cuda")
    kernel(ptx, name).launch((1, 1, 1), (32, 1, 1), [da, db, dd])
    torch.cuda.synchronize()
    got = dd.cpu().to(torch.int64)
    bad = (got != want).nonzero().tolist()
    return bad, got, want


def main():
    cap = torch.cuda.get_device_capability()
    print(f"{torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]}, torch {torch.__version__}")
    if cap < (7, 5):
        raise SystemExit("*** needs sm_75 or later")
    ok = True
    for seed in (1, 2, 3):
        bad, _, _ = run(ONE, "probe_m8n8k16", 8, 8, 16, seed)
        print(f"  m8n8k16, seed {seed}: {64 - len(bad)}/64 outputs exact" + (f"; first wrong {bad[:3]}" if bad else ""))
        ok &= not bad
        bad, _, _ = run(four(), "probe_four", 16, 8, 32, seed)
        print(f"  four m8n8k16 as one m16n8k32, seed {seed}: {128 - len(bad)}/128 outputs exact"
              + (f"; first wrong {bad[:3]}" if bad else ""))
        ok &= not bad
        bad, _, _ = run(four(swap_a1_a2=True), "probe_four", 16, 8, 32, seed)
        print(f"  control (a1 and a2 swapped), seed {seed}: {len(bad)}/128 outputs wrong (must be > 0)")
        ok &= bool(bad)
    print("PROBE", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
