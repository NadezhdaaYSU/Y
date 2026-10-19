"""probe.py: which part of Y's cp.async tile copy costs 1.5x the ideal L2
sectors. Each variant copies a 64 x 1024 f16 matrix through shared memory in
32 K-tiles with Y's global mapping (A: thread t -> row t/4, 16-byte chunk
t%4, a warp covering 8 rows x 64 B; B: row t/8, chunk t%8, 4 rows x 128 B),
varying only the shared-memory row stride (Y's padded 80 / 144 B, dense
64 / 128 B, dense with Y's padding removed by an XOR swizzle) and the copy
form (with and without the src-size operand Y emits). Each copied tile is
summed into the output so the copies are checked, not dead.

    python3 probe.py            # runs every variant once, checks each
"""
import sys

import torch

from y_lang.torch_interop import TorchKernel

ROWS, K, KT = 64, 1024, 32  # A-like: 64 rows of K f16, K-tiles of 32


def ptx(name, row_bytes, chunks_per_row, dst_stride, swizzle, srcsize):
    """One warp-group of 256 threads; each copies one 16-byte chunk a tile."""
    tiles = K * 2 // row_bytes  # tile row segment = row_bytes
    rows_per_tile = 256 // chunks_per_row
    lines = [
        ".version 7.8", ".target sm_89", ".address_size 64",
        f".visible .entry {name}(.param .u64 A, .param .u64 O)", "{",
        "    .reg .b32 %r<40>;", "    .reg .b64 %rd<20>;", "    .reg .f32 %f<8>;", "    .reg .pred %p<4>;",
        f"    .shared .align 128 .b8 sm[{rows_per_tile * dst_stride}];",
        "    ld.param.u64 %rd0, [A];", "    ld.param.u64 %rd1, [O];",
        "    cvta.to.global.u64 %rd0, %rd0;", "    cvta.to.global.u64 %rd1, %rd1;",
        "    mov.u32 %r0, %tid.x;",
        f"    shr.u32 %r1, %r0, {chunks_per_row.bit_length() - 1};",  # row
        f"    and.b32 %r2, %r0, {chunks_per_row - 1};",  # chunk
        f"    mul.lo.u32 %r3, %r1, {K * 2};",  # global row offset (bytes)
        "    shl.b32 %r4, %r2, 4;",
        "    add.u32 %r3, %r3, %r4;",
        "    cvt.u64.u32 %rd2, %r3;", "    add.s64 %rd2, %rd0, %rd2;",
    ]
    if swizzle:
        # chunk ^ (row % chunks_per_row): dense rows, no padding.
        lines += [f"    and.b32 %r5, %r1, {chunks_per_row - 1};", "    xor.b32 %r6, %r2, %r5;", "    shl.b32 %r6, %r6, 4;"]
    else:
        lines += ["    shl.b32 %r6, %r2, 4;"]
    lines += [
        f"    mad.lo.u32 %r7, %r1, {dst_stride}, %r6;",
        "    mov.u32 %r8, sm;", "    add.u32 %r7, %r7, %r8;",
        "    mov.u32 %r9, 16;",
        "    mov.f32 %f0, 0f00000000;",
        "    mov.u32 %r10, 0;",
        "LOOP:",
        "    .pragma \"nounroll\";",
        f"    cp.async.cg.shared.global [%r7], [%rd2], 16{', %r9' if srcsize else ''};",
        "    cp.async.commit_group;",
        "    cp.async.wait_group 0;",
        "    bar.sync 0;",
        "    ld.shared.v4.u32 {%r11, %r12, %r13, %r14}, [%r7];",
        "    add.u32 %r15, %r11, %r12;", "    add.u32 %r15, %r15, %r13;", "    add.u32 %r15, %r15, %r14;",
        "    cvt.rn.f32.u32 %f1, %r15;", "    add.f32 %f0, %f0, %f1;",
        "    bar.sync 0;",
        f"    add.s64 %rd2, %rd2, {row_bytes};",
        "    add.u32 %r10, %r10, 1;",
        f"    setp.lt.u32 %p0, %r10, {tiles};",
        "    @%p0 bra LOOP;",
        "    mul.wide.u32 %rd3, %r0, 4;", "    add.s64 %rd3, %rd1, %rd3;",
        "    st.global.f32 [%rd3], %f0;",
        "    ret;", "}",
    ]
    return "\n".join(lines) + "\n"


VARIANTS = {
    # name: (row_bytes, chunks_per_row, dst_stride, swizzle, srcsize)
    "A_y_pad80_srcsize": (64, 4, 80, False, True),   # Y's A copy as emitted
    "A_pad80": (64, 4, 80, False, False),
    "A_dense64": (64, 4, 64, False, False),
    "A_dense64_srcsize": (64, 4, 64, False, True),
    "A_swz64": (64, 4, 64, True, False),
    "B_y_pad144_srcsize": (128, 8, 144, False, True),  # Y's B copy as emitted
    "B_pad144": (128, 8, 144, False, False),
    "B_dense128": (128, 8, 128, False, False),
    "B_swz128": (128, 8, 128, True, False),
}


def expected(a_bytes, row_bytes, cpr, swizzle):
    """What each thread sums: the four u32 words of the 16-byte shared chunk
    it reads back (its own destination), over every tile."""
    words = a_bytes.view(torch.int32).view(ROWS, K * 2 // 4).to(torch.int64)  # u32 view
    words = words & 0xFFFFFFFF
    tid = torch.arange(256, device=a_bytes.device)
    row, chunk = tid // cpr, tid % cpr
    out = torch.zeros(256, dtype=torch.float64, device=a_bytes.device)
    rows_per_tile = 256 // cpr
    for t in range(K * 2 // row_bytes):
        # thread reads back its own destination; with the XOR swizzle the
        # destination holds the chunk the thread itself copied, so either
        # way it reads the chunk it wrote: global row `row`, chunk `chunk`.
        base = t * row_bytes // 4 + chunk * 4
        s = sum(words[row % ROWS, base + i] for i in range(4)) & 0xFFFFFFFF
        out += s.double()
    return out, rows_per_tile


def main():
    torch.manual_seed(0)
    a = torch.randint(0, 2, (ROWS, K), dtype=torch.int16, device="cuda")  # small words
    ok = True
    for name, (rb, cpr, stride, swz, ss) in VARIANTS.items():
        k = TorchKernel.__new__(TorchKernel)
        k.source, k.kernel_name, k.target_sm = None, name, "sm_89"
        k.ptx = ptx(name, rb, cpr, stride, swz, ss)
        k._module = k._function = None
        k._ensure_loaded()
        o = torch.full((256,), float("nan"), device="cuda")
        k.launch((1, 1, 1), (256, 1, 1), [a, o])
        torch.cuda.synchronize()
        want, _ = expected(a, rb, cpr, swz)
        good = bool(torch.equal(o.double(), want))
        ok &= good
        print(name, "exact", good, flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
