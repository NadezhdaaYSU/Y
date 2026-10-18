#!/usr/bin/env python3
"""mk_xbar.py <committed gemm_f16_512.ptx> <out.ptx>: the same kernel with its K loop
pipelined across the barrier (CUTLASS's multistage order): stage s+1's
wait_group/bar.sync sits between stage s's two k16 MMA steps, and stage s+1's
first fragments load while stage s's second step runs. Prologue and epilogue
are the committed text; only the loop between $GEMM_K_LOOP_0 and $GEMM_K_DONE_1
is replaced."""
import sys
src = open(sys.argv[1]).read()
head, rest = src.split("    $GEMM_K_LOOP_0:\n", 1)
_, tail = rest.split("    $GEMM_K_DONE_1:\n", 1)
assert head.rstrip().endswith("mov.u32 %r60, 0;"), head[-200:]

def ldsm(kk, a_base, b_base, regs, t):
    """kk: 0 or 16. regs: (B lo 4, B hi 4, A 4) register names; t: temp prefix."""
    blo, bhi, a = regs
    return f"""    add.u32 {t}0, %r21, %r25;
    add.u32 {t}1, %r23, {kk};
    mad.lo.u32 {t}2, {t}1, 72, {t}0;
    shl.b32 {t}3, {t}2, 1;
    add.u32 {t}4, {b_base}, {t}3;
    ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{{blo}}}, [{t}4];
    add.u32 {t}5, {t}4, 32;
    ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{{bhi}}}, [{t}5];
    add.u32 {t}6, %r20, %r23;
    add.u32 {t}7, %r25, {kk};
    mad.lo.u32 {t}8, {t}6, 40, {t}7;
    shl.b32 {t}9, {t}8, 1;
    add.u32 {t}10, {a_base}, {t}9;
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{a}}}, [{t}10];
"""

def mma(regs):
    blo, bhi, a = regs
    b = blo.split(",") + bhi.split(",")
    out = ""
    for i in range(4):
        f = ",".join(f"%f{4 * i + j}" for j in range(4))
        out += f"    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{{f}}}, {{{a}}}, {{{b[2 * i]},{b[2 * i + 1]}}}, {{{f}}};\n"
    return out

F0 = ("%r78,%r79,%r80,%r81", "%r88,%r89,%r90,%r91", "%r98,%r99,%r100,%r101")
F1 = ("%r108,%r109,%r110,%r111", "%r118,%r119,%r120,%r121", "%r128,%r129,%r130,%r131")
decl = "    .reg .b32 %x<40>;\n    .reg .b32 %y<40>;\n    .reg .b32 %z<40>;\n"
head = head.replace("    .reg .pred %p<16>;\n", "    .reg .pred %p<16>;\n" + decl, 1)
loop = ("    // [PIPELINED ACROSS THE BARRIER] stage 0 lands; its first fragments load before the loop\n"
        "    cp.async.wait_group 2;\n    bar.sync 0;\n"
        + ldsm(0, "%r26", "%r27", F0, "%x")
        + "    $GEMM_K_LOOP_0:\n"
        "    setp.ge.u32 %p3, %r60, 16;\n    @%p3 bra $GEMM_K_DONE_1;\n"
        "    rem.u32 %r61, %r60, 4;\n    add.u32 %r62, %r60, 3;\n    rem.u32 %r63, %r62, 4;\n"
        "    setp.lt.u32 %p4, %r62, 16;\n    mul.lo.u32 %r64, %r62, 32;\n"
        "    mad.lo.u32 %r65, %r63, 5120, %r26;\n    mad.lo.u32 %r66, %r63, 4608, %r27;\n"
        "    and.pred %p5, %p4, %p2;\n    selp.u32 %r67, 16, 0, %p5;\n"
        "    cvt.u64.u32 %rd29, %r64;\n    shl.b64 %rd28, %rd29, 1;\n    add.s64 %rd30, %rd7, %rd28;\n    add.u32 %r68, %r65, %r35;\n"
        "    cvt.u64.u32 %rd32, %r64;\n    shl.b64 %rd31, %rd32, 10;\n    add.s64 %rd33, %rd9, %rd31;\n    add.u32 %r69, %r66, %r43;\n"
        "    mad.lo.u32 %r70, %r61, 5120, %r26;\n    mad.lo.u32 %r71, %r61, 4608, %r27;\n"
        "    // stage s, second k16 step's fragments\n"
        + ldsm(16, "%r70", "%r71", F1, "%y")
        + "    cp.async.cg.shared.global [%r68], [%rd30], 16, %r67;\n"
        "    cp.async.cg.shared.global [%r69], [%rd33], 16, %r67;\n"
        "    cp.async.commit_group;\n"
        "    // stage s, first k16 step\n"
        + mma(F0)
        + "    // stage s+1 lands; every warp is done reading stage s\n"
        "    cp.async.wait_group 2;\n    bar.sync 0;\n"
        "    add.u32 %z20, %r60, 1;\n    rem.u32 %z21, %z20, 4;\n"
        "    mad.lo.u32 %z22, %z21, 5120, %r26;\n    mad.lo.u32 %z23, %z21, 4608, %r27;\n"
        + ldsm(0, "%z22", "%z23", F0, "%z")
        + "    // stage s, second k16 step\n"
        + mma(F1)
        + "    add.u32 %r60, %r60, 1;\n    bra $GEMM_K_LOOP_0;\n"
        "    $GEMM_K_DONE_1:\n")
open(sys.argv[2], "w").write(head + "    $GEMM_K_LOOP_0:\n".join([""]) + loop + tail)
