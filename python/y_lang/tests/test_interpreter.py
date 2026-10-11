"""The CPU interpreter runs a Y kernel's PTX and stops at what the GPU gets wrong.

Until 2026-10-11 it ran nothing: `TorchKernel.launch` in debug mode printed
"Executed safely under software bounds validation" and wrote no output, and
`debug_interpret` returned "SUCCESS" for a kernel name alone. Every test here
fails against that version. The differential tests compare the interpreter
with the GPU on the same PTX, bit for bit where the instructions are exact.
"""

import os
import subprocess
import unittest
from pathlib import Path

import torch

from y_lang.interpreter import CPUInterpreter, CPUInterpreterError, debug_interpret
from y_lang.torch_interop import TorchKernel

REPO = Path(__file__).resolve().parents[3]
Y_BIN = REPO / "target" / "release" / "Y"

VEC_ADD = """
kernel vec_add(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, N: I32) {
    let idx: I32 = global_thread_id() * 4;
    if idx < N {
        vec_add_v4(A[idx], B[idx], C[idx]);
    }
}
"""

# The launch the old head-to-head benchmark made: no guard, 100,352 lanes of
# 100,000-element buffers.
VEC_ADD_UNGUARDED = """
kernel vec_add(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, N: I32) {
    let idx: I32 = global_thread_id() * 4;
    vec_add_v4(A[idx], B[idx], C[idx]);
}
"""

RMSNORM = """
kernel rms(X: GlobalMemory<F32>, W: GlobalMemory<F32>, Out: GlobalMemory<F32>, hidden_dim: I32, eps: F32) {
    let row: I32 = block_idx_x();
    let col: I32 = thread_idx_x() * 4;
    let idx: I32 = row * hidden_dim + col;
    rmsnorm_v4(X[idx], W[col], Out[idx], hidden_dim, eps);
}
"""

FMA = """
kernel fmak(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, Out: GlobalMemory<F32>) {
    let i: I32 = global_thread_id();
    Out[i] = A[i] * B[i] + C[i];
}
"""

INT_OPS = """
kernel iops(A: GlobalMemory<I32>, B: GlobalMemory<I32>, U: GlobalMemory<U32>, Out: GlobalMemory<I32>, OutU: GlobalMemory<U32>, OutL: GlobalMemory<I64>) {
    let i: I32 = global_thread_id();
    let a: I32 = A[i];
    let b: I32 = B[i];
    let u: U32 = U[i];
    Out[i] = (a / b) + (a % b) * 3 - (a >> 3) + (a << 2);
    OutU[i] = (u / 7) + (u % 13) + (u >> 5) + (u * u);
    let l: I64 = a;
    OutL[i] = l * l * 3 - l;
}
"""


def kernel(src, name, target="sm_80"):
    return TorchKernel(src, kernel_name=name, target_sm=target)


def interpret(k, grid, block, args):
    return CPUInterpreter(k.kernel_name, ptx=k.ptx).run(grid, block, args)


class PtxKernel(TorchKernel):
    """A TorchKernel over PTX that did not come from Y source."""

    def __init__(self, ptx, name):
        self.source, self.kernel_name, self.target_sm, self.ptx = None, name, None, ptx
        self._module = self._function = None


class RunsTheKernel(unittest.TestCase):
    def test_vector_add_is_computed_exactly(self):
        n = 100_000
        a, b = torch.randn(n), torch.randn(n)
        c = torch.full((n,), float("nan"))
        interpret(kernel(VEC_ADD, "vec_add"), ((n // 4 + 127) // 128, 1, 1), (128, 1, 1), [a, b, c, n])
        self.assertTrue(torch.equal(c, a + b))

    def test_debug_mode_launches_write_their_outputs(self):
        k = kernel(VEC_ADD, "vec_add")
        a, b, c = torch.randn(1000), torch.randn(1000), torch.zeros(1000)
        saved = os.environ.get("Y_INTERPRETER")
        os.environ["Y_INTERPRETER"] = "1"
        try:
            k.launch((2, 1, 1), (128, 1, 1), [a, b, c, 1000])
        finally:
            if saved is None:
                del os.environ["Y_INTERPRETER"]
            else:
                os.environ["Y_INTERPRETER"] = saved
        self.assertTrue(torch.equal(c, a + b), "the debug-mode launch did not compute the kernel")

    def test_rmsnorm_matches_float64(self):
        x, w = torch.randn(8, 512), torch.randn(512)
        out = torch.zeros_like(x)
        interpret(kernel(RMSNORM, "rms"), (8, 1, 1), (128, 1, 1), [x, w, out, 512, 0.5])
        xd = x.double()
        ref = xd * torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + 0.5) * w.double()
        self.assertLess(((out.double() - ref).abs().max() / ref.abs().max()).item(), 1e-6)

    def test_a_warp_reduction_sums_each_warp(self):
        src = """
kernel k(A: GlobalMemory<F32>, Out: GlobalMemory<F32>) {
    let t: I32 = global_thread_id();
    let s: F32 = warp_reduce_sum(A[t]);
    Out[t] = s;
}
"""
        out = torch.zeros(64)
        interpret(kernel(src, "k"), (1, 1, 1), (64, 1, 1), [torch.arange(64, dtype=torch.float32), out])
        self.assertEqual(out[0].item(), sum(range(32)))
        self.assertEqual(out[63].item(), sum(range(32, 64)))

    def test_fma_rounds_once(self):
        # a*b = 1 + 2^-11 + 2^-24 sits exactly between two floats; c = +-2^-60
        # decides the rounding. Rounded through float64 first, both cases
        # would tie to even and give 1 + 2^-11.
        k = kernel(FMA, "fmak")
        a = torch.tensor([1 + 2**-12, 1 + 2**-12])
        c = torch.tensor([2**-60, -2**-60])
        out = torch.zeros(2)
        interpret(k, (1, 1, 1), (2, 1, 1), [a, a.clone(), c, out])
        self.assertEqual(out[0].item(), 1 + 2**-11 + 2**-23)
        self.assertEqual(out[1].item(), 1 + 2**-11)

    def test_debug_interpret_runs_source_and_refuses_a_bare_name(self):
        a, b, c = torch.randn(256), torch.randn(256), torch.zeros(256)
        debug_interpret(VEC_ADD, (1, 1, 1), (64, 1, 1), [a, b, c, 256], target_sm="sm_80")
        self.assertTrue(torch.equal(c, a + b))
        with self.assertRaises(CPUInterpreterError):
            debug_interpret("dummy", (1, 1, 1), (256, 1, 1), [torch.zeros(4)])


class StopsAtWhatTheGpuGetsWrong(unittest.TestCase):
    def assert_stops(self, src, name, grid, block, args, *phrases, target="sm_80"):
        with self.assertRaises(CPUInterpreterError) as cm:
            interpret(kernel(src, name, target), grid, block, args)
        for p in phrases:
            self.assertIn(p, str(cm.exception))
        return str(cm.exception)

    def test_out_of_bounds_access_names_buffer_thread_and_line(self):
        n = 100_000
        msg = self.assert_stops(VEC_ADD_UNGUARDED, "vec_add", ((n // 4 + 127) // 128, 1, 1), (128, 1, 1),
                                [torch.randn(n), torch.randn(n), torch.zeros(n), n],
                                "out-of-bounds", "past its end", "PTX line")
        self.assertIn("block (195, 0, 0)", msg)

    def test_trap_is_reported(self):
        x, w = torch.randn(8, 1024), torch.randn(1024)
        self.assert_stops(RMSNORM, "rms", (8, 1, 1), (128, 1, 1), [x, w, torch.zeros_like(x), 1024, 1e-5], "`trap`")

    def test_a_runtime_bounds_check_traps(self):
        src = """
kernel k(A: GlobalMemory<I32>, Out: GlobalMemory<I32>) {
    @bounds(min=0, max=3)
    let i: I32 = A[0];
    Out[0] = i + 1;
}
"""
        self.assert_stops(src, "k", (1, 1, 1), (1, 1, 1),
                          [torch.tensor([1000000], dtype=torch.int32), torch.zeros(1, dtype=torch.int32)], "`trap`")

    def test_a_barrier_in_divergent_code_is_reported(self):
        src = """
kernel k(Out: GlobalMemory<I32>) {
    let t: I32 = thread_idx_x();
    if t < 16 {
        barrier_sync();
    }
    Out[t] = t;
}
"""
        self.assert_stops(src, "k", (1, 1, 1), (64, 1, 1), [torch.zeros(64, dtype=torch.int32)], "only part of a block")

    def test_a_shuffle_in_divergent_code_is_reported(self):
        src = """
kernel k(A: GlobalMemory<F32>, Out: GlobalMemory<F32>) {
    let t: I32 = global_thread_id();
    if t < 20 {
        let s: F32 = warp_reduce_sum(A[t]);
        Out[t] = s;
    }
}
"""
        self.assert_stops(src, "k", (1, 1, 1), (32, 1, 1), [torch.zeros(32), torch.zeros(32)], "shfl.sync")

    def test_a_shuffle_whose_mask_names_absent_lanes_is_reported(self):
        # Every lane here reads a partner that is present (lane ^ 1 < 20), so
        # only the mask - all 32 lanes - says the instruction is undefined.
        src = """
kernel k(A: GlobalMemory<F32>, Out: GlobalMemory<F32>) {
    let t: I32 = global_thread_id();
    if t < 20 {
        let s: F32 = shfl_sync_bfly(A[t], 1);
        Out[t] = s;
    }
}
"""
        self.assert_stops(src, "k", (1, 1, 1), (32, 1, 1), [torch.zeros(32), torch.zeros(32)], "named in the mask")

    def test_conflicting_stores_are_reported(self):
        src = """
kernel k(Out: GlobalMemory<I32>) {
    let t: I32 = thread_idx_x();
    Out[0] = t;
}
"""
        self.assert_stops(src, "k", (1, 1, 1), (32, 1, 1), [torch.zeros(4, dtype=torch.int32)], "different values")

    def test_unmodelled_forms_are_refused(self):
        # Each changes what the instruction does; running it without the
        # modifier, or bar.arrive as a full barrier, would be a guess.
        for line, phrase in [("bar.arrive 0, 64;", "modifier .arrive"),
                             ("barrier.sync 0;", "without .aligned is not modelled"),
                             ("fma.rn.relu.f16 %h0, %h0, %h0, %h0;", "modifier .relu"),
                             ("ld.shared::cluster.u32 %r0, [%r0];", "modifier .shared::cluster"),
                             ("cvt.rn.satfinite.e4m3x2.f32 %h0, %f0, %f0;", "not modelled"),
                             # Not a refusal: with no buffer at all, an address is in none.
                             ("ld.global.u32 %r1, [%rd0];", "in no buffer the kernel was given")]:
            ptx = (".version 7.8\n.target sm_90\n.address_size 64\n.visible .entry k()\n{\n"
                   ".reg .b32 %r<2>; .reg .b16 %h<2>; .reg .f32 %f<2>; .reg .b64 %rd<2>;\nmov.u32 %r0, 0;\n"
                   f"mov.b16 %h0, 0;\nmov.f32 %f0, 0f00000000;\nmov.u64 %rd0, 4096;\n{line}\nret;\n}}\n")
            with self.subTest(line=line):
                with self.assertRaises(CPUInterpreterError) as cm:
                    CPUInterpreter("k", ptx=ptx).run((1, 1, 1), (32, 1, 1), [])
                self.assertIn(phrase, str(cm.exception))

    def test_tensor_core_kernels_are_refused_not_guessed(self):
        src = """
@tile(256, 256, 256)
kernel gemm_fp8_256(A: GlobalMemory<F32>, B: GlobalMemory<F32>, scale_a: F32, scale_b: F32, C: GlobalMemory<F32>) {
}
"""
        z = lambda: torch.zeros(256 * 256)
        self.assert_stops(src, "gemm_fp8_256", (4, 4, 1), (128, 1, 1), [z(), z(), 1.0, 1.0, z()],
                          "not modelled", target="sm_89")


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device for the differential")
class MatchesTheGpu(unittest.TestCase):
    def both(self, k, grid, block, args, exact=True, tol=0.0):
        gpu = [a.cuda() if torch.is_tensor(a) else a for a in args]
        k.launch(grid, block, gpu)
        torch.cuda.synchronize()
        interpret(k, grid, block, args)
        for i, (g, c) in enumerate(zip(gpu, args)):
            if not torch.is_tensor(g):
                continue
            g = g.cpu()
            if exact:
                self.assertTrue(torch.equal(g.view(torch.uint8), c.view(torch.uint8)), f"argument {i} differs from the GPU")
            else:
                err = ((g.double() - c.double()).abs().max() / c.double().abs().max().clamp_min(1e-30)).item()
                self.assertLessEqual(err, tol, f"argument {i}")

    def test_integer_arithmetic(self):
        n = 4096
        gen = torch.Generator().manual_seed(2)
        a = torch.randint(-2**31, 2**31 - 1, (n,), dtype=torch.int32, generator=gen)
        b = torch.randint(1, 1000, (n,), dtype=torch.int32, generator=gen) * (torch.randint(0, 2, (n,), generator=gen) * 2 - 1).int()
        u = torch.randint(0, 2**31 - 1, (n,), dtype=torch.int32, generator=gen).view(torch.uint32)
        out = torch.zeros(n, dtype=torch.int32)
        self.both(kernel(INT_OPS, "iops"), (n // 128, 1, 1), (128, 1, 1),
                  [a, b, u, out, torch.zeros(n, dtype=torch.int32).view(torch.uint32), torch.zeros(n, dtype=torch.int64)])
        self.assertTrue(bool((out != 0).any()))

    def test_fma(self):
        n = 1 << 16
        gen = torch.Generator().manual_seed(6)
        a, b, c = torch.randn(n, generator=gen) * 1e3, torch.randn(n, generator=gen), torch.randn(n, generator=gen) * 1e-3
        a[:2], b[:2], c[:2] = 1 + 2**-12, 1 + 2**-12, torch.tensor([2**-60, -2**-60])
        self.both(kernel(FMA, "fmak"), (n // 256, 1, 1), (256, 1, 1), [a, b, c, torch.zeros(n)])

    def test_carry_chains(self):
        ptx = (REPO / "tests" / "bn254_fr_mul_fast.ptx").read_text()
        n = 512
        gen = torch.Generator().manual_seed(7)
        limbs = lambda: torch.randint(0, 2**31, (8 * n,), dtype=torch.int64, generator=gen).to(torch.int32)
        out = torch.zeros(8 * n, dtype=torch.int32)
        self.both(PtxKernel(ptx, "bn254_fr_mul_fast"), ((n + 255) // 256, 1, 1), (256, 1, 1), [limbs(), limbs(), out, n])
        self.assertTrue(bool((out != 0).any()))

    def test_approximate_functions_are_close(self):
        src = """
kernel swiglu(Gate: GlobalMemory<F32>, Up: GlobalMemory<F32>, Out: GlobalMemory<F32>, N: I32) {
    let idx: I32 = global_thread_id() * 4;
    if idx < N {
        swiglu_v4(Gate[idx], Up[idx], Out[idx]);
    }
}
"""
        n = 10000
        gen = torch.Generator().manual_seed(8)
        self.both(kernel(src, "swiglu"), ((n // 4 + 127) // 128, 1, 1), (128, 1, 1),
                  [torch.randn(n, generator=gen), torch.randn(n, generator=gen), torch.zeros(n), n], exact=False, tol=1e-6)

    def test_cp_async_reads_only_its_source_size(self):
        # Y's GEMM zero-fills tiles past the matrix edge with a register
        # src-size of 0 (`selp.u32 %r, 16, 0, %in_bounds`). Such a thread reads
        # nothing: its address may lie past the buffer, and when it lies inside,
        # the tile still gets zeros. The interpreter used to copy all 16 bytes.
        def ptx(fill):
            return f"""
.version 7.8
.target sm_80
.address_size 64
.visible .entry cpz(.param .u64 src, .param .u64 out, .param .u32 n)
{{
    .reg .pred %p<2>; .reg .b32 %r<8>; .reg .b64 %rd<8>; .reg .f32 %f<4>;
    .shared .align 16 .b8 tile[512];
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    ld.param.u32 %r2, [n];
    setp.lt.u32 %p0, %r1, %r2;
    setp.ge.u32 %p1, %r1, %r2;
    selp.u32 %r3, 16, 0, %p0;
    ld.param.u64 %rd0, [src];
    mul.wide.u32 %rd1, %r1, 4;
    add.s64 %rd2, %rd0, %rd1;
    mov.u32 %r4, tile;
    shl.b32 %r5, %r0, 4;
    add.u32 %r6, %r4, %r5;
    {fill}
    cp.async.commit_group;
    cp.async.wait_group 0;
    bar.sync 0;
    ld.shared.v4.f32 {{%f0, %f1, %f2, %f3}}, [%r6];
    ld.param.u64 %rd3, [out];
    add.s64 %rd4, %rd3, %rd1;
    st.global.v4.f32 [%rd4], {{%f0, %f1, %f2, %f3}};
    ret;
}}
"""
        for fill in ["cp.async.cg.shared.global [%r6], [%rd2], 16, %r3;",      # src-size register
                     "cp.async.cg.shared.global [%r6], [%rd2], 16, %p1;"]:     # ignore-src predicate
            for size in (40, 128):              # masked addresses past the end / inside the buffer
                with self.subTest(fill=fill, buffer=size):
                    src, out = torch.arange(1, size + 1, dtype=torch.float32), torch.full((128,), -1.0)
                    self.both(PtxKernel(ptx(fill), "cpz"), (1, 1, 1), (32, 1, 1), [src, out, 40])
                    self.assertTrue(torch.equal(out, torch.cat([src[:40], torch.zeros(88)])))

    def test_narrow_loads_extend_by_their_type(self):
        ptx = """
.version 7.8
.target sm_80
.address_size 64
.visible .entry ldx(.param .u64 a, .param .u64 out)
{
    .reg .b32 %r<4>; .reg .b64 %rd<12>;
    mov.u32 %r0, %tid.x;
    ld.param.u64 %rd0, [a];
    mul.wide.u32 %rd1, %r0, 4;
    add.s64 %rd2, %rd0, %rd1;
    ld.global.s32 %rd3, [%rd2];
    ld.global.u32 %rd4, [%rd2];
    ld.global.s16 %rd5, [%rd2];
    ld.global.s8 %r1, [%rd2];
    cvt.u64.u32 %rd6, %r1;
    ld.param.u64 %rd7, [out];
    mul.wide.u32 %rd8, %r0, 32;
    add.s64 %rd9, %rd7, %rd8;
    st.global.u64 [%rd9], %rd3;
    st.global.u64 [%rd9+8], %rd4;
    st.global.u64 [%rd9+16], %rd5;
    st.global.u64 [%rd9+24], %rd6;
    ret;
}
"""
        a = torch.tensor([-1, -2**31, 2**31 - 1, 0x00008000, 0x7FFF0080, 5, -300, 0], dtype=torch.int32)
        out = torch.zeros(4 * 8, dtype=torch.int64)
        self.both(PtxKernel(ptx, "ldx"), (1, 1, 1), (8, 1, 1), [a, out])
        got = out.view(8, 4)
        self.assertEqual(got[0].tolist(), [-1, 2**32 - 1, -1, 2**32 - 1])
        self.assertEqual(got[3].tolist(), [0x8000, 0x8000, -0x8000, 0])

    @unittest.skipUnless(Y_BIN.exists(), "needs target/release/Y for --emit-attention-ptx")
    def test_exact_attention_and_its_temperature_trap(self):
        env = dict(os.environ, Y_NO_CERTIFICATE="1")
        ptx = subprocess.run([str(Y_BIN), "--emit-attention-ptx", "32", "256"], capture_output=True, env=env,
                             check=True).stdout.decode()
        B, S, D = 2, 256, 32
        gen = torch.Generator().manual_seed(1)
        q, k, v = (torch.randint(-127, 128, (n,), dtype=torch.int8, generator=gen) for n in (B * D, S * D, S * D))
        scores, m = torch.zeros(B * S, dtype=torch.int32), torch.zeros(B, dtype=torch.int32)
        self.both(PtxKernel(ptx, "attn_scores"), ((S + 127) // 128, B, 1), (128, 1, 1), [q, k, scores, m])
        accum_args = [scores, v, m, torch.zeros(B, dtype=torch.int64), torch.zeros(B * D, dtype=torch.int64),
                      torch.zeros(B * S, dtype=torch.int32), (8 << 16) + 1]
        self.both(PtxKernel(ptx, "attn_accum"), (4, B, 3), (64, 1, 1), accum_args)
        self.assertTrue(bool((accum_args[4] != 0).any()))
        accum_args[-1] = 0
        with self.assertRaises(CPUInterpreterError) as cm:
            interpret(PtxKernel(ptx, "attn_accum"), (4, B, 3), (64, 1, 1), accum_args)
        self.assertIn("`trap`", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
