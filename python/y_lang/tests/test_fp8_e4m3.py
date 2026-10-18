"""`E4M3` buffers and the `@tile` GEMM over e4m3 bytes, run on the GPU.

`GlobalMemory<E4M3>` holds FP8 e4m3 (`torch.float8_e4m3fn`): storing an F32
rounds to nearest-even and saturates at +-448 (NaN stays NaN), loading widens
exactly. Every comparison is of values with their sign, or of bytes, against
torch's own conversion and an independent round-half-even reference.

The GEMM takes A [M][K] and B [N][K] e4m3 and returns
C = scale_a * scale_b * (A @ B^T) in float32. Inputs are small integers, exact
in e4m3, so every product and partial sum is exact and C must equal the
float64 product bit for bit, whatever the tile. That cannot see how precisely
partial sums are added, which a separate test bounds. C starts as NaN, so a tile no
CTA writes fails too. Tiles are pinned with Y_FP8_E4M3_TILE.
"""

import contextlib
import os
import re
import shutil
import subprocess
import sys
import unittest

import torch

from y_lang.torch_interop import TorchKernel

QUANT = """
kernel quant_e4m3(X: GlobalMemory<F32>, inv_s: F32, Q: GlobalMemory<E4M3>, n: I32) {
    let idx: I32 = global_thread_id();
    if idx < n {
        Q[idx] = X[idx] * inv_s;
    }
}
"""

# X [rows][cols] f32 to Q [cols][rows] e4m3, rows a multiple of 4. A warp
# covers 8 columns and 16 rows: each lane reads 4 rows of one column and
# writes their 4 bytes with one 4-byte store.
QUANT_T = """
kernel quant_e4m3_t(X: GlobalMemory<F32>, inv_s: F32, Q: GlobalMemory<E4M3>, rows: I32, cols: I32) {
    let t: I32 = global_thread_id();
    let warp: I32 = t / 32;
    let lane: I32 = t - warp * 32;
    let groups: I32 = (cols + 7) / 8;
    let rg: I32 = warp / groups;
    let c: I32 = (warp - rg * groups) * 8 + lane - (lane / 8) * 8;
    let r: I32 = rg * 16 + (lane / 8) * 4;
    if r < rows {
        if c < cols {
            let v0: F32 = X[r * cols + c] * inv_s;
            let v1: F32 = X[(r + 1) * cols + c] * inv_s;
            let v2: F32 = X[(r + 2) * cols + c] * inv_s;
            let v3: F32 = X[(r + 3) * cols + c] * inv_s;
            block_ptr2d_store_v4(Q, c, r, rows, cols, rows, v0, v1, v2, v3);
        }
    }
}
"""

WIDEN = """
kernel widen_e4m3(Q: GlobalMemory<E4M3>, X: GlobalMemory<F32>, n: I32) {
    let idx: I32 = global_thread_id();
    if idx < n {
        X[idx] = Q[idx];
    }
}
"""

STORE_F64 = """
kernel store_f64(V: GlobalMemory<F64>, Q: GlobalMemory<E4M3>, n: I32) {
    let idx: I32 = global_thread_id();
    if idx < n {
        let v: F64 = V[idx];
        Q[idx] = v;
    }
}
"""

# 1.0625 is the tie between e4m3's 1.0 and 1.125; this literal is above it by
# far less than a float32 ulp, so rounding it to float32 first lands on the
# tie, which rounds to even: 1.0. Rounded once, it is 1.125.
LITERAL = """
kernel store_literal(Q: GlobalMemory<E4M3>) {
    Q[0] = 1.0625000001;
    Q[1] = -1.0625000001;
    Q[2] = 1.0625;
}
"""

# Four e4m3 bytes in one store and one load: row r of a [rows][8] matrix,
# columns c .. c+3; a row past `rows` is neither read nor written.
V4 = """
kernel v4_e4m3(Q: GlobalMemory<E4M3>, X: GlobalMemory<F32>, Out: GlobalMemory<F32>, rows: I32) {
    let t: I32 = global_thread_id();
    let r: I32 = t / 2;
    let c: I32 = (t - r * 2) * 4;
    let i: I32 = t * 4;
    block_ptr2d_store_v4(Q, r, c, 8, rows, 8, X[i], X[i + 1], X[i + 2], X[i + 3]);
    let v: U32x4 = block_ptr2d_load_v4(Q, r, c, 8, rows, 8);
    Out[i] = v.x;
    Out[i + 1] = v.y;
    Out[i + 2] = v.z;
    Out[i + 3] = v.w;
}
"""


def gpu_has_fp8():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


@contextlib.contextmanager
def tile(spec):
    old = os.environ.get("Y_FP8_E4M3_TILE")
    if spec is None:
        os.environ.pop("Y_FP8_E4M3_TILE", None)
    else:
        os.environ["Y_FP8_E4M3_TILE"] = spec
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("Y_FP8_E4M3_TILE", None)
        else:
            os.environ["Y_FP8_E4M3_TILE"] = old


def kernel(src, name):
    k = TorchKernel(src, kernel_name=name, target_sm="sm_89")
    k._ensure_loaded()
    return k


def blocks(threads):
    return ((threads + 255) // 256, 1, 1), (256, 1, 1)


def reference(x64):
    """Round half to even to e4m3 (4 significant bits, exponent >= -6),
    saturating at 448; NaN stays NaN. float64, exact for these inputs."""
    a = x64.abs()
    e = torch.floor(torch.log2(a)).clamp(min=-6, max=8)
    q = torch.exp2(e - 3)
    r = torch.round(a / q) * q
    r = torch.where(a >= 448, torch.full_like(a, 448.0), r)
    r = torch.where(torch.isnan(a), a, r)
    return torch.copysign(r, x64)


def same(got, want):
    """Equal values with equal signs (so -0 is not +0); NaN matches NaN."""
    nan = torch.isnan(want)
    eq = (got == want) & (torch.signbit(got) == torch.signbit(want))
    return torch.where(nan, torch.isnan(got), eq)


def as_e4m3(q):
    return q.view(torch.float8_e4m3fn).double()


@unittest.skipUnless(gpu_has_fp8(), "needs an sm_89 or later GPU")
class E4m3Conversions(unittest.TestCase):
    def test_a_store_rounds_to_nearest_even_and_saturates(self):
        # Every 65537th float32 bit pattern (all exponents, both signs), and
        # the 9 float32 values around every e4m3 value and every midpoint
        # between neighbours, where the rounding decisions are.
        sample = torch.arange(0, 1 << 32, 65537, device="cuda", dtype=torch.int64)
        e4 = torch.arange(0, 256, device="cuda", dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).float()
        e4 = e4[~torch.isnan(e4)].sort().values
        points = torch.cat([e4, (e4[:-1] + e4[1:]) / 2, torch.tensor([464.0, 480.0, 1e30], device="cuda")])
        near = (points.view(torch.int32).long().unsqueeze(1) + torch.arange(-4, 5, device="cuda")).flatten()
        bits = torch.cat([sample, near & 0xFFFFFFFF])
        x = (bits - (1 << 32) * (bits >= (1 << 31))).to(torch.int32).view(torch.float32)
        x = torch.cat([x, torch.tensor([float("inf"), -float("inf"), float("nan"), -0.0, 0.0], device="cuda")])
        q = torch.empty(x.numel(), device="cuda", dtype=torch.uint8)
        kernel(QUANT, "quant_e4m3").launch(*blocks(x.numel()), [x, 1.0, q, x.numel()])
        torch.cuda.synchronize()
        got = as_e4m3(q)
        for name, want in (("torch", x.to(torch.float8_e4m3fn).double()), ("the reference", reference(x.double()))):
            bad = ~same(got, want)
            if bad.any():
                self.fail(f"{int(bad.sum())} of {x.numel()} differ from {name}; first "
                          f"{x[bad][0].item()!r} -> {got[bad][0].item()} vs {want[bad][0].item()}")

    def test_a_load_widens_every_byte_exactly(self):
        q = torch.arange(256, device="cuda", dtype=torch.int32).to(torch.uint8)
        x = torch.full((256,), 12345.0, device="cuda")
        kernel(WIDEN, "widen_e4m3").launch(*blocks(256), [q, x, 256])
        torch.cuda.synchronize()
        bad = ~same(x.double(), as_e4m3(q))
        self.assertFalse(bool(bad.any()), f"bytes {q[bad].tolist()} widen to {x[bad].tolist()}")

    def test_an_f64_is_rounded_once(self):
        # Every midpoint between e4m3 neighbours, and 2^-40 relative above and
        # below it: inside one float32 ulp, so rounding to float32 first would
        # land on the midpoint and round to even.
        e4 = torch.arange(0, 127, device="cuda", dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).double()
        mids = (e4[:-1] + e4[1:]) / 2
        v = torch.cat([mids, mids * (1 + 2.0 ** -40), mids * (1 - 2.0 ** -40), -mids * (1 + 2.0 ** -40),
                       torch.tensor([1e300, -1e300, float("inf"), 1e-300, -1e-300, -0.0, float("nan")],
                                    device="cuda", dtype=torch.float64)])
        q = torch.empty(v.numel(), device="cuda", dtype=torch.uint8)
        kernel(STORE_F64, "store_f64").launch(*blocks(v.numel()), [v, q, v.numel()])
        torch.cuda.synchronize()
        want = reference(v)
        bad = ~same(as_e4m3(q), want)
        if bad.any():
            self.fail(f"{int(bad.sum())} differ; first {v[bad][0].item()!r} -> {as_e4m3(q)[bad][0].item()}")
        # The control: through float32 first, these values do not all survive.
        twice = v.float().to(torch.float8_e4m3fn).double()
        self.assertTrue(bool((~same(twice, want)).any()), "the sample does not tell one rounding from two")

    def test_a_literal_is_rounded_from_its_double(self):
        q = torch.full((4,), 0xAB, device="cuda", dtype=torch.uint8)
        kernel(LITERAL, "store_literal").launch((1, 1, 1), (1, 1, 1), [q])
        torch.cuda.synchronize()
        self.assertEqual(as_e4m3(q[:3]).tolist(), [1.125, -1.125, 1.0])
        self.assertEqual(q[3].item(), 0xAB, "a byte the kernel never stored was written")

    def test_the_v4_forms_move_four_bytes(self):
        rows = 3  # threads cover 4 rows; the 4th is out of range
        x = torch.tensor([1.0, -2.5, 448.0, 1000.0, 0.0078125, -0.0, 3.3, 7.0] * 4, device="cuda")
        q = torch.full((4 * 8,), 0xAB, device="cuda", dtype=torch.uint8)
        out = torch.full((32,), 99.0, device="cuda")
        kernel(V4, "v4_e4m3").launch((1, 1, 1), (8, 1, 1), [q, x, out, rows])
        torch.cuda.synchronize()
        want = x.to(torch.float8_e4m3fn)
        self.assertTrue(torch.equal(q[:24], want.view(torch.uint8)[:24]), (q.tolist(), want.view(torch.uint8).tolist()))
        self.assertEqual(q[24:].tolist(), [0xAB] * 8, "the row past `rows` was written")
        self.assertTrue(bool(same(out[:24].double(), want.double()[:24]).all()), out.tolist())
        self.assertEqual(out[24:].tolist(), [0.0] * 8, "a load past `rows` must give zeros")


@unittest.skipUnless(gpu_has_fp8(), "needs an sm_89 or later GPU")
class E4m3QuantizeKernels(unittest.TestCase):
    def test_both_quantize_kernels_give_torchs_bytes(self):
        qa, qt = kernel(QUANT, "quant_e4m3"), kernel(QUANT_T, "quant_e4m3_t")
        for rows, cols in ((256, 256), (1024, 520), (36, 13), (4, 1), (100, 999)):
            with self.subTest(rows=rows, cols=cols):
                x = torch.randn(rows, cols, device="cuda") * 100
                inv = 0.37
                want = (x * inv).to(torch.float8_e4m3fn).view(torch.uint8)
                q = torch.full((rows, cols), 0xAB, device="cuda", dtype=torch.uint8)
                qa.launch(*blocks(rows * cols), [x, inv, q, rows * cols])
                t = torch.full((cols + 1, rows), 0xAB, device="cuda", dtype=torch.uint8)
                qt.launch(*blocks((cols + 7) // 8 * ((rows + 15) // 16) * 32), [x, inv, t, rows, cols])
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(q, want), int((q != want).sum()))
                self.assertTrue(torch.equal(t[:cols], want.t()), int((t[:cols] != want.t()).sum()))
                self.assertEqual(int((t[cols] != 0xAB).sum()), 0, "bytes past Q's end were written")


# Pinned tiles: cta_m,cta_n,cta_k,warps_m,warps_n,stages,ctas_per_sm,group_m.
# Every K width the swizzle covers (32, 64, 128), 2 to 4 stages, 1 to 16
# warps, groups of 8 and 3 M tiles; None is the default tile (64x64x64 on 4
# warps, groups of 16).
TILES = [
    None,
    "128,128,64,4,2,3,2,8",
    "64,64,64,2,2,2,2,8",
    "128,128,32,4,2,4,2,8",
    "64,128,128,2,2,2,2,3",
    "128,256,64,2,4,3,1,8",
    "32,32,32,1,1,3,4,8",
    "256,128,64,4,4,2,1,3",
]


def shapes_for(spec):
    cm, cn, ck = (64, 64, 64) if spec is None else (int(x) for x in spec.split(",")[:3])
    return [
        (cm * 2, cn * 3, ck * 4),              # whole tiles
        (cm * 9 + 5, cn * 2 + 3, ck * 3 + 16),  # ragged M and N, a K tail of one chunk
        (cm - 1, cn + 7, 16),                  # K is one chunk
        (cm * 11, cn * 2, ck * 2 + 48),        # several launch groups, a partial last one
    ]


def e4m3_gemm(m, n, k):
    name = f"gemm_e4m3_t_{m}_{n}_{k}"
    src = (f"@tile({m}, {n}, {k})\nkernel {name}(A: GlobalMemory<E4M3>, B: GlobalMemory<E4M3>, "
           f"scale_a: F32, scale_b: F32, C: GlobalMemory<F32>) {{\n}}\n")
    kern = TorchKernel(src, kernel_name=name, target_sm="sm_89")
    g = re.search(r"Launch grid required: \((\d+), (\d+), (\d+)\) CTAs, block \((\d+),(\d+),(\d+)\)", kern.ptx)
    v = [int(x) for x in g.groups()]
    return kern, tuple(v[:3]), tuple(v[3:])


def run_gemm(m, n, k, sa=1.0, sb=1.0):
    gen = torch.Generator(device="cuda").manual_seed(m * 7 + n * 3 + k)
    ai = torch.randint(-8, 9, (m, k), device="cuda", generator=gen)
    bi = torch.randint(-8, 9, (n, k), device="cuda", generator=gen)
    c = torch.full((m, n), float("nan"), device="cuda")
    kern, grid, block = e4m3_gemm(m, n, k)
    kern.launch(grid, block, [ai.float().to(torch.float8_e4m3fn), bi.float().to(torch.float8_e4m3fn), sa, sb, c])
    torch.cuda.synchronize()
    return c, ((ai.double() @ bi.double().t()) * (sa * sb)).float(), kern.ptx


@unittest.skipUnless(gpu_has_fp8(), "needs an sm_89 or later GPU")
class E4m3GemmIsExact(unittest.TestCase):
    def check(self, m, n, k, sa=1.0, sb=1.0):
        c, ref, _ = run_gemm(m, n, k, sa, sb)
        bad = ~(c == ref)
        self.assertFalse(bool(bad.any()), f"{int(bad.sum())} of {m * n} differ; first at "
                         f"{tuple(bad.nonzero()[0].tolist()) if bad.any() else None}")

    def test_every_tile_on_whole_and_ragged_shapes(self):
        for spec in TILES:
            with tile(spec):
                for shape in shapes_for(spec):
                    with self.subTest(tile=spec, shape=shape):
                        self.check(*shape)

    def test_the_pinned_tile_is_the_one_compiled(self):
        with tile("64,128,128,2,2,2,2,3"):
            _, _, ptx = run_gemm(256, 256, 256)
        self.assertIn("CTA 64x128x128 | 2x2 warps | 2 stages", ptx)

    def test_partial_sums_are_added_in_float32(self):
        # The tensor core adds an mma's products into its accumulator at about
        # 11 bits, so chaining every mma of the K loop in it left a median
        # relative error of 6.6% at K = 4096 on positive inputs (each step
        # truncates; _scaled_mm, which adds in float32 every two mmas, has
        # 5.7e-4). Each mma starts from zero and is added in float32 now.
        m, n, k = 256, 256, 4096
        gen = torch.Generator(device="cuda").manual_seed(7)
        a8 = (torch.rand(m, k, device="cuda", generator=gen) * 16).to(torch.float8_e4m3fn)
        b8 = (torch.rand(n, k, device="cuda", generator=gen) * 16).to(torch.float8_e4m3fn)
        ref = a8.double() @ b8.double().t()
        c = torch.full((m, n), float("nan"), device="cuda")
        kern, grid, block = e4m3_gemm(m, n, k)
        kern.launch(grid, block, [a8, b8, 1.0, 1.0, c])
        torch.cuda.synchronize()
        err = ((c.double() - ref).abs() / ref.abs()).max().item()
        self.assertLess(err, 4e-4, f"relative error {err:.2e}: the partial sums are not added in float32")

    def test_the_scales_multiply_the_result(self):
        # Products other than 1, on the default tile and a ragged one.
        self.check(1024, 512, 256, sa=2.0, sb=4.0)
        self.check(300, 200, 96, sa=0.5, sb=0.25)


# In the last two K is whole K tiles, so no chunk is masked as past K, and
# A and B are whole multiples of 512 bytes, the allocator's granule: a copy
# of a K tile past the last one - in the prologue (one K tile, three stages)
# or the loop's last iterations - reads outside the allocation.
MEMCHECK_SHAPES = [(300, 200, 96), (129, 135, 16), (1157, 259, 208), (128, 128, 64), (256, 128, 192)]
MEMCHECK_CHILD = """
import sys, torch
sys.path.insert(0, {tests!r})
from test_fp8_e4m3 import run_gemm
for m, n, k in {shapes!r}:
    c, ref, _ = run_gemm(m, n, k)
    assert torch.equal(c, ref), (m, n, k)
print("RAN", len({shapes!r}))
"""


def compute_sanitizer():
    return shutil.which("compute-sanitizer") or next(
        (p for p in ("/opt/cuda/bin/compute-sanitizer", "/usr/local/cuda/bin/compute-sanitizer") if os.path.exists(p)), None)


@unittest.skipUnless(gpu_has_fp8(), "needs an sm_89 or later GPU")
class E4m3GemmReadsOnlyItsOperands(unittest.TestCase):
    def test_no_access_outside_a_b_or_c(self):
        # Rows past M or N and chunks past K are copied with source size 0;
        # their addresses lie outside A and B, which only memcheck can see.
        tool = compute_sanitizer()
        if tool is None:
            self.skipTest("compute-sanitizer not found; the out-of-bounds check did not run")
        child = MEMCHECK_CHILD.format(tests=os.path.dirname(os.path.abspath(__file__)), shapes=MEMCHECK_SHAPES)
        env = dict(os.environ, PYTORCH_NO_CUDA_MEMORY_CACHING="1")
        out = subprocess.run([tool, "--tool", "memcheck", "--error-exitcode", "3", sys.executable, "-c", child],
                             capture_output=True, text=True, env=env, timeout=600)
        text = out.stdout + out.stderr
        self.assertIn(f"RAN {len(MEMCHECK_SHAPES)}", text, text[-3000:])
        self.assertIn("ERROR SUMMARY: 0 errors", text, text[-3000:])
        self.assertEqual(out.returncode, 0, text[-3000:])


if __name__ == "__main__":
    unittest.main()
