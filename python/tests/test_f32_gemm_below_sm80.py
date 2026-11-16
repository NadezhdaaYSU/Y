"""The `@tile` FP32 GEMM built for sm_75 (a Tesla T4), run on the CPU by the
PTX interpreter, through the real liby.so; no GPU needed.

Below sm_80 a copy is a load and a shared store. Until 2026-11-15 a K tail or
a ragged M guarded every copy of the K loop (a predicated load and a zero
fill), which spilled 280-688 bytes at 255 registers, and 1023 x 1025 x 1027
ran at 0.22x cuBLAS on a T4. Now the first K tile is the short one and the
last tile row slides back to end at row M-1, so the loop's copies take no
guards (src/ptx_emitter/f32_simt_gemm.rs, `short_first`, `slide_m`). These
tests check that:

* every element of C is the in-order fma.rn.f32 chain over k = 0..K-1, bit
  for bit, against an oracle computed here in float64 with an error-free sum.
  Every element is stored, and the interpreter refuses any read or write
  outside A, B or C, which is what the guards were for;
* the oracle can tell the order apart: the same chain in another order gives
  other bits for these inputs;
* the K loops of those shapes contain no guarded global load, and issue
  every load of the next tile before the FMAs and every shared store after.

The GPU runs of the same kernels are in tests/f32_simt_gemm.rs
(`built_for_sm_75_each_element_is_the_same_fma_chain`).
"""

import re
import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from y_lang.compiler import compile_to_ptx
from y_lang.interpreter import CPUInterpreter

# (M, N, K): the K tail with a ragged M on the mid tile; K tail alone; M
# alone; a row remainder set sliding into the rows before it; a 1-row GEMM
# (no slide: M < BM); K < 16 (one short tile, still guarded); small K, where
# ptxas unrolls the loop; and the few-column tiles (`emit_f32_slab_kernel`):
# 8 x 16 with two K slices and a tail, 16 x 32 with M and N ragged.
SHAPES = [(1023, 1025, 99), (256, 1024, 35), (1023, 1028, 32), (400, 2049, 40), (67, 37, 35),
          (1, 257, 40), (33, 65, 5), (130, 129, 50), (6, 100, 150), (12, 70, 77)]


def source(m, n, k):
    return (f"@tile({m}, {n}, {k})\nkernel g(A: GlobalMemory<F32>, B: GlobalMemory<F32>, "
            f"C: GlobalMemory<F32>) {{\n}}\n")


def fma32_chain(A, B, order=None):
    """acc = fma32(A[:, k], B[k, :], acc) over k in `order` (default 0..K-1),
    each fma rounded once to float32 (round to nearest, ties to even)."""
    K = A.shape[1]
    acc = np.zeros((A.shape[0], B.shape[1]), np.float32)
    A64, B64 = A.astype(np.float64), B.astype(np.float64)
    for k in (range(K) if order is None else order):
        p = np.outer(A64[:, k], B64[k, :])            # exact: 24 + 24 bits
        c = acc.astype(np.float64)
        s = p + c
        bb = s - p
        e = (p - (s - bb)) + (c - bb)                 # s + e == p + c exactly
        r = s.astype(np.float32)
        r64 = r.astype(np.float64)
        toward = np.where(s > r64, np.float32(np.inf), np.float32(-np.inf)).astype(np.float32)
        other = np.nextafter(r, toward)
        # s rounds to r unless s sits exactly on a float32 midpoint and the
        # exact sum lies off it: then e picks the side.
        tie = (r64 != s) & (s == (r64 + other.astype(np.float64)) / 2) & (e != 0)
        acc = np.where(tie, np.where(e > 0, np.maximum(r, other), np.minimum(r, other)), r).astype(np.float32)
    return acc


def inputs(m, n, k, seed):
    g = np.random.default_rng(seed)

    def mat(r, c):
        x = (g.standard_normal((r, c)) * 2.0 ** g.integers(-6, 7, (r, c))).astype(np.float32)
        x[g.random((r, c)) < 0.05] = 0.0
        x[g.random((r, c)) < 0.05] = -0.0
        return x
    return mat(m, k), mat(k, n)


def k_loops(ptx):
    """The body of each K loop: from its label to its back branch."""
    out = []
    for lab in re.findall(r"^(\$f32_gemm_k_\d+):", ptx, re.M):
        body = ptx.split(lab + ":", 1)[1]
        out.append(body.split(f"bra {lab};", 1)[0])
    return out


@unittest.skipIf(torch is None, "torch is not installed")
class F32GemmBelowSm80(unittest.TestCase):
    def ptx(self, m, n, k):
        p = compile_to_ptx(source(m, n, k), target_sm="sm_75", name="g")
        self.assertIn(".target sm_75", p)
        self.assertNotIn("cp.async", "\n".join(l for l in p.splitlines() if not l.strip().startswith("//")))
        return p

    def run_ptx(self, ptx, A, B, grid):
        m, n = A.shape[0], B.shape[1]
        C = torch.from_numpy(np.full((m, n), 0x7FC0DEAD, np.uint32).view(np.float32).copy())
        block = int(re.search(r"Launch: block \((\d+),", ptx).group(1))
        CPUInterpreter("g", ptx=ptx).run((grid, 1, 1), (block, 1, 1), [torch.from_numpy(A), torch.from_numpy(B), C])
        return C.numpy().view(np.uint32)

    def test_each_element_is_the_in_order_fma_chain(self):
        for m, n, k in SHAPES:
            with self.subTest(shape=(m, n, k)):
                ptx = self.ptx(m, n, k)
                slab = re.search(r"\| CTA \d+x\d+x(\d+) \| \d+ threads of \d+x\d+ outputs \| tile `slab", ptx)
                if slab and k % int(slab.group(1)):
                    self.assertIn(f"then the last {k % int(slab.group(1))} steps from global memory", ptx)
                elif k % 16 and k >= 16 and "tile `gemv" not in ptx and not slab:
                    self.assertIn(f"the first K tile runs its {k % 16} steps", ptx)
                A, B = inputs(m, n, k, m * 7919 + n * 31 + k)
                want = fma32_chain(A, B).view(np.uint32)
                tiles = int(re.search(r"\((\d+), 1, 1\) is one tile per CTA", ptx).group(1))
                for grid in sorted({tiles, 3}):
                    got = self.run_ptx(ptx, A, B, grid)
                    bad = int((got != want).sum())
                    self.assertEqual(bad, 0, f"{m}x{n}x{k} grid {grid}: {bad} of {m * n} elements differ")

    def test_the_oracle_tells_the_order_apart(self):
        """Without this, a kernel summing k in another order could pass."""
        m, n, k = 1023, 1025, 99
        A, B = inputs(m, n, k, m * 7919 + n * 31 + k)
        A, B = A[:64], B[:, :64]
        fwd = fma32_chain(A, B).view(np.uint32)
        short_last = list(range(3, k)) + [0, 1, 2]       # the short tile's steps moved to the end
        self.assertGreater(int((fma32_chain(A, B, short_last).view(np.uint32) != fwd).sum()), 100)

    def test_the_k_loops_have_no_guarded_loads(self):
        for m, n, k in [(1023, 1025, 1027), (1024, 1024, 1027), (1023, 1024, 1024), (400, 2049, 1025)]:
            with self.subTest(shape=(m, n, k)):
                loops = k_loops(self.ptx(m, n, k))
                self.assertTrue(loops)
                for body in loops:
                    self.assertTrue(re.search(r"\bld\.global", body), "a K loop with no copies")
                    guarded = re.findall(r"@!?%p\d+ ld\.global.*", body)
                    self.assertEqual(guarded, [], f"{m}x{n}x{k}: guarded loads in the K loop")

    def test_the_k_loops_load_first_and_store_last(self):
        """A plain load's shared store waits for the load, so every load of
        the next tile comes before the first FMA and every store after the
        last (`prefetch`); spread over the k steps, ptxas left 30-80
        instructions between the two."""
        for m, n, k in [(1024, 1024, 1024), (1023, 1025, 1027), (400, 2049, 1025)]:
            with self.subTest(shape=(m, n, k)):
                for body in k_loops(self.ptx(m, n, k)):
                    lines = body.splitlines()
                    fma = [i for i, l in enumerate(lines) if "fma.rn.f32" in l]
                    ldg = [i for i, l in enumerate(lines) if re.search(r"\bld\.global", l)]
                    sts = [i for i, l in enumerate(lines) if re.search(r"\bst\.shared", l)]
                    self.assertTrue(fma and ldg and sts)
                    self.assertLess(max(ldg), min(fma), f"{m}x{n}x{k}: a load after an FMA")
                    self.assertGreater(min(sts), max(fma), f"{m}x{n}x{k}: a store before an FMA")


if __name__ == "__main__":
    unittest.main()
