"""`drift_add_at` and `drift_finish` under the CPU interpreter: every launch shape gives the exact totals.

The interpreter checks what the GPU cannot show: a `bar.sync` that part of a
block does not reach, and a read of shared memory no thread wrote. The last
block learns it is last through shared memory between two barriers, so a
mutant that moved either barrier, or the flag's store, could pass on a GPU.
"""

import unittest

import numpy as np
import torch

from y_lang.interpreter import CPUInterpreter, CPUInterpreterError
from y_lang.torch_interop import TorchKernel

PAIRS = """
kernel scat(X: GlobalMemory<F32>, IDX: GlobalMemory<I32>, JDX: GlobalMemory<I32>, ACC: GlobalMemory<I64>, OUT: GlobalMemory<F32>, N: I32, K: I32, NT: I32) {
    let t: I32 = thread_idx_x() + block_dim_x() * (thread_idx_y() + block_dim_y() * thread_idx_z());
    let threads: I32 = block_dim_x() * block_dim_y() * block_dim_z();
    let b: I32 = block_idx_x() + grid_dim_x() * (block_idx_y() + grid_dim_y() * block_idx_z());
    let stride: I32 = grid_dim_x() * grid_dim_y() * grid_dim_z() * threads;
    @invariant(j >= 0)
    for j in 0..K step 1 {
        let k: I32 = b * threads + t + j * stride;
        if k < N {
            @ZeroDrift
            @bounds(min=-1000000.0, max=1000000.0)
            let mut acc: F32 = 0.0;
            acc += X[k];
            drift_add_at(ACC, IDX[k], acc);
            acc -= X[k];
            drift_add_at(ACC, JDX[k], acc);
        }
    }
    drift_finish(ACC, OUT, NT);
}
"""

# Partial warps (33, 45 and 7 threads; one thread), full ones, a 3-D grid,
# and blocks with fewer threads than there are totals.
SHAPES = [((33, 1, 1), (3, 1, 1)), ((5, 3, 3), (2, 1, 1)), ((7, 1, 1), (1, 1, 1)), ((1, 1, 1), (1, 1, 1)),
          ((32, 1, 1), (2, 1, 1)), ((64, 1, 1), (1, 2, 2)), ((7, 1, 1), (5, 1, 1))]


def q(v):
    """A term rounded to Q32.32, ties away from zero."""
    return int(np.sign(v) * np.floor(abs(float(v)) * 2.0 ** 32 + 0.5))


def exact_bits(xs, idx, jdx, totals):
    """Each total summed exactly from the definition, read back through f64."""
    s = [0] * totals
    for v, i, j in zip(xs.tolist(), idx.tolist(), jdx.tolist()):
        s[i] += q(v)
        s[j] -= q(v)
    return [int(np.array(np.float32(np.float64(t) * 2.0 ** -32)).view(np.int32)) for t in s]


class ScatterUnderTheInterpreter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.k = TorchKernel(PAIRS, kernel_name="scat", target_sm="sm_80")

    def launch(self, x, idx, jdx, acc, totals, block, grid):
        out = torch.full((max(totals, 1),), 7.0, dtype=torch.float32)
        threads = int(np.prod(block) * np.prod(grid))
        rounds = (x.numel() + threads - 1) // threads
        CPUInterpreter("scat", ptx=self.k.ptx).run(grid, block, [x, idx, jdx, acc, out, x.numel(), rounds, totals])
        return out

    def test_every_shape_gives_the_exact_totals_and_leaves_acc_zero(self):
        g = torch.Generator().manual_seed(3)
        n, totals = 301, 11
        x = (torch.randn(n, generator=g) * torch.pow(10.0, torch.randint(-7, 3, (n,), generator=g).float())).float()
        idx = torch.randint(0, totals, (n,), generator=g, dtype=torch.int32)
        jdx = torch.randint(0, totals, (n,), generator=g, dtype=torch.int32)
        want = exact_bits(x, idx, jdx, totals)
        for block, grid in SHAPES:
            acc = torch.zeros(2 * totals + 1, dtype=torch.int64)
            out = self.launch(x, idx, jdx, acc, totals, block, grid)
            self.assertEqual(out.view(torch.int32).tolist(), want, f"block {block} grid {grid}")
            self.assertEqual(acc.tolist(), [0] * (2 * totals + 1), f"block {block} grid {grid}: ACC not left zero")

    def test_an_acc_that_was_not_zero_stops_the_kernel(self):
        x = torch.ones(40, dtype=torch.float32)
        zeros = torch.zeros(40, dtype=torch.int32)
        acc = torch.tensor([0, 0, 0, 0, 5], dtype=torch.int64)
        with self.assertRaisesRegex(CPUInterpreterError, "trap"):
            self.launch(x, zeros, zeros, acc, 2, (33, 1, 1), (2, 1, 1))

    def test_an_index_outside_the_totals_stops_the_kernel(self):
        # Index 2 of 2 totals is the ticket's word: a term of zero there
        # would change nothing, so only the index test stops it. Index -1 is
        # before ACC, which the interpreter would report as an access out of
        # bounds, not as a trap.
        for bad, term in [(2, 0.0), (2, 1.0), (3, 1.0), (-1, 1.0)]:
            x = torch.ones(40, dtype=torch.float32)
            idx = torch.zeros(40, dtype=torch.int32)
            idx[17], x[17] = bad, term
            acc = torch.zeros(5, dtype=torch.int64)
            with self.assertRaisesRegex(CPUInterpreterError, "trap", msg=f"index {bad}"):
                self.launch(x, idx, torch.zeros(40, dtype=torch.int32), acc, 2, (33, 1, 1), (2, 1, 1))



CLOSED = """
kernel k(X: GlobalMemory<F32>, IDX: GlobalMemory<I32>, ACC: GlobalMemory<I64>, N: I32, K: I32, NT: I32) {
    let t: I32 = thread_idx_x() + block_dim_x() * (thread_idx_y() + block_dim_y() * thread_idx_z());
    let threads: I32 = block_dim_x() * block_dim_y() * block_dim_z();
    let b: I32 = block_idx_x() + grid_dim_x() * (block_idx_y() + grid_dim_y() * block_idx_z());
    let stride: I32 = grid_dim_x() * grid_dim_y() * grid_dim_z() * threads;
    @invariant(j >= 0)
    for j in 0..K step 1 {
        let i: I32 = b * threads + t + j * stride;
        if i < N {
            @ZeroDrift
            @bounds(min=-1000000.0, max=1000000.0)
            let mut acc: F32 = 0.0;
            acc += X[i];
            drift_add_at(ACC, IDX[i], acc);
        }
    }
    drift_close(ACC, NT);
}

kernel c(ACC: GlobalMemory<I64>, OUT: GlobalMemory<F32>, NT: I32) {
    drift_convert(ACC, OUT, NT);
}
"""


class CloseAndConvertUnderTheInterpreter(unittest.TestCase):
    """`drift_close` in one kernel and `drift_convert` in the next, each at its own shape."""

    @classmethod
    def setUpClass(cls):
        cls.ptx = TorchKernel(CLOSED, kernel_name="k", target_sm="sm_80").ptx
        g = torch.Generator().manual_seed(4)
        cls.n, cls.totals = 211, 9
        cls.x = (torch.randn(cls.n, generator=g) * torch.pow(10.0, torch.randint(-7, 3, (cls.n,), generator=g).float())).float()
        cls.idx = torch.randint(0, cls.totals, (cls.n,), generator=g, dtype=torch.int32)

    def close(self, acc, block, grid):
        threads = int(np.prod(block) * np.prod(grid))
        CPUInterpreter("k", ptx=self.ptx).run(grid, block, [self.x, self.idx, acc, self.n, (self.n + threads - 1) // threads, self.totals])

    def convert(self, acc, out, block, grid, totals=None):
        CPUInterpreter("c", ptx=self.ptx).run(grid, block, [acc, out, self.totals if totals is None else totals])

    def test_the_two_kernels_give_the_exact_totals_and_leave_acc_zero(self):
        s = [0] * self.totals
        for v, i in zip(self.x.tolist(), self.idx.tolist()):
            s[i] += q(v)
        want = [int(np.array(np.float32(np.float64(t) * 2.0 ** -32)).view(np.int32)) for t in s]
        for (block, grid), (cblock, cgrid) in zip(SHAPES, [((1, 1, 1), (1, 1, 1)), ((4, 1, 1), (3, 1, 1)), ((33, 1, 1), (2, 1, 1)),
                                                         ((2, 2, 1), (1, 1, 2)), ((64, 1, 1), (1, 1, 1)), ((5, 1, 1), (7, 1, 1)), ((9, 1, 1), (1, 1, 1))]):
            acc = torch.zeros(2 * self.totals + 7, dtype=torch.int64)
            out = torch.full((self.totals,), 7.0, dtype=torch.float32)
            self.close(acc, block, grid)
            self.convert(acc, out, cblock, cgrid)
            self.assertEqual(out.view(torch.int32).tolist(), want, f"close {block} {grid}, convert {cblock} {cgrid}")
            self.assertEqual(acc.tolist(), [0] * (2 * self.totals + 7), f"close {block} {grid}: ACC not left zero")

    def test_every_misuse_of_the_pair_stops(self):
        out = torch.zeros(self.totals, dtype=torch.float32)
        fresh = lambda: torch.zeros(2 * self.totals + 7, dtype=torch.int64)
        def alone():
            self.convert(fresh(), out, (32, 1, 1), (1, 1, 1))
        def twice():
            acc = fresh(); self.close(acc, (32, 1, 1), (2, 1, 1)); self.convert(acc, out, (32, 1, 1), (1, 1, 1)); self.convert(acc, out, (32, 1, 1), (1, 1, 1))
        def fewer():
            acc = fresh(); self.close(acc, (32, 1, 1), (2, 1, 1)); self.convert(acc, out, (32, 1, 1), (1, 1, 1), self.totals - 1)
        def closed_twice():
            acc = fresh(); self.close(acc, (32, 1, 1), (2, 1, 1)); self.close(acc, (32, 1, 1), (2, 1, 1))
        # A negative count, each kernel alone: together the first check would
        # hide the second, and the header it names is before ACC, which the
        # interpreter reports as an access out of bounds, not as a trap.
        def close_negative():
            # No elements, or every hand-over's index test would stop it first.
            CPUInterpreter("k", ptx=self.ptx).run((2, 1, 1), (32, 1, 1), [self.x, self.idx, fresh(), 0, 0, -1])
        def convert_negative():
            self.convert(fresh(), out, (32, 1, 1), (1, 1, 1), -1)
        for name, f in [("convert alone", alone), ("convert twice", twice), ("one total fewer", fewer), ("closed twice", closed_twice),
                        ("close, negative count", close_negative), ("convert, negative count", convert_negative)]:
            with self.assertRaisesRegex(CPUInterpreterError, "trap", msg=name):
                f()

    def test_a_header_that_does_not_match_stops(self):
        # A wrong count usually moves the header, and the tag test stops it
        # first; these headers carry the tag where the converter looks.
        tag = 0x5944524946540001
        out = torch.zeros(self.totals, dtype=torch.float32)
        h = 2 * self.totals
        for name, words in [("written for another count", {h + 1: tag, h + 2: self.totals + 1, h + 4: -(2 ** 63), h + 5: 2 ** 63 - 1}),
                            ("converter's ticket not zero", {h + 1: tag, h + 2: self.totals, h + 3: 32, h + 4: -(2 ** 63), h + 5: 2 ** 63 - 1, h + 6: 5})]:
            acc = torch.zeros(h + 7, dtype=torch.int64)
            for k, v in words.items():
                acc[k] = v
            with self.assertRaisesRegex(CPUInterpreterError, "trap", msg=name):
                self.convert(acc, out, (32, 1, 1), (2, 1, 1))


if __name__ == "__main__":
    unittest.main()
