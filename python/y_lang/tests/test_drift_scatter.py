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


if __name__ == "__main__":
    unittest.main()
