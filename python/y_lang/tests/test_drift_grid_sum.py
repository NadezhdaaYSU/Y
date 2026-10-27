"""`drift_add_to` under the CPU interpreter: every launch shape gives the exact total.

The interpreter checks what the GPU cannot show. A warp the block does not
fill must not shuffle: a `shfl.sync` naming lanes that are not there is
undefined, and an RTX 4070 Ti SUPER returns something harmless, so a mutant
that shuffled in every warp passed every GPU test in
`tests/zero_drift_grid_sum.rs`. The interpreter stops at that shuffle.
"""

import unittest

import numpy as np
import torch

from y_lang.interpreter import CPUInterpreter, CPUInterpreterError
from y_lang.torch_interop import TorchKernel

SUM = """
kernel gsum(X: GlobalMemory<F32>, ACC: GlobalMemory<I64>, OUT: GlobalMemory<F32>, N: I32, K: I32) {
    @ZeroDrift
    @bounds(min=-1000000.0, max=1000000.0)
    let mut acc: F32 = 0.0;
    let t: I32 = thread_idx_x() + block_dim_x() * (thread_idx_y() + block_dim_y() * thread_idx_z());
    let threads: I32 = block_dim_x() * block_dim_y() * block_dim_z();
    let b: I32 = block_idx_x() + grid_dim_x() * (block_idx_y() + grid_dim_y() * block_idx_z());
    let stride: I32 = grid_dim_x() * grid_dim_y() * grid_dim_z() * threads;
    @invariant(j >= 0)
    for j in 0..K step 1 {
        let i: I32 = b * threads + t + j * stride;
        if i < N {
            acc += X[i];
        }
    }
    drift_add_to(ACC, OUT, acc);
}
"""

# Partial warps (33, 45 and 7 threads; one thread), full ones, a 3-D grid.
SHAPES = [((33, 1, 1), (3, 1, 1)), ((5, 3, 3), (2, 1, 1)), ((7, 1, 1), (1, 1, 1)), ((1, 1, 1), (1, 1, 1)),
          ((32, 1, 1), (2, 1, 1)), ((64, 1, 1), (1, 2, 2))]


def exact_bits(xs):
    """Each term rounded to Q32.32, ties away from zero, summed exactly, read back through f64."""
    total = sum(int(np.sign(v) * np.floor(abs(float(v)) * 2.0 ** 32 + 0.5)) for v in xs.tolist())
    return int(np.array(np.float32(np.float64(total) * 2.0 ** -32)).view(np.int32))


class GridSumUnderTheInterpreter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.k = TorchKernel(SUM, kernel_name="gsum", target_sm="sm_80")

    def launch(self, x, acc, block, grid):
        out = torch.full((1,), 7.0, dtype=torch.float32)
        threads = int(np.prod(block) * np.prod(grid))
        rounds = (x.numel() + threads - 1) // threads
        CPUInterpreter("gsum", ptx=self.k.ptx).run(grid, block, [x, acc, out, x.numel(), rounds])
        return out

    def test_every_shape_gives_the_exact_total_and_leaves_acc_zero(self):
        g = torch.Generator().manual_seed(3)
        x = (torch.randn(301, generator=g) * torch.pow(10.0, torch.randint(-7, 3, (301,), generator=g).float())).float()
        want = exact_bits(x)
        for block, grid in SHAPES:
            acc = torch.zeros(3, dtype=torch.int64)
            out = self.launch(x, acc, block, grid)
            self.assertEqual(out.view(torch.int32).item(), want, f"block {block} grid {grid}")
            self.assertEqual(acc.tolist(), [0, 0, 0], f"block {block} grid {grid}: ACC not left zero")

    def test_an_acc_that_was_not_zero_stops_the_kernel(self):
        x = torch.ones(40, dtype=torch.float32)
        with self.assertRaisesRegex(CPUInterpreterError, "trap"):
            self.launch(x, torch.tensor([0, 0, 5], dtype=torch.int64), (33, 1, 1), (2, 1, 1))


if __name__ == "__main__":
    unittest.main()
