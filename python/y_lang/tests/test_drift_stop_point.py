"""Where a `@ZeroDrift` loop stops, under the CPU interpreter: the stores made before it.

A trap ends the GPU's context, so the host cannot read what the kernel stored
before it stopped. The interpreter keeps memory, so here the stores are the
evidence:
- a loop that stores keeps the stop at the term: every element before the bad
  term is stored, and none from it on;
- a loop that stays in its thread stops at its end, before the store after it,
  including when only a partial sum leaves the bounds;
- with good terms both store the exact sums.
"""

import re
import unittest

import numpy as np
import torch

from y_lang.interpreter import CPUInterpreter, CPUInterpreterError
from y_lang.torch_interop import TorchKernel

HEAD = """
kernel k(A: GlobalMemory<F32>, Out: GlobalMemory<F32>, N: I32) {
    @ZeroDrift
    @bounds(min=-10000.0, max=10000.0)
    let mut acc: F32 = 0.0;
    @invariant(j >= 0)
    for j in 0..N step 1 {
        acc += A[j];
"""
STORES = HEAD + """
        Out[j] = acc;
    }
    Out[N] = acc;
}
"""
IN_THREAD = HEAD + """
    }
    Out[N] = acc;
}
"""


def q(v):
    """A term rounded to Q32.32, ties away from zero."""
    return int(np.sign(v) * np.floor(abs(float(v)) * 2.0 ** 32 + 0.5))


def running(terms):
    """Each prefix's exact sum, read back as an f32."""
    s, out = 0, []
    for t in terms:
        s += q(t)
        out.append(np.float32(np.float64(s) * 2.0 ** -32))
    return out


class WhereALoopStops(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stores = TorchKernel(STORES, kernel_name="k", target_sm="sm_80")
        cls.in_thread = TorchKernel(IN_THREAD, kernel_name="k", target_sm="sm_80")

    def run_k(self, k, terms):
        a = torch.tensor(terms, dtype=torch.float32)
        out = torch.full((len(terms) + 1,), 7.0, dtype=torch.float32)
        CPUInterpreter("k", ptx=k.ptx).run((1, 1, 1), (1, 1, 1), [a, out, len(terms)])
        return out

    def test_the_two_loops_are_emitted_differently(self):
        # The premise of the rest: one loop traps inside, the other sets a flag.
        def body(ptx):
            start = re.search(r"^\s*\$LOOP_START_\d+:", ptx, re.M)
            return ptx[start.end():re.search(r"^\s*\$LOOP_END_\d+:", ptx, re.M).start()]
        self.assertIn("trap;", body(self.stores.ptx))
        self.assertNotIn("or.pred", self.stores.ptx)
        self.assertNotIn("trap;", body(self.in_thread.ptx))
        self.assertIn("or.pred", body(self.in_thread.ptx))

    def test_good_terms_store_the_exact_sums(self):
        terms = [0.1, -2.5, 1e-9, 3.75, 9000.0, -9000.0]
        want = running(terms)
        got = self.run_k(self.stores, terms)
        self.assertEqual(got.numpy().view(np.int32).tolist(), np.array(want + [want[-1]]).view(np.int32).tolist())
        got = self.run_k(self.in_thread, terms)
        self.assertEqual(got[-1].item(), float(want[-1]))
        self.assertTrue(bool((got[:-1] == 7.0).all()))

    def test_a_loop_that_stores_stops_at_the_bad_term(self):
        for terms, first_bad in [([1.0, 2.0, float("nan"), 4.0, 5.0], 2), ([6000.0, 6000.0, -6000.0, 1.0], 1),
                                 ([1e30, 1.0], 0), ([1.0, 2.0, 3.0, float("inf")], 3)]:
            a = torch.tensor(terms, dtype=torch.float32)
            out = torch.full((len(terms) + 1,), 7.0, dtype=torch.float32)
            with self.assertRaisesRegex(CPUInterpreterError, "trap", msg=str(terms)):
                CPUInterpreter("k", ptx=self.stores.ptx).run((1, 1, 1), (1, 1, 1), [a, out, len(terms)])
            want = running(terms[:first_bad])
            self.assertEqual(out[:first_bad].tolist(), [float(w) for w in want], msg=str(terms))
            self.assertTrue(bool((out[first_bad:] == 7.0).all()), msg=f"{terms}: stored after the bad term: {out.tolist()}")

    def test_a_loop_in_its_thread_stops_before_the_store_after_it(self):
        for terms in [[1.0, 2.0, float("nan"), 4.0], [6000.0, 6000.0, -6000.0, -6000.0], [1e30], [1.0, float("-inf")]]:
            a = torch.tensor(terms, dtype=torch.float32)
            out = torch.full((len(terms) + 1,), 7.0, dtype=torch.float32)
            with self.assertRaisesRegex(CPUInterpreterError, "trap", msg=str(terms)):
                CPUInterpreter("k", ptx=self.in_thread.ptx).run((1, 1, 1), (1, 1, 1), [a, out, len(terms)])
            self.assertTrue(bool((out == 7.0).all()), msg=f"{terms}: stored before stopping: {out.tolist()}")


if __name__ == "__main__":
    unittest.main()
