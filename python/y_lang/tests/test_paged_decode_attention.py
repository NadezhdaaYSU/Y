"""Y's paged decode attention, both shapes, run on the GPU against a float64
reference that walks the same page table.

`emit_paged_decode_attention_kernel` spreads a token's head_dim row over
`head_dim / epl` lanes inside its KV loop (`attn_loop_elems_per_lane`), so a
warp holds one, two or four tokens at once; lanes past their warp's last
token are masked; the softmax state is rescaled only when a score passes its
reference max by `ATTN_LAZY_RESCALE`; and the next iteration's K/V are
prefetched. The configurations below put each of those on a path a test can
fail on:

- one, two and four tokens per warp (head_dim 128 at GQA 8:1 and 1:1 vs
  4:1, head_dim 64), with warp blocks that end mid-group (odd lengths);
- page sizes 1, 7, 12, 16 and 32, so the per-lane page reload and its
  div/rem path run, and 5 and 6 warps, whose token blocks need a `div`;
- scores that rise steeply (a rescale on every token), rise slowly (one
  every few dozen), fall, spike once, or spread widely;
- lengths 0, below the warp count, around page and group edges, and 4096.

Out, and the split shape's two scratch buffers, start as NaN, so a row or a
split nobody writes fails. A lane past its warp's block, the last
iteration's prefetch and a warp with no tokens read nothing outside the
sequence; that changes no output, so those cases run under memcheck. Each row is compared on its own: an aggregate
error can hide one wrong short row. Kernels are compiled against
test_launch's pinned profile, in a temporary directory, so nothing is probed
or written beside the tests.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import torch

from test_launch import PINNED_PROFILE
from y_lang.torch_interop import TorchKernel

PARAMS7 = ("Q: GlobalMemory<F16>, KCache: GlobalMemory<F16>, VCache: GlobalMemory<F16>, "
           "PageTable: GlobalMemory<I32>, SeqLens: GlobalMemory<I32>, Out: GlobalMemory<F16>")
SINGLE = "kernel {name}(" + PARAMS7 + ", MaxPages: I32) {{\n}}\n"
SPLIT = ("kernel {name}(" + PARAMS7 + ", Partial: GlobalMemory<F32>, PartialMeta: GlobalMemory<F32>, "
         "MaxPages: I32) {{\n}}\n")

# f16 storage of Q, K, V and Out puts the floor near 2e-4.
TOL = 3e-3

# (head_dim, q heads, kv heads, page size, splits (0: single pass), warps)
CONFIGS = [
    (128, 32, 8, 16, 16, 8),   # 16 lanes a token, 2 tokens a warp: the README's batch-1 kernel
    (128, 32, 8, 16, 4, 8),    # ... and its batch-many kernel
    (128, 32, 8, 16, 4, 4),
    (128, 32, 4, 16, 16, 8),   # GQA 8:1: 32 lanes a token (the register bound)
    (64, 16, 4, 16, 8, 8),     # 8 lanes a token, 4 tokens a warp
    (64, 16, 16, 32, 8, 8),    # GQA 1:1, page 32
    (256, 16, 4, 1, 8, 4),     # page size 1: a new page every token
    (128, 32, 8, 12, 3, 6),    # page 12 (div/rem), 3 splits and 6 warps (div)
    (128, 32, 8, 16, 0, 32),   # single pass, 32 warps
    (128, 32, 8, 16, 0, 8),    # single pass, 8 warps
    (64, 8, 2, 7, 0, 5),       # single pass, page 7, 5 warps
]

RAGGED = [0, 1, 2, 3, 5, 15, 16, 17, 31, 33, 64, 100, 257, 1000]
LONG = [4096, 4095]
SHAPES = ["random", "steep_ramp", "ramp", "ramp_down", "spike", "wide"]


def pinned_kernel(src, name):
    saved = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, ".ysu_hw_profile"), "w") as f:
            f.write(PINNED_PROFILE)
        os.chdir(d)
        try:
            return TorchKernel(src, kernel_name=name, target_sm="sm_89")
        finally:
            os.chdir(saved)


def kernel_name(hd, nqh, nkvh, ps, splits, warps):
    if splits:
        return "paged_decode_attention_split_%d_%d_%d_%d_%d_%d" % (hd, nqh, nkvh, ps, splits, warps)
    return "paged_decode_attention_%d_%d_%d_%d_%d" % (hd, nqh, nkvh, ps, warps)


def make_inputs(hd, nqh, nkvh, ps, lens, shape, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ns = len(lens)
    max_pages = max(1, max((L + ps - 1) // ps for L in lens))
    num_pages = ns * max_pages + 3
    q = torch.randn(ns, nqh, hd, generator=g)
    kc = torch.randn(num_pages, ps, nkvh, hd, generator=g)
    vc = torch.randn(num_pages, ps, nkvh, hd, generator=g)
    # A shuffled table, never the identity, so a paging bug cannot cancel out.
    pt = torch.randperm(num_pages, generator=g)[: ns * max_pages].view(ns, max_pages).to(torch.int32)
    gqa = nqh // nkvh
    scale = hd ** -0.5
    if shape == "wide":
        q *= 40.0
    for s, L in enumerate(lens):
        if L == 0 or shape in ("random", "wide"):
            continue
        t = torch.arange(L)
        page, slot = pt[s, t // ps].long(), t % ps
        for j in range(nkvh):
            # K along the first query head of the group, scaled so that head's
            # score is `slope * (t - L/2)`; the group's other heads see their
            # own projections of it.
            qd = q[s, j * gqa]
            d = qd / (qd.norm() ** 2 * scale)
            if shape == "spike":
                kc[page[L // 2], slot[L // 2], j] = 40.0 * d
                continue
            slope = {"steep_ramp": 12.0, "ramp": 0.5, "ramp_down": -0.5}[shape]
            kc[page, slot, j] = (slope * (t.float() - L / 2)).unsqueeze(1) * d
    dev = torch.device("cuda")
    return (q.half().to(dev), kc.half().to(dev), vc.half().to(dev), pt.to(dev),
            torch.tensor(lens, dtype=torch.int32, device=dev), max_pages)


def reference(q, kc, vc, pt, sl, ps):
    ns, nqh, hd = q.shape
    nkvh = kc.shape[2]
    gqa = nqh // nkvh
    out = torch.zeros(ns, nqh, hd, dtype=torch.float64, device=q.device)
    for s in range(ns):
        L = int(sl[s])
        if L == 0:
            continue
        t = torch.arange(L, device=q.device)
        page, slot = pt[s, t // ps].long(), t % ps
        k, v = kc[page, slot].double(), vc[page, slot].double()      # [L, nkvh, hd]
        qh = q[s].double().view(nkvh, gqa, hd)
        p = torch.softmax(torch.einsum("kgd,lkd->kgl", qh, k) * hd ** -0.5, dim=-1)
        out[s] = torch.einsum("kgl,lkd->kgd", p, v).reshape(nqh, hd)
    return out


def run_kernel(cfg, kernels, q, kc, vc, pt, sl, mp):
    hd, nqh, nkvh, ps, splits, warps = cfg
    ns = q.shape[0]
    out = torch.full((ns, nqh, hd), float("nan"), dtype=torch.float16, device=q.device)
    if splits:
        part = torch.full((ns, nqh, splits, hd), float("nan"), device=q.device)
        meta = torch.full((ns, nqh, splits, 2), float("nan"), device=q.device)
        args = [q, kc, vc, pt, sl, out, part, meta, mp]
        kernels[0].launch((nkvh, ns, splits), (warps * 32, 1, 1), args)
        kernels[1].launch((nqh, ns, 1), (32, 1, 1), args)
    else:
        kernels[0].launch((nqh, ns, 1), (warps * 32, 1, 1), [q, kc, vc, pt, sl, out, mp])
    torch.cuda.synchronize()
    return out


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA GPU")
class PagedDecodeAttentionMatchesReference(unittest.TestCase):
    def test_every_configuration_and_score_shape(self):
        for cfg in CONFIGS:
            hd, nqh, nkvh, ps, splits, warps = cfg
            name = kernel_name(*cfg)
            src = (SPLIT if splits else SINGLE).format(name=name)
            kernels = [pinned_kernel(src, name)]
            self.assertIn("[Y PAGED DECODE ATTENTION]", kernels[0].ptx, name)
            if splits:
                kernels.append(pinned_kernel(src, name + "_reduce"))
            for lens in (RAGGED, LONG):
                for i, shape in enumerate(SHAPES):
                    with self.subTest(kernel=name, lens=len(lens), shape=shape):
                        q, kc, vc, pt, sl, mp = make_inputs(hd, nqh, nkvh, ps, lens, shape, seed=i)
                        got = run_kernel(cfg, kernels, q, kc, vc, pt, sl, mp).double()
                        ref = reference(q, kc, vc, pt, sl, ps)
                        for s, L in enumerate(lens):
                            if L == 0:
                                self.assertTrue(torch.equal(got[s], torch.zeros_like(got[s])),
                                                "seq_len 0 must store a zero row")
                                continue
                            self.assertTrue(torch.isfinite(got[s]).all(), "row %d (len %d) not finite" % (s, L))
                            err = ((got[s] - ref[s]).norm() / ref[s].norm()).item()
                            self.assertLessEqual(err, TOL, "row %d (len %d): relative L2 %.3e" % (s, L, err))



# Lengths that leave whole warps without a token (0-3 against 4-8 warps),
# blocks that end mid-group, and a page boundary inside one warp's block.
MEMCHECK_CONFIGS = [(128, 32, 8, 16, 16, 8), (64, 16, 4, 16, 8, 8), (128, 32, 8, 12, 3, 6), (128, 32, 8, 16, 0, 8)]
MEMCHECK_CHILD = """
import sys, torch
sys.path.insert(0, {tests!r})
import test_paged_decode_attention as T
for cfg in {cfgs!r}:
    name = T.kernel_name(*cfg)
    src = (T.SPLIT if cfg[4] else T.SINGLE).format(name=name)
    ks = [T.pinned_kernel(src, name)] + ([T.pinned_kernel(src, name + "_reduce")] if cfg[4] else [])
    for lens in ([1, 2, 3, 0, 17, 33], [5, 255]):
        q, kc, vc, pt, sl, mp = T.make_inputs(cfg[0], cfg[1], cfg[2], cfg[3], lens, "random", 0)
        got = T.run_kernel(cfg, ks, q, kc, vc, pt, sl, mp).double()
        ref = T.reference(q, kc, vc, pt, sl, cfg[3])
        for s, L in enumerate(lens):
            if L:
                assert ((got[s] - ref[s]).norm() / ref[s].norm()).item() <= T.TOL, (name, lens, s)
print("RAN", len({cfgs!r}))
"""


def compute_sanitizer():
    return shutil.which("compute-sanitizer") or next(
        (p for p in ("/opt/cuda/bin/compute-sanitizer", "/usr/local/cuda/bin/compute-sanitizer") if os.path.exists(p)), None)


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA GPU")
class PagedDecodeAttentionReadsOnlyItsBuffers(unittest.TestCase):
    def test_no_access_outside_the_sequence(self):
        tool = compute_sanitizer()
        if tool is None:
            self.skipTest("compute-sanitizer not found; the out-of-bounds check did not run")
        child = MEMCHECK_CHILD.format(tests=os.path.dirname(os.path.abspath(__file__)), cfgs=MEMCHECK_CONFIGS)
        # Each tensor its own cudaMalloc, and padding after each, so a read
        # just before the page table (a token clamped to -1) lands in padding
        # rather than in the allocation before it.
        env = dict(os.environ, PYTORCH_NO_CUDA_MEMORY_CACHING="1")
        out = subprocess.run([tool, "--tool", "memcheck", "--padding", "512", "--error-exitcode", "3",
                              sys.executable, "-c", child], capture_output=True, text=True, env=env, timeout=900)
        text = out.stdout + out.stderr
        self.assertIn(f"RAN {len(MEMCHECK_CONFIGS)}", text, text[-3000:])
        self.assertIn("ERROR SUMMARY: 0 errors", text, text[-3000:])
        self.assertEqual(out.returncode, 0, text[-3000:])


if __name__ == "__main__":
    unittest.main()
