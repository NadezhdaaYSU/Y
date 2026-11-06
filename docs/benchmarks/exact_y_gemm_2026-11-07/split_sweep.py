# Y's int8 GEMM at decode shapes, with K split over gridDim.z: which split
# factor fills the card? At M <= 32 the kernel uses its 64x64 tile, so an
# N = 896 layer is 14 CTAs on a 66-SM card, and only a split of K gives it
# more. A split launch adds its partials into a zeroed C with
# red.global.add.s32, which proofs/Int8GemmSchedule.v covers for any factor.
#
# Bits first: every factor's output must equal factor 1's and the float64
# product. Then the per-call time of the kernel alone (C's zeroing included
# for factors above 1): the minimum over 15 rounds, factors shuffled each round.
#
# Usage, from the repo root: python3 docs/benchmarks/exact_y_gemm_2026-11-07/split_sweep.py [--quick]
import os
import random
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
sys.path.insert(0, os.path.join(REPO, "python"))

import torch                                  # noqa: E402

from y_lang import int8_gemm as G             # noqa: E402

SHAPES = {   # (N, K): q/o, k/v, gate/up, down
    "0.5B": ((896, 896), (128, 896), (4864, 896), (896, 4864)),
    "7B": ((3584, 3584), (512, 3584), (18944, 3584), (3584, 18944)),
}
ROWS = (1, 8, 32)
SPLITS = (1, 2, 3, 4, 6, 8, 12, 16)


def main():
    quick = "--quick" in sys.argv
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    print(f"# {props.name}, {props.multi_processor_count} SMs; torch {torch.__version__}", flush=True)
    rng = random.Random(0)
    gen = torch.Generator().manual_seed(3)
    for model, shapes in SHAPES.items():
        for n, k in shapes:
            b = torch.randint(-128, 128, (n, k), generator=gen, dtype=torch.int8).to(dev)
            for m in ROWS:
                a = torch.randint(-128, 128, (m, k), generator=gen, dtype=torch.int8).to(dev)
                mb = G.rows_bucket(m)
                kern, threads, ctas, policy = G.kernel_for(mb, n, k, dev)
                cta_k = int(re.search(r"\| CTA \d+x\d+x(\d+) \|", kern.ptx).group(1))
                k_tiles = -(-k // cta_k)
                ap = torch.nn.functional.pad(a, (0, 0, 0, mb - m)) if mb > m else a
                splits = sorted({z for z in SPLITS if z <= k_tiles} | {policy})

                def run(z):
                    c = torch.empty((mb, n), dtype=torch.int32, device=dev) if z == 1 else \
                        torch.zeros((mb, n), dtype=torch.int32, device=dev)
                    kern.launch((ctas, 1, z), (threads, 1, 1), [ap, b, c])
                    return c

                want = (a.double() @ b.double().t()).cpu()
                for z in splits:
                    got = run(z)[:m].cpu().double()
                    if not torch.equal(got, want):
                        sys.exit(f"{model} {m}x{n}x{k} split {z}: not the exact product")
                best = {z: float("inf") for z in splits}
                reps = 50
                for _ in range(5 if quick else 15):
                    rng.shuffle(splits)
                    for z in splits:
                        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        s.record()
                        for _ in range(reps):
                            run(z)
                        e.record()
                        e.synchronize()
                        best[z] = min(best[z], s.elapsed_time(e) * 1e3 / reps)
                z1 = best[1]
                fastest = min(best, key=best.get)
                cells = "  ".join(f"z{z}:{best[z]:6.1f}" for z in sorted(best))
                print(f"{model:4} M={m:2} N={n:5} K={k:5} ctas={ctas:3} ktiles={k_tiles:3} | {cells} | "
                      f"best z{fastest} {z1 / best[fastest]:.2f}x over z1; split_factor z{policy} "
                      f"{best.get(policy, float('nan')) / best[fastest]:.2f}x the best  bits ok", flush=True)
    import subprocess
    print("# " + subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu",
                                 "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())


if __name__ == "__main__":
    main()
