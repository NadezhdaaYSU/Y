# Y's exact path against samebit and tritify (2026-11-12)

Two other open-source projects run a language model in integer arithmetic so
that every device gives the same bits: [samebit](https://github.com/Shakhtar-Sankur/samebit)
(Python, CuPy kernels) and [tritify](https://github.com/pelloran-inc/tritify)
(Rust, arXiv 2611.13730). This is Y's exact path beside them on one model,
`Qwen/Qwen2.5-0.5B-Instruct` at revision `7ae557604adf67be50417f59c2c2f167def9a775`,
on two machines:

- **sm_89**: RTX 4070 Ti SUPER and a Ryzen 9 9950X (the development machine),
  torch 2.13.0+cu130.
- **T4**: a Kaggle notebook, Tesla T4 (sm_75) and a 4-core `Intel(R) Xeon(R)
  CPU @ 2.00GHz`, torch 2.13.0 (cu126), Y from the wheel of `84a4429` with the
  repository's `python/` at `2358bfb`.

samebit is at `c6f3530`, tritify at `db52e0c` with Rust 1.90.0 (its pinned
toolchain). A third project, detllm, runs only Qwen3-0.6B and was not run.

**The three are different integer models.** Each quantises the checkpoint its
own way, so their outputs differ from one another; what is compared is each
one's bits across devices, its distance from the float model, and its speed.

## Bits

Every system reproduces itself on every device it ran on.

| system | check | result |
|---|---|---|
| samebit | all logits (256 x 151,936) of 2 WikiText-2 windows, GPU vs its NumPy CPU forward | identical on both machines; model fingerprint `aba0455663f2b5ac` on both |
| Y | the same 2 windows, all logits, GPU vs CPU; compiled vs eager on window 0 | identical on both machines |
| Y | every step's whole logits tensor of a generation, chained into one SHA-256 | `817e619db67c8ab0` (chat request) and `3587dd7d1f6429a2` (128 tokens) on the 4070 Ti, the 9950X, the T4 and the Xeon, eager and compiled |
| tritify | every step's logits (its `logits_hash`) | `d0f296c4...` (chat) and `a8788579...` (128) on all 7 backend runs on sm_89 and all 6 on the T4 machine, and equal across the two machines |

With `Y_EXACT_ACT_LEVELS=511` Y is a different integer model: the 128-token
hash is `46331748029c5a57` on all four devices again, and the chat request's
(`92efa93ea9f6...`, run on the T4 machine only) is the same on the T4 and the
Xeon.

The mean NLL of the integer systems is the same on both GPUs to every printed
digit (Y `3.27957`, samebit `3.23514`); float16's is not (`3.23385` on the
4070 Ti, `3.23396` on the T4).

## Accuracy, on identical tokens

`harness/accuracy_same_tokens.py`: samebit's own protocol (WikiText-2-raw-v1
test, joined, tokenised once, 40 windows of 256, next-token NLL at every
position: 10,240 scored tokens, ids SHA-256 `8f37fdcb021f2ad7...`). Every
system gets the same ids; metrics in float64; the confidence interval is a
bootstrap over windows of the paired per-token difference. Per-token NLLs are
in `sm89/accuracy_act*_nll.npz`.

| | NLL/token | PPL | dNLL vs fp16 [95% CI] | KL vs fp16 | same top-1 as fp16 |
|---|---|---|---|---|---|
| fp16 | 3.23385 | 25.377 | | | |
| fp32 | 3.23421 | 25.386 | +0.0004 [+0.0001, +0.0007] | 0.00003 | 99.7% |
| samebit | 3.23514 | 25.410 | +0.0013 [-0.0038, +0.0065] | 0.017 | 92.8% |
| Y, default (activations at 127 levels) | 3.27957 | 26.564 | **+0.0457 [+0.0365, +0.0545]** | 0.060 | 87.0% |
| Y, `Y_EXACT_ACT_LEVELS=511` | 3.24149 | 25.572 | +0.0076 [+0.0018, +0.0136] | 0.024 | 93.6% |

- **Y's default is less accurate than samebit**: +4.7% perplexity against
  +0.13%, lower NLL on 4 of 40 windows. samebit carries 16-bit activations;
  Y's default is 8-bit.
- **At 511 levels Y and samebit are not separable on this sample**: Y minus
  samebit is +0.0064 NLL/token, 95% CI [-0.0011, +0.0136], Y lower on 17 of
  40 windows. Y is still measurably above fp16 (+0.77% perplexity); samebit's
  difference from fp16 is inside its interval.
- tritify's accuracy was not measured: this harness has no way to read its
  logits as a tensor. Its 128-token greedy continuation leaves Y's at step 17.

## Prefill: samebit against Y

`harness/prefill_vs_samebit.py`. One 256-token prompt (WikiText-2 test window
0), batch 1, logits of every position; one process, every arm warm, 11 rounds
in a shuffled order, each sample the mean of 5 forwards. Milliseconds.

| RTX 4070 Ti SUPER | min | median | mean | sd |
|---|---|---|---|---|
| samebit | 192.11 | 193.49 | 194.13 | 2.48 |
| Y exact, eager | 39.15 | 39.60 | 39.92 | 1.08 |
| Y exact, compiled | 12.22 | 12.53 | 12.53 | 0.19 |
| PyTorch float16 | 6.81 | 6.96 | 6.97 | 0.17 |
| Y exact at 511, eager | 43.52 | 44.13 | 44.25 | 0.43 |
| Y exact at 511, compiled | 14.52 | 14.72 | 14.76 | 0.27 |

| Tesla T4 | min | median | mean | sd |
|---|---|---|---|---|
| samebit | 505.28 | 524.83 | 553.20 | 72.48 |
| Y exact, eager | 208.64 | 212.14 | 214.17 | 6.64 |
| Y exact, compiled | 46.38 | 50.10 | 52.30 | 6.05 |
| PyTorch float16 | 37.68 | 38.42 | 41.43 | 8.45 |
| Y exact at 511, eager | 239.62 | 243.55 | 251.19 | 22.62 |
| Y exact at 511, compiled | 67.93 | 72.75 | 77.40 | 14.02 |

By medians: Y eager is 4.9x samebit's speed on the 4070 Ti and 2.5x on the T4;
compiled, 15.4x and 10.5x. **Eager is the like-for-like row**: samebit has no
compiled mode (CuPy builds its kernels when first called), and `torch.compile`
is something Y's path survives, not something samebit was offered. samebit's
T4 time agrees with its own published 482 ms for the base model. Its v0.2
forward takes a whole sequence and we found no incremental decode, so it is
compared at prefill only.

## Decode: tritify against Y

tritify's own request (`generate_backends`: a system and a user message through
Qwen's chat template, greedy). The 31 prompt ids are equal one by one
(`PROMPT_IDS` in both logs). Two runs: to EOS (both give the same 7 tokens,
"The capital of France is Paris."), and exactly 128 tokens with EOS ignored.
`harness/tritify_bench.patch` adds only timing and printing to tritify's
example; `harness/chat_request_y2.py` is Y's side. Warm, 5 runs of 128 tokens,
medians, milliseconds. tritify feeds the prompt one position at a time, so the
first token (prompt plus first choice) and the per-token decode after it are
reported apart.

| sm_89 machine | first token | decode per token | total, 128 tokens |
|---|---|---|---|
| tritify, CUDA (int8 tensor cores) | 689.4 | 27.5 | 4186 |
| tritify, wgpu | 767.8 | 30.1 | 4574 |
| Y GPU, eager | 40.3 | **38.5** | 4924 |
| Y GPU, compiled | 5.3 | 3.4 | 442 |
| tritify, AVX-512 (9950X) | 1332.1 | 49.1 | 7560 |
| Y CPU (9950X) | 59.2 | 28.5 | 3677 |
| Y GPU at 511, eager / compiled | 43.2 / 5.3 | 41.0 / 3.4 | 5251 / 435 |
| Y CPU at 511 | 81.8 | 37.3 | 4826 |

| T4 machine | first token | decode per token | total, 128 tokens |
|---|---|---|---|
| tritify, AVX-512 (Xeon, 4 cores) | 4276 | **167.3** | 25,377 |
| tritify, AVX2 | 4365 | 167.2 | 25,523 |
| Y GPU, eager | 175.2 | 170.9 | |
| Y GPU, compiled | 19.4 | 17.0 | 2179 |
| Y CPU (Xeon, 4 cores) | 7250 | **451.0** | 64,525 |
| Y GPU at 511, eager / compiled | 190.1 / 31.9 | 185.5 / 29.9 | / 3828 |
| Y CPU at 511 | 14,240 | 794.7 | 115,163 |

What this says, including where Y loses:

- **Eager, Y decodes more slowly than tritify's CUDA backend** on the 4070 Ti
  (38.5 against 27.5 ms per token): Y's eager decode is bound by Python and
  kernel launches. Compiled it is 8x faster than tritify per token.
- **On the 4-core Xeon tritify's CPU path is 2.7x faster than Y's** (167
  against 451 ms per token, and 4.3 s against 7.3 s to the first token). On
  the 16-core 9950X Y's is 1.7x faster. Y's CPU path is torch's CPU kernels.
- tritify's first token is slow everywhere because the prompt goes in one
  position at a time; that is its design, not a measurement artefact.
- **No tritify GPU number on the T4.** Its CUDA backend did not appear among
  the backends there (the build reported no error; cause not found), and its
  wgpu backend got `llvmpipe`, a software rasteriser (1254 ms per token), so
  those rows are left out.
- **511 levels cost little on sm_89 and a lot on the T4.** Compiled decode is
  3.4 ms per token either way on the 4070 Ti, but 17.0 against 29.9 on the T4,
  where the linear layers run on `torch._int_mm` and the wider activation is
  two of them. (After the weight-order fix, 12.75 against 16.7; see below.)

## T4 again, after the weight-order fix (2026-11-18)

On the T4 Y's linear layers run on `torch._int_mm`, and their int8 weights
were stored in the order that keeps cuBLAS off the tensor cores
(`docs/benchmarks/exact_t4_profile_2026-11-17`; fixed in 56847c5). The same
notebook ran again on 2026-11-18 with Y at 366bed6 (version 2 of the
Kaggle job, another T4 machine of the same type). Logs and the printed
output are in `t4_after_weight_order_fix/`.

The fix changes no bits. Every logits hash in the new run equals the old
one, on the GPU and on the CPU, and both accuracy JSON files are identical
to the old ones, all 53 values.

Prefill, 256 tokens, batch 1, milliseconds:

| Tesla T4 | median before | median after | min after |
|---|---|---|---|
| samebit | 524.83 | 559.62 | 540.19 |
| Y exact, eager | 212.14 | 205.31 | 202.16 |
| Y exact, compiled | 50.10 | 43.54 | 40.01 |
| PyTorch float16 | 38.42 | 38.12 | 37.26 |
| Y exact at 511, eager | 243.55 | 229.49 | 226.66 |
| Y exact at 511, compiled | 72.75 | 59.34 | 54.55 |

By medians Y eager is now 2.7x samebit's speed on the T4 (2.5x before), and
compiled 12.9x (10.5x). samebit itself ran 7% slower on this machine.

Decode, batch 1, medians, milliseconds:

| T4 machine | first token | decode per token | total, 128 tokens |
|---|---|---|---|
| tritify, AVX-512 (Xeon, 4 cores) | 4517 | 175.8 | 26,979 |
| tritify, AVX2 | 4669 | 179.0 | 27,407 |
| Y GPU, eager | 171.4 | 166.7 | 21,349 |
| Y GPU, compiled | 13.8 | **12.75** (17.0 before) | 1633 |
| Y CPU (Xeon, 4 cores) | 7297 | 453.2 | 64,685 |
| Y GPU at 511, eager / compiled | 200.4 / 16.7 | 194.2 / **16.71** (29.9 before) | 24,856 / 2140 |
| Y CPU at 511 | 15,060 | 853.7 | 123,643 |

Width 511 now costs 1.3x in compiled decode on the T4 (16.7 against 12.75
ms per token), where it cost 1.8x before. The CPU rows do not use the
fix, and they moved by under 1% (Y) and 5% to 7% (tritify) with the
machine. tritify's CUDA backend again did not appear on the T4.

## Provenance

- samebit's published calibration is for the base model. For Instruct its
  converter was run as shipped with `--recalibrate` (its defaults: the first
  400,000 characters of WikiText-2-raw-v1 train, 16 windows of 256 tokens,
  margin 2.0); the resulting maxima are `harness/samebit_calibration_instruct.json`,
  and the Kaggle run reuses them, which is why both machines build fingerprint
  `aba0455663f2b5ac` (samebit's published base-model one is `e1994bc2e4f9bff7`).
- WikiText: `Salesforce/wikitext` at `b08601e04326c79dfdd32d625aee71d232d685c3`.
- `harness/make_rivals_job.py` writes the Kaggle notebook (every version
  pinned); `t4/notebook_output.txt` is what it printed. Kaggle's file listing
  rate-limited before the T4's per-step decode logs could be fetched, so those
  numbers are read from that output.
- `sm89/` holds the development machine's logs: `tritify_*.log`,
  `y_{chat,128}_{eager,compiled}_act*.{log,json}`, `prefill_act*`,
  `accuracy_act*`.

## Not covered

One model at 0.5B; one prompt per timing; batch 1; no accuracy for tritify;
detllm not run; no ARM or Apple device, where samebit and tritify both report
results and Y has none.
