# The same tokens on an A100 and an H100, 2026-11-21

## What was asked

Does the exact path decode the same tokens on an NVIDIA A100 (sm_80, Ampere)
and an H100 (sm_90, Hopper) as on the Zen 5 CPU, the Tesla T4 and the RTX 4070
Ti SUPER? Until this run every GPU record was from the T4 or the 4070 Ti.

## How

`modal_same_tokens.py` on Modal (modal.com): one container per GPU, an
`NVIDIA A100-SXM4-40GB` and an `NVIDIA H100 80GB HBM3` (`machine.txt` in each
folder). Y was built in the container image from the sources at 264ad40 (Rust
1.97.1; both containers ran the same liby.so, sha256 `b264cfca54401bcf`).
torch 2.13.0+cu126, transformers 5.15.0, driver 620.70.06.
`tools/exact_cross_device.py --record` with Hugging Face model ids, so the
workload hashes match the earlier records. 5 cases x 160 tokens, greedy.

Two records on each card, Qwen2.5-0.5B-Instruct, eager:

- `*_eager.json`: the exact path at its defaults, and stock bf16 as the
  control (it must change across cards, or the pair shows nothing);
- `*_ygemm_every_eager.json`: the exact path with every switch
  (`Y_EXACT_V_INT8=1 Y_EXACT_NORM=triton Y_EXACT_ATTN=y Y_EXACT_KV_PACK=inplace`)
  and Y's own int8 GEMM in every linear layer (`Y_EXACT_GEMM=y`), so Y's
  attention, `exact_pv` and GEMM kernels ran, compiled by Y for sm_80 and sm_90.

Before renting the GPUs, every Y kernel the exact path launches (`exact_pv`,
`exact_decode_attention`, the int8 GEMM at 60 shapes of the 0.5B and 1.5B
models) was compiled for sm_80 and sm_90 on the development machine and
assembled with ptxas 13.4 for each: 124 of 124 assembled.

## Result

`compare.txt` holds every comparison. Exact's token digest is
`30eec169d2f38da4` in all four records, the CPU's and the T4's digest.

| Record | Against | Exact | Stock (control) |
|---|---|---|---|
| A100, defaults | Zen 5 CPU (`../exact_norm_sum_2026-11-10/cpu_0p5b.json`) | 0/5 changed | |
| H100, defaults | Zen 5 CPU | 0/5 changed | |
| A100, every switch, Y's GEMM | Zen 5 CPU | 0/5 changed | |
| H100, every switch, Y's GEMM | Zen 5 CPU | 0/5 changed | |
| A100, defaults | RTX 4070 Ti SUPER (`../exact_y_gemm_2026-11-07/gpu_sm89_triton_eager.json`) | 0/5 changed | 5/5 changed |
| H100, defaults | RTX 4070 Ti SUPER | 0/5 changed | 5/5 changed |
| A100, defaults | Tesla T4 (`../exact_cross_gpu_t4_2026-11-11/t4_eager.json`) | 0/5 changed | 5/5 changed |
| H100, defaults | Tesla T4 | 0/5 changed | 5/5 changed |
| A100, defaults | H100, defaults | 0/5 changed | 5/5 changed |

The CPU record has no stock arm, so the tool reports those four pairs as
having no control (exit 1); the control is in the GPU pairs, where stock moved
on every case (in the four cases the tool lists, first at token 13 to 72). The integer exp's domain digest is the
CPU's on both cards (`0x7170cd39b442d506`). Each card also decoded every case
twice and got the same tokens both times.

Stock's own digests: A100 `a706c0d5c5f7ea3a`, H100 `ba8d5ce33b0f8e50`.

## Not covered yet

Compiled runs (`torch.compile`), the 1.5B model, and speed. Eager only, one
model size, one model family.

## Cost

$0.59 of Modal's free credit (A100 $0.20, H100 $0.32, CPU and memory $0.07,
from `modal billing report`). The GPUs ran about 6 minutes each; building the
image ran on Modal's CPUs.

## Files

- `modal_same_tokens.py`: the job (`--stage eager` here; `--stage
  compiled_0p5b` and `compiled_1p5b` with `--gpus h100` for the compiled
  records).
- `a100_eager/`, `h100_eager/`: the records, their logs and `machine.txt`.
- `compare.txt`: `exact_cross_device.py --compare` on each pair.
