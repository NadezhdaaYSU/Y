# Qwen2.5-7B-Instruct across a CPU and a GPU, 2026-11-03/04

The two `tools/exact_cross_device.py` records behind "At 7B: exact 0/5, stock
5/5" in `docs/bit_identical_decode.md` (Finding 12, the pipeline). Re-check them
with no GPU and no model:

    python3 tools/exact_cross_device.py --compare \
        docs/benchmarks/exact_cross_device_7b_2026-11-04/cpu_zen5_eager.json \
        docs/benchmarks/exact_cross_device_7b_2026-11-04/gpu_sm89_exact-compiled_stock-eager.json

One machine: Zen 5 (32 threads) and an RTX 4070 Ti SUPER (sm_89, driver 13.0),
PyTorch 2.13.0+cu130, transformers 5.15.0, Python 3.14.8. 5 cases x 160 greedy
tokens, both arms. The two CPU arms and the GPU exact arm were recorded on
2026-11-03 at `fc7a981`, the GPU stock arm on 2026-11-04 at `a3aa581`.

- `cpu_zen5_eager.json`: both arms eager on the CPU, recorded in one process
  per arm and merged; the exact arm on the int8 `torch._int_mm` path.
- `gpu_sm89_exact-compiled_stock-eager.json`: the exact arm compiled
  (`torch.compile(dynamic=True)`), the stock arm eager with the token embedding
  and `lm_head` on the host (`--stock-lm-head-on-host`), recorded in separate
  processes and merged. The merge requires equal device blocks, configs and
  integer-exp digests. These records predate the `build` field
  `exact_cross_device` now writes, so how each arm was built is stated here.

Both 7B builds use `--low-memory` (`tools/exact_large_model.py`): bf16 loaded
on the host, each Linear upcast and quantised on the target device. The stock
GPU record fitted the 16 GB card only after Discord was closed; with it open
the batch-32 case ran out of memory.
