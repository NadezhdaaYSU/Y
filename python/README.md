# y-lang

Python bindings for the Y compiler: compile Y kernels to PTX, launch them on
PyTorch tensors, JIT-compile Y for the CPU, and run Y's exact int8 matrix
multiply. The compiler itself is `liby.so`, which the wheel carries with its
GPU probe, `ysu_gpu_probe`.

## Install

From the repository, build the compiler, then the wheel:

```bash
cargo build --release
python3 -m pip wheel ./python --no-deps -w dist
python3 -m pip install dist/y_lang-*.whl
```

The wheel build refuses a missing `liby.so` or `ysu_gpu_probe`, and one older
than a source it was built from (as cargo's dep-info beside it lists them;
without that, any compiler source). `Y_WHEEL_LIBY` and `Y_WHEEL_PROBE` name
files built elsewhere.

Requirements:
- Linux on x86-64. A library built on a current distribution needs glibc
  2.34 or newer (Ubuntu 22.04, Debian 12, RHEL 9).
- PyTorch, and an NVIDIA driver for the GPU paths.
- Optional: a C compiler and this Python's headers. With them, kernel launches
  go through a small native module built on first use; without them, through
  ctypes, which is slower per call.
- For the CPU JIT (`CPUJit`) only: LLVM as a shared library, which it loads
  when first used (`libLLVM-17.so` to `libLLVM-24.so` are looked for, or
  `Y_LLVM_LIBRARY=<path>`). LLVM 21 and 23 run it; AlmaLinux 8's LLVM 17
  compatibility library loads but fails to create the JIT (see the
  repository's `docs/known_issues.md`).
- Older Linux: `tools/wheel_glibc228/build.sh` (Docker) builds the binaries
  on glibc 2.28, where they need no glibc symbol newer than 2.17, and tests
  the wheel there (only 2.28 is tested).

## Reproducible inference

```bash
python3 -m pip install dist/y_lang-*.whl transformers
```

```python
from y_lang.exact import load, generate

model, tokenizer = load("Qwen/Qwen2.5-0.5B-Instruct", device="cuda")
tokens = generate(model, tokenizer, "The capital of France is", max_new_tokens=32)
```

The same prompt decodes to the same tokens whatever else is in its batch, and
the same tokens on a CPU and a GPU: every reduction whose order can change is
done in integers. The model is quantised (int8 weights, and quantised
activations and K/V cache), so its tokens are not the bf16 model's. Measured
results and their limits are in the repository's
`docs/bit_identical_decode.md`.

- Qwen2-architecture checkpoints only; another architecture is refused.
- `load(..., low_memory=True)` builds a bf16 checkpoint layer by layer, for
  models such as Qwen2.5-7B.
- On a GPU, `Y_EXACT_GEMM=y` runs the linear layers on Y's proved int8 kernel
  (default: Triton) and `Y_EXACT_V_INT8=1` runs attention's p.V on Y's
  `exact_pv` kernel; every choice gives the same bits. Set them before
  importing `y_lang.exact.model`. q, k and v run as one GEMM
  (`Y_EXACT_MERGE`: `qkv`, the default; `all` also merges gate and up; `0`
  none), with the same bits.
- For a batch decoded to a fixed length, `y_lang.exact.greedy.greedy(model,
  input_ids, attention_mask, n)` takes the argmax on the device and does not
  wait for the GPU until the end (3% faster than `generate` at 7B, batch 32).
  It applies no logits processor, so it matches `generate(...,
  repetition_penalty=1.0)`, and end-of-sequence tokens do not stop it.
- Loading replaces `transformers`' Qwen2 eager attention for the whole
  process, and importing the model module sets two `torch.compile` rounding
  flags process-wide; see `y_lang.exact`'s docstring.

## Exact int8 matrix multiply

```python
import torch
from y_lang.int8_gemm import int8_mm

a = torch.randint(-128, 128, (64, 4096), dtype=torch.int8, device="cuda")
b = torch.randint(-128, 128, (1024, 4096), dtype=torch.int8, device="cuda")
c = int8_mm(a, b)  # int32 [64, 1024]: c[i][j] = sum_k a[i][k] * b[j][k], exactly
```

Every output is the exact integer dot product, the same bits as
`torch._int_mm(a, b.t())`. The kernel is the one `proofs/Int8GemmSchedule.v` and
`proofs/Int8GemmExact.v` prove. N must be a multiple of 8, K a multiple of 32 up
to 131,040, and both operands contiguous and 16-byte aligned; anything else is
refused before it runs. Under `torch.compile` it is the op
`torch.ops.y_lang.int8_mm`.

## Hardware profile and caches

The first compile runs the bundled `ysu_gpu_probe`, which runs
microbenchmarks on the GPU, and saves what it measured as `hw_profile` in
`$YSU_CACHE_DIR`, else `$XDG_CACHE_HOME/y-lang`, else `~/.cache/y-lang`. Later
compiles read it. Compiled PTX is cached in `$YSU_CACHE_DIR` itself if set,
else in `ptx/` beside the profile. Nothing is written into the working
directory.

- A `.ysu_hw_profile` in the working directory is used instead, if present.
- `YSU_HW_PROFILE=<file>` names the profile outright.
- Delete the profile after changing the GPU or driver, so it is measured again.
- Compiles print nothing on stdout; a compile that measures the machine says
  so in one line on stderr. `Y_VERBOSE=1` prints the profile as the `Y`
  command line does.

## Licence

Business Source License 1.1 (`LICENSE`): free for personal, educational,
research and internal evaluation use. A commercial product or hosted service
needs a commercial licence. On 2030-01-01 the licence becomes Apache 2.0.
Third-party files in the repository are listed in `NOTICE`. The binaries also
contain eight MIT-licensed Rust crates (`THIRD_PARTY_NOTICES`, generated by
`tools/third_party_notices.py`) and the Rust standard library
(`y_lang/RUST_STD_COPYRIGHT.html`, copied from the toolchain at build time).
