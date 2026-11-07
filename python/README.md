# y-lang

Python bindings for the Y compiler: compile Y kernels to PTX, launch them on
PyTorch tensors, JIT-compile Y for the CPU, and run Y's exact int8 matrix
multiply. The compiler itself is `liby.so`, which the wheel carries.

## Install

From the repository, build the compiler, then the wheel:

```bash
cargo build --release
python3 -m pip wheel ./python --no-deps -w dist
python3 -m pip install dist/y_lang-*.whl
```

The wheel build refuses a missing `liby.so`, and one older than the compiler
sources beside it. `Y_WHEEL_LIBY=/path/to/liby.so` names a library built
elsewhere.

Requirements:
- Linux on x86-64. A library built on a current distribution needs glibc
  2.34 or newer (Ubuntu 22.04, Debian 12, RHEL 9).
- PyTorch, and an NVIDIA driver for the GPU paths.
- Optional: a C compiler and this Python's headers. With them, kernel launches
  go through a small native module built on first use; without them, through
  ctypes, which is slower per call.

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

## Known limitation

A compile reads `.ysu_hw_profile` from the working directory and, where there
is none, writes one there. Outside the repository the hardware probe is not
found, so that file names an unknown GPU and kernels that take their target
from it are built for `sm_80`. `int8_mm` takes its target from the tensor's
device instead.

## Licence

Business Source License 1.1 (`LICENSE`): free for personal, educational,
research and internal evaluation use. A commercial product or hosted service
needs a commercial licence. On 2030-01-01 the licence becomes Apache 2.0.
Third-party files are listed in `NOTICE`.
