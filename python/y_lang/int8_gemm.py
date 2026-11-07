"""Exact int8 matmul from PyTorch, on the kernel Y emits and proves.

    from y_lang.int8_gemm import int8_mm
    c = int8_mm(a, b)   # a [M][K] int8, b [N][K] int8, one CUDA device -> c [M][N] int32

`c[i][j]` is the exact integer `sum_k a[i][k] * b[j][k]`: the contract of
`torch._int_mm(a, b.t())`, and the same bits, since both sums are exact. The
kernel is the one `@tile(M, N, K)` emits for `GlobalMemory<I8>` operands
(`src/ptx_emitter/int8_staged_gemm.rs`). `proofs/Int8GemmSchedule.v` and
`proofs/Int8GemmExact.v` prove that, launched as its header says (which is how
this module launches it), it stores every output's exact dot product, for K up
to 131,040.

Refused before anything runs, naming the limit:
  * operands that are not 2-D int8 with the same K (b is the [N][K] matrix
    itself, not its transpose);
  * N not a positive multiple of 8, K not a multiple of 32 from 32 to 131,040
    (the mma is m16n8k32; past 131,040 a full-range dot product leaves int32);
  * operands not on one CUDA device, not contiguous, or not 16-byte aligned
    (each row is copied 16 bytes at a time).
A copy that would make an operand acceptable is not made here, because it
would hide a whole extra pass over a weight matrix.

M is free. The kernel is compiled for a fixed M, and M changes with the batch,
so the rows are zero-padded to `rows_bucket(M)` (a multiple of 16 up to 128
rows, of 128 above) and the extra rows dropped: a zero row adds nothing to any
other row. Compiled kernels are cached for the process per device and shape,
and their PTX on disk by `y_lang.compile_to_ptx`.

Where the output tiles fill at most half the device's SMs, K is split over
`gridDim.z` (`split_factor`) and the partials are added into a zeroed C with
`red.global.add.s32`; a decode step's long-K, few-tile layers need it. The
proofs cover the split for any factor and an integer add lands the same in
any order, so the factor - which depends on the SM count - changes the time,
not the bits.

The PTX targets the tensor's device (`sm_XY` from its compute capability). The
compile still reads the hardware profile, as every `y_lang` compile does
(`YCompilerLib.hw_profile_path`: the working directory's `.ysu_hw_profile`,
or for a wheel's library the user cache), and probes the machine where there
is none; the int8
kernel's PTX depends only on the target, the shape and `Y_INT8_TILE` if set
(`tests/test_int8_mm.py` checks two profiles give the same PTX).

Under `torch.compile` the call is the op `torch.ops.y_lang.int8_mm`, so dynamo
keeps it in the graph rather than breaking at the launch.
"""

import re
import threading

import torch

from .compiler import is_interpreter_enabled
from .torch_interop import TorchKernel

# `INT8_MAX_EXACT_K` in src/ptx_emitter.rs: K * 128^2 <= i32::MAX, rounded down
# to a multiple of 32. The compiler refuses past it too; this says so first.
MAX_K = 131_040

KERNEL_NAME = "y_int8_mm"

_HEADER = re.compile(r"\[Y INT8 TENSOR CORE GEMM\] staged \| M=(\d+) N=(\d+) K=(\d+) \| "
                     r"CTA \d+x\d+x(\d+) \|")
_LAUNCH = re.compile(r"// Launch: block \((\d+),1,1\); any other block traps\. "
                     r"Grid: any; \((\d+), 1, z\)")

_KERNELS = {}
_LOCK = threading.Lock()


def rows_bucket(m: int) -> int:
    """The row count the kernel for an `m`-row product is compiled for."""
    if m <= 128:
        return max(16, -(-m // 16) * 16)
    return -(-m // 128) * 128


def source(m: int, n: int, k: int) -> str:
    """The Y program whose PTX is the m x n x k kernel."""
    return (f"@tile({m}, {n}, {k})\n"
            f"kernel {KERNEL_NAME}(A: GlobalMemory<I8>, B: GlobalMemory<I8>, "
            f"C: GlobalMemory<I32>) {{\n}}\n")


def launch_geometry(ptx: str, m: int, n: int, k: int):
    """`(threads per block, CTAs)` from the kernel's header, which states the
    launch the proofs assume. RuntimeError if the header is missing, so the
    launch is unknown, or names another shape."""
    return _header(ptx, m, n, k)[:2]


def _header(ptx, m, n, k):
    """`(threads per block, CTAs, cta_k)`, as `launch_geometry` checks them."""
    head, launch = _HEADER.search(ptx), _LAUNCH.search(ptx)
    if head is None or launch is None:
        raise RuntimeError(
            "the compiled kernel has no '[Y INT8 TENSOR CORE GEMM] staged' header or no "
            "'Launch: block' line, so its launch is unknown; not launching it")
    if tuple(int(v) for v in head.groups()[:3]) != (m, n, k):
        raise RuntimeError(f"the compiled kernel's header names M, N, K = {head.groups()[:3]}, "
                           f"not {(m, n, k)}")
    return int(launch.group(1)), int(launch.group(2)), int(head.group(4))


def split_factor(ctas: int, k_tiles: int, sms: int) -> int:
    """K's split over `gridDim.z` for a launch of `ctas` output tiles of
    `k_tiles` K tiles each, on a device of `sms` SMs.

    None unless the tiles fill at most half the SMs. Then the smallest factor
    within 10% of the least `waves(z) * ceil(k_tiles / z)`, with
    `waves(z) = ceil(ctas * z / sms)` and at least 8 K tiles a split: a launch
    is as slow as its most loaded SM, so 14 tiles split 5 ways (70 CTAs on 66
    SMs) measured slower than 4 ways (56) or 8 (112). A split of a short K or
    of a full card only adds C's zeroing and the atomics
    (docs/benchmarks/exact_y_gemm_2026-11-07/)."""
    if 2 * ctas > sms:
        return 1
    top = max(1, k_tiles // 8)
    cost = [-(-ctas * z // sms) * -(-k_tiles // z) for z in range(1, top + 1)]
    least = min(cost)
    return next(z for z, c in enumerate(cost, start=1) if c <= 1.1 * least)


def _target(device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}"


def kernel_for(m: int, n: int, k: int, device):
    """`(TorchKernel, threads per block, CTAs, K split)` for an m x n x k
    product on `device`, compiled on first use and cached for the process."""
    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, m, n, k)
    with _LOCK:
        hit = _KERNELS.get(key)
        if hit is None:
            kern = TorchKernel(source(m, n, k), kernel_name=KERNEL_NAME,
                               target_sm=_target(device))
            threads, ctas, cta_k = _header(kern.ptx, m, n, k)
            sms = torch.cuda.get_device_properties(index).multi_processor_count
            hit = (kern, threads, ctas, split_factor(ctas, -(-k // cta_k), sms))
            _KERNELS[key] = hit
    return hit


def check_operands(a, b):
    """Raise TypeError/ValueError, naming the limit, for operands the kernel
    does not take. Shape limits are checked before the device, so they are
    reported for host tensors too."""
    for name, t in (("a", a), ("b", b)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"int8_mm: {name} is a {type(t).__name__}, not a torch.Tensor")
        if t.dtype != torch.int8 or t.dim() != 2:
            raise ValueError(f"int8_mm: {name} must be a 2-D int8 tensor, not a "
                             f"{t.dim()}-D {t.dtype} one")
    m, k = a.shape
    n, kb = b.shape
    if kb != k:
        raise ValueError(f"int8_mm: a is [M][K] = [{m}][{k}], so b must be [N][{k}], not "
                         f"[{n}][{kb}]; b is the [N][K] matrix itself, not its transpose")
    if k % 32 or not 0 < k <= MAX_K:
        raise ValueError(f"int8_mm: K = {k}; the kernel takes K a multiple of 32 (the "
                         f"m16n8k32 mma) from 32 to {MAX_K}, where a full-range int8 dot "
                         f"product still fits int32")
    if n % 8 or n == 0:
        raise ValueError(f"int8_mm: N = {n}; the kernel takes N a positive multiple of 8 "
                         f"(the mma's n8)")
    if not (a.is_cuda and b.is_cuda) or a.device != b.device:
        raise ValueError(f"int8_mm: a and b must be on one CUDA device, not {a.device} and "
                         f"{b.device}")
    for name, t in (("a", a), ("b", b)):
        if not t.is_contiguous():
            raise ValueError(f"int8_mm: {name} must be contiguous, K-contiguous rows; its "
                             f"strides are {tuple(t.stride())}. It is not copied here, "
                             f"because that would hide a whole extra pass over it")
        if t.data_ptr() % 16:
            raise ValueError(f"int8_mm: {name} starts at {t.data_ptr():#x}, which is not "
                             f"16-byte aligned; each row is copied 16 bytes at a time")


def _int8_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    check_operands(a, b)
    if is_interpreter_enabled():
        raise RuntimeError("int8_mm: Y_INTERPRETER or Y_DEBUG is set, which sends y_lang "
                           "launches to the CPU interpreter; this op runs on the GPU only")
    m, k = a.shape
    n = b.shape[0]
    if m == 0:
        return torch.empty((0, n), dtype=torch.int32, device=a.device)
    mb = rows_bucket(m)
    dev = a.device
    kern, threads, ctas, z = _KERNELS.get((dev.index, mb, n, k)) or kernel_for(mb, n, k, dev)
    ap = a if mb == m else torch.nn.functional.pad(a, (0, 0, 0, mb - m))
    # gridDim.z = 1 STORES every output, so C needs no zeroing; a split launch
    # adds its partials into C, which must start at zero.
    if z == 1:
        c = torch.empty((mb, n), dtype=torch.int32, device=dev)
    else:
        c = torch.zeros((mb, n), dtype=torch.int32, device=dev)
    if dev.index == torch.cuda.current_device():
        kern.launch((ctas, 1, z), (threads, 1, 1), [ap, b, c])
    else:
        with torch.cuda.device(dev):
            kern.launch((ctas, 1, z), (threads, 1, 1), [ap, b, c])
    return c if mb == m else c[:m]


def _fake_int8_mm(a, b):
    return a.new_empty((a.shape[0], b.shape[0]), dtype=torch.int32)


# A registration made through `torch.library.Library` lasts as long as the
# library object does, so the object is kept here. A reload of this module
# runs in the same namespace and finds it again instead of dropping it.
_LIB = globals().get("_LIB")


def _register_op():
    """`torch.ops.y_lang.int8_mm`, or None on a torch without `torch.library`.

    Registered through the low-level `torch.library.Library`: a schema, the
    body above for CUDA tensors and for CPU ones (which it refuses with its
    own message rather than the dispatcher's), and a fake for
    `torch.compile`. `torch.library.custom_op`, used until 2026-11-07, wraps
    every call in Python-side dispatch. That cost 9.1 us of host time a call
    against 5.8, and inside a compiled graph 34.7 against 28.3, where a Triton
    launch took 32.4 (docs/benchmarks/exact_y_gemm_2026-11-07/dispatch/). A
    module reloaded in the same process finds the op registered and reuses
    it."""
    global _LIB
    if not hasattr(torch, "library") or not hasattr(torch.library, "Library"):
        return None
    if not hasattr(torch.ops.y_lang, "int8_mm"):
        lib = torch.library.Library("y_lang", "FRAGMENT")
        lib.define("int8_mm(Tensor a, Tensor b) -> Tensor")
        lib.impl("int8_mm", _int8_mm, "CUDA")
        lib.impl("int8_mm", _int8_mm, "CPU")
        torch.library.register_fake("y_lang::int8_mm", _fake_int8_mm, lib=lib)
        _LIB = lib
    return torch.ops.y_lang.int8_mm


_OP = _register_op()


def int8_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`c[M][N] = a[M][K] @ b[N][K]^T` in int32, exact, on Y's proven kernel.

    See the module docstring for what is refused and why."""
    if _OP is not None and torch.compiler.is_compiling():
        return _OP(a, b)
    return _int8_mm(a, b)
