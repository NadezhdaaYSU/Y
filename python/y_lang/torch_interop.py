import ctypes
import os
import re
import struct
from typing import Dict, Tuple, Any
from .compiler import compile_to_ptx, is_interpreter_enabled

_F32 = struct.Struct("<f")
_U32 = struct.Struct("<I")


def _raw_stream_reader():
    """A function returning the current CUDA stream's raw handle, resolved
    once. `torch.cuda.current_stream()` builds a Stream object on every call
    and was 40% of a launch; the raw accessor (Triton's launcher uses it too)
    is a single C call. Without torch, the legacy default stream."""
    try:
        import torch
    except ImportError:
        return lambda: 0
    get_raw = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    get_device = getattr(torch._C, "_cuda_getDevice", None)
    if get_raw is not None and get_device is not None:
        return lambda: get_raw(get_device())
    return lambda: torch.cuda.current_stream().cuda_stream


_current_raw_stream = None

# What a kernel's PTX states it needs beyond its static `.shared`: the tile
# GEMMs stage through `.extern .shared` and say how much in this line.
_DYNAMIC_SHARED = re.compile(r"Dynamic shared memory required: (\d+) bytes")
# The most a kernel may use without opting in through cuFuncSetAttribute.
_DEFAULT_DYNAMIC_SHARED_LIMIT = 48 * 1024
_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8


def _dynamic_shared_bytes(ptx: str, kernel_name: str) -> int:
    """The dynamic shared memory `kernel_name`'s body says it needs, 0 if it
    says nothing. Only that kernel's body is read: a module can hold
    several entries."""
    entry = re.search(r"\.entry\s+" + re.escape(kernel_name) + r"\b", ptx)
    if entry is None:
        return 0
    body = ptx[entry.end():]
    following = re.search(r"\.entry\s+\w+", body)
    if following is not None:
        body = body[:following.start()]
    found = _DYNAMIC_SHARED.search(body)
    return int(found.group(1)) if found else 0

# Load CUDA Driver API (libcuda.so on Linux)
_cuda_lib = None
def _get_cuda_driver():
    global _cuda_lib
    if _cuda_lib is not None:
        return _cuda_lib

    for libname in ["libcuda.so", "libcuda.so.1", "nvcuda.dll", "libcuda.dylib"]:
        try:
            _cuda_lib = ctypes.CDLL(libname)
            break
        except Exception:
            continue

    if _cuda_lib is None:
        raise RuntimeError("CUDA Driver API library (libcuda.so) could not be loaded.")
    return _cuda_lib

_MODULE_CACHE: Dict[str, Tuple[Any, Any]] = {}

class TorchKernel:
    """Wrapper class for loading PTX kernels and executing them directly on PyTorch Tensors."""
    def __init__(self, ysu_source: str, kernel_name: str = "main", target_sm: str = "auto"):
        self.source = ysu_source
        self.kernel_name = kernel_name
        self.target_sm = target_sm
        self.ptx = compile_to_ptx(ysu_source, target_sm=target_sm)
        self._module = None
        self._function = None

    def _ensure_loaded(self):
        if self._function is not None:
            return

        cuda = _get_cuda_driver()

        # cuModuleLoadData(CUmodule *module, const void *image)
        cuModuleLoadData = cuda.cuModuleLoadData
        cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        cuModuleLoadData.restype = ctypes.c_int

        # cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name)
        cuModuleGetFunction = cuda.cuModuleGetFunction
        cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        cuModuleGetFunction.restype = ctypes.c_int

        module = ctypes.c_void_p()
        res = cuModuleLoadData(ctypes.byref(module), self.ptx.encode("utf-8"))
        if res != 0:
            raise RuntimeError(f"CUDA Error cuModuleLoadData failed with error code {res}")

        function = ctypes.c_void_p()
        res = cuModuleGetFunction(ctypes.byref(function), module, self.kernel_name.encode("utf-8"))
        if res != 0:
            # This fell back to an entry named `main`, so a misspelt kernel
            # name ran whatever else the module held under that name.
            raise RuntimeError(
                f"CUDA Error cuModuleGetFunction found no kernel '{self.kernel_name}' in the module (error code {res})"
            )

        # The launch used to pass no dynamic shared memory at all, so every
        # kernel staging through `.extern .shared` (the F16 tile GEMMs) died
        # with an illegal address when launched from Python.
        shared_bytes = _dynamic_shared_bytes(self.ptx, self.kernel_name)
        if shared_bytes > _DEFAULT_DYNAMIC_SHARED_LIMIT:
            cuFuncSetAttribute = cuda.cuFuncSetAttribute
            cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            cuFuncSetAttribute.restype = ctypes.c_int
            res = cuFuncSetAttribute(function, _CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_bytes)
            if res != 0:
                raise RuntimeError(
                    f"'{self.kernel_name}' needs {shared_bytes} bytes of dynamic shared memory and "
                    f"cuFuncSetAttribute refused them (error code {res})"
                )
        self._shared_bytes = shared_bytes

        self._module = module
        self._function = function

        cuLaunchKernel = cuda.cuLaunchKernel
        cuLaunchKernel.argtypes = [
            ctypes.c_void_p,  # f
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, # gridDimX, Y, Z
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, # blockDimX, Y, Z
            ctypes.c_uint, # sharedMemBytes
            ctypes.c_void_p, # hStream
            ctypes.POINTER(ctypes.c_void_p), # kernelParams
            ctypes.POINTER(ctypes.c_void_p)  # extra
        ]
        cuLaunchKernel.restype = ctypes.c_int
        self._cuLaunchKernel = cuLaunchKernel

    def launch(self, grid: Tuple[int, int, int], block: Tuple[int, int, int], args: list, stream: Any = None):
        """Launch the compiled kernel on the GPU, or, with `Y_INTERPRETER=1` or
        `Y_DEBUG=1`, run the same PTX on the CPU interpreter, which writes the
        outputs and stops at the first out-of-bounds access, `trap` or
        divergent barrier (`y_lang.interpreter`)."""
        if is_interpreter_enabled():
            # This used to print "Executed safely under software bounds
            # validation" and return, running nothing and writing nothing.
            from .interpreter import CPUInterpreter
            CPUInterpreter(self.kernel_name, ptx=self.ptx).run(grid, block, args)
            return

        self._ensure_loaded()

        # Prepare kernel argument pointers (Zero Allocation Fast Path)
        num_args = len(args)
        if not hasattr(self, "_args_array") or len(self._args_array) != num_args:
            self._storage_array = [ctypes.c_uint64(0) for _ in range(num_args)]
            self._args_array = (ctypes.c_void_p * num_args)()
            for i in range(num_args):
                self._args_array[i] = ctypes.addressof(self._storage_array[i])

        for slot, arg in zip(self._storage_array, args):
            data_ptr = getattr(arg, "data_ptr", None)
            if data_ptr is not None and callable(data_ptr):
                slot.value = data_ptr()
            elif isinstance(arg, int):
                slot.value = arg
            elif isinstance(arg, float):
                slot.value = _U32.unpack(_F32.pack(arg))[0]
            else:
                raise TypeError(f"Unsupported argument type: {type(arg)}")

        if stream is not None and hasattr(stream, "cuda_stream"):
            stream_ptr = stream.cuda_stream
        else:
            global _current_raw_stream
            if _current_raw_stream is None:
                _current_raw_stream = _raw_stream_reader()
            stream_ptr = _current_raw_stream()

        grid_x, grid_y, grid_z = grid
        block_x, block_y, block_z = block

        res = self._cuLaunchKernel(
            self._function,
            grid_x, grid_y, grid_z,
            block_x, block_y, block_z,
            self._shared_bytes,
            stream_ptr,
            self._args_array,
            None
        )

        if res != 0:
            raise RuntimeError(f"cuLaunchKernel failed with CUDA error code {res}")


    def create_cuda_graph(self, grid: Tuple[int, int, int], block: Tuple[int, int, int], args: list):
        """Captures kernel execution sequence into a persistent CUDA Graph to eliminate launch latency."""
        import torch
        if torch.cuda.is_available():
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self.launch(grid, block, args)
            return g
        return None

    def launch_cuda_graph(self, cuda_graph: Any):
        """Executes a previously captured CUDA Graph with zero CPU dispatch overhead."""
        if cuda_graph is not None:
            cuda_graph.replay()


