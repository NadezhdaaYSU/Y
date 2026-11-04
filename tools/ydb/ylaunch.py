"""Launch one Y kernel from the command line: the host program a kernel needs,
so `ydb --gpu` can debug a kernel nobody has written a host for.

    python3 tools/ydb/ylaunch.py prog.ptx 'saxpy<<<1, 32>>>(arange(32), ones(32), 20)'

The PTX is `Y prog.ysu --emit-ptx -g`'s. Each parameter's type is read from
the debug information it carries - the description cuda-gdb uses - and a
launch whose arguments do not fit is refused before anything runs.

    KERNEL<<<GRID, BLOCK>>>(ARG, ...)           GRID, BLOCK: N or (x, y, z)
    KERNEL<<<GRID, BLOCK, SHARED>>>(ARG, ...)   SHARED: dynamic shared bytes

Each ARG, in the kernel's parameter order or as NAME=ARG:

    7, -2, 0.5          a scalar parameter, in the parameter's own type
    arange(n)           a buffer 0, 1, ..., n-1, in the buffer's element type
    zeros(n), ones(n)   n zeros, n ones
    full(n, v)          n copies of v
    [a, b, ...]         these values

A value the element type cannot hold exactly (0.5 into an I32 buffer, 300
into a U8 one, 2**24 + 1 into F32) is refused, not rounded. After the kernel,
every buffer is printed, so a run outside the debugger shows what it computed.

The kernel runs in the device's primary context, through the CUDA driver API
(libcuda), with nothing compiled: the driver JIT-compiles the PTX, and for a
`.target ..., debug` module that is debug code cuda-gdb can stop in.
"""
import ast
import ctypes
import math
import os
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ydwarf  # noqa: E402


class LaunchError(Exception):
    pass


# Y base type -> (struct code, integer?, ctypes type for a scalar parameter).
_TYPES = {
    "I8": ("b", True, ctypes.c_int8), "U8": ("B", True, ctypes.c_uint8),
    "I16": ("h", True, ctypes.c_int16), "U16": ("H", True, ctypes.c_uint16),
    "I32": ("i", True, ctypes.c_int32), "U32": ("I", True, ctypes.c_uint32),
    "I64": ("q", True, ctypes.c_int64), "U64": ("Q", True, ctypes.c_uint64),
    "F16": ("e", False, None), "F32": ("f", False, ctypes.c_float), "F64": ("d", False, ctypes.c_double),
    "E4M3_bits": ("B", True, ctypes.c_uint8),
}


def _exact(value, tname):
    """`value` as element type `tname`, or LaunchError if it would change."""
    code, integer, _ = _TYPES[tname]
    if integer:
        if isinstance(value, float):
            if not value.is_integer():
                raise LaunchError("%r is not an integer, and %s holds integers" % (value, tname))
            value = int(value)
        try:
            struct.pack("<" + code, value)
        except struct.error:
            raise LaunchError("%d does not fit in %s" % (value, tname)) from None
        return value
    value = float(value)
    try:
        back = struct.unpack("<" + code, struct.pack("<" + code, value))[0]
    except (struct.error, OverflowError):
        raise LaunchError("%r does not fit in %s" % (value, tname)) from None
    if back != value and not (math.isnan(back) and math.isnan(value)):
        raise LaunchError("%r is not exactly representable in %s (it would be %r)" % (value, tname, back))
    return value


def _number(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _number(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        return _number(node.left) ** _number(node.right)
    if isinstance(node, ast.Name) and node.id in ("inf", "nan"):
        return float(node.id)
    raise LaunchError("expected a number, got `%s`" % ast.unparse(node))


def _count(node):
    n = _number(node)
    if not isinstance(n, int) or n <= 0:
        raise LaunchError("a buffer's length must be a positive integer, got %r" % n)
    return n


def _buffer(node):
    """A buffer argument as a list of numbers, or None if it is a scalar."""
    if isinstance(node, ast.List):
        if not node.elts:
            raise LaunchError("an empty buffer has nothing to launch on")
        return [_number(e) for e in node.elts]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        f, args = node.func.id, node.args
        if node.keywords:
            raise LaunchError("%s takes positional arguments only" % f)
        if f == "arange" and len(args) == 1:
            return list(range(_count(args[0])))
        if f in ("zeros", "ones") and len(args) == 1:
            return [0 if f == "zeros" else 1] * _count(args[0])
        if f == "full" and len(args) == 2:
            return [_number(args[1])] * _count(args[0])
        raise LaunchError("unknown buffer `%s`: arange(n), zeros(n), ones(n), full(n, v) or [a, b, ...]"
                          % ast.unparse(node))
    return None


def _dims(text):
    try:
        v = ast.literal_eval(text.strip())
    except (ValueError, SyntaxError):
        raise LaunchError("`%s` is not a dimension: N or (x, y, z)" % text) from None
    v = (v,) if isinstance(v, int) else tuple(v)
    if not 1 <= len(v) <= 3 or any(not isinstance(x, int) or x <= 0 for x in v):
        raise LaunchError("`%s` is not a dimension: N or (x, y, z), positive" % text)
    return v + (1,) * (3 - len(v))


def parse_launch(spec, ptx):
    """(kernel, grid, block, shared, [(param, value or list)]) for `spec`,
    checked against the kernel's debug information in `ptx`."""
    m = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*<<<(.*?)>>>\s*\((.*)\)\s*", spec, re.S)
    if not m:
        raise LaunchError("a launch is KERNEL<<<GRID, BLOCK>>>(ARG, ...): got `%s`" % spec)
    name, conf, args = m.groups()
    try:
        described = ydwarf.kernels(ptx)
    except ydwarf.DwarfError as e:
        raise LaunchError(str(e)) from None
    if name not in described:
        raise LaunchError("the module has no kernel `%s`; it has %s" % (name, ", ".join(sorted(described)) or "none"))
    params = described[name]["params"]
    try:
        tree = ast.parse("_(%s)" % conf, mode="eval").body
        dims = [ast.unparse(a) for a in tree.args]
    except SyntaxError:
        raise LaunchError("`<<<%s>>>` is not GRID, BLOCK[, SHARED]" % conf) from None
    if len(dims) not in (2, 3) or tree.keywords:
        raise LaunchError("`<<<%s>>>` is not GRID, BLOCK[, SHARED]" % conf)
    grid, block = _dims(dims[0]), _dims(dims[1])
    shared = 0
    if len(dims) == 3:
        shared = _number(tree.args[2])
        if not isinstance(shared, int) or shared < 0:
            raise LaunchError("SHARED is a byte count, got %r" % shared)
    try:
        call = ast.parse("_(%s)" % args, mode="eval").body
    except SyntaxError as e:
        raise LaunchError("the arguments do not parse: %s" % e.msg) from None
    given = {}
    for i, a in enumerate(call.args):
        if i >= len(params):
            raise LaunchError("%s takes %d arguments, got %d" % (name, len(params), len(call.args)))
        given[params[i]["name"]] = a
    for kw in call.keywords:
        if kw.arg is None or kw.arg not in [p["name"] for p in params]:
            raise LaunchError("%s has no parameter `%s`" % (name, kw.arg))
        if kw.arg in given:
            raise LaunchError("`%s` is given twice" % kw.arg)
        given[kw.arg] = kw.value
    missing = [p["name"] for p in params if p["name"] not in given]
    if missing:
        raise LaunchError("no argument for %s" % ", ".join(missing))
    out = []
    for p in params:
        node, t = given[p["name"]], p["type"]
        buf = _buffer(node)
        if t["kind"] == "pointer":
            elem = t["target"].get("name")
            if elem not in _TYPES:
                raise LaunchError("`%s` points to %s, which the launcher cannot fill" % (p["name"], elem))
            if buf is None:
                raise LaunchError("`%s` is a buffer of %s: give arange(n), zeros(n), ones(n), full(n, v) "
                                  "or [a, b, ...]" % (p["name"], elem))
            try:
                out.append((p, [_exact(v, elem) for v in buf]))
            except LaunchError as e:
                raise LaunchError("`%s`: %s" % (p["name"], e)) from None
        elif t["kind"] == "base" and t["name"] in _TYPES and _TYPES[t["name"]][2] is not None:
            if buf is not None:
                raise LaunchError("`%s` is a %s, not a buffer" % (p["name"], t["name"]))
            try:
                out.append((p, _exact(_number(node), t["name"])))
            except LaunchError as e:
                raise LaunchError("`%s`: %s" % (p["name"], e)) from None
        else:
            raise LaunchError("`%s` is a %s, which the launcher cannot pass" % (p["name"], t.get("name")))
    return name, grid, block, shared, out


class Driver:
    """The few CUDA driver calls a launch needs."""

    def __init__(self):
        try:
            self.cu = ctypes.CDLL("libcuda.so.1")
        except OSError as e:
            raise LaunchError("no CUDA driver (libcuda.so.1): %s" % e) from None
        self.check(self.cu.cuInit(0), "cuInit")
        dev = ctypes.c_int()
        self.check(self.cu.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
        self.ctx = ctypes.c_void_p()
        self.check(self.cu.cuDevicePrimaryCtxRetain(ctypes.byref(self.ctx), dev), "cuDevicePrimaryCtxRetain")
        self.check(self.cu.cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")

    def check(self, err, what):
        if err != 0:
            s = ctypes.c_char_p()
            self.cu.cuGetErrorString(err, ctypes.byref(s))
            raise LaunchError("%s failed: %s (%d)" % (what, (s.value or b"?").decode(), err))

    def alloc(self, data):
        ptr = ctypes.c_uint64()
        self.check(self.cu.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(len(data))), "cuMemAlloc")
        self.check(self.cu.cuMemcpyHtoD_v2(ptr, data, ctypes.c_size_t(len(data))), "cuMemcpyHtoD")
        return ptr

    def read(self, ptr, n):
        buf = ctypes.create_string_buffer(n)
        self.check(self.cu.cuMemcpyDtoH_v2(buf, ptr, ctypes.c_size_t(n)), "cuMemcpyDtoH")
        return buf.raw

    def launch(self, ptx, kernel, grid, block, shared, slots):
        mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
        log = ctypes.create_string_buffer(16384)
        # CU_JIT_ERROR_LOG_BUFFER (5) and its size (6), so a module the JIT
        # refuses says why.
        opts = (ctypes.c_int * 2)(5, 6)
        vals = (ctypes.c_void_p * 2)(ctypes.cast(log, ctypes.c_void_p).value, ctypes.c_void_p(16384))
        err = self.cu.cuModuleLoadDataEx(ctypes.byref(mod), ctypes.c_char_p(ptx.encode()), 2, opts, vals)
        if err:
            raise LaunchError("the driver refused the PTX (%d): %s" % (err, log.value.decode(errors="replace")))
        self.check(self.cu.cuModuleGetFunction(ctypes.byref(fn), mod, kernel.encode()), "cuModuleGetFunction")
        if shared > 48 * 1024:
            # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            self.check(self.cu.cuFuncSetAttribute(fn, 8, shared), "cuFuncSetAttribute")
        args = (ctypes.c_void_p * len(slots))(*[ctypes.cast(ctypes.byref(s), ctypes.c_void_p) for s in slots])
        self.check(self.cu.cuLaunchKernel(fn, *grid, *block, shared, None, args, None), "cuLaunchKernel")
        self.check(self.cu.cuCtxSynchronize(), "the kernel")


def _show(x):
    return ("%.9g" % x) if isinstance(x, float) else str(x)


def run(ptx_path, spec, out=sys.stdout):
    with open(ptx_path) as f:
        ptx = f.read()
    kernel, grid, block, shared, args = parse_launch(spec, ptx)
    drv = Driver()
    slots, buffers = [], []
    for p, v in args:
        if isinstance(v, list):
            code = _TYPES[p["type"]["target"]["name"]][0]
            ptr = drv.alloc(struct.pack("<%d%s" % (len(v), code), *v))
            slots.append(ptr)
            buffers.append((p, ptr, len(v), code))
        else:
            slots.append(_TYPES[p["type"]["name"]][2](v))
    drv.launch(ptx, kernel, grid, block, shared, slots)
    print("%s<<<(%d,%d,%d), (%d,%d,%d)>>> finished" % ((kernel,) + grid + block), file=out)
    for p, ptr, n, code in buffers:
        vals = struct.unpack("<%d%s" % (n, code), drv.read(ptr, n * struct.calcsize(code)))
        shown = [_show(x) for x in vals] if n <= 16 else \
            [_show(x) for x in vals[:12]] + ["..."] + [_show(x) for x in vals[-4:]]
        print("  %s: %d x %s = [%s]" % (p["name"], n, p["type"]["target"]["name"], ", ".join(shown)), file=out)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__.strip().split("\n\n")[1], file=sys.stderr)
        return 2
    try:
        run(argv[0], argv[1])
    except (LaunchError, OSError) as e:
        print("ylaunch: %s" % e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
