import ctypes, sys, torch
torch.cuda.init(); torch.zeros(1, device="cuda")
cu = ctypes.CDLL("libcuda.so.1")
for path in sys.argv[1:]:
    mod = ctypes.c_void_p(); f = ctypes.c_void_p()
    data = open(path, "rb").read() + b"\0"
    r = cu.cuModuleLoadData(ctypes.byref(mod), ctypes.c_char_p(data)); assert r == 0, r
    r = cu.cuModuleGetFunction(ctypes.byref(f), mod, b"exact_decode_attention"); assert r == 0, r
    regs = ctypes.c_int(); cu.cuFuncGetAttribute(ctypes.byref(regs), 4, f)
    out = []
    for threads in (224, 256):
        n = ctypes.c_int()
        r = cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(n), f, threads, 0); assert r == 0, r
        out.append(f"{threads} thr: {n.value} blocks/SM")
    print(path.split("/")[-1], f"regs {regs.value}", *out)
