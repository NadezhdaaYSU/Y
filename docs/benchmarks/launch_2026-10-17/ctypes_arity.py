"""What ctypes costs per call, by argument count: an empty C function
(`noop.c`, built to libnoop.so beside this file) called with cuLaunchKernel's
eleven argument types and with one. Minimum and median of 7 rounds of
200,000 calls, microseconds per call.

    gcc -O2 -shared -fPIC noop.c -o libnoop.so && python3 ctypes_arity.py
"""
import ctypes
import os
import statistics
import time

lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libnoop.so"))
eleven = lib.noop11
eleven.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
eleven.restype = ctypes.c_int
one = lib.noop1
one.argtypes = [ctypes.c_void_p]
one.restype = ctypes.c_int
params = (ctypes.c_void_p * 4)()
function = ctypes.c_void_p(0x1234)


def per_call(f, n=200_000):
    rounds = []
    for _ in range(7):
        t0 = time.perf_counter()
        for _ in range(n):
            f()
        rounds.append((time.perf_counter() - t0) / n * 1e6)
    return min(rounds), statistics.median(rounds)


for name, f in (("eleven arguments", lambda: eleven(function, 98, 1, 1, 128, 1, 1, 0, 777, params, None)),
                ("one argument", lambda: one(1234567)),
                ("an empty lambda", lambda: None)):
    print("%-16s min %.3f median %.3f us" % ((name,) + per_call(f)))
