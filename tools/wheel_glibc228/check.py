"""Run inside the glibc 2.28 container: does the wheel built from the
container's binaries work there, and does a host build not?

    python3.11 check.py <wheel> <host liby.so> <scratch dir>

Prints one JSON line; exits 1 if any check failed. No GPU is visible in the
container, so the probe finds no CUDA driver and the profile names no GPU.
"""

import ctypes
import json
import os
import sys
import zipfile

wheel, host_lib, scratch = sys.argv[1:4]
site = os.path.join(scratch, "site")
os.environ["XDG_CACHE_HOME"] = os.path.join(scratch, "xdg")
for name in ("YSU_CACHE_DIR", "YSU_HW_PROFILE", "Y_LIB_PATH", "Y_VERBOSE"):
    os.environ.pop(name, None)

VEC_ADD = """
kernel vec_add(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, N: I32) {
    let idx: I32 = global_thread_id() * 4;
    if idx < N {
        vec_add_v4(A[idx], B[idx], C[idx]);
    }
}
"""

out = {"glibc": os.confstr("CS_GNU_LIBC_VERSION")}
ok = True


def check(name, passed, detail=""):
    global ok
    out[name] = passed if not detail else [passed, detail]
    ok = ok and bool(passed)


# The control: a library linked on the host needs a newer glibc than this one.
try:
    ctypes.CDLL(host_lib)
    check("host_build_refused", False, "the host build loaded, so this glibc is not older")
except OSError as e:
    check("host_build_refused", "GLIBC_2." in str(e), str(e)[-120:])

# Install the wheel as pip would; zipfile drops the mode, which pip keeps.
with zipfile.ZipFile(wheel) as z:
    z.extractall(site)
    for info in z.infolist():
        if (info.external_attr >> 16) & 0o111:
            os.chmod(os.path.join(site, info.filename), 0o755)
sys.path.insert(0, site)

import y_lang  # noqa: E402
from y_lang import CPUJit, compiler  # noqa: E402

check("library_is_the_wheels", compiler._find_liby() == os.path.join(site, "y_lang", "liby.so"),
      compiler._find_liby())


def targets(ptx):
    return [line.split()[1] for line in ptx.splitlines() if line.startswith(".target")]


# A pinned profile in the working directory.
pinned = os.path.join(scratch, "pinned")
os.makedirs(pinned)
with open(os.path.join(pinned, ".ysu_hw_profile"), "w") as f:
    f.write("SM_VERSION=8.9\nSM_COUNT=66\n")
os.chdir(pinned)
# Rust prints to file descriptor 1, not to Python's sys.stdout, so that is
# what is captured: redirecting sys.stdout would see nothing either way.
capture = os.path.join(scratch, "stdout_of_a_compile")
sys.stdout.flush()
saved = os.dup(1)
with open(capture, "w") as f:
    os.dup2(f.fileno(), 1)
    try:
        ptx = y_lang.compile_to_ptx(VEC_ADD, target_sm="sm_89")
    finally:
        os.dup2(saved, 1)
        os.close(saved)
printed = open(capture).read()
check("pinned_compile", ".visible .entry vec_add(" in ptx and targets(ptx) == ["sm_89"], targets(ptx))

# No profile anywhere: the wheel's probe runs, finds no GPU, and the profile
# goes to the user cache, not the working directory.
empty = os.path.join(scratch, "empty")
os.makedirs(empty)
os.chdir(empty)
ptx = y_lang.compile_to_ptx(VEC_ADD)
profile = os.path.join(scratch, "xdg", "y-lang", "hw_profile")
check("probed_compile", ".visible .entry vec_add(" in ptx and targets(ptx) == ["sm_80"], targets(ptx))
check("profile_in_user_cache", os.path.isfile(profile), profile)
check("working_directory_empty", os.listdir(empty) == [], os.listdir(empty))

# The CPU JIT, through the libLLVM the loader finds (the image's LLVM 21).
with CPUJit("fn square(x: I64) -> I64 { return x * x; }") as jit:
    check("cpu_jit", jit("square", 123456) == 123456 * 123456)

check("library_compiles_quiet", printed == "", printed[:200])

print(json.dumps(out))
sys.exit(0 if ok else 1)
