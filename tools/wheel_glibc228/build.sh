#!/bin/bash
# A y-lang wheel whose binaries need only an old glibc, tested on glibc 2.28.
#
#   tools/wheel_glibc228/build.sh          (as a user who can run docker)
#
# 1. Builds the AlmaLinux 8 image (glibc 2.28, gcc, Python 3.11, LLVM).
# 2. Compiles liby.so and ysu_gpu_probe in it, with the host's rustup and cargo
#    registry mounted, into target/glibc228/release/.
# 3. Builds the wheel on the host from those two files (Y_WHEEL_LIBY,
#    Y_WHEEL_PROBE; python/setup.py still refuses them if they are older than
#    the sources) into dist/glibc228/.
# 4. Runs check.py in the container: the host's own liby.so must fail to load
#    there (the control), and the wheel must compile, probe into the user
#    cache, run the CPU JIT on the system's LLVM 21 and print nothing.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
HERE="$ROOT/tools/wheel_glibc228"
IMAGE=y-glibc228
USER_FLAGS=(--user "$(id -u):$(id -g)" -e HOME="$HOME"
            -v "$HOME/.rustup:$HOME/.rustup" -v "$HOME/.cargo:$HOME/.cargo"
            -e PATH="$HOME/.cargo/bin:/usr/local/bin:/usr/bin:/bin")

docker build -q -t "$IMAGE" "$HERE" >/dev/null
docker run --rm "$IMAGE" bash -c 'ldd --version | head -1'

# Mounted at its own path, so cargo's dep-info (target/glibc228/release/*.d)
# names files that exist on the host, where python/setup.py reads it.
docker run --rm "${USER_FLAGS[@]}" -v "$ROOT:$ROOT" -w "$ROOT" \
    -e CARGO_TARGET_DIR="$ROOT/target/glibc228" \
    "$IMAGE" cargo build --release --offline --lib --bin ysu_gpu_probe
BIN="$ROOT/target/glibc228/release"
for f in "$BIN/liby.so" "$BIN/ysu_gpu_probe"; do
    need=$(objdump -T "$f" | grep -v ' w ' | grep -o 'GLIBC_[0-9.]*' | sort -V -u | tail -1)
    echo "$(basename "$f"): newest glibc symbol required $need"
done

rm -rf "$ROOT/dist/glibc228"
(cd "$ROOT" && Y_WHEEL_LIBY="$BIN/liby.so" Y_WHEEL_PROBE="$BIN/ysu_gpu_probe" \
    python3 -m pip wheel ./python --no-deps --no-build-isolation -q -w dist/glibc228)
WHEEL=$(ls "$ROOT"/dist/glibc228/y_lang-*.whl)
echo "wheel: ${WHEEL#$ROOT/}"

SCRATCH=$(mktemp -d)
trap 'rm -rf "$SCRATCH"' EXIT
docker run --rm "${USER_FLAGS[@]}" -v "$ROOT:/src:ro" -v "$SCRATCH:/scratch" "$IMAGE" \
    python3.11 /src/tools/wheel_glibc228/check.py "/src/${WHEEL#$ROOT/}" \
    /src/target/release/liby.so /scratch
