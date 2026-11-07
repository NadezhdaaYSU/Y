#!/usr/bin/env bash
# The recorded sweep: run.sh OUT.log (Y_PROFILE: the .ysu_hw_profile to compile with)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
shapes=(256x256x256 512x512x512 1024x1024x1024 2048x2048x2048 4096x4096x4096 8192x8192x8192
        1000x1000x1000 4000x4000x4000 768x768x768 1024x1024x8192)
for m in 32 128 512 2048; do
  for nk in 4096x4096 11008x4096 4096x11008 14336x4096 4096x14336; do
    shapes+=("${m}x${nk}")
  done
done
python3 "$here/bf16_bench.py" "${shapes[@]}" > "$1" 2>&1
