#!/usr/bin/env bash
# The recorded sweeps:
#   run.sh OUT.log          every shape, operands repeated (L2-resident where
#                           they fit), the old kernel too
#   run.sh OUT.log --cold   the LLM shapes, operands rotated over 128 MB
# Needs YBIN_OLD (a compiler built from the commit before the change, 8e3e2ce)
# and Y_PROFILE (the machine's .ysu_hw_profile); WORKDIR for the PTX.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
llm=()
for m in 32 128 512 2048; do
  for nk in 4096x4096 11008x4096 4096x11008 14336x4096 4096x14336 1024x4096; do
    llm+=("${m}x${nk}")
  done
done
if [[ "${2:-}" == "--cold" ]]; then
  python3 "$here/int8_bench.py" --cold --kernels rule,large,mid,small "${llm[@]}" > "$1" 2>&1
  exit
fi
shapes=(256x256x256 512x512x512 1024x1024x1024 2048x2048x2048 3072x3072x3072
        4096x4096x4096 8192x8192x8192 4000x4000x4000 1008x1008x1024
        4096x4096x4000 4096x4096x4064 256x256x65536 1024x1024x16384 "${llm[@]}")
python3 "$here/int8_bench.py" --kernels rule,large,mid,small,old "${shapes[@]}" > "$1" 2>&1
