#!/usr/bin/env bash
# Decode throughput of the exact arm with each GEMM under it: Triton (the
# default), cuBLASLt (`torch._int_mm`) and Y's kernel. Compiled, batch 32,
# 64 new tokens, best of 5, one mode per process, in ABCCBA order so a drift
# lands on every mode alike. Run on an idle GPU.
# Usage: e2e.sh OUT_DIR [extra exact_throughput.py args, e.g. --model ... --low-memory]
set -u
OUT="$(realpath "$1")"; shift
mkdir -p "$OUT"
cd "$(dirname "$0")/../../.."
n=0
for mode in triton torch y y torch triton; do
  n=$((n + 1))
  log="$OUT/$n-$mode.log"
  nvidia-smi --query-gpu=utilization.gpu,clocks.sm,power.draw,memory.used --format=csv,noheader > "$log"
  Y_THR_ARMS=exact Y_EXACT_GEMM=$mode timeout 3600 \
    python3 tools/exact_throughput.py --batch 32 --tokens 64 --compile "$@" >> "$log" 2>&1
  echo "$n $mode exit $?: $(grep -E '^exact' "$log" | tr -s ' ')"
done
