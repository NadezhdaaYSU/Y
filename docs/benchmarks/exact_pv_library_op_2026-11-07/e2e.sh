#!/usr/bin/env bash
# Decode throughput of the exact arm with each exact_pv route under it, compiled,
# batch 32, 64 new tokens, best of 5, one route per process, ABCCBA:
#   A  digit split (float32 V cache, the default)
#   B  int8 V cache, the route until 2026-11-07, from a worktree at OLD_TREE
#      (custom_op, synchronising ctypes launch, casts inside the op)
#   C  int8 V cache, the route now (Library op, TorchKernel launch, casts outside)
# Run on an idle GPU. Usage: e2e.sh OUT_DIR OLD_TREE
# OLD_TREE is `git worktree add --detach OLD_TREE 83da64d` with `target` linked to
# this checkout's and `.ysu_exact_gemm` and `.ysu_hw_profile` copied from it, so
# the GEMM tuning and the compiler profile are the same in both trees.
set -u
OUT="$(realpath "$1")"; OLD="$(realpath "$2")"
mkdir -p "$OUT"
cd "$(dirname "$0")/../../.."
PY=/tmp/y_exact_nospace/venv/bin/python   # a path with no space, for torch.compile
n=0
for arm in A B C C B A; do
  n=$((n + 1))
  log="$OUT/$n-$arm.log"
  nvidia-smi --query-gpu=utilization.gpu,clocks.sm,power.draw,memory.used --format=csv,noheader > "$log"
  case $arm in
    A) Y_THR_ARMS=exact Y_EXACT_V_INT8=0 timeout 3600 python3 tools/exact_throughput.py --batch 32 --tokens 64 --compile >> "$log" 2>&1 ;;
    B) Y_THR_ARMS=exact Y_EXACT_V_INT8=1 timeout 3600 "$PY" "$OLD/tools/exact_throughput.py" --batch 32 --tokens 64 --compile >> "$log" 2>&1 ;;
    C) Y_THR_ARMS=exact Y_EXACT_V_INT8=1 timeout 3600 python3 tools/exact_throughput.py --batch 32 --tokens 64 --compile >> "$log" 2>&1 ;;
  esac
  echo "$n $arm exit $?: $(grep -E '^exact' "$log" | tr -s ' ')"
done
