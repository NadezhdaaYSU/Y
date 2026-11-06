#!/usr/bin/env bash
# GPU records of the 0.5B model: Y's GEMM eager and compiled, and the default
# (Triton) path eager as a regression check. Each is compared with the CPU record.
# Usage: records.sh DIR, where DIR holds cpu_0p5b.json, recorded on the host with
#   python3 tools/exact_cross_device.py --record DIR/cpu_0p5b.json --device cpu
set -u
S="$(realpath "$1")"
cd "$(dirname "$0")/../../.."
rec() {   # name, gemm ("" = default), extra args...
  local name=$1 gemm=$2; shift 2
  date +%T
  if [ -n "$gemm" ]; then export Y_EXACT_GEMM=$gemm; else unset Y_EXACT_GEMM; fi
  timeout 3600 python3 tools/exact_cross_device.py --record "$S/$name.json" "$@" > "$S/$name.log" 2>&1
  echo "$name exit $?"
  grep -E "tokens [0-9a-f]{16}|gate:|self-check" "$S/$name.log" | tr -s ' ' | tr '\n' ' '; echo
  python3 tools/exact_cross_device.py --compare "$S/cpu_0p5b.json" "$S/$name.json" > "$S/$name.compare.txt" 2>&1
  echo "compare exit $?"; tail -12 "$S/$name.compare.txt"
}
rec gpu_y_eager y
rec gpu_y_compiled y --compile
rec gpu_triton_eager ""
date +%T
