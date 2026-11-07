#!/usr/bin/env bash
# The 0.5B GPU record with the int8 V cache on Y's exact_pv kernel, compiled,
# compared with the CPU record (which stores V as float32: the CPU has no
# kernel route). Both routes are exact, so the exact arm's tokens must match.
# Usage: record.sh  (run on a GPU host; writes beside this script)
set -u
H="$(realpath "$(dirname "$0")")"
cd "$H/../../.."
CPU=docs/benchmarks/exact_y_gemm_2026-11-07/cpu_zen5_0p5b.json
date +%T
Y_EXACT_V_INT8=1 timeout 3600 python3 tools/exact_cross_device.py --record "$H/gpu_sm89_vint8_compiled.json" --compile > "$H/gpu_sm89_vint8_compiled.log" 2>&1
echo "record exit $?"
grep -E "tokens [0-9a-f]{16}|gate:|self-check" "$H/gpu_sm89_vint8_compiled.log" | tr -s ' '
python3 tools/exact_cross_device.py --compare "$CPU" "$H/gpu_sm89_vint8_compiled.json" > "$H/compare_cpu_vs_vint8_compiled.txt" 2>&1
echo "compare exit $?"; tail -12 "$H/compare_cpu_vs_vint8_compiled.txt"
date +%T
