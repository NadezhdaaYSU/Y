#!/bin/bash
# Setup A again after the exact_throughput fix (2026-11-17): each arm now runs
# with its own attention, and the fixed-order control is really compiled.
# Compiled twice, then once eager (to see what compiling does for the control).
# Usage: run_fixed.sh <this directory>
S=$(realpath "$1")
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=16
L=/tmp/y_exact_nospace; PY=$L/venv/bin/python3
others() { for n in sober steam Discord chromium; do pgrep -x $n >/dev/null && echo -n "$n "; done; }
util() { for i in 1 2 3 4 5; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; done | awk '{s+=$1} END {printf "%.0f", s/NR}'; }
run() {   # name, extra args
  local t=$(date +%H%M%S) log
  log="$S/$1.$t.log"
  echo "== $1 $t  util $(util)%  others: $(others)"
  nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,memory.used --format=csv,noheader > "$log"
  (cd $L/tools && Y_THR_ARMS=stock,exact,fixedfloat nice -n 5 timeout 5400 $PY exact_throughput.py \
    --batch 32 --tokens 64 --reps 5 $2 >> "$log" 2>&1)
  echo "exit $?"; grep -E 'tok/s|^(stock|exact|fixed)|slower|NOTE|sanity|\*\*\*' "$log" | cut -c1-160
}
git -C $L diff --stat HEAD -- tools/exact_throughput.py | tee "$S/commit_fixed.txt"
git -C $L log -1 --format='commit %H' | tee -a "$S/commit_fixed.txt"
run A2_127_control_fixed --compile
run A2_127_control_fixed --compile
run A2_127_control_fixed_eager ""
echo "done $(date +%T)"
