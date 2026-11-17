#!/bin/bash
# Qwen2.5-0.5B decode speed on the current code, RTX 4070 Ti SUPER.
# Compiled, batch 32, 64 new tokens, generate(), best of 5, arms interleaved
# inside each process. One configuration per process, order A B C D D C B A.
#   A: stock bf16, exact (127 levels), fixed-order fp32 control
#   B: stock bf16, exact at 511 levels
#   C: stock bf16, exact with every switch (V int8 on exact_pv, Triton norm,
#      Y's decode attention, in-place packed cache)
#   D: as C, with Y's own int8 GEMM (the 7B configuration)
# A, B and C match the Tesla T4 job (which uses float16 for stock).
# Run again after 328c6aa: the tool's forward wrappers no longer hide
# `logits_to_keep`, so no arm's prefill computes every position's logits.
# Usage: run_after_logits_fix.sh <a directory> [order, default "A B C D D C B A"]
# A stopped run is resumed by passing the setups it did not reach.
mkdir -p "$1"; S=$(realpath "$1")
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=16
L=/tmp/y_exact_nospace; PY=$L/venv/bin/python3
others() { for n in sober steam Discord chromium; do pgrep -x $n >/dev/null && echo -n "$n "; done; }
util() { for i in 1 2 3 4 5; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; done | awk '{s+=$1} END {printf "%.0f", s/NR}'; }
run() {   # name, arms, env
  local t=$(date +%H%M%S) log
  log="$S/$1.$t.log"
  echo "== $1 $t  util $(util)%  others: $(others)"
  nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,memory.used --format=csv,noheader > "$log"
  (cd $L/tools && env $3 Y_THR_ARMS=$2 nice -n 5 timeout 3600 $PY exact_throughput.py \
    --batch 32 --tokens 64 --reps 5 --compile >> "$log" 2>&1)
  echo "exit $?"; grep -E 'tok/s|^(stock|exact|fixed)|slower|NOTE|sanity|\*\*\*' "$log" | cut -c1-160
}
date '+start %F %T' | tee -a "$S/commit.txt"
git -C $L log -1 --format='commit %H' | tee -a "$S/commit.txt"
sha256sum $L/target/release/liby.so | tee -a "$S/commit.txt"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit --format=csv,noheader | tee -a "$S/commit.txt"
ALL="Y_EXACT_V_INT8=1 Y_EXACT_NORM=triton Y_EXACT_ATTN=y Y_EXACT_KV_PACK=inplace"
for c in ${2:-A B C D D C B A}; do
  case $c in
    A) run A_127_control stock,exact,fixedfloat "" ;;
    B) run B_511 stock,exact "Y_EXACT_ACT_LEVELS=511" ;;
    C) run C_every_switch stock,exact "$ALL" ;;
    D) run D_every_switch_ygemm stock,exact "$ALL Y_EXACT_GEMM=y" ;;
  esac
done
echo "done $(date +%T)"
