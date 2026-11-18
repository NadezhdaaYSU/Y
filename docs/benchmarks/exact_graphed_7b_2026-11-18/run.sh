#!/bin/bash
# Qwen2.5-7B-Instruct, exact path only (Y_THR_ARMS=exact), compiled, batch 32,
# 64 new tokens, best of 5, `--low-memory` (the 7B build), RTX 4070 Ti SUPER.
#   A: the 2026-11-08 setting (Y GEMM, int8 V cache, q/k/v merged), --loop greedy.
#      It bridges to that day's vLLM FP8 runs, which are not repeated here.
#   B: the same setting, --loop graphed (each decode step one CUDA graph).
#   C: every switch on top of Y GEMM, --loop graphed.
# One configuration per process, order A B C C B A. Usage: run.sh <output directory>
mkdir -p "$1"; S=$(realpath "$1")
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/tmp/y_exact_nospace; PY=$L/venv/bin/python3; M=Qwen/Qwen2.5-7B-Instruct
# gpu-screen-reco: gpu-screen-recorder (process names are cut at 15 characters)
others() { for n in sober steam Discord chromium gpu-screen-reco; do pgrep -x $n >/dev/null && echo -n "$n "; done; }
util() { for i in 1 2 3 4 5; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; done | awk '{s+=$1} END {printf "%.0f", s/NR}'; }
date '+start %F %T' | tee -a "$S/commit.txt"
git -C $L log -1 --format='commit %H' | tee -a "$S/commit.txt"
git -C $L status --short -- python tools | tee -a "$S/commit.txt"
sha256sum $L/target/release/liby.so | tee -a "$S/commit.txt"
BASE="Y_EXACT_GEMM=y Y_EXACT_V_INT8=1"
ALL="$BASE Y_EXACT_NORM=triton Y_EXACT_ATTN=y Y_EXACT_KV_PACK=inplace"
for c in A B C C B A; do
  case $c in
    A) E="$BASE"; LOOP=greedy;  name=A_greedy ;;
    B) E="$BASE"; LOOP=graphed; name=B_graphed ;;
    C) E="$ALL";  LOOP=graphed; name=C_graphed_every_switch ;;
  esac
  t=$(date +%H%M%S); log="$S/$name.$t.log"
  echo "== $name $t  util $(util)%  others: $(others)"
  nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,memory.used --format=csv,noheader > "$log"
  echo "env: $E" >> "$log"
  (cd $L/tools && env $E Y_THR_ARMS=exact nice -n 5 timeout 3600 $PY exact_throughput.py \
    --compile --loop $LOOP --model $M --batch 32 --tokens 64 --reps 5 --low-memory >> "$log" 2>&1)
  echo "exit $?"; grep -E '^exact|loop |sanity|Error|\*\*\*' "$log" | cut -c1-160
done
echo "done $(date +%T)"
