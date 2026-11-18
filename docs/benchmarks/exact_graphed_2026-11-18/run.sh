#!/bin/bash
# Qwen2.5-0.5B decode, by default with --loop graphed (each decode step one CUDA graph,
# y_lang.exact.graphed), RTX 4070 Ti SUPER. Compiled, batch 32, 64 new
# tokens, best of 5, stock and exact taken in turns in each process.
#   D: exact at its defaults;  E: exact with every switch.
# Usage: run.sh <output directory> [loop, default graphed] [order, default "D E E D"]
# (runs/ is `run.sh runs`; runs_greedy/ is `run.sh runs_greedy greedy "E D"`.)
mkdir -p "$1"; S=$(realpath "$1"); LOOP=${2:-graphed}; ORDER=${3:-D E E D}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=16
L=/tmp/y_exact_nospace; PY=$L/venv/bin/python3
# gpu-screen-reco: gpu-screen-recorder (process names are cut at 15 characters)
others() { for n in sober steam Discord chromium gpu-screen-reco; do pgrep -x $n >/dev/null && echo -n "$n "; done; }
util() { for i in 1 2 3 4 5; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; done | awk '{s+=$1} END {printf "%.0f", s/NR}'; }
date '+start %F %T' | tee -a "$S/commit.txt"
git -C $L log -1 --format='commit %H' | tee -a "$S/commit.txt"
git -C $L status --short -- python tools | tee -a "$S/commit.txt"
sha256sum $L/target/release/liby.so | tee -a "$S/commit.txt"
ALL="Y_EXACT_V_INT8=1 Y_EXACT_NORM=triton Y_EXACT_ATTN=y Y_EXACT_KV_PACK=inplace"
for c in $ORDER; do
  t=$(date +%H%M%S)
  if [ $c = E ]; then E="$ALL"; name=E_every_switch; else E=""; name=D_default; fi
  log="$S/$name.$t.log"
  echo "== $name $t  util $(util)%  others: $(others)"
  nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,memory.used --format=csv,noheader > "$log"
  (cd $L/tools && env $E Y_THR_ARMS=stock,exact nice -n 5 timeout 3600 $PY exact_throughput.py \
    --batch 32 --tokens 64 --reps 5 --compile --loop $LOOP >> "$log" 2>&1)
  echo "exit $?"; grep -E 'tok/s|^(stock|exact)|slower|faster|NOTE|sanity|loop |\*\*\*' "$log" | cut -c1-140
done
echo "done $(date +%T)"
