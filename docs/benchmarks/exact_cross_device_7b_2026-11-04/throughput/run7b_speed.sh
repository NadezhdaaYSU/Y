#!/bin/zsh
# ABBA, one arm per process (7B stock and exact do not fit one card together).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16
L=/tmp/y_exact_nospace; S=$1; cd $L/tools; PY=$L/venv/bin/python3
for arm in stock exact exact stock; do
  t=$(date +%H%M%S)
  extra=(--low-memory)
  util=$(for i in 1 2 3 4 5; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; done | awk '{s+=$1} END {printf "%.0f", s/NR}')
  games=$(for n in sober steam Discord chromium; do pgrep -x $n >/dev/null && echo -n "$n "; done)
  echo "== $arm $t  util ${util}%  others: ${games:-none}"
  Y_THR_ARMS=$arm $PY exact_throughput.py --compile --model Qwen/Qwen2.5-7B-Instruct --batch 32 --tokens 64 --reps 5 $extra > $S/b7_speed_$arm.$t.log 2>&1
  echo "exit $?"
  grep -E 'sanity|bf16|int8|NOTE|Error|\*\*\*' $S/b7_speed_$arm.$t.log | cut -c1-200 | tail -4
done
echo "== done $(date +%T)"
