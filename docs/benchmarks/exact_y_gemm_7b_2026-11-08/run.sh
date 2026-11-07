#!/bin/zsh
# 7B on Y's int8 GEMM: the CPU==GPU record first, then decode speed with the
# linear layers on Triton (the default), on Y's GEMM (Y_EXACT_GEMM=y), and on
# Y's GEMM with the int8 V cache on Y's exact_pv (Y_EXACT_V_INT8=1).
# One arm per process, ABCCBA. Usage: run.sh <output dir>
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16 HF_HUB_OFFLINE=1
L=/tmp/y_exact_nospace; S=$1; cd $L/tools; PY=$L/venv/bin/python3
M=Qwen/Qwen2.5-7B-Instruct
others() { for n in sober steam Discord chromium isaac-ng.exe; do pgrep -x $n >/dev/null && echo -n "$n "; done; }
util() { for i in 1 2 3 4 5; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; done | awk '{s+=$1} END {printf "%.0f", s/NR}'; }
echo "== record (Y GEMM, compiled) $(date +%T)  util $(util)%  others: $(others)"
Y_EXACT_GEMM=y $PY exact_cross_device.py --record $S/gpu_sm89_y_compiled.json --device cuda \
  --arms exact --compile --model $M --low-memory > $S/record_y_compiled.log 2>&1
echo "exit $?"
for arm in triton y y_vint8 y_vint8 y triton; do
  case $arm in
    triton)  envs=(Y_EXACT_GEMM=triton) ;;
    y)       envs=(Y_EXACT_GEMM=y) ;;
    y_vint8) envs=(Y_EXACT_GEMM=y Y_EXACT_V_INT8=1) ;;
  esac
  t=$(date +%H%M%S)
  echo "== $arm $t  util $(util)%  others: $(others)"
  env $envs Y_THR_ARMS=exact $PY exact_throughput.py --compile --model $M --batch 32 \
    --tokens 64 --reps 5 --low-memory > $S/b7_$arm.$t.log 2>&1
  echo "exit $?"
  grep -E 'sanity|int8|NOTE|Error|\*\*\*' $S/b7_$arm.$t.log | cut -c1-200 | tail -3
done
echo "== done $(date +%T)"
