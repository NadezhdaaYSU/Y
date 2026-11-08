#!/bin/zsh
# 7B exact (Y GEMM, int8 V cache, q/k/v merged, greedy loop) with inner
# reductions of up to 4096 elements persistent (Y_EXACT_PERSISTENT_ROWS, the
# default since this run) against Inductor's own rule (=0), beside vLLM FP8
# with CUDA graphs. First the CPU==GPU record with the new default. One
# configuration per process, A B C C B A. Usage: run.sh <this directory>
S=$1; PYV=/home/yumin/vllm_env/bin/python; V=$(cd $S/../vllm_fp8_7b_2026-11-08 && pwd)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16
L=/tmp/y_exact_nospace; PY=$L/venv/bin/python3; M=Qwen/Qwen2.5-7B-Instruct
others() { for n in sober steam Discord chromium; do pgrep -x $n >/dev/null && echo -n "$n "; done; }
util() { for i in 1 2 3 4 5; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; done | awk '{s+=$1} END {printf "%.0f", s/NR}'; }
echo "== record (persistent rows 4096) $(date +%T)  util $(util)%  others: $(others)"
(cd $L/tools && Y_EXACT_GEMM=y Y_EXACT_V_INT8=1 $PY exact_cross_device.py \
  --record $S/gpu_sm89_y_vint8_persistent_compiled.json --device cuda --arms exact --compile \
  --model $M --low-memory > $S/record_persistent.log 2>&1)
echo "exit $?"
y_arm() {   # Y_EXACT_PERSISTENT_ROWS value
  local t=$(date +%H%M%S)
  echo "== y_rows_$1 $t  util $(util)%  others: $(others)"
  (cd $L/tools && Y_EXACT_GEMM=y Y_EXACT_V_INT8=1 Y_EXACT_PERSISTENT_ROWS=$1 Y_THR_ARMS=exact \
    $PY exact_throughput.py --compile --loop greedy --model $M --batch 32 --tokens 64 --reps 5 \
    --low-memory > $S/b7_rows_$1.$t.log 2>&1)
  echo "exit $?"; grep -E 'sanity|loop|int8|Error|\*\*\*' $S/b7_rows_$1.$t.log | tail -3 | cut -c1-200
}
vllm_arm() {
  local t=$(date +%H%M%S)
  echo "== vllm_fp8_graphs $t  util $(util)%  others: $(others)"
  (cd $V && $PYV vllm_7b.py graphs $S/vllm_fp8_graphs.$t.json > $S/vllm_fp8_graphs.$t.log 2>&1)
  echo "exit $?"; grep -E "^RESULT|Error" $S/vllm_fp8_graphs.$t.log | tail -1 | cut -c1-250
}
vllm_arm; y_arm 0; y_arm 4096; y_arm 4096; y_arm 0; vllm_arm
echo "== done $(date +%T)"
