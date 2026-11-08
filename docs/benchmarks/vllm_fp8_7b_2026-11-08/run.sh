#!/bin/zsh
# vLLM FP8 7B, one configuration per process, and Y's fastest exact
# configuration (Y GEMM + int8 V) interleaved with it: A B B A C.
S=$1; PYV=/home/yumin/vllm_env/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
L=/tmp/y_exact_nospace; PY=$L/venv/bin/python3
others() { for n in sober steam Discord chromium; do pgrep -x $n >/dev/null && echo -n "$n "; done; }
util() { for i in 1 2 3 4 5; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; done | awk '{s+=$1} END {printf "%.0f", s/NR}'; }
vllm_arm() {   # name, mode, extra env
  local t=$(date +%H%M%S)
  echo "== vllm $1 $t  util $(util)%  others: $(others)"
  (cd $S && env $3 $PYV vllm_7b.py $2 $S/vllm_$1.$t.json > $S/vllm_$1.$t.log 2>&1)
  echo "exit $?"; grep -E "^RESULT|Error|error:" $S/vllm_$1.$t.log | tail -2 | cut -c1-300
}
y_arm() {
  local t=$(date +%H%M%S)
  echo "== y_vint8 $t  util $(util)%  others: $(others)"
  (cd $L/tools && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16 \
    Y_EXACT_GEMM=y Y_EXACT_V_INT8=1 Y_THR_ARMS=exact $PY exact_throughput.py --compile \
    --model Qwen/Qwen2.5-7B-Instruct --batch 32 --tokens 64 --reps 5 --low-memory > $S/y_vint8.$t.log 2>&1)
  echo "exit $?"; grep -E 'sanity|int8|\*\*\*' $S/y_vint8.$t.log | tail -2 | cut -c1-200
}
vllm_arm fp8_graphs graphs ""
y_arm
y_arm
vllm_arm fp8_graphs graphs ""
vllm_arm fp8_eager eager ""
vllm_arm fp8_graphs_bi graphs VLLM_BATCH_INVARIANT=1
echo "== done $(date +%T)"
