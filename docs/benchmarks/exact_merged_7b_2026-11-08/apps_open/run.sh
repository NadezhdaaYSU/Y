#!/bin/zsh
# q/k/v and gate/up merged (Y_EXACT_MERGE, the default) against not merged, at
# 7B on Y's GEMM with the int8 V cache, beside vLLM FP8: the CPU==GPU record
# first, then speed in ABCCBA order, one configuration per process.
S=$1; PYV=/home/yumin/vllm_env/bin/python; V=$(cd $S/../vllm_fp8_7b_2026-11-08 && pwd)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
L=/tmp/y_exact_nospace; PY=$L/venv/bin/python3; M=Qwen/Qwen2.5-7B-Instruct
others() { for n in sober steam Discord chromium; do pgrep -x $n >/dev/null && echo -n "$n "; done; }
util() { for i in 1 2 3 4 5; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; done | awk '{s+=$1} END {printf "%.0f", s/NR}'; }
echo "== record merged $(date +%T)  util $(util)%  others: $(others)"
(cd $L/tools && Y_EXACT_GEMM=y Y_EXACT_V_INT8=1 $PY exact_cross_device.py --record $S/gpu_sm89_y_vint8_merged_compiled.json \
  --device cuda --arms exact --compile --model $M --low-memory > $S/record_merged.log 2>&1)
echo "exit $?"
y_arm() {   # name, Y_EXACT_MERGE value
  local t=$(date +%H%M%S)
  echo "== $1 $t  util $(util)%  others: $(others)"
  (cd $L/tools && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16 \
    Y_EXACT_GEMM=y Y_EXACT_V_INT8=1 Y_EXACT_MERGE=$2 Y_THR_ARMS=exact $PY exact_throughput.py --compile \
    --model $M --batch 32 --tokens 64 --reps 5 --low-memory > $S/b7_$1.$t.log 2>&1)
  echo "exit $?"; grep -E 'sanity|int8|\*\*\*' $S/b7_$1.$t.log | tail -2 | cut -c1-200
}
vllm_arm() {
  local t=$(date +%H%M%S)
  echo "== vllm_fp8_graphs $t  util $(util)%  others: $(others)"
  (cd $V && $PYV vllm_7b.py graphs $S/vllm_fp8_graphs.$t.json > $S/vllm_fp8_graphs.$t.log 2>&1)
  echo "exit $?"; grep -E "^RESULT|Error" $S/vllm_fp8_graphs.$t.log | tail -1 | cut -c1-250
}
vllm_arm; y_arm merged 1; y_arm unmerged 0; y_arm unmerged 0; y_arm merged 1; vllm_arm
echo "== done $(date +%T)"
