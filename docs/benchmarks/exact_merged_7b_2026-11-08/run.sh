#!/bin/zsh
# Merging projections (Y_EXACT_MERGE: 0 = none, qkv = q/k/v only, all = q/k/v
# and gate/up, the default) at 7B on Y's GEMM with the int8 V cache, beside
# vLLM FP8 with CUDA graphs. First the CPU==GPU record with everything merged,
# compared with the CPU record and the unmerged GPU record; then speed, one
# configuration per process, in the order A B C D D C B A.
# Usage: run.sh <this directory>   (the first try, with apps open, is in apps_open/)
S=$1; PYV=/home/yumin/vllm_env/bin/python; V=$(cd $S/../vllm_fp8_7b_2026-11-08 && pwd)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16
L=/tmp/y_exact_nospace; PY=$L/venv/bin/python3; M=Qwen/Qwen2.5-7B-Instruct
CPU_REC=$S/../exact_cross_device_7b_2026-11-04/cpu_zen5_eager.json
UNMERGED_REC=$S/../exact_y_gemm_7b_2026-11-08/gpu_sm89_y_compiled.json
others() { for n in sober steam Discord chromium; do pgrep -x $n >/dev/null && echo -n "$n "; done; }
util() { for i in 1 2 3 4 5; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; done | awk '{s+=$1} END {printf "%.0f", s/NR}'; }
echo "== record merged=all $(date +%T)  util $(util)%  others: $(others)"
(cd $L/tools && Y_EXACT_GEMM=y Y_EXACT_V_INT8=1 Y_EXACT_MERGE=all $PY exact_cross_device.py \
  --record $S/gpu_sm89_y_vint8_merged_compiled.json --device cuda --arms exact --compile \
  --model $M --low-memory > $S/record_merged.log 2>&1)
echo "exit $?"
for ref in $CPU_REC $UNMERGED_REC; do
  echo "-- compare with ${ref:t}"
  (cd $L/tools && $PY exact_cross_device.py --compare $ref $S/gpu_sm89_y_vint8_merged_compiled.json 2>&1 | tail -6)
done
y_arm() {   # Y_EXACT_MERGE value
  local t=$(date +%H%M%S)
  echo "== y_merge_$1 $t  util $(util)%  others: $(others)"
  (cd $L/tools && Y_EXACT_GEMM=y Y_EXACT_V_INT8=1 Y_EXACT_MERGE=$1 Y_THR_ARMS=exact \
    $PY exact_throughput.py --compile --model $M --batch 32 --tokens 64 --reps 5 --low-memory \
    > $S/b7_merge_$1.$t.log 2>&1)
  echo "exit $?"; grep -E 'sanity|int8|Error|\*\*\*' $S/b7_merge_$1.$t.log | tail -2 | cut -c1-200
}
vllm_arm() {
  local t=$(date +%H%M%S)
  echo "== vllm_fp8_graphs $t  util $(util)%  others: $(others)"
  (cd $V && $PYV vllm_7b.py graphs $S/vllm_fp8_graphs.$t.json > $S/vllm_fp8_graphs.$t.log 2>&1)
  echo "exit $?"; grep -E "^RESULT|Error" $S/vllm_fp8_graphs.$t.log | tail -1 | cut -c1-250
}
vllm_arm; y_arm 0; y_arm qkv; y_arm all; y_arm all; y_arm qkv; y_arm 0; vllm_arm
echo "== done $(date +%T)"
