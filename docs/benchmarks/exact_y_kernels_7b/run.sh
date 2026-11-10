#!/bin/zsh
# 7B exact (Y GEMM, int8 V cache, q/k/v merged, greedy loop) with the Y-side
# changes added one at a time: decode attention on Y's exact_decode_attention
# kernel (Y_EXACT_ATTN=y), ExactRMSNorm on y_lang.exact.norm_kernel
# (Y_EXACT_NORM=triton), the cache kept quantised and packed
# (Y_EXACT_KV_PACK=1), then appended in place (Y_EXACT_KV_PACK=inplace).
# First the CPU==GPU record with all of them on (in place); then speed, one
# configuration per process, in the order vLLM, A B C D E E D C B A, vLLM;
# last, where the all-on step's time goes (profile_all.py, in place).
# Usage: run.sh <this directory>
S=$1; PYV=/home/yumin/vllm_env/bin/python; V=$(cd $S/../vllm_fp8_7b_2026-11-08 && pwd)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16
L=/tmp/y_exact_nospace; PY=$L/venv/bin/python3; M=Qwen/Qwen2.5-7B-Instruct
others() { for n in sober steam Discord chromium; do pgrep -x $n >/dev/null && echo -n "$n "; done; }
util() { for i in 1 2 3 4 5; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; done | awk '{s+=$1} END {printf "%.0f", s/NR}'; }
echo "== record (attention y, norm triton, kv pack inplace) $(date +%T)  util $(util)%  others: $(others)"
(cd $L/tools && Y_EXACT_GEMM=y Y_EXACT_V_INT8=1 Y_EXACT_NORM=triton Y_EXACT_ATTN=y Y_EXACT_KV_PACK=inplace \
  $PY exact_cross_device.py --record $S/gpu_sm89_y_kernels_compiled.json --device cuda --arms exact \
  --compile --model $M --low-memory > $S/record_y_kernels.log 2>&1)
echo "exit $?"
(cd $L/tools && $PY -c "
import json
c=json.load(open('$S/../exact_norm_sum_2026-11-10/cpu_zen5_eager_7b.json'))['arms']['exact']
g=json.load(open('$S/gpu_sm89_y_kernels_compiled.json'))['arms']['exact']
print('digest', g['digest'], 'cpu', c['digest'], 'weights', g['weight_fingerprint'], c['weight_fingerprint'],
      'cases differing:', [k for k in c['cases'] if g['cases'][k] != c['cases'][k]])")
y_arm() {   # name, attn, norm, pack
  local t=$(date +%H%M%S)
  echo "== y_$1 $t  util $(util)%  others: $(others)"
  (cd $L/tools && Y_EXACT_GEMM=y Y_EXACT_V_INT8=1 Y_EXACT_ATTN=$2 Y_EXACT_NORM=$3 Y_EXACT_KV_PACK=$4 \
    Y_THR_ARMS=exact $PY exact_throughput.py --compile --loop greedy --model $M --batch 32 \
    --tokens 64 --reps 5 --low-memory > $S/b7_$1.$t.log 2>&1)
  echo "exit $?"; grep -E 'sanity|loop|int8|Error|\*\*\*' $S/b7_$1.$t.log | tail -3 | cut -c1-200
}
vllm_arm() {
  local t=$(date +%H%M%S)
  echo "== vllm_fp8_graphs $t  util $(util)%  others: $(others)"
  (cd $V && $PYV vllm_7b.py graphs $S/vllm_fp8_graphs.$t.json > $S/vllm_fp8_graphs.$t.log 2>&1)
  echo "exit $?"; grep -E "^RESULT|Error" $S/vllm_fp8_graphs.$t.log | tail -1 | cut -c1-250
}
vllm_arm
y_arm base torch torch 0; y_arm attn y torch 0; y_arm attn_norm y triton 0; y_arm all y triton 1
y_arm inplace y triton inplace; y_arm inplace y triton inplace
y_arm all y triton 1; y_arm attn_norm y triton 0; y_arm attn y torch 0; y_arm base torch torch 0
vllm_arm
echo "== profile (attention y, norm triton, kv pack inplace) $(date +%T)  util $(util)%  others: $(others)"
(cd $L/tools && Y_EXACT_GEMM=y Y_EXACT_V_INT8=1 Y_EXACT_NORM=triton Y_EXACT_ATTN=y Y_EXACT_KV_PACK=inplace \
  $PY $L/docs/benchmarks/exact_y_kernels_7b/profile_all.py $S/profile_all.json > $S/profile_all.log 2>&1)
echo "exit $?"; grep -E '^greedy|^one decode|^launches' $S/profile_all.log
echo "== done $(date +%T)"
