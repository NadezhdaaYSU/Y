#!/bin/zsh
# 0.5B GPU records with the staged attention kernel: every Y switch (in-place
# cache), and Y attention on the cat cache; each compared with the CPU record
# of the integer norm sum. Usage: records.sh OUT_DIR [python]  (from a
# checkout path without spaces, or a symlink to one)
S=${1:A}; PY=${2:-python3}; R=${0:A:h}/../../..
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$R/python
M=Qwen/Qwen2.5-0.5B-Instruct
cd $R/tools
echo "gpu all $(date +%T)"
Y_EXACT_GEMM=y Y_EXACT_V_INT8=1 Y_EXACT_NORM=triton Y_EXACT_ATTN=y Y_EXACT_KV_PACK=inplace \
  $PY exact_cross_device.py --record $S/gpu_all_compiled.json --device cuda --arms exact \
  --compile --model $M > $S/gpu_all.log 2>&1; echo "exit $?"
echo "gpu attn, cat cache $(date +%T)"
Y_EXACT_GEMM=y Y_EXACT_V_INT8=1 Y_EXACT_ATTN=y Y_EXACT_KV_PACK=1 \
  $PY exact_cross_device.py --record $S/gpu_attn_cat_compiled.json --device cuda --arms exact \
  --compile --model $M > $S/gpu_attn_cat.log 2>&1; echo "exit $?"
command $PY - $S $R <<'PYEOF'
import json, sys
S, R = sys.argv[1], sys.argv[2]
arm = lambda p: json.load(open(p))["arms"]["exact"]
c = arm(f"{R}/docs/benchmarks/exact_norm_sum_2026-11-10/cpu_0p5b.json")
print("cpu", c["digest"])
for n in ("gpu_all_compiled", "gpu_attn_cat_compiled"):
    g = arm(f"{S}/{n}.json")
    print(n, g["digest"], "weights equal", g["weight_fingerprint"] == c["weight_fingerprint"],
          "self-check", g.get("self_check"),
          "cases differing:", [k for k in c["cases"] if g["cases"].get(k) != c["cases"][k]])
PYEOF
