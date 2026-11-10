#!/bin/zsh
# 0.5B records for the integer sum of squares: the CPU, then the GPU compiled
# with the default (torch) norm and with every Y switch; each GPU record is
# compared with the CPU one, and the CPU one with the CPU record before the
# change. Usage: records.sh OUT_DIR [python]   (run from a checkout whose path
# has no spaces, or through a symlink to one: Inductor's C++ build breaks on
# spaces)
S=${1:A}; PY=${2:-python3}; R=${0:A:h}/../../..
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$R/python
M=Qwen/Qwen2.5-0.5B-Instruct
cd $R/tools
echo "cpu $(date +%T)"
OMP_NUM_THREADS=16 $PY exact_cross_device.py --record $S/cpu_0p5b.json --device cpu \
  --arms exact --model $M > $S/cpu.log 2>&1; echo "exit $?"
echo "gpu default $(date +%T)"
Y_EXACT_GEMM=y Y_EXACT_V_INT8=1 $PY exact_cross_device.py --record $S/gpu_default_compiled.json \
  --device cuda --arms exact --compile --model $M > $S/gpu_default.log 2>&1; echo "exit $?"
echo "gpu all $(date +%T)"
Y_EXACT_GEMM=y Y_EXACT_V_INT8=1 Y_EXACT_NORM=triton Y_EXACT_ATTN=y Y_EXACT_KV_PACK=inplace \
  $PY exact_cross_device.py --record $S/gpu_all_compiled.json --device cuda --arms exact \
  --compile --model $M > $S/gpu_all.log 2>&1; echo "exit $?"
command $PY - $S $R <<'EOF'
import json, sys
S, R = sys.argv[1], sys.argv[2]
arm = lambda p: json.load(open(p))["arms"]["exact"]
old = arm(f"{R}/docs/benchmarks/exact_y_gemm_2026-11-07/cpu_zen5_0p5b.json")
c = arm(f"{S}/cpu_0p5b.json")
print("cpu", c["digest"], "old cpu", old["digest"],
      "cases differing:", [k for k in c["cases"] if c["cases"][k] != old["cases"][k]])
for n in ("gpu_default_compiled", "gpu_all_compiled"):
    g = arm(f"{S}/{n}.json")
    print(n, g["digest"], "weights equal", g["weight_fingerprint"] == c["weight_fingerprint"],
          "self-check", g.get("self_check"),
          "cases differing:", [k for k in c["cases"] if g["cases"].get(k) != c["cases"][k]])
EOF
