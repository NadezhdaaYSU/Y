#!/bin/bash
# run_data.sh K "columns...": time on small integers, then on normal data.
W=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/8cb989c6-e604-4dd6-849f-ecfdae469440/scratchpad/sk
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad; X=$S/xbar
exec 9>$S/gpu.lock; flock 9
K=$1; calls=50; [ $K = 16384 ] && calls=20; [ $K -ge 65536 ] && calls=6
cols=""; for c in $2; do cols="$cols $c=$W/sw$K/$c.ptx"; done
for d in ${3:-int normal}; do
  { echo "data $d K=$K $(date)"; for j in 1 2 3; do nvidia-smi --query-gpu=utilization.gpu,clocks.sm,power.draw --format=csv,noheader; sleep 1; done; } >> $W/sw$K/env_data.txt
  Y_LIB_PATH=$X/target/release/liby.so PYTHONPATH=$X/python python3 $W/sbench.py $W/sw$K/data_${d}${4}.json 512 512 $K $cols calls=$calls data=$d 2>&1 | grep -v Warn
done
