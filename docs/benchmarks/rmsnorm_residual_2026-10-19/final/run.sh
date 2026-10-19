#!/bin/bash
# run.sh <tag>: one probe_final.py run. Started by gpu_quiet_then.sh.
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
F=$SP/nr/final
M="/home/yumin/NVME files/YSU-engine-main/YSU-engine-main/src/Y_lang/Y"
cd $F || exit 9
{ date; nvidia-smi --query-gpu=name,driver_version,utilization.gpu,clocks.sm --format=csv,noheader; "$SP/games.sh"; echo; } > $F/env_$1.txt
Y_LIB_PATH=$SP/xbar/target/release/liby.so PYTHONPATH=$SP/xbar/python python3 $F/probe_final.py "$M/tests" $SP/f16sq $F/ptx 896,2048,3072,4096,5120,8192,12032,16384 4,17,34,134 > $F/final_$1.log 2>&1
echo "final $1 exit $?" >> $F/summary.txt
{ date; nvidia-smi --query-gpu=utilization.gpu,clocks.sm --format=csv,noheader; "$SP/games.sh"; echo; } >> $F/env_$1.txt
