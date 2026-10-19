#!/bin/bash
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
A=$SP/cdiv/ab
cd $A || exit 9
{ date; nvidia-smi --query-gpu=name,driver_version,utilization.gpu,clocks.sm --format=csv,noheader; "$SP/games.sh"; python3 -c "import torch, triton; print(torch.__version__, triton.__version__)"; } > $A/env_$1.txt
Y_LIB_PATH=$SP/xbar/target/release/liby.so PYTHONPATH=$SP/xbar/python python3 $SP/f16sq/bench_ab.py base=$A/base,final=$A/final,unrolled=$A/new $A/ab_$1.json 256,512,1024,2048,4096,8192,1x4096x4096,4x4096x4096,8x4096x4096,32x4096x4096 > $A/ab_$1.log 2>&1
echo "ab $1 exit $?" >> $A/summary.txt
{ date; nvidia-smi --query-gpu=utilization.gpu,clocks.sm --format=csv,noheader; "$SP/games.sh"; } >> $A/env_$1.txt
