#!/bin/bash
# run.sh <tag>: one probe_final.py run.
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
N=$SP/rope/final
M="/home/yumin/NVME files/YSU-engine-main/YSU-engine-main/src/Y_lang/Y"
cd $N || exit 9
{ date; nvidia-smi --query-gpu=name,driver_version,utilization.gpu,clocks.sm --format=csv,noheader; "$SP/games.sh"; python3 -c "import torch, triton, flashinfer; print(torch.__version__, triton.__version__, flashinfer.__version__)"; echo; } > $N/env_$1.txt
Y_LIB_PATH=$SP/xbar/target/release/liby.so PYTHONPATH=$SP/fishim/y_flashinfer_shim:$SP/xbar/python python3 $N/probe_final.py "$M/python/examples" $SP/f16sq $N/ptx > $N/final_$1.log 2>&1
echo "final $1 exit $?" >> $N/summary.txt
{ date; nvidia-smi --query-gpu=utilization.gpu,clocks.sm --format=csv,noheader; "$SP/games.sh"; echo; } >> $N/env_$1.txt
