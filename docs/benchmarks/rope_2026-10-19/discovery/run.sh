#!/bin/bash
# run.sh <tag> <hd> <rows,...>: one probe_rope.py run over v<hd>/.
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
N=$SP/rope
M="/home/yumin/NVME files/YSU-engine-main/YSU-engine-main/src/Y_lang/Y"
cd $N || exit 9
{ date; nvidia-smi --query-gpu=name,driver_version,utilization.gpu,clocks.sm --format=csv,noheader; "$SP/games.sh"; echo; } > $N/env_$1.txt
Y_LIB_PATH=$SP/xbar/target/release/liby.so PYTHONPATH=$SP/xbar/python python3 $N/probe_rope.py "$M/python/examples" $SP/f16sq $SP/xbar/tests/rope_$2.ptx $N/v$2 $2 $3 > $N/rope_$1.log 2>&1
echo "rope $1 exit $?" >> $N/summary.txt
{ date; nvidia-smi --query-gpu=utilization.gpu,clocks.sm --format=csv,noheader; "$SP/games.sh"; echo; } >> $N/env_$1.txt
