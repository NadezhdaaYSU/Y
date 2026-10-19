#!/bin/bash
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
A=$SP/g512s/ab
cd $A || exit 9
{ date; nvidia-smi --query-gpu=utilization.gpu,clocks.sm --format=csv,noheader; "$SP/games.sh"; } > $A/env_$1.txt
Y_LIB_PATH=$SP/xbar/target/release/liby.so PYTHONPATH=$SP/xbar/python python3 $SP/f16sq/bench_ab.py base=$A/base,pow2=$A/pow2 $A/ab_$1.json 256,512,1024,2048,4096,8192 > $A/ab_$1.log 2>&1
echo "ab $1 exit $?" >> $A/summary.txt
