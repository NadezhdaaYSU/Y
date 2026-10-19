#!/bin/bash
# run.sh <i>: the committed kernels (base) against the swizzled ones (swz),
# with Triton and cuBLAS, at square and decode shapes, holding gpu.lock but
# not waiting for a quiet GPU; utilisation and games logged first.
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
A=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/swz/ab
exec 9>"$SP/gpu.lock"; flock 9
cd $A || exit 9
mean() { for i in $(seq 10); do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0; sleep 0.1; done | awk '{s+=$1} END {printf "%.1f", s/NR}'; }
echo "$1: $(date +%T) mean $(mean)% games: $("$SP/games.sh" | grep -oE 'isaac-ng.exe|sober|AppId=[0-9]+|rungameid/[0-9]+' | sort -u | tr '\n' ' ')" >> $A/queue.txt
{ date; nvidia-smi --query-gpu=name,driver_version,utilization.gpu,clocks.sm --format=csv,noheader; "$SP/games.sh"; python3 -c "import torch, triton; print(torch.__version__, triton.__version__)"; } > $A/env_$1.txt
Y_LIB_PATH=$SP/xbar/target/release/liby.so PYTHONPATH=$SP/xbar/python python3 $A/bench_ab.py base=$A/base,swz=$A/swz $A/ab_$1.json 256,512,1024,2048,4096,8192,1x4096x4096,4x4096x4096,8x4096x4096,32x4096x4096 > $A/ab_$1.log 2>&1
echo "ab $1 exit $?" >> $A/summary.txt
{ date; nvidia-smi --query-gpu=utilization.gpu,clocks.sm --format=csv,noheader; "$SP/games.sh"; } >> $A/env_$1.txt
