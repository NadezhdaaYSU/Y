#!/bin/bash
# queue.sh <tag> <hd> <rows,...>
N=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/rope
D="/home/yumin/NVME files/YSU-engine-main/YSU-engine-main/src/Y_lang/Y/docs/benchmarks/paged_attention_2026-10-18"
echo "$1: $(WAIT_S=7200 bash "$D/gpu_quiet_then.sh" bash $N/run.sh "$@" | head -1)" >> $N/queue.txt
