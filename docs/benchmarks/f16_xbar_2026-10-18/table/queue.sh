#!/bin/bash
T=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/xbar_ptx/table
D="/home/yumin/NVME files/YSU-engine-main/YSU-engine-main/src/Y_lang/Y/docs/benchmarks/paged_attention_2026-10-18"
for t in "$@"; do
  echo "table $t: $(bash "$D/gpu_quiet_then.sh" bash $T/bench.sh $t | head -1)" >> $T/queue.txt
done
