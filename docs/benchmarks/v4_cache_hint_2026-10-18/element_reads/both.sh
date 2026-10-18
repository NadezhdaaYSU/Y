#!/bin/bash
P=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
D="/home/yumin/NVME files/YSU-engine-main/YSU-engine-main/src/Y_lang/Y/docs/benchmarks/paged_attention_2026-10-18"
for r in 1 2; do
  echo "run $r: $(bash "$D/gpu_quiet_then.sh" bash "$P/big/ca2/run.sh" $r | head -1)" >> "$P/big/ca2/queue.txt"
done
sha256sum "$P/snap957/target/release/liby.so" >> "$P/big/ca2/queue.txt"
