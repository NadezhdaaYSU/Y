#!/bin/bash
W=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/ksweep
X=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/xbar
exec 9>"/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/gpu.lock"; flock 9
cd "$W" || exit 9
for i in 1 2; do
  { echo "run $i $(date)"; for j in $(seq 10); do nvidia-smi --query-gpu=utilization.gpu,clocks.sm --format=csv,noheader; sleep 1; done; pgrep -af 'isaac|Isaac|roblox|Roblox|steam' | grep -v pgrep; } >> "$W/queue.txt" 2>&1
  nvidia-smi > "$W/env_$i.txt" 2>&1
  Y_LIB_PATH=$X/target/release/liby.so PYTHONPATH=$X/python python3 "$W/ksweep.py" "$W/v" "$W/ks_$i.json" > "$W/ks_$i.log" 2>&1
  echo "run $i exit $?" >> "$W/summary.txt"
done
