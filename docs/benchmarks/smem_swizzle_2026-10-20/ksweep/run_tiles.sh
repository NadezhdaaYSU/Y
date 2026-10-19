#!/bin/bash
W=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/ksweep
X=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/xbar
exec 9>"/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/gpu.lock"; flock 9
cd "$W" || exit 9
for i in 1 2; do
  { echo "tiles run $i $(date)"; for j in $(seq 10); do nvidia-smi --query-gpu=utilization.gpu,clocks.sm --format=csv,noheader; sleep 1; done; pgrep -af 'rungameid|isaac|Isaac|roblox|Roblox' | grep -v pgrep | cut -c1-120; } >> "$W/queue_tiles.txt" 2>&1
  Y_LIB_PATH=$X/target/release/liby.so PYTHONPATH=$X/python python3 "$W/ktiles.py" "64_64_32_4_2_4=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/ksweep/t/64_64_32_4_2_4,64_64_32_4_2_6=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/ksweep/t/64_64_32_4_2_6,64_64_32_2_2_4=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/ksweep/t/64_64_32_2_2_4,64_64_64_4_2_4=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/ksweep/t/64_64_64_4_2_4,64_64_64_2_2_4=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/ksweep/t/64_64_64_2_2_4,64_64_64_2_2_5=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/ksweep/t/64_64_64_2_2_5" "$W/tiles_$i.json" 512,1024,4096 > "$W/tiles_$i.log" 2>&1
  echo "tiles run $i exit $?" >> "$W/summary.txt"
done
