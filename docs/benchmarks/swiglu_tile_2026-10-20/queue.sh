#!/bin/bash
# queue.sh: fused SwiGLU tiles (Y_SWIGLU_TILE) at 512-4096, bench_swiglu.py, two runs, under gpu.lock.
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
W=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/swiglu_tiles
X=$SP/xbar
exec 9>"$SP/gpu.lock"; flock 9
cd $W || exit 9
mean() { for i in $(seq 10); do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0; sleep 0.1; done | awk "{s+=\$1} END {printf \"%.1f\", s/NR}"; }
for i in 1 2; do
  echo "run $i: $(date +%T) mean $(mean)% games: $("$SP/games.sh" | tr "\n" " ")" >> $W/queue.txt
  Y_LIB_PATH=$X/target/release/liby.so PYTHONPATH=$X/python python3 $SP/bench2/bench_swiglu.py $SP/swz/ab "128_128_32_4_4=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/swiglu_tiles/128_128_32_4_4,64_64_32_2_2=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/swiglu_tiles/64_64_32_2_2,64_64_32_4_2=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/swiglu_tiles/64_64_32_4_2,128_64_32_4_2=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/swiglu_tiles/128_64_32_4_2,64_128_32_2_4=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/swiglu_tiles/64_128_32_2_4,64_64_64_2_2=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/swiglu_tiles/64_64_64_2_2" $SP/acc/ptx $W/sw_$i.json > $W/sw_$i.log 2>&1
  echo "run $i exit $?" >> $W/summary.txt
done
