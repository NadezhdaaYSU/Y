#!/bin/bash
# queue.sh: bias+ReLU with the candidate tiles, and 16384^3 tile candidates, two runs each, under gpu.lock.
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
R=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/retune2
X=$SP/xbar
exec 9>"$SP/gpu.lock"; flock 9
cd $R || exit 9
mean() { for i in $(seq 10); do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0; sleep 0.1; done | awk "{s+=\$1} END {printf \"%.1f\", s/NR}"; }
for i in 1 2; do
  echo "bias $i: $(date +%T) mean $(mean)% games: $("$SP/games.sh" | tr "\n" " ")" >> $R/queue.txt
  Y_LIB_PATH=$X/target/release/liby.so PYTHONPATH=$X/python python3 $SP/bench2/bench_bias.py $SP/swz/ab "current=$R/bias_cur,candidate=$R/bias_new" $R/bias_$i.json 2048,4096,8192 > $R/bias_$i.log 2>&1
  echo "bias $i exit $?" >> $R/summary.txt
  echo "16384 $i: $(date +%T) mean $(mean)% games: $("$SP/games.sh" | tr "\n" " ")" >> $R/queue.txt
  Y_LIB_PATH=$X/target/release/liby.so PYTHONPATH=$X/python python3 $SP/swz/ab/bench_ab.py "128_128_32_2_2_2=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/retune2/p16384/128_128_32_2_2_2,128_128_32_2_2_3=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/retune2/p16384/128_128_32_2_2_3,128_128_32_4_2_3=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/retune2/p16384/128_128_32_4_2_3" $R/p16384_$i.json 16384 > $R/p16384_$i.log 2>&1
  echo "16384 $i exit $?" >> $R/summary.txt
done
