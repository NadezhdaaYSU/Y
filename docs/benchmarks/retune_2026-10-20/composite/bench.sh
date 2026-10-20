#!/bin/bash
# bench.sh: the re-tuned kernels against the current ones, L2-hot and from DRAM, two runs each, under gpu.lock.
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
T=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/retune3
X=$SP/xbar
exec 9>"$SP/gpu.lock"; flock 9
cd $T || exit 9
mean() { for i in $(seq 10); do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0; sleep 0.1; done | awk "{s+=\$1} END {printf \"%.1f\", s/NR}"; }
run() { name=$1; shift; echo "$name: $(date +%T) mean $(mean)% games: $("$SP/games.sh" | tr "\n" " ")" >> $T/queue.txt; env Y_LIB_PATH=$X/target/release/liby.so PYTHONPATH=$X/python "$@" > $T/$name.log 2>&1; echo "$name exit $?" >> $T/bench_summary.txt; }
for i in 1 2; do
  run hot_$i python3 $SP/swz/ab/bench_ab.py "current=$T/cur,new=$T/new" $T/hot_$i.json 256,512,1024,2048,4096,8192,16384,1x4096x4096,4x4096x4096,8x4096x4096,32x4096x4096,512x1024x256,1280x1024x1024
  run cold_$i python3 /tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/autotune2/bench_cold_sq.py $SP/swz/ab "current=$T/cur,new=$T/new" $T/cold_$i.json 1024,2048,4096,8192
  run bias_$i python3 $SP/bench2/bench_bias.py $SP/swz/ab "current=$T/cur,new=$T/new" $T/bias_$i.json
  run dec_$i python3 $X/docs/benchmarks/decode_dram_2026-10-18/bench_cold.py $SP/swz/ab "current=$T/cur,new=$T/new" $T/dec_$i.json 4 8
done
echo DONE >> $T/bench_summary.txt
