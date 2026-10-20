#!/bin/bash
# queue.sh: the quiet-GPU benchmarks after 9d64faa, one after another under
# gpu.lock; each logs the 10-sample mean utilisation and any game first.
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
B=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/bench2
X=$SP/xbar
exec 9>"$SP/gpu.lock"; flock 9
cd $B || exit 9
mean() { for i in $(seq 10); do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0; sleep 0.1; done | awk '{s+=$1} END {printf "%.1f", s/NR}'; }
note() { echo "$1: $(date +%T) mean $(mean)% games: $("$SP/games.sh" | grep -oE 'isaac-ng.exe|sober|AppId=[0-9]+|rungameid/[0-9]+' | sort -u | tr '\n' ' ')" >> $B/queue.txt; }
run() { name=$1; shift; note "$name"; env Y_LIB_PATH=$X/target/release/liby.so PYTHONPATH=$X/python "$@" > $B/$name.log 2>&1; echo "$name exit $?" >> $B/summary.txt; }
{ date; nvidia-smi --query-gpu=name,driver_version,utilization.gpu,clocks.sm,memory.used --format=csv,noheader; "$SP/games.sh"; python3 -c "import torch, triton; print(torch.__version__, triton.__version__)"; } > $B/env.txt
run acc_1 python3 $SP/acc/accuracy.py $SP/swz/ab $SP/acc/ptx $B/acc_1.json 256 512 1024 2048 4096 8192 1x4096x4096 4x4096x4096 8x4096x4096 32x4096x4096 512x512x1024 512x512x4096 512x512x16384 512x512x65536
for i in 1 2; do
  run ksweep_$i python3 $SP/ksweep/ktiles.py "padded=$B/k_pad,swizzled=$B/k_swz" $B/ksweep_$i.json 256,512,1024,2048,4096,8192
  run bias_$i python3 $B/bench_bias.py $SP/swz/ab "base=$B/bias_base,swz=$B/bias_swz" $B/bias_$i.json
  run cold_$i python3 $X/docs/benchmarks/decode_dram_2026-10-18/bench_cold.py $SP/swz/ab "base=$B/dec_base,swz=$B/dec_swz" $B/cold_$i.json 4
done
run ab_3 python3 $SP/swz/ab/bench_ab.py "base=$SP/swz/ab/base,swz=$SP/swz/ab/swz" $B/ab_3.json 256,512,1024,2048,4096,8192,1x4096x4096,4x4096x4096,8x4096x4096,32x4096x4096
{ date; nvidia-smi --query-gpu=utilization.gpu,clocks.sm --format=csv,noheader; "$SP/games.sh"; } >> $B/env.txt
echo DONE >> $B/summary.txt
