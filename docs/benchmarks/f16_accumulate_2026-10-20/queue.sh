#!/bin/bash
# queue.sh: the F16 GEMM accumulating each mma in float32 (Y_F16_ACCUMULATE=f32) against 9d64faa+swiglu (mma chains), speed and accuracy.
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
F=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/f32acc
X=$SP/xbar
exec 9>"$SP/gpu.lock"; flock 9
cd $F || exit 9
mean() { for i in $(seq 10); do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0; sleep 0.1; done | awk "{s+=\$1} END {printf \"%.1f\", s/NR}"; }
note() { echo "$1: $(date +%T) mean $(mean)% games: $("$SP/games.sh" | tr "\n" " ")" >> $F/queue.txt; }
run() { name=$1; shift; note "$name"; env Y_LIB_PATH=$X/target/release/liby.so PYTHONPATH=$X/python "$@" > $F/$name.log 2>&1; echo "$name exit $?" >> $F/summary.txt; }
run acc_f32 python3 $SP/acc/accuracy.py $SP/swz/ab $F/acc_ptx $F/acc_f32.json 256 512 1024 2048 4096 8192 1x4096x4096 4x4096x4096 8x4096x4096 32x4096x4096 512x512x1024 512x512x4096 512x512x16384 512x512x65536
for i in 1 2; do run ab_$i python3 $SP/swz/ab/bench_ab.py "mma=$SP/swz/ab/swz,f32=$F/ab_f32" $F/ab_$i.json 256,512,1024,2048,4096,8192,1x4096x4096,4x4096x4096,8x4096x4096,32x4096x4096; done
echo DONE >> $F/summary.txt
