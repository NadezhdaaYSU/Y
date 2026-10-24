#!/bin/bash
# run_cold.sh OUT: the decode kernels of OUT/ptx in three modes from DRAM and from L2, with
# ../decode_dram_2026-10-18/bench_cold.py (4 copies of B), two runs; utilisation before each.
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
X=$S/xbar; O=$1
exec 9>$S/gpu.lock; flock 9
export Y_LIB_PATH=$X/target/release/liby.so PYTHONPATH=$X/python
for i in 1 2; do
  { echo "cold run $i $(date)"; for j in 1 2 3 4 5; do nvidia-smi --query-gpu=utilization.gpu,clocks.sm,power.draw --format=csv,noheader; sleep 1; done; pgrep -af 'rungameid|sober|steam|roblox' | grep -v pgrep | cut -c1-80; } >> $O/env.txt 2>&1
  python3 $X/docs/benchmarks/decode_dram_2026-10-18/bench_cold.py $S/swz/ab "f32=$O/ptx/f32,kahan=$O/ptx/kahan,mma=$O/ptx/mma" $O/cold_$i.json 4 > $O/cold_$i.log 2>&1
  echo "cold $i exit $?" >> $O/summary.txt
done
echo COLD_DONE >> $O/summary.txt
