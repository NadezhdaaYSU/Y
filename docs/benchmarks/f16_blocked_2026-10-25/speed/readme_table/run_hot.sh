#!/bin/bash
# run_hot.sh: the README table's harness (f16_square_2026-10-18/bench_ab.py: one B reused, interleaved rounds,
# minimum) on the committed 8192 and 16384 kernels, this tree's (new, blocked) against HEAD's (old, the chain),
# with cuBLAS and Triton; two runs. Times nothing while a game runs (exit 4).
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad; W=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/8cb989c6-e604-4dd6-849f-ecfdae469440/scratchpad/blk/hot; X=$S/xbar
exec 9>$S/gpu.lock; flock 9
no_game() { [ -z "$($S/games.sh)" ] || { echo "a game is running ($1, $(date +%T)): not timed" >> $W/summary.txt; exit 4; }; }
export Y_LIB_PATH=$X/target/release/liby.so PYTHONPATH=$X/python
for i in 1 2; do
  no_game "before run $i"
  { echo "run $i $(date)"; for j in 1 2 3; do nvidia-smi --query-gpu=utilization.gpu,clocks.sm,power.draw --format=csv,noheader; sleep 1; done; } >> $W/env.txt
  python3 $X/docs/benchmarks/f16_square_2026-10-18/bench_ab.py "new=$W/ptx/new,old=$W/ptx/old" $W/hot_$i.json 8192,16384 > $W/hot_$i.log 2>&1
  echo "run $i exit $?" >> $W/summary.txt
  no_game "after run $i"
done
echo DONE >> $W/summary.txt
