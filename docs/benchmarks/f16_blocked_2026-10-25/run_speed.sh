#!/bin/bash
# run_speed.sh PTX OUT: each shape's columns of PTX (gen.sh) against cuBLAS with
# f16_splitk_2026-10-20/sweep/sbench.py (CUDA graphs, interleaved rounds), two runs x
# small-integer and N(0,1) timing data; utilisation, clocks and any game before each run in
# OUT/env.txt. Times nothing while a game runs: exits 4 if one is running once the lock is held,
# or before or after any run.
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
X=${X:-$S/xbar}; P=$1; O=$2; mkdir -p $O
exec 9>$S/gpu.lock; flock 9
no_game() { [ -z "$($S/games.sh)" ] || { echo "a game is running ($1, $(date +%T)): not timed" >> $O/summary.txt; exit 4; }; }
no_game "lock taken"
export Y_LIB_PATH=$X/target/release/liby.so PYTHONPATH=$X/python
env_log() { { echo "$1 $(date)"; for j in 1 2 3; do nvidia-smi --query-gpu=utilization.gpu,clocks.sm,power.draw --format=csv,noheader; sleep 1; done; $S/games.sh; echo; } >> $O/env.txt 2>&1; }
for i in 1 2; do for sh in ${SHAPES:-8192 512x512x4096 512x512x16384 512x512x65536 512x512x1024 1024}; do for d in int normal; do
  IFS=x read m n k <<< "$sh"; [ -z "$n" ] && n=$m && k=$m
  calls=$(python3 -c "print(max(3, min(200, int(5e-3 / max(2*$m*$n*$k/80e12, 2e-6)))))")
  cols=""; for c in ${COLS:-Y_mma Y_f32 Y_blk64 Y_blk128 Y_blk256 Y_blk512 Y_blk1024}; do cols="$cols $c=$P/$sh/$c.ptx"; done
  no_game "before $sh run $i $d"
  env_log "$sh run $i $d"
  python3 $X/docs/benchmarks/f16_splitk_2026-10-20/sweep/sbench.py $O/${sh}_run${i}_$d.json $m $n $k $cols calls=$calls data=$d > $O/${sh}_run${i}_$d.log 2>&1
  echo "$sh run $i $d exit $?" >> $O/summary.txt
  no_game "after $sh run $i $d"
done; done; done
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > $O/gpu.txt; ptxas --version | tail -1 >> $O/gpu.txt
echo DONE >> $O/summary.txt
