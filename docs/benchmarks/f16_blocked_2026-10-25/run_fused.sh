#!/bin/bash
# run_fused.sh FUSED OUT: two runs of bench_swiglu.py (4096), bench_bias.py (8192) and sbench.py (16384^3,
# int and N(0,1) data) over FUSED/{mma,blk128,blk256} (gen_fused.sh). Times nothing while a game runs:
# exits 4 if one is running once the lock is held, or before or after any run.
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
X=${X:-$S/xbar}; F=$1; O=$2; B=$X/docs/benchmarks; mkdir -p $O
exec 9>$S/gpu.lock; flock 9
no_game() { [ -z "$($S/games.sh)" ] || { echo "a game is running ($1, $(date +%T)): not timed" >> $O/summary.txt; exit 4; }; }
no_game "lock taken"
export Y_LIB_PATH=$X/target/release/liby.so PYTHONPATH=$X/python
env_log() { { echo "$1 $(date)"; for j in 1 2 3; do nvidia-smi --query-gpu=utilization.gpu,clocks.sm,power.draw --format=csv,noheader; sleep 1; done; $S/games.sh; echo; } >> $O/env.txt 2>&1; }
cols="mma=$F/mma,blk128=$F/blk128,blk256=$F/blk256"
for i in 1 2; do
  no_game "before swiglu $i"; env_log "swiglu run $i"
  python3 $B/swiglu_tile_2026-10-20/bench_swiglu.py $B/f16_square_2026-10-18 $cols,blk256cap=$F/blk256cap $F/mma $O/swiglu_$i.json 4096 > $O/swiglu_$i.log 2>&1
  echo "swiglu $i exit $?" >> $O/summary.txt
  no_game "before bias $i"; env_log "bias run $i"
  python3 $B/quiet_2026-10-20/bias_relu/bench_bias.py $B/f16_square_2026-10-18 $cols $O/bias_$i.json 8192 > $O/bias_$i.log 2>&1
  echo "bias $i exit $?" >> $O/summary.txt
  for d in int normal; do
    no_game "before 16384 $i $d"; env_log "16384 run $i $d"
    python3 $B/f16_splitk_2026-10-20/sweep/sbench.py $O/16384_run${i}_$d.json 16384 16384 16384 Y_mma=$F/mma/gemm_f16_16384.ptx Y_blk128=$F/blk128/gemm_f16_16384.ptx Y_blk256=$F/blk256/gemm_f16_16384.ptx calls=3 data=$d > $O/16384_run${i}_$d.log 2>&1
    echo "16384 $i $d exit $?" >> $O/summary.txt
  done
  no_game "after run $i"
done
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > $O/gpu.txt; ptxas --version | tail -1 >> $O/gpu.txt
echo DONE >> $O/summary.txt
