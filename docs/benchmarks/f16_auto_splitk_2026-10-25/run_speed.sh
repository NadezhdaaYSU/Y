#!/bin/bash
# run_speed.sh PTX OUT [SHAPES]: every column of each shape (gen.sh) against cuBLAS with ./sbench.py (CUDA
# graphs, 21 interleaved rounds, minimum), two runs x small-integer and N(0,1) timing data; utilisation, clocks
# and any game before each run in OUT/env.txt. Holds the GPU lock throughout; times nothing while a game runs
# (exits 4 if one is running once the lock is held, or before or after any run). With DRAM=1 each call reads
# the next of R copies of A and B, R x (A + B) at least 144 MB (three L2s), so operands come from DRAM.
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
X=${X:-$S/xbar}; P=$1; O=$2; mkdir -p $O; H=$X/docs/benchmarks/f16_auto_splitk_2026-10-25
. $H/shapes.sh
exec 9>$S/gpu.lock; flock 9
no_game() { [ -z "$($S/games.sh)" ] || { echo "a game is running ($1, $(date +%T)): not timed" >> $O/summary.txt; exit 4; }; }
no_game "lock taken"
export Y_LIB_PATH=$X/target/release/liby.so PYTHONPATH=$X/python
env_log() { { echo "$1 $(date)"; for j in 1 2 3; do nvidia-smi --query-gpu=utilization.gpu,clocks.sm,power.draw,memory.used --format=csv,noheader; sleep 1; done; $S/games.sh; echo; } >> $O/env.txt 2>&1; }
for i in 1 2; do for sh in ${3:-$SHAPES_A $SHAPES_D $SHAPES_B}; do for d in int normal; do
  IFS=x read m n k <<< "$sh"
  calls=$(python3 -c "print(max(3, min(200, int(5e-3 / max(2*$m*$n*$k/80e12, 2e-6)))))")
  cols=""; for f in $P/$sh/*.ptx; do c=$(basename $f .ptx); cols="$cols $c=$f"; done
  no_game "before $sh run $i $d"
  env_log "$sh run $i $d"
  rot=1; [ -n "$DRAM" ] && rot=$(python3 -c "print(max(1, -(-144_000_000 // (2 * $k * ($m + $n)))))")
  python3 $H/sbench.py $O/${sh}_run${i}_$d.json $m $n $k $cols calls=$calls data=$d rot=$rot > $O/${sh}_run${i}_$d.log 2>&1
  echo "$sh run $i $d exit $? $(date +%T)" >> $O/summary.txt
  no_game "after $sh run $i $d"
done; done; done
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > $O/gpu.txt; ptxas --version | tail -1 >> $O/gpu.txt
python3 -c "import torch; print('torch', torch.__version__)" >> $O/gpu.txt
echo DONE >> $O/summary.txt
