#!/bin/bash
# run.sh OUT: OUT/ptx from gen.sh, then two runs of the decode GEMMs with
# ../f16_square_2026-10-18/bench_ab.py (f32, kahan, mma, cuBLAS); utilisation, clocks and any
# game process before each run in OUT/env.txt. Times nothing while a game runs: exits 4 if one is
# running once the lock is held, or before or after either run.
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
X=${X:-$S/xbar}; R=$X/docs/benchmarks/f16_kahan_default_2026-10-24; O=$1; mkdir -p $O
exec 9>$S/gpu.lock; flock 9
no_game() { [ -z "$($S/games.sh)" ] || { echo "a game is running ($1, $(date +%T)): not timed" >> $O/summary.txt; exit 4; }; }
no_game "lock taken"
export Y_LIB_PATH=$X/target/release/liby.so PYTHONPATH=$X/python
env_log() { { echo "$1 $(date)"; for j in 1 2 3 4 5; do nvidia-smi --query-gpu=utilization.gpu,clocks.sm,power.draw --format=csv,noheader; sleep 1; done; pgrep -af 'rungameid|sober|steam|roblox' | grep -v pgrep | cut -c1-80; } >> $O/env.txt 2>&1; }
X=$X $R/gen.sh $O/ptx > $O/gen.txt 2>&1
G=1x11008x4096,8x11008x4096,32x11008x4096,1x4096x11008,8x4096x11008,32x4096x11008,1x14336x4096,8x14336x4096,32x14336x4096,1x4096x14336,8x4096x14336,32x4096x14336
for i in 1 2; do
  no_game "before run $i"
  env_log "gemm run $i"
  python3 $X/docs/benchmarks/f16_square_2026-10-18/bench_ab.py "f32=$O/ptx/f32,kahan=$O/ptx/kahan,mma=$O/ptx/mma" $O/gemm_$i.json $G > $O/gemm_$i.log 2>&1
  echo "gemm $i exit $?" >> $O/summary.txt
  no_game "after run $i"
done
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > $O/gpu.txt; ptxas --version | tail -1 >> $O/gpu.txt
python3 -c "import torch; print('torch', torch.__version__)" >> $O/gpu.txt
echo DONE >> $O/summary.txt
