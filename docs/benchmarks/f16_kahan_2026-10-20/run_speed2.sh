#!/bin/bash
# run_speed2.sh OUT PERMMA_DIR: two-level Kahan ("kahan2", this compiler) against per-mma Kahan
# ("kahan1", the kernels in PERMMA_DIR), float32 and the chain: decode from L2 (bench_ab.py) and
# from DRAM (bench_cold.py), two runs each, then 512 x 512 x 4096 and 65536 with sbench.py.
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
X=$S/xbar; R=$X/docs/benchmarks/f16_kahan_2026-10-20; O=$1; K1=$2; mkdir -p $O
exec 9>$S/gpu.lock; flock 9
export Y_LIB_PATH=$X/target/release/liby.so PYTHONPATH=$X/python
env_log() { { echo "$1 $(date)"; for j in 1 2 3 4 5; do nvidia-smi --query-gpu=utilization.gpu,clocks.sm,power.draw --format=csv,noheader; sleep 1; done; pgrep -af 'rungameid|sober|steam|roblox' | grep -v pgrep | cut -c1-80; } >> $O/env.txt 2>&1; }
$R/gen_speed.sh $O/ptx > $O/gen.txt 2>&1
dirs="f32=$O/ptx/f32,kahan1=$K1,kahan2=$O/ptx/kahan,mma=$O/ptx/mma"
for i in 1 2; do
  env_log "decode L2 run $i"
  python3 $X/docs/benchmarks/f16_square_2026-10-18/bench_ab.py "$dirs" $O/decode_$i.json 1x4096x4096,4x4096x4096,8x4096x4096,32x4096x4096 > $O/decode_$i.log 2>&1
  echo "decode $i exit $?" >> $O/summary.txt
  env_log "decode DRAM run $i"
  python3 $X/docs/benchmarks/decode_dram_2026-10-18/bench_cold.py $S/swz/ab "$dirs" $O/cold_$i.json 4 > $O/cold_$i.log 2>&1
  echo "cold $i exit $?" >> $O/summary.txt
done
$R/gen.sh $O/big > $O/gen_big.txt 2>&1
for i in 1 2; do for K in 4096 65536; do for d in int normal; do
  calls=50; [ $K = 65536 ] && calls=6
  cols=""; for f in $O/big/512x512x$K/*.ptx; do cols="$cols $(basename $f .ptx)=$f"; done
  for f in $R/accuracy/ptx/512x512x$K/Y_kahan*.ptx; do cols="$cols $(basename $f .ptx)_1=$f"; done
  env_log "512 x 512 x $K run $i $d"
  python3 $X/docs/benchmarks/f16_splitk_2026-10-20/sweep/sbench.py $O/k${K}_run${i}_$d.json 512 512 $K $cols calls=$calls data=$d > $O/k${K}_run${i}_$d.log 2>&1
  echo "k$K run $i $d exit $?" >> $O/summary.txt
done; done; done
echo DONE >> $O/summary.txt
