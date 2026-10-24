#!/bin/bash
# run_speed.sh OUT: the decode kernels in three modes with ../f16_square_2026-10-18/bench_ab.py
# (two runs), then the 512 x 512 kernels of accuracy/ptx with ../f16_splitk_2026-10-20/sweep/sbench.py
# (two runs, small-integer and normal timing data); utilisation before each run in OUT/env.txt.
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
X=$S/xbar; R=$X/docs/benchmarks/f16_kahan_2026-10-20; O=$1; mkdir -p $O
exec 9>$S/gpu.lock; flock 9
export Y_LIB_PATH=$X/target/release/liby.so PYTHONPATH=$X/python
env_log() { { echo "$1 $(date)"; for j in 1 2 3 4 5; do nvidia-smi --query-gpu=utilization.gpu,clocks.sm,power.draw --format=csv,noheader; sleep 1; done; pgrep -af 'rungameid|sober|steam|roblox' | grep -v pgrep | cut -c1-80; } >> $O/env.txt 2>&1; }
$R/gen_speed.sh $O/ptx > $O/gen.txt 2>&1
for i in 1 2; do
  env_log "decode run $i"
  python3 $X/docs/benchmarks/f16_square_2026-10-18/bench_ab.py "f32=$O/ptx/f32,kahan=$O/ptx/kahan,mma=$O/ptx/mma" $O/decode_$i.json 1x4096x4096,4x4096x4096,8x4096x4096,32x4096x4096 > $O/decode_$i.log 2>&1
  echo "decode $i exit $?" >> $O/summary.txt
done
for i in 1 2; do for K in 4096 65536; do for d in int normal; do
  calls=50; [ $K = 65536 ] && calls=6
  cols=""; for f in $R/accuracy/ptx/512x512x$K/*.ptx; do cols="$cols $(basename $f .ptx)=$f"; done
  env_log "512 x 512 x $K run $i $d"
  python3 $S/xbar/docs/benchmarks/f16_splitk_2026-10-20/sweep/sbench.py $O/k${K}_run${i}_$d.json 512 512 $K $cols calls=$calls data=$d > $O/k${K}_run${i}_$d.log 2>&1
  echo "k$K run $i $d exit $?" >> $O/summary.txt
done; done; done
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > $O/gpu.txt; ptxas --version | tail -1 >> $O/gpu.txt
echo DONE >> $O/summary.txt
