#!/bin/bash
# run_final.sh DIR: two runs of each K's columns, small-integer then normal timing data.
W=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/8cb989c6-e604-4dd6-849f-ecfdae469440/scratchpad/sk
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad; X=$S/xbar; D=$1
exec 9>$S/gpu.lock; flock 9
for i in 1 2; do for K in 4096 16384 65536; do for d in int normal; do
  calls=50; [ $K = 16384 ] && calls=20; [ $K = 65536 ] && calls=6
  cols=""; for f in $D/$K/*.ptx; do cols="$cols $(basename $f .ptx)=$f"; done
  { echo "run $i K=$K data=$d $(date)"; for j in 1 2 3 4 5; do nvidia-smi --query-gpu=utilization.gpu,clocks.sm,power.draw --format=csv,noheader; sleep 1; done; pgrep -af 'rungameid|isaac|roblox|Roblox|sober|steam' | grep -v pgrep | cut -c1-100; } >> $D/env.txt 2>&1
  Y_LIB_PATH=$X/target/release/liby.so PYTHONPATH=$X/python python3 $W/sbench.py $D/$K/run${i}_$d.json 512 512 $K $cols calls=$calls data=$d > $D/$K/run${i}_$d.log 2>&1
  echo "run $i K=$K $d exit $?" >> $D/summary.txt
done; done; done
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > $D/gpu.txt; ptxas --version | tail -1 >> $D/gpu.txt
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" >> $D/gpu.txt
echo DONE >> $D/summary.txt
