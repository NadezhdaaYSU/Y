#!/bin/bash
# queue.sh: the full accuracy grid twice (outputs are hashed, so the two runs can be compared).
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
F=$S/acc2; X=$S/xbar
exec 9>"$S/gpu.lock"; flock 9
cd $F || exit 9
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > $F/gpu.txt
for i in 1 2; do
  echo "run $i start $(date +%T)" >> $F/summary.txt
  env Y_LIB_PATH=$X/target/release/liby.so PYTHONPATH=$X/python python3 $F/accuracy.py $S/swz/ab $F/ptx $F/acc_$i.json > $F/acc_$i.log 2> $F/acc_$i.err
  echo "run $i exit $? $(date +%T)" >> $F/summary.txt
done
echo DONE >> $F/summary.txt
