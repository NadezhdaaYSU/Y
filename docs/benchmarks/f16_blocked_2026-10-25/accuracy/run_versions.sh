#!/bin/bash
# run_versions.sh: the accuracy cases again on the final kernels (ptx, ptx2, ptx_split: v3), then their outputs
# against the first runs' (the v1 kernels) column by column. Waits for the GPU lock; no timing.
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
W=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/8cb989c6-e604-4dd6-849f-ecfdae469440/scratchpad/blk
X=$S/xbar; B=$X/docs/benchmarks
export ACC_DIR=$B/f16_accuracy_2026-10-20 SPLIT_DIR=$B/f16_splitk_2026-10-20 Y_LIB_PATH=$X/target/release/liby.so PYTHONPATH=$X/python
cd $W
exec 9>$S/gpu.lock; flock 9
for t in "ptx acc" "ptx2 acc2" "ptx_split acc_split"; do set -- $t
  python3 accuracy_blocked.py $B/f16_square_2026-10-18 $W/$1 $W/${2}_v3.json > ${2}_v3.log 2>&1; echo "$2 exit $?"
  python3 versions.py $W/$2.json $W/${2}_v3.json
done
