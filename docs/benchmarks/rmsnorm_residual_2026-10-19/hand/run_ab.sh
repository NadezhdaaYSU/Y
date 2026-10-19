#!/bin/bash
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
M="/home/yumin/NVME files/YSU-engine-main/YSU-engine-main/src/Y_lang/Y"
cd $SP/nr/hand || exit 9
Y_LIB_PATH=$SP/xbar/target/release/liby.so PYTHONPATH=$SP/xbar/python python3 ab.py "$M/python/examples" $SP/f16sq "$M/tests/rmsnorm_residual_4096.ptx" $SP/nr/emit/new_4096.ptx $SP/nr/hand/reg_4096.ptx > $SP/nr/hand/ab_$1.log 2>&1
echo "ab $1 exit $?" >> $SP/nr/hand/summary.txt
