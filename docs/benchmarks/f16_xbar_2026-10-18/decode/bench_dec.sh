#!/bin/bash
# bench_dec.sh <tag>: decode shapes, committed schedule (off) against the switched one (on), and cuBLAS.
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
O=$SP/xbar_ptx
cd $O || exit 9
Y_LIB_PATH=$SP/snap957/target/release/liby.so PYTHONPATH=$SP/snap957/python python3 $SP/f16sq/bench_ab.py "off=$O/bench_dec/off,on=$O/bench_dec/on" $O/bench_dec_$1.json 1x4096x4096,4x4096x4096,8x4096x4096,32x4096x4096 > $O/bench_dec_$1.log 2>&1
echo "bench_dec $1 exit $?" >> $O/summary.txt
