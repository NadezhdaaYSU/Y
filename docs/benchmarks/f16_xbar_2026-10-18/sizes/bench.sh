#!/bin/bash
# bench.sh <tag>: committed schedule (off) against the switched one (on) at 256, 512, 1024.
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
O=$SP/xbar_ptx
cd $O || exit 9
Y_LIB_PATH=$SP/snap957/target/release/liby.so PYTHONPATH=$SP/snap957/python python3 $SP/f16sq/bench_ab.py "off=$O/bench/off,on=$O/bench/on" $O/bench_$1.json 256,512,1024 > $O/bench_$1.log 2>&1
echo "bench $1 exit $?" >> $O/summary.txt
