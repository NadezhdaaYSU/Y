#!/bin/bash
# bench.sh <tag> <label=dir,...>: one bench_ab run at 512 with the 957899d snapshot.
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
P=$SP/g512/pipe
cd $P || exit 9
Y_LIB_PATH=$SP/snap957/target/release/liby.so PYTHONPATH=$SP/snap957/python python3 $SP/f16sq/bench_ab.py "$2" $P/bench_$1.json 512 > $P/bench_$1.log 2>&1
echo "bench $1 exit $?" >> $P/summary.txt
