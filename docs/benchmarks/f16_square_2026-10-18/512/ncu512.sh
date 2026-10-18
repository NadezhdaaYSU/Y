#!/bin/bash
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
G=$SP/g512
F=$SP/f16sq
M=launch__grid_size,launch__block_size,launch__registers_per_thread,launch__shared_mem_per_block_dynamic,launch__shared_mem_per_block_static,gpu__time_duration.sum,smsp__inst_executed.sum,sm__warps_active.avg.pct_of_peak_sustained_active,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,launch__waves_per_multiprocessor
cd $G || exit 9
Y_LIB_PATH=$F/snap/target/release/liby.so PYTHONPATH=$F/snap/python ncu --cache-control none --metrics $M -f -o $G/p512 python3 $G/prof512.py $F $F/ptx/g512.ptx > $G/ncu.out 2>&1
echo "ncu exit $?" >> $G/summary.txt
ncu -i $G/p512.ncu-repz --page raw --csv > $G/p512.csv 2>&1
