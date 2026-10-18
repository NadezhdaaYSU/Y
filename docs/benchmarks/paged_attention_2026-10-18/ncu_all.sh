#!/bin/bash
# Instruction counts (Nsight Compute, no cache flush) of the old and new
# split kernels and FlashInfer on two README shapes. Counts only: ncu's
# durations are of replayed passes and are not used.
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
B=$SP/bench_attn
export PYTHONPATH=$SP/fishim/y_flashinfer_shim Y_TESTS=$SP/snap_new/tests
cd $B/ptx || exit 9
M=smsp__inst_executed.sum,launch__registers_per_thread,launch__grid_size,launch__block_size,gpu__time_duration.sum
run() { # label ptx splits warps batch ctx
  ncu --cache-control none -k "regex:paged_decode|BatchDecode|MergeStates" --metrics $M -c 6 -f -o $B/ncu_$1 python3 $B/prof_attn.py $2 $3 --warps $4 $5 $6 --fi --check --iters 1 > $B/ncu_$1.out 2>&1
  echo "$1 exit $?" >> $B/ncu_summary.txt
  python3 $B/ncu_tab.py $B/ncu_$1.ncu-repz smsp__inst_executed.sum launch__registers_per_thread launch__grid_size launch__block_size > $B/ncu_$1.txt 2>&1
}
run old_b1c4096 old_16_8.ptx 16 8 1 4096
run new_b1c4096 new_16_8.ptx 16 8 1 4096
run old_b8c1024 old_4_8.ptx 4 8 8 1024
run new48_b8c1024 new_4_8.ptx 4 8 8 1024
run new44_b8c1024 new_4_4.ptx 4 4 8 1024
date >> $B/ncu_summary.txt
