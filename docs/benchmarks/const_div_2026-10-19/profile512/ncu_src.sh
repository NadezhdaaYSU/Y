#!/bin/bash
# ncu_src.sh: SourceCounters (per-instruction SASS execution counts) for Y's
# committed gemm_f16_512 and cuBLAS's torch.mm at 512^3.
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
G=$SP/g512s; F=$SP/f16sq
cd $G || exit 9
{ date; nvidia-smi --query-gpu=utilization.gpu,clocks.sm --format=csv,noheader; "$SP/games.sh"; } > $G/env.txt
Y_LIB_PATH=$SP/xbar/target/release/liby.so PYTHONPATH=$SP/xbar/python ncu --cache-control none --section SourceCounters --section LaunchStats --section ComputeWorkloadAnalysis --section WarpStateStats -k 'regex:gemm_f16_512|s16816gemm' -c 2 -f -o $G/src512 python3 $G/prof512.py $F $G/gemm_f16_512.ptx > $G/ncu.out 2>&1
echo "ncu exit $?" >> $G/summary.txt
for i in 0 1; do ncu -i $G/src512.ncu-rep --page source --print-source sass --csv --launch-skip $i --launch-count 1 > $G/sass_$i.csv 2>&1; done
ncu -i $G/src512.ncu-rep --page details --csv > $G/details.csv 2>&1
echo "export done" >> $G/summary.txt
