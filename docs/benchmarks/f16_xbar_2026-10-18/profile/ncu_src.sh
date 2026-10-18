#!/bin/bash
# Per-SASS-instruction stall samples for Y's gemm_f16_512 (committed PTX) and cuBLAS's 512^3 kernel.
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
G=$SP/g512; R=$G/regions; F=$SP/f16sq
cd $G || exit 9
Y_LIB_PATH=$SP/snap957/target/release/liby.so PYTHONPATH=$SP/snap957/python ncu --cache-control none \
  -k "regex:gemm_f16_512|s16816gemm" -c 4 \
  --section SourceCounters --section WarpStateStats --section InstructionStats --section LaunchStats \
  -f -o $R/src512 python3 $G/prof512.py $F $F/ptx/g512.ptx > $R/ncu.out 2>&1
echo "ncu exit $?" >> $R/summary.txt
