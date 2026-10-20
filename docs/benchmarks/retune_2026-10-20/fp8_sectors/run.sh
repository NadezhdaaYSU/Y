#!/bin/bash
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
W=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/fp8sec
X=$SP/xbar
exec 9>"$SP/gpu.lock"; flock 9
cd $W || exit 9
for n in 1024 4096; do
  Y_LIB_PATH=$X/target/release/liby.so PYTHONPATH=$X/python ncu --cache-control none --section SourceCounters --metrics lts__t_sectors_srcunit_tex_op_read.sum -k regex:gemm_fp8 -c 1 -f -o $W/fp8_$n python3 $W/prof.py $X/tests/gemm_fp8_$n.ptx $n > $W/ncu_$n.out 2>&1
  ncu -i $W/fp8_$n.ncu-repz --page source --print-source sass --csv > $W/sass_$n.csv 2>&1
done
echo DONE > $W/summary.txt
