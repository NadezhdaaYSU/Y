#!/bin/bash
# ncu.sh: shared-memory bank conflicts, instruction counts and per-instruction
# stall samples for two Y tiles and cuBLAS at 512x512x1024, no cache flush.
W=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/ksweep
X=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/xbar
exec 9>"/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/gpu.lock"; flock 9
cd $W/prof || exit 9
{ date; for j in $(seq 5); do nvidia-smi --query-gpu=utilization.gpu,clocks.sm --format=csv,noheader; sleep 1; done; pgrep -af 'rungameid' | grep -v pgrep | cut -c1-100; } > $W/prof/env.txt 2>&1
M=l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_cmd_write.sum,sm__inst_executed_op_ldsm.sum,smsp__inst_executed.sum,gpu__time_duration.sum,lts__t_sector_hit_rate.pct,lts__t_sectors_srcunit_tex_op_read.sum
Y_LIB_PATH=$X/target/release/liby.so PYTHONPATH=$X/python ncu --cache-control none --metrics $M --section SourceCounters --section WarpStateStats --section LaunchStats -k 'regex:gemm_f16|s16816gemm' -c 6 -f -o $W/prof/p1024 python3 $W/prof/prof1024.py "tuned=$W/t/64_64_32_4_2_4,cublas_shape=$W/t/64_64_64_2_2_5" > $W/prof/ncu.out 2>&1
echo "ncu exit $?" >> $W/prof/summary.txt
ncu -i $W/prof/p1024.ncu-repz --page raw --csv --metrics $M > $W/prof/metrics.csv 2>&1
for i in 1 3 5; do ncu -i $W/prof/p1024.ncu-repz --page source --print-source sass --csv --launch-skip $i --launch-count 1 > $W/prof/sass_$i.csv 2>&1; done
echo "export done" >> $W/prof/summary.txt
