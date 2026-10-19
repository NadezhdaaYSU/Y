#!/bin/bash
W=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/ksweep
X=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/xbar
exec 9>"/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/gpu.lock"; flock 9
cd "$W" || exit 9
for i in 1 2; do
  { echo "pad run $i $(date)"; for j in $(seq 10); do nvidia-smi --query-gpu=utilization.gpu,clocks.sm --format=csv,noheader; sleep 1; done; pgrep -af 'rungameid' | grep -v pgrep | cut -c1-100; } >> "$W/queue_pad.txt" 2>&1
  Y_LIB_PATH=$X/target/release/liby.so PYTHONPATH=$X/python python3 "$W/ktiles.py" "tuned=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/ksweep/t/64_64_32_4_2_4,tuned_pad0=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/ksweep/t/64_64_32_4_2_4_pad0,cublas_shape=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/ksweep/t/64_64_64_2_2_5,cublas_shape_pad0=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/ksweep/t/64_64_64_2_2_5_pad0" "$W/pad_$i.json" 512,1024,4096 > "$W/pad_$i.log" 2>&1
  echo "pad run $i exit $?" >> "$W/summary.txt"
done
MM=lts__t_sectors_srcunit_tex_op_read.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,gpu__time_duration.sum
cd $W/prof && Y_LIB_PATH=$X/target/release/liby.so PYTHONPATH=$X/python ncu --cache-control none --metrics $MM -k 'regex:gemm_f16' -f -o $W/prof/pad0 python3 $W/prof/prof1024.py "tuned_pad0=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/ksweep/t/64_64_32_4_2_4_pad0,cublas_shape_pad0=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/ksweep/t/64_64_64_2_2_5_pad0" > $W/prof/pad0.out 2>&1
ncu -i $W/prof/pad0.ncu-repz --page raw --csv --metrics $MM > $W/prof/pad0_metrics.csv 2>&1
echo "pad ncu exit $?" >> "$W/summary.txt"
